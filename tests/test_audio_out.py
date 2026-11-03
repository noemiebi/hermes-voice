"""0.7.0 출력 오디오: 24 kHz 기본 레이트·μ-law 부호화·세션 시간 계산(aec_ref·_sched·pos)·송신 진단."""
import asyncio
import json
import logging
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_session_hold import FakeHermes, FakeSTT, FakeWS  # noqa: E402

from server import audio_out, config  # noqa: E402
from server.aec_gate import ReferenceBuffer, resample_44k_to_16k, resample_to_16k  # noqa: E402
from server.session import VoiceSession  # noqa: E402
from server.tts import SAMPLE_RATE, TTSEngine, resample_to  # noqa: E402
from server.tts_qwen import QwenPostProcessor  # noqa: E402


def tone(sec, sr, f=220.0, amp=0.3):
    t = np.arange(int(sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * f * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def snr_db(ref, y):
    ref = ref.astype(np.float64)
    err = y.astype(np.float64) - ref
    return 10 * np.log10(np.sum(ref ** 2) / max(np.sum(err ** 2), 1e-12))


# ---------------------------------------------------------------- 설정·리샘플
def test_default_out_rate_and_codec():
    assert config.AUDIO_OUT_RATE in (24000, 44100) and config.AUDIO_OUT_CODEC in audio_out.CODECS


def test_resample_lengths():
    x = tone(1.0, 44100)
    assert resample_to(x, 44100, 24000).size == 24000
    assert resample_to(x, 44100, 44100).size == 44100
    assert resample_to(tone(0.37, 44100), 44100, 24000).size == int(np.ceil(int(0.37 * 44100) * 80 / 147))
    assert resample_to_16k(np.zeros(24000, np.int16), 24000).size == 16000
    assert abs(resample_44k_to_16k(np.zeros(44100, np.int16)).size - 16000) <= 1


def test_resample_keeps_tone():
    x = tone(1.0, 44100, f=440, amp=0.5)
    y = resample_to(x, 44100, 24000)
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    assert abs(np.argmax(spec) * 24000 / y.size - 440) < 3
    assert abs(np.sqrt(np.mean(y[2000:-2000] ** 2)) / np.sqrt(np.mean(x[4000:-4000] ** 2)) - 1) < 0.05


def test_supertonic_synth_outputs_out_rate():
    """엔진 출력 44.1 kHz -> shape_sentence 뒤 out_rate로(길이 비율 24000/44100)."""
    wav = np.concatenate([np.zeros(4410, np.float32), tone(1.0, SAMPLE_RATE), np.zeros(4410, np.float32)])
    lens = {}
    for rate in (24000, 44100):
        eng = TTSEngine(out_rate=rate)
        eng.tts = lambda *a, w=wav: (w[None, :], None)
        pcm = eng.synth_sync("가")
        assert pcm.dtype == np.int16 and eng.out_rate == rate
        lens[rate] = pcm.size
    assert abs(lens[24000] / lens[44100] - 24000 / 44100) < 0.002


def test_qwen_post_processor_out_rate():
    """Qwen 워커 24 kHz: out_rate 24000이면 리샘플 없이 통과, 44100이면 147/80배."""
    sr = 24000
    x = np.concatenate([np.zeros(sr // 4, np.float32), tone(1.0, sr), np.zeros(sr // 4, np.float32)])
    out = {}
    for rate in (24000, 44100):
        pp = QwenPostProcessor(sr, gain_db=0.0, out_rate=rate)
        assert pp.out_rate == rate
        y = np.concatenate([pp.push(c) for c in np.array_split(x, 9)] + [pp.flush()])
        out[rate] = y
        tail = int(rate * config.QWEN_TTS_TAIL_MS / 1000)
        assert abs(y.size - rate * 1.0 - tail) < rate * 0.15, (rate, y.size)
    assert abs(out[44100].size / out[24000].size - 44100 / 24000) < 0.01


def test_reference_buffer_24k():
    ref = ReferenceBuffer(sr_in=24000)
    ref.add(1, (tone(1.0, 24000) * 32767).astype(np.int16))
    assert ref.samples == 16000 and abs(ref.duration_ms - 1000) < 1e-6
    assert ReferenceBuffer().sr_in == config.AUDIO_OUT_RATE


# ---------------------------------------------------------------- μ-law
def test_mulaw_known_values_and_table():
    enc = audio_out.mulaw_encode(np.array([0, -1, 32767, -32768], np.int16))
    assert list(enc) == [0xFF, 0x7F, 0x80, 0x00]
    t = audio_out.MULAW_TABLE
    assert t.size == 256 and t[0] == -32124 and t[128] == 32124 and t[255] == 0 and t[127] == 0
    # 표의 모든 값은 다시 부호화하면 같은 코드(0x7f/0xff 두 0은 0xff로)
    codes = audio_out.mulaw_encode(t)
    assert np.array_equal(t[codes], t)


def test_mulaw_round_trip_snr():
    for amp in (0.05, 0.3, 0.9):
        x = (tone(1.0, 24000, f=330, amp=amp) * 32767).astype(np.int16)
        y = audio_out.mulaw_decode(audio_out.mulaw_encode(x))
        assert y.dtype == np.int16 and y.size == x.size
        assert snr_db(x, y) > 30, (amp, snr_db(x, y))
    rng = np.random.default_rng(0)
    x = np.clip(rng.standard_normal(24000) * 3000, -32768, 32767).astype(np.int16)
    assert snr_db(x, audio_out.mulaw_decode(audio_out.mulaw_encode(x).tobytes())) > 30


def test_encode_body_sizes():
    x = np.arange(-100, 100, dtype=np.int16)
    assert audio_out.encode(x, "pcm16") == x.astype("<i2").tobytes()
    assert len(audio_out.encode(x, "mulaw")) == x.size


# ---------------------------------------------------------------- 세션
class ToneTTS:
    loaded = True

    def __init__(self, rate):
        self.out_rate = rate

    async def synth(self, text, is_current=lambda: True):
        await asyncio.sleep(0.005)
        return ((tone(1.0, self.out_rate) * 32767).astype(np.int16), 1) if is_current() else None  # 1초


async def speak_one(rate, codec=None):
    ws = FakeWS()
    s = VoiceSession(ws, FakeSTT(), ToneTTS(rate), FakeHermes())
    if codec:
        s.out_codec = codec
    s.style_note = ""
    await s.send_ready()
    s.generation = gen = 1
    s.speaker_task = asyncio.create_task(s._speaker())
    s.queue.put_nowait((gen, "첫 문장이에요."))
    for _ in range(100):
        await asyncio.sleep(0.01)
        if len(ws.audio) >= 2 and not s._speaking_now:
            break
    t0 = s.clock()
    await s._on_text(json.dumps({"type": "playback", "state": "speaking", "generation": gen, "pos_ms": 250}))
    est = s._est_play_end(gen) - t0
    s.speaker_task.cancel()
    return s, ws, gen, est


def test_session_24k_ready_frames_ref_and_pos():
    s, ws, gen, est = asyncio.run(speak_one(24000))
    ready = ws.of("ready")[0]
    assert ready["out_sample_rate"] == 24000 and ready["out_codec"] == s.out_codec
    assert s.out_rate == 24000 and s.chunk_samples == 12000
    frames = [b for b in ws.audio if struct.unpack("<I", b[:4])[0] == gen]
    bps = 1 if s.out_codec == "mulaw" else 2
    assert [len(b) for b in frames] == [4 + 12000 * bps] * 2      # 0.5초 청크 2개
    assert abs(s.aec_ref.duration_ms - 1000) < 1                  # 참조 = 보낸 길이(24k -> 16k)
    assert s.aec_ref.sr_in == 24000
    assert len(s._sched) == 1 and abs(s._sched[0][1] - 1.0) < 1e-9
    assert abs(est - 0.75) < 0.05                                 # pos 250 ms 보고 -> 남은 0.75초
    assert s._tx["gen"] == gen and s._tx["bytes"] == sum(len(b) for b in frames)


def test_session_follows_engine_rate_44k():
    s, ws, gen, est = asyncio.run(speak_one(44100, codec="pcm16"))
    assert ws.of("ready")[0]["out_sample_rate"] == 44100 and s.chunk_samples == 22050
    assert [len(b) for b in ws.audio] == [4 + 22050 * 2] * 2
    assert abs(s.aec_ref.duration_ms - 1000) < 1 and abs(s._sched[0][1] - 1.0) < 1e-9
    assert abs(est - 0.75) < 0.05


def test_session_mulaw_frames_decode():
    s, ws, gen, _ = asyncio.run(speak_one(24000, codec="mulaw"))
    assert [len(b) for b in ws.audio] == [4 + 12000] * 2
    y = np.concatenate([audio_out.mulaw_decode(b[4:]) for b in ws.audio])
    x = (tone(1.0, 24000) * 32767).astype(np.int16)
    assert snr_db(x, y) > 30


def test_send_slow_and_lag_logs(caplog):
    class SlowWS(FakeWS):
        async def send_bytes(self, b):
            await asyncio.sleep(0.08)
            self.audio.append(b)

    async def main():
        ws = SlowWS()
        s = VoiceSession(ws, FakeSTT(), ToneTTS(24000), FakeHermes())
        await s.send_audio(3, np.zeros(12000, np.int16))
        s._sched_gen, s._sched = 3, [(s.clock() - 5.0, 5.0)]
        s._check_lag(3, 0.0)
        return s

    with caplog.at_level(logging.INFO, logger="session"):
        s = asyncio.run(main())
    text = caplog.text
    assert "송신 지연" in text
    assert "재생 지연" in text and "첫 전송 뒤" in text and "KB" in text
    assert s._tx["bytes"] == 4 + (12000 if s.out_codec == "mulaw" else 24000)
