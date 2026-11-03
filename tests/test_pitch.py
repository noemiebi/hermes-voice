"""음높이(F0) 판별(server/pitch.py)과 세션 연결 시험."""
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_duck_hold import Clock, feed  # noqa: E402
from test_session_hold import FakeEP, FakeHermes, FakeSTT, FakeTTS, FakeWS, settle  # noqa: E402

from server import config, pitch  # noqa: E402
from server.session import VoiceSession  # noqa: E402
from server.vad import VADEvent  # noqa: E402

SR = 16000


def tone(f0: float, ms: int, db: float = -20.0, phase0: float = 0.0, vib: float = 0.0) -> np.ndarray:
    """기본파 + 2·3배음(진폭 1, 0.6, 0.4) 합성 톤(float). vib: 5 Hz 비브라토 깊이(비율)."""
    t = np.arange(int(SR * ms / 1000)) / SR + phase0
    f = f0 * (1 + vib * np.sin(2 * np.pi * 5 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) + 0.6 * np.sin(2 * ph) + 0.4 * np.sin(3 * ph)
    x = x / np.sqrt(np.mean(x * x)) * 10 ** (db / 20)
    return x


def pcm(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


# ---------- 추정기 ----------
@pytest.mark.parametrize("f0", [110.0, 220.0, 95.0, 250.0, 75.0, 380.0])
def test_f0_harmonic_tone_no_octave_error(f0):
    r = pitch.f0_track(tone(f0, 600))
    assert r.size >= 50
    assert abs(np.median(r) / f0 - 1) < 0.05
    assert np.all(np.abs(r / f0 - 1) < 0.05)   # 한 창도 옥타브 튐 없음


def test_f0_strong_second_harmonic_stays_low():
    # 2배음이 기본파보다 센 소리(모음에서 흔함): 220으로 올라가면 안 됨
    t = np.arange(SR) / SR
    x = 0.4 * np.sin(2 * np.pi * 110 * t) + 0.8 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 330 * t)
    r = pitch.f0_track(x * 0.2)
    assert abs(np.median(r) / 110 - 1) < 0.05


def test_f0_int16_and_44k_input():
    x = tone(220, 500)
    assert abs(np.median(pitch.f0_track((x * 32767).astype(np.int16))) / 220 - 1) < 0.05
    t = np.arange(int(44100 * 0.5)) / 44100
    y = (np.sin(2 * np.pi * 110 * t) + 0.6 * np.sin(2 * np.pi * 220 * t)) * 0.1
    assert abs(np.median(pitch.f0_track(pitch.resample(y, 44100))) / 110 - 1) < 0.05
    assert abs(np.median(pitch.f0_track(y, sr=44100)) / 110 - 1) < 0.05


def test_f0_skips_noise_and_silence():
    rng = np.random.default_rng(0)
    assert pitch.f0_track(np.zeros(SR)).size == 0
    assert pitch.f0_track(rng.standard_normal(SR) * 10 ** (-50 / 20)).size == 0   # -50 dBFS: 잡음 창 건너뜀
    loud_noise = pitch.f0_track(rng.standard_normal(SR) * 0.1)
    assert loud_noise.size < 10   # 큰 백색 잡음도 상관 0.5 미만이라 거의 없음


# ---------- 분류 ----------
def profile(user_hz=109.0, tts_hz=218.0, n_user=8, mad=0.08, key="supertonic:F1.json"):
    rng = np.random.default_rng(1)
    u = pitch.Dist([[float(np.log(user_hz) + rng.normal(0, 0.05)), mad] for _ in range(n_user)])
    t = pitch.Dist([[float(np.log(tts_hz) + rng.normal(0, 0.03)), 0.06] for _ in range(4)])
    return pitch.VoiceProfile(u, {key: t})


def frames(hz, n=30):
    return np.full(n, float(hz))


def test_classify_separated():
    p = profile()
    assert p.classify(frames(110))[0] == "user"
    assert p.classify(frames(225))[0] == "tts"
    lab, sc = p.classify(frames(109))
    assert lab == "user" and 0.5 < sc <= 1.0
    assert p.classify(frames(160))[0] == "unknown"   # 두 구간 사이
    assert p.classify(frames(390))[0] == "unknown"   # 어느 쪽도 아님


def test_classify_overlap_is_unknown():
    p = profile(user_hz=205.0)   # 사용자가 TTS와 비슷한 음역
    assert p.classify(frames(212))[0] == "unknown"


def test_classify_too_few_frames_or_untrained():
    p = profile()
    assert p.classify(frames(110, n=config.PITCH_MIN_FRAMES - 1))[0] == "unknown"
    assert profile(n_user=config.PITCH_USER_MIN_UTTS - 1).classify(frames(110))[0] == "unknown"
    q = profile()
    q.tts = {}
    assert q.classify(frames(220))[0] == "unknown"
    q = profile()
    q.active_tts = ["qwen:Sohee"]   # 지금 음색은 아직 안 잼
    assert not q.ready() and q.classify(frames(220))[0] == "unknown"


def test_min_width_interval():
    d = pitch.Dist([[float(np.log(200)), 0.0]] * 4)   # 퍼짐 0이어도 ±15%
    lo, hi = d.range_hz()
    assert abs(lo / 200 - 1 / 1.15) < 1e-6 and abs(hi / 200 - 1.15) < 1e-6


def test_user_learning_and_cap():
    p = pitch.VoiceProfile()
    assert not p.add_user(np.full(5, 110.0))       # 창 부족
    for _ in range(pitch.MAX_UTTS + 20):
        assert p.add_user(pitch.f0_track(tone(110, 300, vib=0.05)))
    assert p.user.n == pitch.MAX_UTTS
    assert abs(p.user.center_hz() / 110 - 1) < 0.05


# ---------- 저장 ----------
def test_store_roundtrip_and_throttle(tmp_path):
    clk = Clock()
    path = tmp_path / "voice_profile.json"
    st = pitch.VoiceProfileStore(path, interval_s=60, clock=clk)
    assert st.profile.user.n == 0
    st.profile.add_user(np.full(30, 110.0))
    st.profile.set_tts("supertonic:F1.json", [np.full(30, 220.0)] * 4)
    assert st.save() and path.exists()
    st.profile.add_user(np.full(30, 112.0))
    clk.t += 30
    assert not st.save() and st.dirty                    # 60초 안: 미룸
    assert json.loads(path.read_text())["user"].__len__() == 1
    clk.t += 31
    assert st.save() and not st.dirty
    st2 = pitch.VoiceProfileStore(path)
    assert st2.profile.user.n == 2 and st2.profile.tts["supertonic:F1.json"].n == 4
    assert abs(st2.profile.tts["supertonic:F1.json"].center_hz() - 220) < 0.01
    assert not list(tmp_path.glob(".voice_profile-*"))   # 임시 파일 안 남음


def test_store_broken_file(tmp_path):
    path = tmp_path / "voice_profile.json"
    path.write_text("{broken")
    assert pitch.VoiceProfileStore(path).profile.user.n == 0


def test_tts_keys_and_learn():
    class ST:
        loaded, voice = True, "/x/voice_styles/F1.json"

        async def synth(self, text, is_current=lambda: True):
            t = np.arange(44100) / 44100
            x = (np.sin(2 * np.pi * 218 * t) + 0.5 * np.sin(2 * np.pi * 436 * t)) * 0.2
            return (x * 32767).astype(np.int16), 1

    class Router:
        engine, qwen = "supertonic", None

        def __init__(self):
            self.supertonic = ST()

    st = pitch.VoiceProfileStore(None)
    r = Router()
    assert pitch.tts_keys(r) == [("supertonic:F1.json", "supertonic")]
    assert asyncio.run(pitch.learn_tts(r, st)) == ["supertonic:F1.json"]
    assert abs(st.profile.tts["supertonic:F1.json"].center_hz() / 218 - 1) < 0.03
    assert asyncio.run(pitch.learn_tts(r, st)) == []    # 이미 잰 키는 건너뜀


# ---------- 세션 연결 ----------
def learned_store():
    st = pitch.VoiceProfileStore(None)
    st.profile = profile()
    return st


async def setup_pitch(store):
    ws, stt, hermes = FakeWS(), FakeSTT(), FakeHermes()
    s = VoiceSession(ws, stt, FakeTTS(), hermes, pitch_store=store)
    s.pitch_gate = True
    s.clock = Clock()
    s.endpointer = FakeEP()
    s.hold_mode = "duck"
    s.speaker_task = asyncio.create_task(s._speaker())
    await s._on_text(json.dumps({"type": "hello", "resume": None}))
    await s._on_text(json.dumps({"type": "conversation", "on": True}))
    await s._on_text(json.dumps({"type": "text", "text": "서울 날씨 어때?"}))
    await settle()
    gen = s.generation
    await s._on_text(json.dumps({"type": "playback", "state": "speaking", "generation": gen}))
    return s, ws, stt, gen


async def feed_tone(s, f0, ms, start_ms=0):
    """96 ms 조각으로 톤을 보내며 시계를 진행(위상 연속)."""
    for i in range(ms // 96):
        s.clock.t += 0.096
        await s._on_audio(pcm(tone(f0, 96, phase0=(start_ms + i * 96) / 1000, vib=0.03)))


async def start(s, speech_ms=416):
    s.endpointer.in_speech, s.endpointer.speech_ms = True, speech_ms
    await s._on_vad(VADEvent("start", barge=True))


def test_session_tts_pitch_skips_duck():
    async def go():
        s, ws, stt, gen = await setup_pitch(learned_store())
        await feed(s, -60, 288)
        await feed_tone(s, 220, 480)
        await start(s)
        assert not ws.of("hold") and not ws.of("interrupt")
        assert s._echo_check and s._pitch_tts is not None and s.state == "speaking"
        await feed_tone(s, 220, 384, 480)              # 900 ms 전: 재검사 없음
        await feed_tone(s, 220, 576, 864)              # 재검사: 여전히 TTS 음역 -> 유지
        assert s._pitch_tts["rechecked"] and not ws.of("interrupt") and not ws.of("hold")
        s.endpointer.in_speech = False
        await s._on_vad(VADEvent("end", pcm=pcm(tone(220, 900)), speech_ms=900))
        await settle()
        assert stt.calls == 1 and not ws.of("interrupt")   # 말 끝 텍스트 대조(빈 결과 -> 재생 계속)
        s.speaker_task.cancel()
    asyncio.run(go())


def test_session_user_pitch_interrupts_without_duck():
    async def go():
        s, ws, stt, gen = await setup_pitch(learned_store())
        await feed_tone(s, 110, 480)
        await start(s)
        assert ws.of("interrupt") == [{"type": "interrupt", "generation": gen + 1}]
        assert not ws.of("hold") and s.state == "listening"
        s.speaker_task.cancel()
    asyncio.run(go())


def test_session_waits_for_onset_then_decides():
    async def go():
        s, ws, stt, gen = await setup_pitch(learned_store())
        await feed_tone(s, 110, 192)
        await start(s, speech_ms=128)                    # 앞부분 128 ms뿐: 판정 대기, 재생 그대로
        assert s._pitch_wait is not None and not ws.of("interrupt") and not ws.of("hold")
        assert s.state == "speaking"
        await feed_tone(s, 110, 192, 192)                # 128+192 < 400: 아직
        assert s._pitch_wait is not None
        await feed_tone(s, 110, 96, 384)                 # 416 ms 모임 -> 판정
        assert s._pitch_wait is None and ws.of("interrupt") and not ws.of("hold")
        s.speaker_task.cancel()
    asyncio.run(go())


def test_session_tts_then_user_recheck_interrupts():
    async def go():
        s, ws, stt, gen = await setup_pitch(learned_store())
        await feed_tone(s, 220, 480)
        await start(s)
        assert s._pitch_tts is not None
        await feed_tone(s, 110, 1056, 480)               # 사용자가 이어서 말함 -> 900 ms 넘어 재검사
        assert ws.of("interrupt") == [{"type": "interrupt", "generation": gen + 1}]
        assert not ws.of("hold")
        s.speaker_task.cancel()
    asyncio.run(go())


def test_session_untrained_profile_uses_duck():
    async def go():
        s, ws, stt, gen = await setup_pitch(pitch.VoiceProfileStore(None))
        await feed_tone(s, 110, 480)
        await start(s)
        assert ws.of("hold") and ws.of("hold")[0]["mode"] == "duck" and not ws.of("interrupt")
        s.speaker_task.cancel()
    asyncio.run(go())


def test_session_unknown_pitch_uses_duck_and_gate_off():
    async def go():
        s, ws, stt, gen = await setup_pitch(learned_store())
        await feed_tone(s, 160, 480)                     # 두 음역 사이 -> 기존 덕킹
        await start(s)
        assert ws.of("hold") and not ws.of("interrupt")
        s.speaker_task.cancel()

        s, ws, stt, gen = await setup_pitch(learned_store())
        s.pitch_gate = False
        await feed_tone(s, 110, 480)
        await start(s)
        assert ws.of("hold") and not ws.of("interrupt")
        s.speaker_task.cancel()
    asyncio.run(go())


def test_session_learns_user_from_turn():
    async def go():
        st = pitch.VoiceProfileStore(None)
        s, ws, stt, gen = await setup_pitch(st)
        await s._on_text(json.dumps({"type": "playback", "state": "idle", "generation": gen}))
        s._speaking_now = False
        stt.text = "내일 일정 알려줘"
        s._start_turn(pcm=pcm(tone(110, 1000, vib=0.05)))
        await settle()
        for _ in range(50):
            if st.profile.user.n:
                break
            await asyncio.sleep(0.01)
        assert st.profile.user.n == 1 and abs(st.profile.user.center_hz() / 110 - 1) < 0.05
        s._start_turn(pcm=pcm(tone(220, 1000)), echo_check=True)   # 재생과 겹친 발화는 학습 안 함
        await settle()
        assert st.profile.user.n == 1
        s.speaker_task.cancel()
    asyncio.run(go())
