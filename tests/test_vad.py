"""Silero VAD 래퍼·말 시작/끝 판정 단위 시험(합성 신호 + TTS로 만든 음성)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.vad import SAMPLE_RATE, WINDOW, Endpointer, SileroVAD, VADParams  # noqa: E402

SPEECH_WAV = Path(__file__).resolve().parent / "out" / "smoke_ko.wav"


def pcm(a: np.ndarray) -> bytes:
    return np.clip(a * 32767, -32768, 32767).astype("<i2").tobytes()


def silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SAMPLE_RATE), dtype=np.float32)


def noise(sec: float, dbfs: float = -40.0, seed: int = 0) -> np.ndarray:
    rms = 10 ** (dbfs / 20)
    return (np.random.default_rng(seed).standard_normal(int(sec * SAMPLE_RATE)) * rms).astype(np.float32)


def sine(sec: float, hz: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sec * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    if not SPEECH_WAV.exists():
        pytest.skip("tests/out/smoke_ko.wav 없음 (tests/smoke_tts.py로 생성)")
    import soundfile as sf
    a, sr = sf.read(SPEECH_WAV, dtype="float32")
    assert sr == SAMPLE_RATE
    # TTS 앞뒤 무음을 잘라 실제 음성 구간만
    e = np.array([np.sqrt(np.mean(a[i:i + WINDOW] ** 2)) for i in range(0, len(a), WINDOW)])
    idx = np.where(e > 0.01)[0]
    return a[idx[0] * WINDOW: (idx[-1] + 1) * WINDOW]


def feed_chunks(ep: Endpointer, a: np.ndarray, chunk_ms: int = 100):
    step = SAMPLE_RATE * chunk_ms // 1000
    out = []
    for i in range(0, len(a), step):
        for ev in ep.feed(pcm(a[i:i + step])):
            out.append((i + step, ev))
    return out


def test_window_probs_silence_and_noise():
    v = SileroVAD()
    probs = v.feed(pcm(np.concatenate([silence(1.0), noise(2.0)])))
    assert len(probs) == int(3.0 * SAMPLE_RATE) // WINDOW
    assert max(probs) < 0.3, max(probs)


def test_sine_is_not_speech():
    v = SileroVAD()
    probs = v.feed(pcm(sine(2.0)))
    assert np.mean(probs) < 0.3, np.mean(probs)


def test_speech_probs(speech):
    v = SileroVAD()
    probs = np.array(v.feed(pcm(speech)))
    assert (probs > 0.5).mean() > 0.4, (probs > 0.5).mean()


def test_reset_restores_state(speech):
    v = SileroVAD()
    a = v.feed(pcm(speech[:16000]))
    v.reset()
    b = v.feed(pcm(speech[:16000]))
    assert np.allclose(a, b)


def test_endpointer_start_end_with_preroll(speech):
    p = VADParams()
    ep = Endpointer(p)
    sig = np.concatenate([silence(1.0), speech, silence(1.5)])
    evs = feed_chunks(ep, sig)
    kinds = [e.kind for _, e in evs]
    assert kinds == ["start", "end"], kinds
    end = evs[1][1]
    # 프리롤 300 ms + 발화 + 끝 침묵 200 ms 정도
    dur = len(end.pcm) / 2 / SAMPLE_RATE
    sp = len(speech) / SAMPLE_RATE
    assert sp + 0.3 < dur < sp + 0.8, (dur, sp)
    assert end.speech_ms >= 1000
    # 말 끝 판정 위치: 음성 끝 뒤 end_silence_ms 근처(±300 ms)
    end_at = evs[1][0] / SAMPLE_RATE - 1.0 - len(speech) / SAMPLE_RATE
    assert p.end_silence_ms / 1000 - 0.1 < end_at < p.end_silence_ms / 1000 + 0.3, end_at


def test_endpointer_noise_only_no_events():
    ep = Endpointer(VADParams())
    sig = np.concatenate([silence(2.0), noise(8.0, -40, seed=1)])
    assert feed_chunks(ep, sig) == []


def test_short_blip_discarded(speech):
    p = VADParams(min_speech_ms=300)
    ep = Endpointer(p)
    blip = speech[int(0.3 * SAMPLE_RATE): int(0.55 * SAMPLE_RATE)]  # 250 ms
    evs = feed_chunks(ep, np.concatenate([silence(1.0), blip, silence(1.5)]))
    assert all(e.kind != "end" for _, e in evs), [e.kind for _, e in evs]


def test_barge_threshold_stricter(speech):
    # 시작 기준이 엄격하면 시작 판정이 늦어진다
    normal = Endpointer(VADParams())
    strict = Endpointer(VADParams(), strict=lambda: True)
    sig = np.concatenate([silence(0.5), speech, silence(1.0)])
    t_n = next(i for i, e in feed_chunks(normal, sig, 32) if e.kind == "start")
    evs = feed_chunks(strict, sig, 32)
    t_s = next(i for i, e in evs if e.kind == "start")
    assert next(e for _, e in evs if e.kind == "start").barge
    assert t_s > t_n


class _FakeTurn:
    def __init__(self, complete):
        self.complete, self.calls = complete, 0

    def is_complete(self, pcm):
        self.calls += 1
        return self.complete, 0.9 if self.complete else 0.1, 1


@pytest.mark.parametrize("complete", [True, False])
def test_smart_turn_shortens_end(speech, complete):
    p = VADParams(turn_detector="smart-turn", smart_turn_silence_ms=300, end_silence_ms=700)
    tm = _FakeTurn(complete)
    ep = Endpointer(p, turn_model=tm)
    sig = np.concatenate([silence(0.5), speech, silence(1.2)])
    evs = feed_chunks(ep, sig, 32)
    end_i, end = next((i, e) for i, e in evs if e.kind == "end")
    lag = end_i / SAMPLE_RATE - 0.5 - len(speech) / SAMPLE_RATE
    assert tm.calls == 1
    if complete:
        assert lag < 0.55 and end.smart[0], lag
    else:
        assert lag > 0.6, lag


def test_smart_turn_model_runs():
    from server.smart_turn import DEFAULT_MODEL, SmartTurn
    if not Path(DEFAULT_MODEL).exists():
        pytest.skip("models/smart-turn-v3.2-cpu.onnx 없음")
    ok, prob, ms = SmartTurn().is_complete(pcm(sine(1.0)))
    assert 0.0 <= prob <= 1.0 and ms < 500


class _ScriptVAD:
    """창마다 정해 둔 확률을 돌려주는 가짜 VAD."""

    def __init__(self, probs):
        self.probs, self.i = list(probs), 0

    def reset(self):
        pass

    def prob(self, w):
        p = self.probs[self.i] if self.i < len(self.probs) else 0.0
        self.i += 1
        return p


def _run_script(probs, **kw):
    ep = Endpointer(VADParams(), vad=_ScriptVAD(probs), **kw)
    evs = []
    for k in range(len(probs)):
        for ev in ep.feed(b"\0\0" * WINDOW):
            evs.append((k, ev))
    return evs


def test_tail_guard_raises_start_threshold():
    # 확률 0.55가 1초: 평소(0.5)엔 말 시작, 꼬리 가드(0.6)에선 무시
    probs = [0.0] * 10 + [0.55] * 31 + [0.0] * 40
    assert [e.kind for _, e in _run_script(probs)][:1] == ["start"]
    assert _run_script(probs, tail=lambda: True) == []


def test_tail_guard_uses_normal_start_ms():
    # 0.65가 이어지면 꼬리 가드에서도 start_ms(160 ms = 5창) 만에 시작, barge 아님
    probs = [0.0] * 10 + [0.65] * 20 + [0.0] * 40
    evs = _run_script(probs, tail=lambda: True)
    k, start = evs[0]
    assert start.kind == "start" and start.tail and not start.barge
    assert k == 10 + 5 - 1
    # strict가 우선: 0.65는 barge 기준(0.7) 미달
    assert _run_script(probs, tail=lambda: True, strict=lambda: True) == []


def test_tail_guard_setter_and_speech_ms():
    probs = [0.0] * 5 + [0.9] * 20 + [0.0] * 40
    ep = Endpointer(VADParams(), vad=_ScriptVAD(probs))
    ep.set_tail(lambda: True)
    seen = []
    for _ in range(15):
        for ev in ep.feed(b"\0\0" * WINDOW):
            seen.append(ev)
    assert seen and seen[0].tail
    assert ep.in_speech and ep.speech_ms == 10 * 32
