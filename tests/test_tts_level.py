"""TTS 라우드니스 정규화."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.tts import SAMPLE_RATE, normalize_loudness, speech_rms_db  # noqa: E402


def _speechy(rms_db: float, sec: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(int(sec * SAMPLE_RATE)).astype(np.float32)
    a *= 10 ** (rms_db / 20) / np.sqrt(np.mean(a ** 2))
    return np.concatenate([np.zeros(SAMPLE_RATE // 2, np.float32), a, np.zeros(SAMPLE_RATE // 2, np.float32)])


def test_quiet_speech_raised_to_target():
    w = _speechy(-25) * 0.5  # 피크 여유가 있도록 크레스트 낮은 신호를 더 작게
    out, gain = normalize_loudness(w, target_db=-18, peak_db=-1)
    assert gain > 0
    assert abs(speech_rms_db(out) - (-18)) < 0.3 or np.max(np.abs(out)) >= 10 ** (-1 / 20) - 1e-4


def test_peak_limited():
    w = _speechy(-24)
    w[SAMPLE_RATE] = 0.9  # 튀는 피크
    out, _ = normalize_loudness(w, target_db=-10, peak_db=-1)
    assert np.max(np.abs(out)) <= 10 ** (-1 / 20) + 1e-6


def test_silence_untouched():
    w = np.zeros(SAMPLE_RATE, np.float32)
    out, gain = normalize_loudness(w)
    assert gain == 0.0 and not out.any()
