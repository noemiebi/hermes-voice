"""TTS로 만든 한국어 wav(16k)를 STT에 넣어 왕복 확인. 먼저 smoke_tts.py 실행."""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.stt import STTEngine, is_hallucination  # noqa: E402

WAV = Path(__file__).resolve().parent / "out" / "smoke_ko.wav"
EXPECT = "오늘 서울 날씨가 어떤지 알려 줄래?"


def strip(s):
    return "".join(ch for ch in s if ch.isalnum())


if __name__ == "__main__":
    pcm, sr = sf.read(WAV, dtype="int16")
    assert sr == 16000
    eng = STTEngine()
    t = time.perf_counter()
    eng.load_sync()
    assert eng.loaded, eng.error
    print(f"load+warmup {int((time.perf_counter() - t) * 1000)} ms")
    for _ in range(3):
        text, ms = eng.transcribe_sync(pcm.tobytes())
        print(f"STT {ms} ms ({len(pcm) / sr:.2f}s 음성): {text!r}")
    a, b = strip(text), strip(EXPECT)
    common = sum(1 for x, y in zip(a, b) if x == y)
    print("기대:", EXPECT, f"| 앞부분 일치 {common}/{len(b)}")
    assert "날씨" in text, text
    # 무음·환각 필터
    assert eng.transcribe_sync(np.zeros(16000 * 2, dtype=np.int16).tobytes())[0] == ""
    assert is_hallucination("시청해 주셔서 감사합니다.") and is_hallucination("감사합니다.")
    assert not is_hallucination("네, 감사합니다. 내일 다시 알려 줘.")
    print("OK")
