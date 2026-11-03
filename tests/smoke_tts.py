"""TTS 상주 로드 후 문장 3개 합성 시간 출력. tests/out/smoke_ko.wav 저장(smoke_stt·e2e 입력)."""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.tts import TTSEngine  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
SENTENCES = ["안녕하세요, 반가워요.", "오늘 서울 날씨가 어떤지 알려 줄래?", "내일 오전 열 시에 회의가 있으니까 잊지 말고 알려 줘."]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    eng = TTSEngine()
    t = time.perf_counter()
    eng.load_sync()
    assert eng.loaded, eng.error
    SAMPLE_RATE = eng.out_rate   # synth_sync 출력 레이트(config.AUDIO_OUT_RATE)
    print(f"load+warmup {int((time.perf_counter() - t) * 1000)} ms (provider={eng.provider})")
    for i, s in enumerate(SENTENCES):
        t = time.perf_counter()
        pcm = eng.synth_sync(s)
        ms = int((time.perf_counter() - t) * 1000)
        dur = len(pcm) / SAMPLE_RATE
        print(f"[{len(s):2d}자] {ms:4d} ms -> {dur:.2f}s 음성 (RTF {ms / 1000 / dur:.2f})  {s}")
        assert pcm.dtype == np.int16 and len(pcm) > SAMPLE_RATE * 0.5
        sf.write(OUT / f"tts_{i}.wav", pcm, SAMPLE_RATE, subtype="PCM_16")
    # STT용 16k 시험 음성
    pcm = eng.synth_sync(SENTENCES[1]).astype(np.float32)
    idx = np.arange(0, len(pcm), SAMPLE_RATE / 16000)
    pcm16k = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.int16)
    sf.write(OUT / "smoke_ko.wav", pcm16k, 16000, subtype="PCM_16")
    print("saved", OUT / "smoke_ko.wav")
