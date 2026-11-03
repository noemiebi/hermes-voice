"""CPU 대 CoreML Supertonic 비교."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import config
from server.tts import build_tts
from server.supertonic_helper import load_voice_style

S = ["안녕하세요, 반가워요.", "오늘 날씨가 꽤 쌀쌀하네요.", "잠깐 확인해 볼게요. 내일 일정은 세 개가 있어요."]
for prov in sys.argv[1:] or ["cpu", "coreml"]:
    t = time.perf_counter()
    tts = build_tts(config.SUPERTONIC_MODEL_DIR, prov)
    style = load_voice_style([config.SUPERTONIC_VOICE])
    tts("예열입니다.", "ko", style, 5, 1.1, 0.1)
    load = time.perf_counter() - t
    res = []
    for rep in range(2):
        for s in S:
            t = time.perf_counter(); tts(s, "ko", style, 5, 1.1, 0.2); res.append(int((time.perf_counter() - t) * 1000))
    print(prov, f"load+warm {load:.1f}s", "ms:", res, flush=True)
