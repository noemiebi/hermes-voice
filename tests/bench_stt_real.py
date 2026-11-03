"""실사용 클립(tests/clips/*.wav)을 server/stt.py의 STTEngine(필터·교정표 포함)으로 받아쓰기. .venv에서 실행.

  .venv/bin/python tests/bench_stt_real.py --model mlx-community/whisper-large-v3-turbo --key A
  .venv/bin/python tests/bench_stt_real.py --model jongmyeong-jeong/whisper-large-v3-turbo-ko-mlx --key B

결과: tests/out/stt-real/<key>.json  {model, load_ms, clips: {name: {text, raw, ms, runs, reason, seconds}}}
지연은 클립마다 --repeat회(기본 3) 돌린 중앙값. 텍스트는 첫 회 결과(temperature 0, 결정적). 클립 저장은 끔.
Qwen3-ASR은 tests/bench_stt_real_qwen.py(.venv-tts), 분석·페이지는 tests/stt_real_report.py.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server.stt import STTEngine  # noqa: E402

CLIPS = ROOT / "tests" / "clips"
OUT = ROOT / "tests" / "out" / "stt-real"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    eng = STTEngine(model=a.model, save_clips_dir="")
    eng.load_sync()
    if not eng.loaded:
        sys.exit(f"load failed: {eng.error}")
    print(f"loaded {a.model} in {eng.load_ms} ms", flush=True)
    wavs = sorted(CLIPS.glob("*.wav"))
    if a.limit:
        wavs = wavs[: a.limit]
    res = {}
    for i, p in enumerate(wavs):
        with wave.open(str(p), "rb") as w:
            pcm = w.readframes(w.getnframes())
        runs, first = [], None
        for _ in range(a.repeat):
            d = eng.transcribe_detail(pcm, save=False)
            runs.append(d["ms"])
            first = first or d
        res[p.stem] = {"text": first["text"], "raw": first["raw"], "ms": int(statistics.median(runs)), "runs": runs,
                       "reason": first["reason"], "seconds": round(len(pcm) / 2 / 16000, 2)}
        if i % 20 == 0:
            print(i, p.stem, runs, first["raw"][:60], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.key}.json").write_text(json.dumps({"model": a.model, "load_ms": eng.load_ms, "clips": res},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    print("ms median", statistics.median(v["ms"] for v in res.values()))


if __name__ == "__main__":
    main()
