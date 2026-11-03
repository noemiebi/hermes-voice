"""실사용 클립(tests/clips/*.wav)을 Qwen3-ASR(mlx-audio)로 받아쓰기. .venv-tts에서 실행.

  [HF_HOME=<HF 캐시 상위 폴더>] .venv-tts/bin/python tests/bench_stt_real_qwen.py [--model ...] [--key C] [--limit N]

결과: tests/out/stt-real/<key>.json  {model, load_ms, clips: {name: {text, raw, ms, runs, reason, seconds}}}
지연은 클립마다 --repeat회(기본 3) 돌린 중앙값. 텍스트는 첫 회 결과.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import wave
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", os.environ.get("QWEN_TTS_HF_HOME") or os.path.expanduser("~/.cache/huggingface"))
ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "tests" / "clips"
OUT = ROOT / "tests" / "out" / "stt-real"


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-ASR-1.7B-8bit")
    ap.add_argument("--key", default="C")
    ap.add_argument("--language", default="Korean")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from mlx_audio.stt.utils import load_model

    t = time.perf_counter()
    model = load_model(a.model)
    model.generate(np.zeros(16000, dtype=np.float32), language=a.language)  # 워밍업
    load_ms = int((time.perf_counter() - t) * 1000)
    print(f"loaded {a.model} in {load_ms} ms", flush=True)
    wavs = sorted(CLIPS.glob("*.wav"))
    if a.limit:
        wavs = wavs[: a.limit]
    res = {}
    for i, p in enumerate(wavs):
        audio = read_wav(p)
        runs, text = [], None
        for _ in range(a.repeat):
            t = time.perf_counter()
            r = model.generate(audio, language=a.language)
            runs.append(int((time.perf_counter() - t) * 1000))
            if text is None:
                text = (getattr(r, "text", None) or "").strip()
        res[p.stem] = {"text": text, "raw": text, "ms": int(statistics.median(runs)), "runs": runs,
                       "reason": "", "seconds": round(audio.size / 16000, 2)}
        if i % 20 == 0:
            print(i, p.stem, runs, text[:60], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.key}.json").write_text(json.dumps({"model": a.model, "load_ms": load_ms, "clips": res},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    print("ms median", statistics.median(v["ms"] for v in res.values()))


if __name__ == "__main__":
    main()
