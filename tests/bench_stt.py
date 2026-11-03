"""STT 모델·옵션 벤치: Supertonic TTS로 한국어 20문장을 합성(16 kHz)해 깨끗/잡음 버전 CER과 지연을 잰다.

  .venv/bin/python tests/bench_stt.py            # 세트가 없으면 합성 후 전체 설정 비교
  .venv/bin/python tests/bench_stt.py --only A,B # 일부 설정만

결과: tests/out/stt-bench/ (wav, set.json, results.json). 교정표는 끄고 모델 출력만 비교한다.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server import config  # noqa: E402
from server.stt import STTEngine, _norm  # noqa: E402

OUT = ROOT / "tests" / "out" / "stt-bench"
SR = 16000

SENTENCES = [
    "내일 오후 세 시에 치과 예약 일정 잡아 줘.",
    "이번 주 금요일 회의가 몇 시였지?",
    "네이버 메일에 새로 온 거 있어?",
    "지메일에서 김민수 씨가 보낸 메일 읽어 줘.",
    "오늘 서울 날씨 어때? 우산 챙겨야 해?",
    "부산은 내일 비가 온대?",
    "헤르메스, 지금 몇 시야?",
    "카카오톡에 엄마한테 온 메시지 확인해 줘.",
    "클로드한테 이 코드 리뷰 부탁해 줘.",
    "오픈클로 설정 파일 어디 있는지 찾아 줘.",
    "다음 달 근무일이 며칠인지 세어 줘.",
    "미리 알림에 우유 사기 추가해.",
    "삼만 오천 원을 네 명이서 나누면 얼마야?",
    "텔레그램으로 온 알림 다 읽어 줘.",
    "네.",
    "아니.",
    "응, 그렇게 해.",
    "잠깐만, 그거 말고 다른 거.",
    "Claude Code로 테스트 돌려 줘.",
    "이번 meeting은 Zoom으로 하자.",
]
# 같은 말의 다른 표기(숫자·영어 음차). CER은 정답·대안 중 가장 가까운 것으로 잰다.
ALTS = {
    "내일 오후 세 시에 치과 예약 일정 잡아 줘.": ["내일 오후 3시에 치과 예약 일정 잡아 줘."],
    "삼만 오천 원을 네 명이서 나누면 얼마야?": ["35000원을 4명이서 나누면 얼마야?", "3만 5천 원을 4명이서 나누면 얼마야?",
                                         "3만5천원을 네 명이서 나누면 얼마야?", "35000원을 네 명이서 나누면 얼마야?"],
    "Claude Code로 테스트 돌려 줘.": ["클로드 코드로 테스트 돌려 줘."],
    "이번 meeting은 Zoom으로 하자.": ["이번 미팅은 Zoom으로 하자.", "이번 미팅은 줌으로 하자.", "이번 meeting은 줌으로 하자."],
}
# 어려운 버전: 앞 5개 문장 인덱스는 분홍 잡음, 다음 5개는 다른 TTS 문장(에코 모사)
HARD_PINK = [0, 4, 6, 12, 14]
HARD_ECHO = [2, 7, 8, 15, 18]


def pink(n: int, rng) -> np.ndarray:
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    k = np.arange(len(f))
    k[0] = 1
    p = np.fft.irfft(f / np.sqrt(k), n)
    return (p / (np.std(p) + 1e-12)).astype(np.float32)


def rms(x):
    x = x[np.abs(x) > 1e-4] if np.any(np.abs(x) > 1e-4) else x
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def build_set() -> list[dict]:
    from scipy.signal import resample_poly
    from server.tts import TTSEngine
    OUT.mkdir(parents=True, exist_ok=True)
    tts = TTSEngine(speed=1.0)  # 앱 기본 1.3은 사람 말보다 빠르고 "네." 같은 짧은 문장이 무음으로 나옴
    tts.load_sync()
    assert tts.loaded, tts.error
    clean = []
    for s in SENTENCES:
        pcm = tts.synth_sync(s).astype(np.float32) / 32768.0
        a = resample_poly(pcm, 160, 441).astype(np.float32)
        pad = np.zeros(int(0.3 * SR), np.float32)
        clean.append(np.concatenate([pad, a, pad]))
    rng = np.random.default_rng(7)
    items = []
    for i, (s, a) in enumerate(zip(SENTENCES, clean)):
        p = OUT / f"c{i:02d}.wav"
        sf.write(p, a, SR, subtype="PCM_16")
        items.append({"id": f"c{i:02d}", "ref": s, "wav": p.name, "hard": False})
    for i in HARD_PINK + HARD_ECHO:
        a = clean[i]
        if i in HARD_PINK:
            noise = pink(len(a), rng) * rms(a) * 10 ** (-20 / 20)
            kind = "pink-20dB"
        else:
            j = (i + 5) % len(SENTENCES)
            o = clean[j]
            o = np.resize(o, len(a)) if len(o) < len(a) else o[: len(a)]
            noise = o / rms(o) * rms(a) * 10 ** (-15 / 20)
            kind = f"echo-15dB({j})"
        mix = np.clip(a + noise, -1, 1).astype(np.float32)
        p = OUT / f"h{i:02d}.wav"
        sf.write(p, mix, SR, subtype="PCM_16")
        items.append({"id": f"h{i:02d}", "ref": SENTENCES[i], "wav": p.name, "hard": True, "kind": kind})
    # 말 없는 클립(환각 확인용): 정답은 빈 문자열
    for k, (sec, db) in enumerate([(0.6, -50), (1.2, -42), (2.0, -38), (3.0, -45), (1.0, -35)]):
        a = (pink(int(sec * SR), rng) * 10 ** (db / 20)).astype(np.float32)
        p = OUT / f"n{k:02d}.wav"
        sf.write(p, a, SR, subtype="PCM_16")
        items.append({"id": f"n{k:02d}", "ref": "", "wav": p.name, "hard": None})
    (OUT / "set.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    return items


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pairs) -> float:
    err = tot = 0
    for h, r in pairs:
        best = min([r] + ALTS.get(r, []), key=lambda x: lev(_norm(h), _norm(x)))
        err += lev(_norm(h), _norm(best))
        tot += len(_norm(best))
    return err / max(tot, 1)


PROMPT = ("헤르메스, Hermes, 카카오톡, 캘린더, 미리 알림, 네이버 메일, Gmail, 클로드, Claude, "
          "오픈클로, Nous Research, 근무일, 텔레그램")
HF = config.HF_HUB_CACHE
CONFIGS = {
    "A": ("turbo 현재", dict(model="mlx-community/whisper-large-v3-turbo", prompt="", temperatures=(0.0,))),
    "B": ("turbo +프롬프트", dict(model="mlx-community/whisper-large-v3-turbo", prompt=PROMPT, temperatures=(0.0,))),
    "C": ("turbo +온도폴백", dict(model="mlx-community/whisper-large-v3-turbo", prompt="",
                                temperatures=(0.0, 0.2, 0.4, 0.6))),
    "D": ("turbo +프롬프트+폴백", dict(model="mlx-community/whisper-large-v3-turbo", prompt=PROMPT,
                                  temperatures=(0.0, 0.2, 0.4, 0.6))),
    "E": ("large-v3", dict(model="mlx-community/whisper-large-v3-mlx", prompt="", temperatures=(0.0,))),
    "F": ("large-v3 +프롬프트", dict(model="mlx-community/whisper-large-v3-mlx", prompt=PROMPT, temperatures=(0.0,))),
    "G": ("ghost613 turbo-ko 4bit", dict(model="youngouk/ghost613-turbo-korean-4bit-mlx", prompt="",
                                         temperatures=(0.0,))),
    "H": ("turbo-ko (jongmyeong)", dict(model="jongmyeong-jeong/whisper-large-v3-turbo-ko-mlx", prompt="",
                                        temperatures=(0.0,))),
}


def run(key: str, items: list[dict], reps: int) -> dict:
    name, kw = CONFIGS[key]
    eng = STTEngine(corrections="/nonexistent", save_clips_dir="", **kw)
    eng.load_sync()
    if not eng.loaded:
        return {"key": key, "name": name, "error": eng.error}
    rows, lat = [], []
    for it in items:
        pcm, sr = sf.read(OUT / it["wav"], dtype="int16")
        assert sr == SR
        b = pcm.tobytes()
        ds = [eng.transcribe_detail(b, save=False) for _ in range(reps if it["hard"] is False else 1)]
        d = ds[0]
        if it["hard"] is False:
            lat.append(statistics.median(x["ms"] for x in ds))
        rows.append({"id": it["id"], "ref": it["ref"], "hyp": d["text"], "raw": d["raw"], "reason": d["reason"]})
    clean = [(r["hyp"], r["ref"]) for r, it in zip(rows, items) if it["hard"] is False]
    hard = [(r["hyp"], r["ref"]) for r, it in zip(rows, items) if it["hard"] is True]
    hard_src = [(r["hyp"], r["ref"]) for r, it in zip(rows, items)
                if it["hard"] is False and int(it["id"][1:]) in HARD_PINK + HARD_ECHO]
    res = {
        "key": key, "name": name, "model": kw["model"],
        "cer_clean": cer(clean), "cer_hard": cer(hard), "cer_clean_same10": cer(hard_src),
        "lat_median_ms": statistics.median(lat),
        "silence_false": sum(1 for r, it in zip(rows, items) if it["hard"] is None and r["hyp"]),
        "silence_raw": [r["raw"] for r, it in zip(rows, items) if it["hard"] is None],
        "prompt_echo": sum(1 for r in rows if r["reason"] == "prompt-echo"),
        "rows": rows,
    }
    print(f"{key} {name:24s} CER 깨끗 {res['cer_clean']*100:5.1f}% | 잡음 {res['cer_hard']*100:5.1f}% "
          f"(같은 10문장 깨끗 {res['cer_clean_same10']*100:4.1f}%) | 지연 {res['lat_median_ms']:.0f} ms | "
          f"무음오탐 {res['silence_false']}/5 | 프롬프트에코 {res['prompt_echo']}", flush=True)
    del eng
    import gc
    gc.collect()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(CONFIGS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    sj = OUT / "set.json"
    items = build_set() if a.rebuild or not sj.exists() else json.loads(sj.read_text(encoding="utf-8"))
    results = []
    for k in a.only.split(","):
        results.append(run(k.strip(), items, a.reps))
    prev = {}
    rp = OUT / "results.json"
    if rp.exists():
        prev = {r["key"]: r for r in json.loads(rp.read_text(encoding="utf-8"))}
    prev.update({r["key"]: r for r in results})
    rp.write_text(json.dumps(sorted(prev.values(), key=lambda r: r["key"]), ensure_ascii=False, indent=1),
                  encoding="utf-8")
