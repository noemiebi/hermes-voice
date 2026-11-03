"""문장 사이 톤 편차 비교(pytest 밖). 사용: .venv/bin/python tests/bench_tts_tone.py [--only A,D] [--runs 2]

실제 답처럼 세 문장을 문장별로 따로 합성(앱과 같은 QwenTTSEngine.synth_stream 경로: 무음 정리·템포·리샘플·이득,
config 그대로)해 0.4초 간격으로 이어 붙여 web/samples/tone/<변형><회차>.m4a로 저장하고 index.html을 만든다.
변형마다 워커를 새로 띄운다. 문장별 길이와 기본 주파수(F0) 중앙값, 세 문장 F0 중앙값의 표준편차(Hz·반음)를 찍는다.
F0는 librosa 없이 정규화 자기상관(40 ms 창, 75~400 Hz, 유성음 프레임만)으로 구한 참고값이다."""
import argparse
import asyncio
import html
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import config  # noqa: E402
from server.tts_qwen import WORKER, QwenTTSEngine  # noqa: E402
from server.tts_text import normalize_for_tts  # noqa: E402

SR = 44100
GAP_S = 0.4
S = ["오늘 서울은 맑고 낮 최고 기온은 열여섯 도예요.",
     "저녁에는 쌀쌀해지니까 겉옷을 챙기세요.",
     "내일 오전 열 시 팀 회의도 잊지 마시고요."]
INSTRUCT = "차분하고 일정한 어조로 자연스럽게 말해 줘"
OUT = Path(__file__).resolve().parent.parent / "web" / "samples" / "tone"

# 이름: (설명, 워커 추가 인자, instruct(None=config 그대로), 회차별 seed(None=문장마다 무작위))
VARIANTS = {
    "A": ("지금 설정(기본 샘플링, seed 없음)", [], None, [None, None]),
    "B": ("temperature 0.6, top_k 20", ["--temperature", "0.6", "--top-k", "20"], None, [None, None]),
    "C": ("temperature 0.4, top_k 10", ["--temperature", "0.4", "--top-k", "10"], None, [None, None]),
    "D": ("기본 샘플링 + 세 문장 같은 seed", [], None, [7, 8]),
    "E": ("temperature 0.6 + 세 문장 같은 seed", ["--temperature", "0.6"], None, [7, 8]),
    "F": (f"지시문 \"{INSTRUCT}\"(기본 샘플링)", [], INSTRUCT, [None, None]),
    "G": (f"지시문 \"{INSTRUCT}\" + temperature 0.6", ["--temperature", "0.6"], INSTRUCT, [None, None]),
}


def worker_cmd(extra: list[str], instruct) -> list[str]:
    ins = config.QWEN_TTS_INSTRUCT if instruct is None else instruct
    return ([config.QWEN_TTS_PYTHON, "-I", str(WORKER), "--model", config.QWEN_TTS_MODEL,
             "--voice", config.QWEN_TTS_VOICE, "--interval", str(config.QWEN_TTS_STREAM_INTERVAL)]
            + (["--instruct", ins] if ins else []) + extra)


def f0_median(pcm: np.ndarray, sr: int = SR) -> float:
    """정규화 자기상관 F0(Hz) 중앙값. 16 kHz로 줄여 40 ms 창·10 ms 간격, 상관 0.5 넘는 유성음 프레임만."""
    from scipy.signal import resample_poly
    x = resample_poly(pcm.astype(np.float64) / 32768, 160, 441)
    fs = 16000
    win, hop = int(0.04 * fs), int(0.01 * fs)
    lo, hi = int(fs / 400), int(fs / 75)
    rms_all = np.sqrt(np.mean(x ** 2)) + 1e-12
    out = []
    for s in range(0, x.size - win - hi, hop):
        fr = x[s:s + win + hi]
        a = fr[:win] - fr[:win].mean()
        if np.sqrt(np.mean(a ** 2)) < 0.3 * rms_all:
            continue
        e0 = np.dot(a, a)
        rs = np.zeros(hi + 2)
        for L in range(lo, hi + 1):
            b = fr[L:L + win] - fr[L:L + win].mean()
            rs[L] = np.dot(a, b) / np.sqrt(e0 * np.dot(b, b) + 1e-12)
        best = rs.max()
        lag = 0
        for L in range(lo, hi + 1):   # 옥타브 아래로 빠지지 않게: 최댓값의 85% 넘는 첫 봉우리
            if rs[L] >= 0.85 * best and rs[L] >= rs[L - 1] and rs[L] >= rs[L + 1]:
                lag = L
                break
        if best > 0.5 and lag:
            out.append(fs / lag)
    return float(np.median(out)) if out else float("nan")


async def synth(eng: QwenTTSEngine, text: str, seed) -> np.ndarray:
    chunks = [c async for c in eng.synth_stream(normalize_for_tts(text), seed=seed)]
    return np.concatenate(chunks) if chunks else np.zeros(0, np.int16)


def save_m4a(pcm: np.ndarray, name: str) -> Path:
    wav = OUT / f"{name}.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    m4a = OUT / f"{name}.m4a"
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "96000", str(wav), str(m4a)], check=True)
    wav.unlink()
    return m4a


async def run_variant(key: str, runs: int) -> list[dict]:
    desc, extra, instruct, seeds = VARIANTS[key]
    eng = QwenTTSEngine(cmd=worker_cmd(extra, instruct))
    t = time.perf_counter()
    await eng.start()
    if not eng.ready:
        print(f"{key}: 워커 실패 {eng.error}")
        return []
    print(f"{key} ({desc}) 준비 {time.perf_counter() - t:.0f}s")
    res = []
    gap = np.zeros(int(GAP_S * SR), np.int16)
    for i in range(runs):
        seed = seeds[i] if i < len(seeds) else None
        parts = [await synth(eng, s, seed) for s in S]
        durs = [p.size / SR for p in parts]
        f0s = [f0_median(p) for p in parts]
        joined = np.concatenate([x for p in parts for x in (p, gap)][:-1])
        save_m4a(joined, f"{key}{i + 1}")
        st_hz = float(np.nanstd(f0s))
        st_semi = float(np.nanstd(12 * np.log2(np.array(f0s) / 100)))
        res.append({"key": key, "run": i + 1, "seed": seed, "durs": durs, "f0s": f0s, "std_hz": st_hz,
                    "std_semi": st_semi, "total": joined.size / SR})
        print(f"  {key}{i + 1} seed={seed}: 길이 {[round(d, 2) for d in durs]} s, F0 {[round(f) for f in f0s]} Hz, "
              f"편차 {st_hz:.1f} Hz ({st_semi:.2f} 반음)")
    await eng.aclose()
    return res


def write_index(results: list[dict]) -> None:
    by = {}
    for r in results:
        by.setdefault(r["key"], []).append(r)
    cards = []
    for key, (desc, _, _, _) in VARIANTS.items():
        if key not in by:
            continue
        rows = []
        for r in by[key]:
            rows.append(f'<div class="r"><span>{key}{r["run"]}</span><audio controls preload="none" '
                        f'src="{key}{r["run"]}.m4a"></audio><small>{r["total"]:.1f}초 · F0 '
                        f'{"/".join(str(round(f)) for f in r["f0s"])} Hz · 편차 {r["std_semi"]:.2f} 반음</small></div>')
        title = f"{key}. {html.escape(desc)}" + (" (지금 설정)" if key == "A" else "")
        cards.append(f'<div class="c"><b>{title}</b>{"".join(rows)}</div>')
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>문장 사이 톤 비교</title>
<style>body{{font-family:-apple-system,sans-serif;background:#111;color:#eee;margin:0;padding:20px 16px}}h1{{font-size:20px;margin:0 0 6px}}p{{color:#aaa;margin:0 0 18px;font-size:14px}}
.c{{background:#1c1c1e;border-radius:12px;padding:14px;margin-bottom:12px}}.c b{{display:block;margin-bottom:8px;font-size:16px}}.r{{margin-bottom:8px}}.r span{{color:#888;font-size:13px}}audio{{width:100%}}small{{color:#999}}</style></head><body>
<h1>문장 사이 톤 비교</h1><p><b style="color:#fff">A가 지금 설정입니다.</b> 세 문장을 앱처럼 문장마다 따로 합성해 0.4초 간격으로 이었습니다.
변형마다 두 번(다른 무작위·다른 seed) 만들었습니다. 편차는 세 문장 음높이 중앙값의 표준편차(참고값)이고, 판단은 들어 보고 해 주세요.</p>
{"".join(cards)}
</body></html>
"""
    (OUT / "index.html").write_text(page, encoding="utf-8")


async def main(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = [k.strip() for k in args.only.split(",")] if args.only else list(VARIANTS)
    print(f"model={config.QWEN_TTS_MODEL} voice={config.QWEN_TTS_VOICE} tempo={config.QWEN_TTS_TEMPO} "
          f"instruct={config.QWEN_TTS_INSTRUCT!r} 샘플링 env={config.QWEN_TTS_TEMPERATURE!r}/{config.QWEN_TTS_TOP_K!r}/"
          f"{config.QWEN_TTS_TOP_P!r}")
    results = []
    for k in keys:
        results += await run_variant(k, args.runs)
    write_index(results)
    print("\n| 변형 | 회차 | seed | 길이 s1/s2/s3 (s) | F0 s1/s2/s3 (Hz) | 편차 Hz | 편차 반음 |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['key']} | {r['run']} | {r['seed']} | {'/'.join(f'{d:.2f}' for d in r['durs'])} | "
              f"{'/'.join(f'{f:.0f}' for f in r['f0s'])} | {r['std_hz']:.1f} | {r['std_semi']:.2f} |")
    print("\n| 변형 | 평균 편차 Hz | 평균 편차 반음 |")
    print("|---|---|---|")
    for k in keys:
        rs = [r for r in results if r["key"] == k]
        if rs:
            print(f"| {k} | {np.mean([r['std_hz'] for r in rs]):.1f} | {np.mean([r['std_semi'] for r in rs]):.2f} |")
    print(f"\n페이지: {OUT / 'index.html'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="예: A,D")
    ap.add_argument("--runs", type=int, default=2)
    asyncio.run(main(ap.parse_args()))
