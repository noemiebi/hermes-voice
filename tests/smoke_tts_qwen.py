"""실제 Qwen3-TTS 워커 스모크(pytest 밖). 사용: .venv/bin/python tests/smoke_tts_qwen.py [--instruct "..."] [--model id]
문장 3개의 첫 청크 지연·총 합성 시간·RTF, 워커 메모리(RSS·MLX 최대)를 찍고 tests/out/qwen-smoke/에 출력 레이트(config.AUDIO_OUT_RATE) wav 저장.
--tempos 1.0,1.2: 같은 seed로 속도별 합성(tempo_<t>_<i>.wav), 길이 비율과 프로젝트 STT로 받아쓴 CER 비교."""
import argparse
import asyncio
import subprocess
import sys
import time
from math import gcd
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import config  # noqa: E402
from server.tts import speech_rms_db  # noqa: E402
from server.tts_qwen import WORKER, QwenTTSEngine  # noqa: E402
from server.tts_text import normalize_for_tts  # noqa: E402

S = ["안녕하세요, 오늘 일정은 세 개예요. 오후 2시에 회의가 있고, 저녁엔 약속이 있어요.",
     "잠깐 확인해 볼게요. 음, 내일 서울은 흐리고 낮 최고 기온은 12도예요.",
     "그건 제가 잘 모르겠어요. 대신 관련된 메일을 찾아봤는데, 두 통이 있네요."]
OUT = Path(__file__).resolve().parent / "out" / "qwen-smoke"


def rss_mb(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 if out else float("nan")


async def main(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [config.QWEN_TTS_PYTHON, "-I", str(WORKER), "--model", args.model, "--voice", config.QWEN_TTS_VOICE,
           "--interval", str(config.QWEN_TTS_STREAM_INTERVAL)] + (["--instruct", args.instruct] if args.instruct else [])
    eng = QwenTTSEngine(cmd=cmd)
    t = time.perf_counter()
    await eng.start()
    print(f"ready={eng.ready} load {time.perf_counter() - t:.1f}s info={eng.info} error={eng.error}")
    if not eng.ready:
        return
    if args.tempos:
        await tempo_compare(eng, [float(x) for x in args.tempos.split(",")])
        await eng.aclose()
        return
    tag = args.tag or ("instruct" if args.instruct else "base")
    for i, s in enumerate(S, 1):
        text = normalize_for_tts(s)
        t0 = time.perf_counter()
        first = None
        chunks = []
        async for c in eng.synth_stream(text):
            if first is None:
                first = time.perf_counter() - t0
            chunks.append(c)
        total = time.perf_counter() - t0
        pcm = np.concatenate(chunks)
        dur = pcm.size / eng.out_rate
        f = pcm.astype(np.float32) / 32767
        print(f"s{i}: 첫 청크 {first * 1000:.0f} ms (워커 {eng.last.get('worker_first_ms')} ms), 총 {total * 1000:.0f} ms, "
              f"음성 {dur:.2f}s, RTF {total / dur:.2f}, 청크 {len(chunks)}개, RMS {speech_rms_db(f):.1f} dBFS, "
              f"피크 {20 * np.log10(np.max(np.abs(f))):.1f} dBFS")
        with wave.open(str(OUT / f"{tag}_{i}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(eng.out_rate)
            w.writeframes(pcm.tobytes())
    print(f"워커 RSS {rss_mb(eng.proc.pid):.0f} MB, MLX 최대(로드·예열) {eng.info.get('peak_mem_gb')} GB")
    await eng.aclose()


def _norm(t: str) -> str:
    import re
    return re.sub(r"[\s.,?!'\"~·…]", "", t)


def cer(ref: str, hyp: str) -> float:
    r, h = _norm(ref), _norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


async def tempo_compare(eng, tempos) -> None:
    from scipy.signal import resample_poly
    from server.stt import STTEngine
    stt = STTEngine(save_clips_dir="")
    await stt.load()
    res = {t: [] for t in tempos}
    for i, s in enumerate(S, 1):
        text = normalize_for_tts(s)
        for t in tempos:
            eng.tempo = t
            t0 = time.perf_counter()
            first = None
            chunks = []
            async for c in eng.synth_stream(text, seed=1000 + i):
                if first is None:
                    first = time.perf_counter() - t0
                chunks.append(c)
            pcm = np.concatenate(chunks)
            with wave.open(str(OUT / f"tempo_{t:g}_{i}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(eng.out_rate)
                w.writeframes(pcm.tobytes())
            a16 = resample_poly(pcm.astype(np.float32) / 32768, 16000 // gcd(16000, eng.out_rate), eng.out_rate // gcd(16000, eng.out_rate))
            hyp, _ = await stt.transcribe((np.clip(a16, -1, 1) * 32767).astype("<i2").tobytes())
            res[t].append((pcm.size / eng.out_rate, first * 1000, cer(s, hyp), hyp))
            print(f"s{i} tempo {t:g}: {pcm.size / eng.out_rate:.2f}s 첫 청크 {first * 1000:.0f} ms CER {cer(s, hyp):.3f} | {hyp}")
    base = tempos[0]
    for t in tempos:
        L = sum(x[0] for x in res[t])
        print(f"tempo {t:g}: 총 {L:.2f}s (비율 {L / sum(x[0] for x in res[base]):.3f}), "
              f"평균 CER {np.mean([x[2] for x in res[t]]):.3f}, 첫 청크 {[round(x[1]) for x in res[t]]} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct", default="")
    ap.add_argument("--model", default=config.QWEN_TTS_MODEL)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tempos", default="", help="예: 1.0,1.2 (같은 seed로 속도 비교 + STT CER)")
    asyncio.run(main(ap.parse_args()))
