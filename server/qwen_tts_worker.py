"""Qwen3-TTS(mlx-audio) 스트리밍 합성 워커. 별도 venv(.venv-tts)에서 서브프로세스로 돈다(server/tts_qwen.py가 띄움).

프로토콜(JSON lines):
  시작: 모델 로드·예열 뒤 stdout {"ready":true,"model":...,"voice":...,"load_ms":n}
  stdin  {"id":n,"text":"...","voice":"Sohee","seed":n?}   합성 요청(큐, 한 번에 하나, seed는 선택)
         선택 필드 "instruct","temperature","top_k","top_p","repetition_penalty"는 그 요청만 덮어씀
  stdin  {"cancel":n}                             생성 중단(다음 청크 경계에서) -> {"id":n,"cancelled":true}
  stdout {"id":n,"sr":24000,"pcm":"<base64 int16 LE mono>"}  청크(여러 개)
  stdout {"id":n,"done":true,"ms":합성ms,"first_ms":첫청크ms,"samples":n}
  stdout {"id":n,"error":"..."}
로그는 stderr로만 쓴다(stdout은 프로토콜 전용).

인자/env: --model(QWEN_TTS_MODEL) --voice(QWEN_TTS_VOICE) --interval(QWEN_TTS_STREAM_INTERVAL, 초)
          --instruct(QWEN_TTS_INSTRUCT, 1.7B만) --lang(QWEN_TTS_LANG). HF 위치는 env QWEN_TTS_HF_HOME(없으면 HF_HOME).
          샘플링 --temperature --top-k --top-p --repetition-penalty(생략=모델 기본, mlx-audio 0.9/50/1.0/1.05).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import sys
import threading
import time

# Qwen3-TTS 모델은 HF_HOME(env QWEN_TTS_HF_HOME, 기본 ~/.cache/huggingface) 아래 hub/에 있다(메인 서버 whisper의
# HF_HUB_CACHE와 다를 수 있음). 부모에게서 물려받은 HF_HUB_CACHE를 HF_HOME/hub로 덮어쓴다.
_HF_HOME = (os.environ.get("QWEN_TTS_HF_HOME") or os.environ.get("HF_HOME")
            or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
os.environ["HF_HOME"] = _HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(_HF_HOME, "hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"

_out_lock = threading.Lock()
_stdout = sys.stdout


def emit(obj: dict) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    with _out_lock:
        _stdout.write(line + "\n")
        _stdout.flush()


def log(msg: str) -> None:
    print(f"[qwen-tts-worker] {msg}", file=sys.stderr, flush=True)


class Requests:
    """stdin 리더 스레드: 합성 요청은 큐로, cancel은 집합으로(생성 루프가 청크마다 확인)."""

    def __init__(self, stream):
        self.q: queue.Queue = queue.Queue()
        self.cancelled: set[int] = set()
        self.lock = threading.Lock()
        self.eof = False
        self.t = threading.Thread(target=self._run, args=(stream,), daemon=True)
        self.t.start()

    def _run(self, stream) -> None:
        for raw in stream:
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except ValueError:
                log(f"잘못된 요청 줄: {raw[:80]!r}")
                continue
            if "cancel" in msg:
                with self.lock:
                    self.cancelled.add(int(msg["cancel"]))
            elif "id" in msg:
                self.q.put(msg)
        self.eof = True
        self.q.put(None)

    def is_cancelled(self, rid: int) -> bool:
        with self.lock:
            return rid in self.cancelled

    def forget(self, rid: int) -> None:
        with self.lock:
            self.cancelled.discard(rid)


SAMPLING_KEYS = (("temperature", float), ("top_k", int), ("top_p", float), ("repetition_penalty", float))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QWEN_TTS_MODEL", DEFAULT_MODEL))
    ap.add_argument("--voice", default=os.environ.get("QWEN_TTS_VOICE", "Sohee"))
    ap.add_argument("--interval", type=float, default=float(os.environ.get("QWEN_TTS_STREAM_INTERVAL", "0.32")))
    ap.add_argument("--instruct", default=os.environ.get("QWEN_TTS_INSTRUCT", ""))
    ap.add_argument("--lang", default=os.environ.get("QWEN_TTS_LANG", "korean"))
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-k", dest="top_k", type=int, default=None)
    ap.add_argument("--top-p", dest="top_p", type=float, default=None)
    ap.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=None)
    return ap


def sampling_kwargs(args: argparse.Namespace) -> dict:
    """CLI에서 지정한 샘플링 값만 generate kwargs로(None이면 모델 기본)."""
    return {k: getattr(args, k) for k, _ in SAMPLING_KEYS if getattr(args, k, None) is not None}


def request_kwargs(base: dict, msg: dict) -> dict:
    """요청별 덮어쓰기: instruct·샘플링 선택 필드(값이 null이 아니면)."""
    kw = dict(base)
    if msg.get("instruct"):
        kw["instruct"] = msg["instruct"]
    for k, typ in SAMPLING_KEYS:
        if msg.get(k) is not None:
            kw[k] = typ(msg[k])
    return kw


def main() -> int:
    args = build_parser().parse_args()

    # mlx/transformers가 stdout에 무언가 찍어도 프로토콜이 깨지지 않게 stdout을 stderr로 돌려 둔다.
    sys.stdout = sys.stderr

    import numpy as np
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model

    t = time.perf_counter()
    try:
        model = load_model(args.model)
        kw = dict(voice=args.voice, lang_code=args.lang)
        if args.instruct:
            kw["instruct"] = args.instruct
        kw.update(sampling_kwargs(args))
        for _ in model.generate("안녕하세요.", **kw):  # 예열(전체 합성 1회 + 스트리밍 1회)
            pass
        for _ in model.generate("네, 알겠어요.", stream=True, streaming_interval=args.interval, **kw):
            pass
    except Exception as e:  # noqa: BLE001
        log(f"모델 로드 실패: {e!r}")
        emit({"ready": False, "error": repr(e)})
        return 1
    load_ms = int((time.perf_counter() - t) * 1000)
    log(f"ready: {args.model} voice={args.voice} interval={args.interval} instruct={args.instruct!r} "
        f"sampling={sampling_kwargs(args)} "
        f"load {load_ms} ms, peak mem {mx.get_peak_memory() / 1e9:.2f} GB")
    emit({"ready": True, "model": args.model, "voice": args.voice, "load_ms": load_ms,
          "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2)})

    reqs = Requests(sys.stdin)
    while True:
        msg = reqs.q.get()
        if msg is None:
            break
        rid = int(msg["id"])
        if reqs.is_cancelled(rid):  # 큐에서 기다리다 취소됨
            reqs.forget(rid)
            emit({"id": rid, "cancelled": True})
            continue
        text = str(msg.get("text", "")).strip()
        if not text:
            emit({"id": rid, "done": True, "ms": 0, "first_ms": None, "samples": 0})
            continue
        rkw = request_kwargs(kw, msg)
        rkw["voice"] = msg.get("voice") or args.voice
        if msg.get("seed") is not None:  # 재현용(스모크 비교): 같은 seed면 같은 음성
            mx.random.seed(int(msg["seed"]))
        t0 = time.perf_counter()
        first_ms = None
        samples = 0
        cancelled = False
        gen = None
        try:
            gen = model.generate(text, stream=True, streaming_interval=args.interval, **rkw)
            for r in gen:
                if reqs.is_cancelled(rid):
                    cancelled = True
                    break
                a = np.asarray(r.audio, dtype=np.float32).reshape(-1)
                if a.size == 0:
                    continue
                pcm = (np.clip(a, -1.0, 1.0) * 32767).astype("<i2")
                if first_ms is None:
                    first_ms = int((time.perf_counter() - t0) * 1000)
                samples += pcm.size
                emit({"id": rid, "sr": int(getattr(r, "sample_rate", 24000) or 24000),
                      "pcm": base64.b64encode(pcm.tobytes()).decode("ascii")})
                if reqs.is_cancelled(rid):
                    cancelled = True
                    break
        except Exception as e:  # noqa: BLE001
            log(f"합성 실패 id={rid}: {e!r}")
            emit({"id": rid, "error": repr(e)})
            reqs.forget(rid)
            continue
        finally:
            if gen is not None and hasattr(gen, "close"):
                try:
                    gen.close()
                except Exception:  # noqa: BLE001
                    pass
        reqs.forget(rid)
        ms = int((time.perf_counter() - t0) * 1000)
        if cancelled:
            emit({"id": rid, "cancelled": True, "ms": ms})
        else:
            emit({"id": rid, "done": True, "ms": ms, "first_ms": first_ms, "samples": samples})
    return 0


if __name__ == "__main__":
    sys.exit(main())
