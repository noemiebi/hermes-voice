"""헤르메스 API 첫 토큰 지연 측정. 게이트웨이(8642)가 떠 있어야 한다.
사용: .venv/bin/python tests/bench_hermes.py [--reasoning default|off|low|medium] [--model voice-fast|voice-deep]
                                             [--repeat N] [--question "..."] [--no-second]
각 반복마다 새 세션 id로 1턴(콜드), 같은 세션으로 2턴(캐시 적중)을 보내
첫 토큰 ms, 전체 ms, prompt_tokens, 답변을 출력한다.
"""
import argparse
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import config  # noqa: E402

Q1 = "오늘 저녁 뭐 먹을까, 한 문장으로"
Q2 = "그거 말고 다른 거 하나만, 한 문장으로"


def reasoning_options(mode: str):
    if mode == "off":
        return {"reasoning": {"enabled": False}}
    if mode in ("minimal", "low", "medium", "high"):
        return {"reasoning": {"enabled": True, "effort": mode}}
    return None  # default: 헤르메스 설정값


def one_turn(client: httpx.Client, sid: str, model: str, text: str, mode: str) -> dict:
    body = {"model": model, "stream": True, "messages": [{"role": "user", "content": text}]}
    opts = reasoning_options(mode)
    if opts:
        body["model_options"] = opts
    headers = {"Authorization": f"Bearer {config.HERMES_API_KEY}", "Accept": "text/event-stream",
               "X-Hermes-Session-Id": sid}
    t0 = time.perf_counter()
    first = first_reason = None
    parts, usage, event = [], None, None
    with client.stream("POST", f"{config.HERMES_API_URL}/v1/chat/completions", headers=headers, json=body) as r:
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {r.read()[:300]!r}")
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
                continue
            if not line.startswith("data:"):
                if not line.strip():
                    event = None
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            if event:
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                if (d.get("reasoning_content") or d.get("reasoning")) and first_reason is None:
                    first_reason = time.perf_counter()
                if d.get("content"):
                    if first is None:
                        first = time.perf_counter()
                    parts.append(d["content"])
    t1 = time.perf_counter()
    ms = lambda t: round((t - t0) * 1000) if t else None  # noqa: E731
    return {"first_ms": ms(first), "first_reason_ms": ms(first_reason), "total_ms": ms(t1),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "reasoning_tokens": ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "text": "".join(parts).strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoning", default="default", choices=["default", "off", "minimal", "low", "medium", "high"])
    ap.add_argument("--model", default="voice-fast")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--question", default=Q1)
    ap.add_argument("--no-second", action="store_true", help="2턴(같은 세션) 측정 생략")
    a = ap.parse_args()
    if not config.HERMES_API_KEY:
        sys.exit("HERMES_API_KEY 없음(.env)")
    rows: dict[str, list] = {"turn1": [], "turn2": []}
    with httpx.Client(timeout=httpx.Timeout(10.0, read=300.0)) as c:
        for i in range(a.repeat):
            sid = f"bench-{uuid.uuid4().hex[:12]}"
            turns = [("turn1", a.question)] + ([] if a.no_second else [("turn2", Q2)])
            for name, q in turns:
                r = one_turn(c, sid, a.model, q, a.reasoning)
                rows[name].append(r)
                print(f"[{i + 1}/{a.repeat}] {name} model={a.model} reasoning={a.reasoning} "
                      f"first={r['first_ms']}ms reason_first={r['first_reason_ms']} total={r['total_ms']}ms "
                      f"prompt={r['prompt_tokens']} completion={r['completion_tokens']} "
                      f"reasoning_tok={r['reasoning_tokens']} | {r['text'][:80]}", flush=True)
    for name, rs in rows.items():
        f = [r["first_ms"] for r in rs if r["first_ms"] is not None]
        if f:
            print(f"SUMMARY {name} model={a.model} reasoning={a.reasoning} n={len(f)} "
                  f"first_median={statistics.median(f)}ms first_min={min(f)} first_max={max(f)} "
                  f"prompt_tokens={rs[-1]['prompt_tokens']}")


if __name__ == "__main__":
    main()
