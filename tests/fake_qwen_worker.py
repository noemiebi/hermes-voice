"""tests/test_tts_qwen.py용 가짜 Qwen TTS 워커(server/qwen_tts_worker.py와 같은 JSON lines 프로토콜).

텍스트에 든 표지로 동작을 바꾼다(세션 경로는 독음 정규화를 거치므로 한글 표지도 받음):
SLOW·느림(첫 청크 전 2초), ERR·오류(오류), LONG·길게(청크 40개, 50 ms 간격), DIE·종료(즉시 종료).
그 밖에는 440 Hz 사인 0.2초 청크 3개. 받은 cancel은 env FAKE_QWEN_LOG 파일에 한 줄씩 적는다.
실제 워커의 CLI 인자(--temperature 등)는 무시하고 'argv ...' 줄로, 요청의 seed는 'seed id n' 줄로 적는다."""
import base64
import json
import os
import queue
import sys
import threading
import time

import numpy as np

SR = 24000
LOG = os.environ.get("FAKE_QWEN_LOG")
if os.environ.get("FAKE_QWEN_NOT_READY"):
    print(json.dumps({"ready": False, "error": "fake load fail"}), flush=True)
    sys.exit(1)
cancelled, lock, q = set(), threading.Lock(), queue.Queue()


def note(s):
    if LOG:
        with open(LOG, "a") as f:
            f.write(s + "\n")


def reader():
    for line in sys.stdin:
        m = json.loads(line)
        if "cancel" in m:
            with lock:
                cancelled.add(m["cancel"])
            note(f"cancel {m['cancel']}")
        else:
            q.put(m)
    q.put(None)


def emit(o):
    sys.stdout.write(json.dumps(o) + "\n")
    sys.stdout.flush()


note("argv " + " ".join(sys.argv[1:]))
threading.Thread(target=reader, daemon=True).start()
emit({"ready": True, "model": "fake"})
pos = 0
while True:
    m = q.get()
    if m is None:
        break
    rid, text = m["id"], m["text"]
    note(f"req {rid} {text}")
    if m.get("seed") is not None:
        note(f"seed {rid} {m['seed']}")
    if ("DIE" in text or "종료" in text):
        sys.exit(3)
    if ("ERR" in text or "오류" in text):
        emit({"id": rid, "error": "fake error"})
        continue
    if ("SLOW" in text or "느림" in text):
        time.sleep(2.0)
    n, gap = (40, 0.05) if ("LONG" in text or "길게" in text) else (3, 0.01)
    stop = False
    for i in range(n):
        with lock:
            if rid in cancelled:
                stop = True
                break
        t = (pos + np.arange(SR // 5)) / SR
        pos += SR // 5
        a = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
        emit({"id": rid, "sr": SR, "pcm": base64.b64encode(a.tobytes()).decode()})
        time.sleep(gap)
    if stop:
        emit({"id": rid, "cancelled": True})
    else:
        emit({"id": rid, "done": True, "ms": 1, "first_ms": 1, "samples": n * SR // 5})
