"""WebSocket 왕복 시험. 서버를 먼저 띄운다(예: HERMES_MOCK=1 PORT=8481 .venv/bin/python -m server).
사용: .venv/bin/python tests/e2e_ws.py [ws://127.0.0.1:8481/ws]"""
import asyncio
import json
import struct
import sys
import time
from pathlib import Path

import soundfile as sf
import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8481/ws"
WAV = Path(__file__).resolve().parent / "out" / "smoke_ko.wav"


async def recv(ws, timeout=60):
    m = await asyncio.wait_for(ws.recv(), timeout)
    if isinstance(m, bytes):
        return ("bin", struct.unpack("<I", m[:4])[0], len(m) - 4)
    return ("json", json.loads(m))


async def wait_ready(ws):
    for _ in range(20):
        r = await recv(ws)
        if r[0] == "json" and r[1]["type"] == "ready":
            return r[1]
    raise AssertionError("no ready")


async def main():
    pcm, sr = sf.read(WAV, dtype="int16")
    assert sr == 16000
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "resume": None, "client": "e2e"}))
        ready = await wait_ready(ws)
        print("ready:", ready)
        rate, codec = ready["out_sample_rate"], ready.get("out_codec", "pcm16")
        assert rate in (16000, 22050, 24000, 32000, 44100, 48000) and codec in ("pcm16", "mulaw"), ready
        assert ready["session_id"]

        # ---- 1) 음성 턴 ----
        await ws.send(json.dumps({"type": "start_utterance"}))
        raw = pcm.tobytes()
        step = 16000 * 2 // 10  # 100ms 단위
        for i in range(0, len(raw), step):
            await ws.send(raw[i:i + step])
        t_end = time.perf_counter()
        await ws.send(json.dumps({"type": "end_utterance"}))
        seen = {"transcript": None, "delta": 0, "speak_start": 0, "bin": 0, "tool": [], "states": [],
                "first_audio_ms": None, "done": None}
        gen = None
        while True:
            r = await recv(ws, 120)
            if r[0] == "bin":
                seen["bin"] += 1
                if seen["first_audio_ms"] is None:
                    seen["first_audio_ms"] = int((time.perf_counter() - t_end) * 1000)
                assert gen is None or r[1] == gen, (r[1], gen)
                continue
            m = r[1]
            t = m["type"]
            if t == "status":
                seen["states"].append(m["state"])
            elif t == "transcript":
                seen["transcript"] = m
                print(f"transcript ({int((time.perf_counter() - t_end) * 1000)} ms 경과):", m)
            elif t == "assistant_delta":
                seen["delta"] += 1
            elif t == "assistant_done":
                seen["done"] = m["text"]
            elif t == "tool":
                seen["tool"].append((m["name"], m["state"]))
            elif t == "speak_start":
                seen["speak_start"] += 1
                gen = m["generation"]
                print("  speak_start", gen, m["text"])
            elif t == "speak_end":
                assert m["generation"] == gen
                break
            elif t == "error":
                raise AssertionError(m)
        print("states:", seen["states"], "tool:", seen["tool"])
        print(f"delta {seen['delta']}개, speak_start {seen['speak_start']}개, 오디오 프레임 {seen['bin']}개, "
              f"말 끝→첫 오디오 {seen['first_audio_ms']} ms")
        print("assistant_done:", seen["done"])
        assert seen["transcript"] and "날씨" in seen["transcript"]["text"]
        assert seen["delta"] > 0 and seen["speak_start"] > 0 and seen["bin"] > 0
        r = await recv(ws)
        assert r[0] == "json" and r[1] == {"type": "status", "state": "idle"}, r

        # ---- 2) 끼어들기 ----
        await ws.send(json.dumps({"type": "text", "text": "긴 이야기를 해 줘"}))
        gen2 = None
        while True:
            r = await recv(ws, 120)
            if r[0] == "json" and r[1]["type"] == "speak_start":
                gen2 = r[1]["generation"]
            if r[0] == "bin":
                break
        assert gen2 is not None and gen2 > gen
        await ws.send(json.dumps({"type": "interrupt"}))
        idle = False
        late = 0
        try:
            while True:
                r = await recv(ws, 3)
                if r[0] == "json" and r[1] == {"type": "status", "state": "idle"}:
                    idle = True
                elif r[0] == "bin" and idle and r[1] == gen2:
                    late += 1
                elif r[0] == "json" and r[1]["type"] in ("speak_end", "speak_start", "assistant_done"):
                    raise AssertionError(f"interrupt 뒤 {r[1]}")
        except asyncio.TimeoutError:
            pass
        assert idle and late == 0, (idle, late)
        print("interrupt OK: idle 수신, 이후 옛 generation 오디오 0개")

        # ---- 3) 끼어든 뒤 다음 턴도 정상 ----
        await ws.send(json.dumps({"type": "text", "text": "짧게 다시"}))
        while True:
            r = await recv(ws, 120)
            if r[0] == "json" and r[1]["type"] == "speak_end":
                assert r[1]["generation"] > gen2
                break
        print("E2E OK")


asyncio.run(main())
