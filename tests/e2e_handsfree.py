"""핸즈프리(양방향) WebSocket e2e. 서버를 먼저 띄운다(예: PORT=8482 .venv/bin/python -m server).
사용: .venv/bin/python tests/e2e_handsfree.py [ws://127.0.0.1:8482/ws]

1) conversation on -> 무음 1초 -> TTS 질문 음성 + 무음(실시간 100 ms 프레임, 마이크처럼 계속 전송)
   -> vad·transcript·speak_start·오디오 확인. 클라이언트 playback 상태도 흉내 낸다.
2) 첫 오디오를 받은 뒤 다시 말함(끼어들기) -> interrupt 수신, 이후 옛 generation 오디오 0개,
   끼어든 발화가 새 턴으로 처리되는지 확인 -> conversation off.
3) 새 연결에서 무음 + 백색잡음(-40 dBFS) 10초 -> 턴이 생기지 않는지 확인.
"""
import asyncio
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8482/ws"
OUT = Path(__file__).resolve().parent / "out"
SR = 16000
FRAME = SR // 10  # 100 ms
OUT_RATE = 44100

QUESTION = "가을에 가 볼 만한 국내 여행지 세 군데를 하나씩 설명해 줘."
BARGE = "잠깐만, 그만하고 삼 더하기 사는 얼마인지 말해 줘."


def tts_clip(name: str, text: str) -> np.ndarray:
    """TTS로 만든 16 kHz int16 음성(캐시). 앞뒤 무음은 잘라 둔다."""
    path = OUT / f"{name}.wav"
    if not path.exists():
        from server.tts import SAMPLE_RATE, TTSEngine
        eng = TTSEngine()
        eng.load_sync()
        assert eng.loaded, eng.error
        pcm = eng.synth_sync(text).astype(np.float32)
        idx = np.arange(0, len(pcm), SAMPLE_RATE / SR)
        pcm16k = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.int16)
        OUT.mkdir(exist_ok=True)
        sf.write(path, pcm16k, SR, subtype="PCM_16")
    a, sr = sf.read(path, dtype="int16")
    assert sr == SR
    e = np.array([np.sqrt(np.mean((a[i:i + 512].astype(np.float32) / 32768) ** 2)) for i in range(0, len(a), 512)])
    nz = np.where(e > 0.01)[0]
    return a[nz[0] * 512:(nz[-1] + 1) * 512]


def noise(sec: float, dbfs: float = -40.0, seed: int = 0) -> np.ndarray:
    rms = 10 ** (dbfs / 20)
    return np.clip(np.random.default_rng(seed).standard_normal(int(sec * SR)) * rms * 32768,
                   -32768, 32767).astype(np.int16)


class Mic:
    """실시간 속도로 100 ms 프레임을 계속 보내는 가짜 마이크. 평소엔 무음(또는 잡음)."""

    def __init__(self, ws, bed_dbfs=None):
        self.ws = ws
        self.queue: list[np.ndarray] = []
        self.bed_dbfs = bed_dbfs
        self.speech_started_at = None   # 지금 말하는 클립의 첫 프레임 전송 시각
        self.speech_ended_at = None     # 마지막 음성 프레임 전송 시각
        self._rng = np.random.default_rng(7)
        self._buf = np.zeros(0, dtype=np.int16)
        self._speaking = False
        self.stop = False

    def say(self, clip: np.ndarray, tail_silence: float = 0.0):
        self.queue.append(("speech", clip))
        if tail_silence:
            self.queue.append(("pad", np.zeros(int(tail_silence * SR), dtype=np.int16)))

    def _bed(self, n):
        if self.bed_dbfs is None:
            return np.zeros(n, dtype=np.int16)
        rms = 10 ** (self.bed_dbfs / 20) * 32768
        return np.clip(self._rng.standard_normal(n) * rms, -32768, 32767).astype(np.int16)

    async def run(self):
        t0 = time.perf_counter()
        n = 0
        cur = None
        pos = 0
        while not self.stop:
            if cur is None and self.queue:
                kind, cur = self.queue.pop(0)
                pos = 0
                if kind == "speech":
                    self._speaking = True
                    self.speech_started_at = None
            if cur is not None:
                frame = cur[pos:pos + FRAME]
                pos += FRAME
                if len(frame) < FRAME:
                    frame = np.concatenate([frame, self._bed(FRAME - len(frame))])
                if self._speaking and self.speech_started_at is None:
                    self.speech_started_at = time.perf_counter()
                if pos >= len(cur):
                    if self._speaking:
                        self.speech_ended_at = time.perf_counter()
                        self._speaking = False
                    cur = None
            else:
                frame = self._bed(FRAME)
            await self.ws.send(frame.astype("<i2").tobytes())
            n += 1
            # 실시간 속도 유지
            delay = t0 + n * 0.1 - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

    def idle(self):
        return not self.queue and not self._speaking


class Client:
    """수신 루프 + 클라이언트 재생 흉내(playback speaking/idle)."""

    def __init__(self, ws):
        self.ws = ws
        self.events: list[tuple[float, str, dict]] = []
        self.gen = None
        self.play_until = 0.0
        self.playing = False
        self.audio_by_gen: dict[int, int] = {}
        self.audio_after_interrupt: dict[int, int] = {}
        self.interrupts: list[tuple[float, int]] = []
        self.first_audio: dict[int, float] = {}

    async def recv_loop(self):
        async for m in self.ws:
            now = time.perf_counter()
            if isinstance(m, bytes):
                g = struct.unpack("<I", m[:4])[0]
                n = (len(m) - 4) // 2
                self.audio_by_gen[g] = self.audio_by_gen.get(g, 0) + 1
                for _, ig in self.interrupts:
                    if g < ig:
                        self.audio_after_interrupt[g] = self.audio_after_interrupt.get(g, 0) + 1
                if self.gen is not None and g != self.gen:
                    continue  # 클라이언트는 다른 generation 오디오를 버림
                self.first_audio.setdefault(g, now)
                self.play_until = max(now, self.play_until) + n / OUT_RATE
                if not self.playing:
                    self.playing = True
                    await self.ws.send(json.dumps({"type": "playback", "state": "speaking", "generation": g}))
                self.events.append((now, "bin", {"generation": g}))
                continue
            msg = json.loads(m)
            self.events.append((now, msg["type"], msg))
            t = msg["type"]
            if t == "speak_start":
                self.gen = msg["generation"]
            elif t == "interrupt":
                self.interrupts.append((now, msg["generation"]))
                self.gen = msg["generation"]
                self.play_until = 0
                if self.playing:
                    self.playing = False
                    await self.ws.send(json.dumps({"type": "playback", "state": "idle", "generation": self.gen}))
            elif t == "error":
                print("  ERROR", msg)

    async def playback_loop(self):
        while True:
            await asyncio.sleep(0.05)
            if self.playing and time.perf_counter() > self.play_until:
                self.playing = False
                await self.ws.send(json.dumps({"type": "playback", "state": "idle", "generation": self.gen}))

    async def wait_for(self, pred, timeout, start=0):
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            for i in range(start, len(self.events)):
                if pred(self.events[i]):
                    return i, self.events[i]
            await asyncio.sleep(0.02)
        raise AssertionError(f"시간 초과: {pred.__doc__ or pred}")


def ms(a, b):
    return int((b - a) * 1000)


async def hello(ws):
    await ws.send(json.dumps({"type": "hello", "resume": None, "client": "e2e-handsfree"}))
    while True:
        m = json.loads(await asyncio.wait_for(ws.recv(), 10))
        if m["type"] == "ready":
            return m


async def scenario_talk_and_barge():
    q = tts_clip("hf_question", QUESTION)
    b = tts_clip("hf_barge", BARGE)
    async with websockets.connect(URL, max_size=None) as ws:
        ready = await hello(ws)
        print("ready:", ready["session_id"])
        cl = Client(ws)
        rt = asyncio.create_task(cl.recv_loop())
        await ws.send(json.dumps({"type": "conversation", "on": True}))
        await ws.send(json.dumps({"type": "settings", "aec": True, "half_duplex": False}))
        mic = Mic(ws)
        mt = asyncio.create_task(mic.run())
        pt = asyncio.create_task(cl.playback_loop())
        try:
            await cl.wait_for(lambda e: e[1] == "status" and e[2]["state"] == "waiting", 5)
            await asyncio.sleep(1.0)
            mic.say(q, tail_silence=1.0)
            i, ev = await cl.wait_for(lambda e: e[1] == "vad" and e[2]["speech"], 10)
            print(f"[1] vad speech:true  (음성 시작 후 {ms(mic.speech_started_at, ev[0])} ms)")
            i, ev = await cl.wait_for(lambda e: e[1] == "vad" and not e[2]["speech"], 20, i)
            t_end = mic.speech_ended_at
            print(f"    vad speech:false (말 끝 후 {ms(t_end, ev[0])} ms)")
            i2, ev2 = await cl.wait_for(lambda e: e[1] == "status" and e[2]["state"] == "transcribing", 5, i)
            print(f"    status transcribing (말 끝 후 {ms(t_end, ev2[0])} ms)")
            i, ev = await cl.wait_for(lambda e: e[1] == "transcript", 20, i)
            print(f"    transcript {ev[2]['text']!r} STT {ev[2]['ms']} ms (말 끝 후 {ms(t_end, ev[0])} ms)")
            assert "여행" in ev[2]["text"], ev[2]
            i, ev = await cl.wait_for(lambda e: e[1] == "speak_start", 90, i)
            gen1 = ev[2]["generation"]
            print(f"    speak_start gen {gen1}: {ev[2]['text']}")
            i, ev = await cl.wait_for(lambda e: e[1] == "bin", 30, i)
            t_first = ev[0]
            print(f"    첫 오디오: 말 끝 -> {ms(t_end, t_first)} ms")

            # ---- 끼어들기 ----
            mic.say(b, tail_silence=1.5)
            i, ev = await cl.wait_for(lambda e: e[1] == "interrupt", 10, i)
            gen_int = ev[2]["generation"]
            assert gen_int > gen1, (gen_int, gen1)
            print(f"[2] interrupt gen {gen_int}: 끼어든 음성 시작 후 {ms(mic.speech_started_at, ev[0])} ms")
            i, ev = await cl.wait_for(lambda e: e[1] == "transcript", 30, i)
            print(f"    끼어든 발화 transcript {ev[2]['text']!r}")
            i, ev = await cl.wait_for(lambda e: e[1] == "speak_start", 90, i)
            gen2 = ev[2]["generation"]
            assert gen2 > gen1
            print(f"    새 턴 speak_start gen {gen2}: {ev[2]['text']}")
            await cl.wait_for(lambda e: e[1] == "speak_end", 90, i)
            late = sum(c for g, c in cl.audio_after_interrupt.items() if g <= gen1)
            print(f"    interrupt 뒤 옛 generation 오디오 {late}개, gen별 프레임 {cl.audio_by_gen}")
            assert late == 0, cl.audio_after_interrupt
            await cl.wait_for(lambda e: e[1] == "status" and e[2]["state"] == "waiting", 10, i)
            await ws.send(json.dumps({"type": "conversation", "on": False}))
            await cl.wait_for(lambda e: e[1] == "status" and e[2]["state"] == "idle", 5, i)
            states = [e[2]["state"] for e in cl.events if e[1] == "status"]
            print("    states:", states)
            print("    assistant_done:", [e[2]["text"] for e in cl.events if e[1] == "assistant_done"])
        finally:
            mic.stop = True
            for t in (mt, pt, rt):
                t.cancel()
    return {"end_to_first_audio_ms": ms(t_end, t_first)}


async def scenario_noise(seconds: float = 10.0):
    async with websockets.connect(URL, max_size=None) as ws:
        await hello(ws)
        cl = Client(ws)
        rt = asyncio.create_task(cl.recv_loop())
        await ws.send(json.dumps({"type": "conversation", "on": True}))
        mic = Mic(ws, bed_dbfs=None)
        mt = asyncio.create_task(mic.run())
        try:
            await asyncio.sleep(2.0)                       # 무음 2초
            mic.bed_dbfs = -40.0                            # 백색잡음 -40 dBFS 8초
            await asyncio.sleep(seconds - 2.0)
            await asyncio.sleep(1.0)
            bad = [e for e in cl.events if e[1] in ("vad", "transcript", "speak_start", "bin")
                   or (e[1] == "status" and e[2]["state"] not in ("waiting", "idle"))]
            print(f"[3] 무음+잡음 {seconds:.0f}초: 턴 관련 이벤트 {len(bad)}개 {[(e[1], e[2]) for e in bad][:5]}")
            assert not bad
            await ws.send(json.dumps({"type": "conversation", "on": False}))
        finally:
            mic.stop = True
            mt.cancel()
            rt.cancel()


async def main():
    r = await scenario_talk_and_barge()
    await scenario_noise()
    print("HANDSFREE E2E OK", r)


asyncio.run(main())
