"""잠정 끼어들기(hold/resume)·에코 필터 흐름: 가짜 WS/STT/TTS/헤르메스로 session 레벨 시험."""
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.hermes import TurnResult  # noqa: E402
from server.session import VoiceSession  # noqa: E402
from server.vad import VADEvent  # noqa: E402

ANSWER = "오늘 서울은 맑고 낮 기온은 십오 도 정도예요."


class FakeWS:
    def __init__(self):
        self.msgs, self.audio = [], []

    async def send_text(self, t):
        self.msgs.append(json.loads(t))

    async def send_bytes(self, b):
        self.audio.append(b)

    def types(self):
        return [m["type"] for m in self.msgs]

    def of(self, t):
        return [m for m in self.msgs if m["type"] == t]


class FakeSTT:
    loaded = True

    def __init__(self):
        self.text, self.calls = "", 0

    async def transcribe(self, pcm):
        self.calls += 1
        await asyncio.sleep(0.01)
        return self.text, 5


class FakeTTS:
    loaded = True

    async def synth(self, text, is_current=lambda: True):
        await asyncio.sleep(0.005)
        return (np.zeros(44100, dtype=np.int16), 1) if is_current() else None


class FakeHermes:
    """첫 문장을 내고, release가 set될 때까지 계속 '생각 중'."""

    def __init__(self):
        self.calls, self.release = [], asyncio.Event()
        self.more = []

    async def chat_stream(self, text, sid, model, on_delta, on_tool):
        self.calls.append(text)
        await on_delta(ANSWER + " ")
        await self.release.wait()
        for s in self.more:
            await on_delta(s + " ")
        return TurnResult(text=ANSWER, session_id=sid)


class FakeEP:
    def __init__(self):
        self.in_speech, self.speech_ms, self.last_probs, self.events = False, 0, [], []

    def reset(self):
        pass

    def feed(self, data):
        ev, self.events = self.events, []
        return ev


async def settle(n=20):
    for _ in range(n):
        await asyncio.sleep(0.01)


async def setup():
    ws, stt, hermes = FakeWS(), FakeSTT(), FakeHermes()
    s = VoiceSession(ws, stt, FakeTTS(), hermes)
    s.endpointer = FakeEP()
    s.hold_mode = "pause"   # 이 파일은 일시정지 hold 동작을 시험(덕킹은 tests/test_duck_hold.py)
    s.style_note = ""
    s.speaker_task = asyncio.create_task(s._speaker())
    await s._on_text(json.dumps({"type": "hello", "resume": None}))
    await s._on_text(json.dumps({"type": "conversation", "on": True}))
    await s._on_text(json.dumps({"type": "text", "text": "서울 날씨 어때?"}))
    await settle()
    assert ws.of("speak_start"), ws.types()
    gen = s.generation
    await s._on_text(json.dumps({"type": "playback", "state": "speaking", "generation": gen}))
    return s, ws, stt, hermes, gen


async def barge_start(s):
    s.endpointer.in_speech, s.endpointer.speech_ms = True, 416
    await s._on_vad(VADEvent("start", barge=True))


async def speech_end(s, kind="end"):
    s.endpointer.in_speech, s.endpointer.speech_ms = False, 0
    await s._on_vad(VADEvent(kind, pcm=b"\0\0" * 16000, speech_ms=600))
    await settle()


def run(coro):
    asyncio.run(coro)


def test_echo_resumes_without_transcript():
    async def go():
        s, ws, stt, hermes, gen = await setup()
        await barge_start(s)
        assert ws.of("hold") == [{"type": "hold", "generation": gen}]
        assert s.generation == gen and s.state == "listening"
        # hold 중에는 새 문장이 와도 speak_start를 보내지 않는다
        n_start = len(ws.of("speak_start"))
        hermes.more = ["저녁에는 조금 쌀쌀해질 수 있어요."]
        hermes.release.set()
        await settle()
        assert len(ws.of("speak_start")) == n_start
        stt.text = "오늘 서울은 맑고 낮 기온은"   # 에코를 받아씀
        await speech_end(s)
        assert ws.of("resume") == [{"type": "resume", "generation": gen}]
        assert not ws.of("transcript") and not ws.of("interrupt")
        assert s.generation == gen and not s.holding
        await settle()
        assert len(ws.of("speak_start")) == n_start + 1   # 재개 뒤 남은 문장 전송
        assert ws.of("speak_end")
        assert hermes.calls == ["서울 날씨 어때?"]
    run(go())


def test_long_speech_confirms_interrupt():
    async def go():
        s, ws, stt, hermes, gen = await setup()
        await barge_start(s)
        assert s.holding
        s.endpointer.speech_ms = 416 + 900
        await s._on_audio(b"\0\0" * 1600)
        assert ws.of("interrupt") == [{"type": "interrupt", "generation": gen + 1}]
        assert not s.holding and s.generation == gen + 1
        stt.text = "아니 그거 말고 내일 부산 날씨 알려 줘"
        await speech_end(s)
        assert ws.of("transcript")[0]["text"] == stt.text
        assert len(hermes.calls) == 2
        assert "중간에 끊었음" in hermes.calls[1] and stt.text in hermes.calls[1]
        assert not ws.of("resume")
    run(go())


def test_short_real_speech_interrupts_with_transcript():
    async def go():
        s, ws, stt, hermes, gen = await setup()
        await barge_start(s)
        stt.text = "그만 멈춰"
        await speech_end(s)
        assert ws.of("interrupt") == [{"type": "interrupt", "generation": gen + 1}]
        assert ws.of("transcript")[0]["text"] == "그만 멈춰"
        assert stt.calls == 1      # 다시 받아쓰지 않음
        assert hermes.calls[-1].endswith("그만 멈춰") and len(hermes.calls) == 2
        assert not ws.of("resume")
        assert s.state in ("thinking", "speaking")
    run(go())


def test_short_noise_discard_resumes():
    async def go():
        s, ws, stt, hermes, gen = await setup()
        await barge_start(s)
        await speech_end(s, kind="discard")
        assert ws.of("resume") == [{"type": "resume", "generation": gen}]
        assert stt.calls == 0 and s.generation == gen
    run(go())


def test_tail_echo_after_playback_dropped():
    async def go():
        s, ws, stt, hermes, gen = await setup()
        hermes.release.set()
        await settle()
        assert ws.of("speak_end")
        await s._on_text(json.dumps({"type": "playback", "state": "idle", "generation": gen}))
        assert s._tail_active() and not s._barge_mode()
        s.endpointer.in_speech = True
        await s._on_vad(VADEvent("start", tail=True))
        stt.text = "십오 도 정도예요"
        await speech_end(s)
        assert not ws.of("transcript") and not ws.of("hold") and not ws.of("interrupt")
        assert len(hermes.calls) == 1 and s.state == "waiting"
        # 꼬리 구간이 지나면 같은 말도 사용자 말로 처리
        s._tail_until = 0
        s.endpointer.in_speech = True
        await s._on_vad(VADEvent("start"))
        await speech_end(s)
        assert ws.of("transcript") and len(hermes.calls) == 2
    run(go())


def test_barge_mode_not_stale_after_playback():
    async def go():
        s, ws, stt, hermes, gen = await setup()
        hermes.release.set()
        await settle()
        s.state = "speaking"   # stale state여도
        s.playback = "idle"
        assert not s._barge_mode()
        # playback idle을 못 받아도 추정 재생 끝 + 여유가 지나면 재생 아님
        s.playback = "speaking"
        s._play_until = s.clock() - 10
        assert not s._barge_mode()
        s._note_cut_answer()
        assert s._interrupt_note is None
    run(go())


def test_speaking_falls_back_to_thinking_between_sentences():
    async def go():
        s, ws, stt, hermes, gen = await setup()
        assert s.state == "speaking"
        await asyncio.sleep(1.0 + 0.6 + 0.3)   # 1초 오디오 추정 재생 끝 + 600 ms 유예
        assert s.state == "thinking"
        hermes.more = ["저녁에는 조금 쌀쌀해질 수 있어요."]
        hermes.release.set()
        await settle()
        states = [m["state"] for m in ws.of("status")]
        assert states[-3:] == ["thinking", "speaking", "waiting"], states
    run(go())


def test_thinking_barge_inside_tail_holds_then_resumes_on_echo():
    async def go():
        s, ws, stt, hermes, gen = await setup()
        await s._on_text(json.dumps({"type": "playback", "state": "idle", "generation": gen}))
        assert s._barge_mode()              # 턴이 아직 생각 중
        await barge_start(s)
        assert ws.of("hold") and not ws.of("interrupt")
        stt.text = "낮 기온은 십오 도"
        await speech_end(s)
        assert ws.of("resume") and s._turn_running() and s.state == "thinking"
    run(go())


def test_diag_logged_without_reply(caplog):
    """클라이언트 diag는 서버 로그 `client:`로만 남고 응답·오류가 없다. 문자열이 아니면 무시."""
    async def go():
        ws = FakeWS()
        s = VoiceSession(ws, FakeSTT(), FakeTTS(), FakeHermes())
        await s._on_text(json.dumps({"type": "hello", "resume": None}))
        n = len(ws.msgs)
        with caplog.at_level("INFO", logger="session"):
            await s._on_text(json.dumps({"type": "diag", "text": "[mic] track mute " + "x" * 600}))
            await s._on_text(json.dumps({"type": "diag", "text": 123}))
        assert len(ws.msgs) == n, ws.types()[n:]
        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("client: ")]
        assert len(lines) == 1
        assert lines[0].startswith("client: [mic] track mute")
        assert len(lines[0]) == len("client: ") + 400

    run(go())
