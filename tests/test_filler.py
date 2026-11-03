"""긴 대기 채움 말 스케줄러(_think): 가짜 WS/TTS/헤르메스로 session 레벨 시험. 시간 상수는 monkeypatch로 줄인다."""
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import config  # noqa: E402
from server.hermes import TurnResult  # noqa: E402
from server.session import FILLER_GENERIC, FILLER_TEXTS, FILLERS, VoiceSession, filler_for_tool  # noqa: E402
from tests.test_session_hold import FakeEP, FakeSTT, FakeWS  # noqa: E402

ANSWER = "오늘 서울은 맑아요."


class ShortTTS:
    """아주 짧은 오디오(10 ms)라 재생 여유(_play_lead_s)가 다음 채움 말을 막지 않는다."""
    loaded = True

    def __init__(self):
        self.texts = []

    async def synth(self, text, is_current=lambda: True):
        await asyncio.sleep(0.005)
        if not is_current():
            return None
        self.texts.append(text)
        return np.zeros(441, dtype=np.int16), 1


class SlowHermes:
    """first_delay 초 뒤 첫 토큰. tools=[(지연, 이름)]이면 그때 도구 started 이벤트."""

    def __init__(self, first_delay=0.0, tools=()):
        self.first_delay, self.tools, self.calls = first_delay, list(tools), []

    async def chat_stream(self, text, sid, model, on_delta, on_tool):
        self.calls.append(text)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        for at, name in self.tools:
            await asyncio.sleep(max(0.0, at - (loop.time() - t0)))
            await on_tool(name, "started", "")
        await asyncio.sleep(max(0.0, self.first_delay - (loop.time() - t0)))
        await on_delta(ANSWER + " ")
        return TurnResult(text=ANSWER, session_id=sid)


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(config, "FILLER_ENABLED", True)
    monkeypatch.setattr(config, "FILLER_FIRST_S", 0.2)
    monkeypatch.setattr(config, "FILLER_EVERY_S", 0.3)
    monkeypatch.setattr(config, "FILLER_MAX", 4)


async def ask(hermes, tts=None, wait=1.2):
    ws, tts = FakeWS(), tts or ShortTTS()
    s = VoiceSession(ws, FakeSTT(), tts, hermes)
    s.endpointer = FakeEP()
    s.style_note = ""
    s.speaker_task = asyncio.create_task(s._speaker())
    await s._on_text(json.dumps({"type": "hello", "resume": None}))
    await s._on_text(json.dumps({"type": "text", "text": "서울 날씨 어때?"}))
    await asyncio.sleep(wait)
    s.speaker_task.cancel()
    return s, ws, tts


def spoken(ws):
    return [m["text"] for m in ws.of("speak_start")]


def run(coro):
    return asyncio.run(coro)


def test_fillers_on_schedule_without_tool(fast):
    # 첫 토큰 0.65 s: 0.2 s에 첫 채움 말(FILLERS), 0.5 s에 일반 문구 하나, 0.8 s 전에 답 → 총 2개
    s, ws, _ = run(ask(SlowHermes(first_delay=0.65)))
    said = spoken(ws)
    assert said[-1] == ANSWER
    fills = said[:-1]
    assert len(fills) == 2, said
    assert fills[0] in FILLERS and fills[1] == FILLER_GENERIC[0]


def test_tool_event_speaks_now_and_uses_tool_phrase(fast):
    # 도구 시작 0.05 s → 즉시 첫 채움 말(4 s 안 기다림), 다음 간격에 도구 문구, 그다음은 일반 문구(같은 도구 반복 안 함)
    h = SlowHermes(first_delay=0.95, tools=[(0.05, "mcp__naver_mail__list")])
    s, ws, _ = run(ask(h))
    fills = [t for t in spoken(ws) if t != ANSWER]
    assert fills[0] in FILLERS
    assert fills[1] == "메일을 보고 있어요, 조금만요."
    assert fills[2] == FILLER_GENERIC[0]
    assert filler_for_tool("web_search") == "웹에서 찾아보는 중이에요, 조금만요."
    assert filler_for_tool("tool_describe") is None and filler_for_tool("weird") is None


def test_tool_event_triggers_before_first_delay(fast, monkeypatch):
    monkeypatch.setattr(config, "FILLER_FIRST_S", 5.0)
    s, ws, _ = run(ask(SlowHermes(first_delay=0.3, tools=[(0.05, "web_search")]), wait=0.6))
    said = spoken(ws)
    assert said[0] in FILLERS and said[-1] == ANSWER and len(said) == 2


def test_queued_filler_skipped_when_answer_arrives(fast, caplog):
    # 채움 말이 큐에 있는 동안(TTS가 앞 문장 합성 중) 첫 토큰이 오면 안 읽은 채움 말은 건너뜀
    import logging
    caplog.set_level(logging.INFO, logger="session")

    class BlockTTS(ShortTTS):
        async def synth(self, text, is_current=lambda: True):
            if not self.texts:
                await asyncio.sleep(0.5)   # 첫 채움 말 합성이 오래 걸림 → 그동안 다음 항목은 큐에 대기
            return await super().synth(text, is_current)

    async def main():
        ws, tts = FakeWS(), BlockTTS()
        h = SlowHermes(first_delay=0.4)
        s = VoiceSession(ws, FakeSTT(), tts, h)
        s.endpointer = FakeEP()
        s.style_note = ""
        s.speaker_task = asyncio.create_task(s._speaker())
        await s._on_text(json.dumps({"type": "hello", "resume": None}))
        await s._on_text(json.dumps({"type": "text", "text": "서울 날씨 어때?"}))
        await asyncio.sleep(0.25)            # 첫 채움 말이 합성 중
        from server.session import _Filler
        s.queue.put_nowait((s.generation, _Filler(FILLER_GENERIC[1])))  # 미재생 채움 말이 큐에 남은 상황
        await asyncio.sleep(1.0)
        s.speaker_task.cancel()
        return ws, tts
    ws, tts = run(main())
    said = spoken(ws)
    assert said[0] in FILLERS and said[-1] == ANSWER
    assert FILLER_GENERIC[1] not in said
    assert "채움 말 건너뜀: 답 도착" in caplog.text


def test_answer_sentence_equal_to_filler_text_not_skipped(fast):
    # 답 문장이 채움 말과 글이 같아도 답이면 읽고 heard에 들어간다(타입으로 구분)
    class H(SlowHermes):
        async def chat_stream(self, text, sid, model, on_delta, on_tool):
            await on_delta("거의 다 됐어요. ")
            return TurnResult(text="거의 다 됐어요.", session_id=sid)
    s, ws, _ = run(ask(H(), wait=0.3))
    assert spoken(ws) == ["거의 다 됐어요."]
    assert s._answer["heard"] == ["거의 다 됐어요."]


def test_disabled_no_fillers(fast, monkeypatch):
    monkeypatch.setattr(config, "FILLER_ENABLED", False)
    s, ws, _ = run(ask(SlowHermes(first_delay=0.65, tools=[(0.05, "web_search")])))
    assert spoken(ws) == [ANSWER]


def test_fillers_not_in_heard_but_in_echo_memory(fast):
    s, ws, tts = run(ask(SlowHermes(first_delay=0.65)))
    said = spoken(ws)
    assert len(said) >= 2 and all(t in FILLER_TEXTS for t in said[:-1])
    assert s._answer["heard"] == [ANSWER]
    assert said[0] in s.echo.recent()   # 에코 메모리에는 들어감(채움 말 에코도 받아쓰기에 걸릴 수 있음)


def test_interrupt_cancels_scheduler(fast):
    async def main():
        ws, tts = FakeWS(), ShortTTS()
        s = VoiceSession(ws, FakeSTT(), tts, SlowHermes(first_delay=5.0))
        s.endpointer = FakeEP()
        s.style_note = ""
        s.speaker_task = asyncio.create_task(s._speaker())
        await s._on_text(json.dumps({"type": "hello", "resume": None}))
        await s._on_text(json.dumps({"type": "text", "text": "서울 날씨 어때?"}))
        await asyncio.sleep(0.3)
        await s.interrupt()
        n = len(ws.of("speak_start"))
        await asyncio.sleep(0.8)
        s.speaker_task.cancel()
        return s, ws, n
    s, ws, n = run(main())
    assert n == 1 and len(ws.of("speak_start")) == 1
    assert s._filler_task is None
