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
from server.history import HistoryStore  # noqa: E402
from server.session import (FILLER_GENERIC, FILLER_GENERIC_CASUAL, FILLER_TEXTS, FILLERS, FILLERS_CASUAL,  # noqa: E402
                            VoiceSession, filler_for_tool, filler_style_of)
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
    monkeypatch.setattr(config, "FILLER_STYLE", "auto")
    monkeypatch.setattr(config, "FILLER_TOOL_DELAY_S", 0.0)  # 아래 전용 시험에서만 켠다
    monkeypatch.setattr(config, "FILLER_GAP_S", 0.0)


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


# ---------- 타이밍 완화(FILLER_TOOL_DELAY_S, FILLER_GAP_S) ----------
def test_new_defaults_in_config_source():
    # .env를 다시 읽지 않게 config.py 소스의 기본값 문자열을 확인한다
    src = (Path(config.__file__)).read_text(encoding="utf-8")
    for name, val in (("FILLER_FIRST_S", "6.0"), ("FILLER_EVERY_S", "14.0"), ("FILLER_MAX", "3"),
                      ("FILLER_TOOL_DELAY_S", "2.0"), ("FILLER_GAP_S", "0.5")):
        assert f'os.environ.get("{name}", "{val}")' in src, name


def test_tool_delay_skips_filler_when_token_comes_soon(fast, monkeypatch):
    # 도구 시작 0.05 s, 지연 0.4 s → 0.45 s 전에 첫 토큰(0.25 s)이 오면 채움 말 없음
    monkeypatch.setattr(config, "FILLER_FIRST_S", 5.0)
    monkeypatch.setattr(config, "FILLER_TOOL_DELAY_S", 0.4)
    s, ws, _ = run(ask(SlowHermes(first_delay=0.25, tools=[(0.05, "web_search")]), wait=0.6))
    assert spoken(ws) == [ANSWER]


def test_tool_delay_then_filler(fast, monkeypatch):
    monkeypatch.setattr(config, "FILLER_FIRST_S", 5.0)
    monkeypatch.setattr(config, "FILLER_TOOL_DELAY_S", 0.2)
    s, ws, _ = run(ask(SlowHermes(first_delay=0.5, tools=[(0.05, "web_search")]), wait=0.8))
    said = spoken(ws)
    assert said[0] in FILLERS and said[-1] == ANSWER and len(said) == 2


def test_tool_delay_capped_by_first_delay(fast, monkeypatch):
    # 지연이 길어도 턴 시작 뒤 FILLER_FIRST_S(0.3 s)를 넘기지 않는다
    monkeypatch.setattr(config, "FILLER_FIRST_S", 0.3)
    monkeypatch.setattr(config, "FILLER_TOOL_DELAY_S", 5.0)
    monkeypatch.setattr(config, "FILLER_EVERY_S", 5.0)
    s, ws, _ = run(ask(SlowHermes(first_delay=0.5, tools=[(0.2, "web_search")]), wait=0.8))
    said = spoken(ws)
    assert said[0] in FILLERS and said[-1] == ANSWER and len(said) == 2


class TimedTTS(ShortTTS):
    """채움 말은 0.3 s 오디오, 합성 시작 시각을 기록한다."""

    def __init__(self):
        super().__init__()
        self.at = {}

    async def synth(self, text, is_current=lambda: True):
        self.at[text] = asyncio.get_running_loop().time()
        await asyncio.sleep(0.005)
        if not is_current():
            return None
        self.texts.append(text)
        n = int(0.3 * 44100) if text in FILLER_TEXTS else 441
        return np.zeros(n, dtype=np.int16), 1


def test_gap_after_played_filler(fast, monkeypatch, caplog):
    # 채움 말(0.3 s 재생) 직후 답이 와도 본문은 채움 말 재생 끝 + FILLER_GAP_S(0.4 s) 뒤에 나간다
    import logging
    caplog.set_level(logging.INFO, logger="session")
    monkeypatch.setattr(config, "FILLER_GAP_S", 0.4)
    s, ws, tts = run(ask(SlowHermes(first_delay=0.25), tts=TimedTTS(), wait=1.4))
    said = spoken(ws)
    assert said[0] in FILLERS and said[-1] == ANSWER and len(said) == 2
    gap = tts.at[ANSWER] - tts.at[said[0]]
    assert 0.3 + 0.4 - 0.05 <= gap < 1.1, gap
    assert "채움 말 뒤 여유" in caplog.text


def test_no_gap_without_filler(fast, monkeypatch, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="session")
    monkeypatch.setattr(config, "FILLER_GAP_S", 2.0)
    s, ws, tts = run(ask(SlowHermes(first_delay=0.05), tts=TimedTTS(), wait=0.4))
    assert spoken(ws) == [ANSWER]
    assert "채움 말 뒤 여유" not in caplog.text


def test_gap_aborts_on_interrupt(fast, monkeypatch):
    # 여유를 기다리는 중 끼어들면 본문은 안 나가고, 다음 턴 답은 바로 나간다(여유 대기에 묶이지 않음)
    monkeypatch.setattr(config, "FILLER_GAP_S", 5.0)

    async def main():
        ws, tts = FakeWS(), TimedTTS()
        h = SlowHermes(first_delay=0.25)
        s = VoiceSession(ws, FakeSTT(), tts, h)
        s.endpointer = FakeEP()
        s.style_note = ""
        s.speaker_task = asyncio.create_task(s._speaker())
        await s._on_text(json.dumps({"type": "hello", "resume": None}))
        await s._on_text(json.dumps({"type": "text", "text": "서울 날씨 어때?"}))
        await asyncio.sleep(0.6)             # 채움 말 재생 뒤 본문이 여유 대기 중
        assert ANSWER not in spoken(ws)
        await s.interrupt()
        h.first_delay = 0.0
        await s._on_text(json.dumps({"type": "text", "text": "다시 알려 줘"}))
        await asyncio.sleep(0.3)
        s.speaker_task.cancel()
        return ws
    said = spoken(run(main()))
    assert said[0] in FILLERS and said[1:] == [ANSWER], said


# ---------- 채움 말 문체(FILLER_STYLE) ----------
def test_style_judge_formal_casual_mixed_empty():
    assert filler_style_of(["오늘 서울은 맑아요.", "우산은 필요 없습니다.", "내일 다시 볼게요!", "괜찮으시죠?"]) == ("formal", 4)
    assert filler_style_of(["오늘 서울은 맑아. 우산은 필요 없어.", "내일 다시 볼게! 괜찮지?", "그건 좋은 생각이다.",
                            "뭐 먹을까?"]) == ("casual", 6)
    assert filler_style_of(["좋아요. 그래."])[0] == "formal"          # 같으면 formal
    assert filler_style_of([]) == ("formal", 0)
    assert filler_style_of(["", "OK", "🙂"]) == ("formal", 0)        # 근거 없음
    # 니다·니까는 다·까보다 먼저(formal), 최근 6문장만 본다
    assert filler_style_of(["확인했습니다. 그럴까요? 맞습니까?"]) == ("formal", 3)
    old_casual = "그래. " * 10
    assert filler_style_of([old_casual, "네, 알겠어요. " * 6])[0] == "formal"


def test_tool_phrase_casual():
    assert filler_for_tool("web_search", "casual") == "웹에서 찾아보는 중이야, 조금만."
    assert filler_for_tool("mcp__naver_mail__list", "casual") == "메일 보고 있어, 조금만."
    assert filler_for_tool("weird", "casual") is None
    assert "거의 다 됐어." in FILLER_TEXTS and "잠깐 확인해 볼게." in FILLER_TEXTS
    assert len(FILLERS_CASUAL) == len(FILLERS) and len(FILLER_GENERIC_CASUAL) == len(FILLER_GENERIC)


async def _ask_after(store, sid, hermes, wait=1.2):
    ws, tts = FakeWS(), ShortTTS()
    s = VoiceSession(ws, FakeSTT(), tts, hermes, store=store)
    s.endpointer = FakeEP()
    s.style_note = ""
    s.speaker_task = asyncio.create_task(s._speaker())
    await s._on_text(json.dumps({"type": "hello", "resume": sid}))
    await s._on_text(json.dumps({"type": "text", "text": "서울 날씨 어때?"}))
    await asyncio.sleep(wait)
    s.speaker_task.cancel()
    return s, ws


def _casual_store(tmp_path):
    store = HistoryStore(tmp_path / "history.json")
    store.append("sid-1", {"r": "user", "t": "안녕"})
    store.append("sid-1", {"r": "bot", "t": "안녕! 오늘 뭐 할까? 날씨 좋더라."})
    store.set_last("sid-1")
    return store


def test_casual_history_gives_casual_fillers(fast, tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="session")
    h = SlowHermes(first_delay=0.95, tools=[(0.05, "web_search")])
    s, ws = run(_ask_after(_casual_store(tmp_path), "sid-1", h))
    fills = [t for t in spoken(ws) if t != ANSWER]
    assert fills[0] in FILLERS_CASUAL
    assert fills[1] == "웹에서 찾아보는 중이야, 조금만."
    assert fills[2] == FILLER_GENERIC_CASUAL[0]
    assert "채움 말 문체: casual (근거 2문장)" in caplog.text


def test_casual_previous_answer_in_session_memory(fast):
    # 기록 저장소 없이도 같은 세션에서 앞서 받은 답(result.text)이 반말이면 다음 턴 채움 말은 반말
    class H(SlowHermes):
        async def chat_stream(self, text, sid, model, on_delta, on_tool):
            if not self.calls:
                self.calls.append(text)
                await on_delta("응, 알겠어. ")
                return TurnResult(text="응, 알겠어. 금방 찾아볼게.", session_id=sid)
            return await super().chat_stream(text, sid, model, on_delta, on_tool)

    async def main():
        ws, tts = FakeWS(), ShortTTS()
        s = VoiceSession(ws, FakeSTT(), tts, H(first_delay=0.65))
        s.endpointer = FakeEP()
        s.style_note = ""
        s.speaker_task = asyncio.create_task(s._speaker())
        await s._on_text(json.dumps({"type": "hello", "resume": None}))
        await s._on_text(json.dumps({"type": "text", "text": "찾아 줘"}))
        await asyncio.sleep(0.3)
        await s._on_text(json.dumps({"type": "text", "text": "서울 날씨 어때?"}))
        await asyncio.sleep(1.2)
        s.speaker_task.cancel()
        return ws
    said = spoken(run(main()))
    fills = [t for t in said if t in FILLER_TEXTS]
    assert fills and fills[0] in FILLERS_CASUAL and fills[1] == FILLER_GENERIC_CASUAL[0], said


def test_forced_formal_ignores_casual_history(fast, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FILLER_STYLE", "formal")
    h = SlowHermes(first_delay=0.65)
    s, ws = run(_ask_after(_casual_store(tmp_path), "sid-1", h))
    fills = [t for t in spoken(ws) if t != ANSWER]
    assert fills[0] in FILLERS and fills[1] == FILLER_GENERIC[0]
