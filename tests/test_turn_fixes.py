"""2026-11-04 서버 버그 수정 회귀 시험: 연결 끊김·대화 끔 뒤 판정 정리, 말 이어짐 합치기, 덜 끝난 말 대기,
채움 말 문체(사용자 말), 에코 오판(사용자 질문), 스피커 예외 생존, 턴 덮어쓰기, 헤르메스 전체 시한, netpath, 들은 부분."""
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import config, netpath  # noqa: E402
from server.hermes import HermesClient, HermesError, TurnResult  # noqa: E402
from server.session import VoiceSession, _sentence_style, looks_unfinished, user_style_of  # noqa: E402
from server.vad import VADEvent  # noqa: E402
from tests.test_session_hold import FakeEP, FakeHermes, FakeSTT, FakeTTS, FakeWS, settle, setup  # noqa: E402


def run(coro):
    return asyncio.run(coro)


class RecSTT(FakeSTT):
    """받은 PCM 길이를 기록하고 delay만큼 걸리는 STT."""

    def __init__(self, delay=0.01):
        super().__init__()
        self.delay, self.pcms = delay, []

    async def transcribe(self, pcm):
        self.calls += 1
        self.pcms.append(len(pcm))
        await asyncio.sleep(self.delay)
        return self.text, 5


class WaitHermes:
    """release 전까지 첫 토큰을 내지 않는 헤르메스(생각 중)."""

    def __init__(self):
        self.calls, self.release = [], asyncio.Event()

    async def chat_stream(self, text, sid, model, on_delta, on_tool):
        self.calls.append(text)
        await self.release.wait()
        await on_delta("네. ")
        return TurnResult(text="네.", session_id=sid)


async def voice_session(stt=None, hermes=None, tts=None):
    ws = FakeWS()
    s = VoiceSession(ws, stt or RecSTT(), tts or FakeTTS(), hermes or WaitHermes())
    s.endpointer = FakeEP()
    s.style_note = ""
    s.speaker_task = asyncio.create_task(s._speaker())
    await s._on_text(json.dumps({"type": "hello", "resume": None}))
    await s._on_text(json.dumps({"type": "conversation", "on": True}))
    return s, ws


async def utter(s, pcm_s=1.0, barge=False):
    s.endpointer.in_speech = True
    await s._on_vad(VADEvent("start", barge=barge))
    s.endpointer.in_speech = False
    await s._on_vad(VADEvent("end", pcm=b"\0\0" * int(16000 * pcm_s), speech_ms=600))


# ---------- 2. 연결 끊김 뒤 hold 판정이 새 턴을 열지 않음 ----------

class DiscWS(FakeWS):
    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    async def receive(self):
        await self.gate.wait()
        return {"type": "websocket.disconnect"}


def test_disconnect_cancels_pending_hold():
    async def go():
        s, ws, stt, hermes, gen = await setup()
        stt.delay = 0.3

        async def slow(pcm):
            stt.calls += 1
            await asyncio.sleep(0.3)
            return "진짜 사용자 질문이에요", 5
        stt.transcribe = slow
        s.endpointer.in_speech, s.endpointer.speech_ms = True, 416
        await s._on_vad(VADEvent("start", barge=True))
        assert s.holding
        s.endpointer.in_speech = False
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=600))
        assert s._hold_task is not None and not s._hold_task.done()
        gate = asyncio.Event()
        s.ws = DiscWS(gate)
        gate.set()
        await s.run()
        await asyncio.sleep(0.5)
        assert hermes.calls == ["서울 날씨 어때?"]   # 끊긴 뒤 보이지 않는 헤르메스 요청 없음
        assert s._hold_task is None and not s.holding
        s._start_turn(text="닫힌 뒤")                  # 닫힌 세션은 턴을 열지 않음
        await settle()
        assert hermes.calls == ["서울 날씨 어때?"]
    run(go())


# ---------- 3. 대화 끔: 에코 판정 받아쓰기가 새 턴을 열지 않음 ----------

def test_conversation_off_cancels_echo_task():
    async def go():
        s, ws, stt, hermes, gen = await setup()
        hermes.release.set()
        await settle(40)
        stt.text = "새로운 질문 하나 할게요"

        async def slow(pcm):
            await asyncio.sleep(0.3)
            return stt.text, 5
        stt.transcribe = slow
        s._echo_pcm = b"\0\0" * 16000
        s._echo_task = asyncio.create_task(s._resolve_pending("echo", s.generation, s._echo_pcm))
        s._pitch_wait = {"at": 0, "n0": 0, "gen": gen, "have": 0}
        s._duck_skip = {"at": 0, "nonecho": 0}
        await s._on_text(json.dumps({"type": "conversation", "on": False}))
        assert s._echo_task is None and s._pitch_wait is None and s._duck_skip is None
        await asyncio.sleep(0.5)
        assert len(hermes.calls) == 1 and not ws.of("transcript")
    run(go())


# ---------- 4a. 받아쓰기 뒤(답 전) 바로 이어진 말은 끼어들기 대신 합침 ----------

def test_barge_right_after_stt_merges_without_note(monkeypatch):
    monkeypatch.setattr(config, "TURN_CONTINUE_WAIT_MS", 0)

    async def go():
        stt, hermes = RecSTT(), WaitHermes()
        s, ws = await voice_session(stt, hermes)
        stt.text = "오늘 일정 요약해주고"
        await utter(s, 1.0)
        await settle()
        assert s.state == "thinking" and len(hermes.calls) == 1
        stt.text = "오늘 일정 요약해주고 오늘 온 메일이랑 같이 알려줘"
        await utter(s, 0.5, barge=True)
        await settle()
        assert not ws.of("interrupt")
        assert stt.pcms[-1] == int(16000 * 1.5) * 2       # 앞 발화 + 새 발화를 같이 받아씀
        assert len(hermes.calls) == 2
        assert "끊었음" not in hermes.calls[-1] and hermes.calls[-1].endswith("같이 알려줘")
        tr = ws.of("transcript")
        assert tr[-1].get("replace_last") is True and "replace_last" not in tr[0]
    run(go())


def test_barge_after_merge_window_interrupts(monkeypatch):
    monkeypatch.setattr(config, "TURN_CONTINUE_WAIT_MS", 0)

    async def go():
        stt, hermes = RecSTT(), WaitHermes()
        s, ws = await voice_session(stt, hermes)
        now = [100.0]
        s.clock = lambda: now[0]
        stt.text = "오늘 일정 요약해 줘"
        await utter(s, 1.0)
        await settle()
        now[0] += config.TURN_MERGE_WINDOW_S + 0.5
        stt.text = "아니 그만"
        await utter(s, 0.5, barge=True)
        await settle()
        assert ws.of("interrupt")                    # 창이 지나면 예전처럼 끼어들기
        assert stt.pcms[-1] == int(16000 * 0.5) * 2
        assert "끊었음" in hermes.calls[-1]
    run(go())


# ---------- 4b. 덜 끝난 말은 잠깐 기다려 이어지는 말과 합침 ----------

def test_looks_unfinished():
    for t in ("오늘 일정 요약해주고", "오늘", "비가 오는데", "내일 비 오면", "철수랑", "가거나", "좋지만"):
        assert looks_unfinished(t), t
    for t in ("오늘 도착 예정인 택배 있냐고", "택배 있냐고", "간다고", "그렇다라고", "같이 가자고",
              "오늘 일정 알려줘", "그거 뭐야?", "확인했습니까", "고마워요", ""):
        assert not looks_unfinished(t), t


def test_unfinished_text_waits_and_merges(monkeypatch):
    monkeypatch.setattr(config, "TURN_CONTINUE_WAIT_MS", 300)

    async def go():
        stt, hermes = RecSTT(), WaitHermes()
        s, ws = await voice_session(stt, hermes)
        stt.text = "오늘 일정 요약해주고"
        await utter(s, 1.0)
        await settle(5)
        assert s.state == "transcribing" and not hermes.calls and not ws.of("transcript")
        stt.text = "오늘 일정 요약해주고 메일도 알려줘"
        await utter(s, 0.5)
        await asyncio.sleep(0.4)
        assert hermes.calls == ["오늘 일정 요약해주고 메일도 알려줘"]
        assert [m["text"] for m in ws.of("transcript")] == ["오늘 일정 요약해주고 메일도 알려줘"]
        assert stt.pcms[-1] == int(16000 * 1.5) * 2
    run(go())


def test_finished_quotative_does_not_wait(monkeypatch):
    monkeypatch.setattr(config, "TURN_CONTINUE_WAIT_MS", 2000)

    async def go():
        stt, hermes = RecSTT(), WaitHermes()
        s, ws = await voice_session(stt, hermes)
        stt.text = "오늘 도착 예정인 택배 있냐고"
        await utter(s, 1.0)
        await settle(10)
        assert hermes.calls == ["오늘 도착 예정인 택배 있냐고"]
    run(go())


# ---------- 5. 채움 말 문체: 답 근거가 없으면 사용자 말로 ----------

def test_casual_quotative_endings():
    for t in ("오늘 도착 예정인 택배 있냐고", "간다고", "그렇다라고", "같이 가자고", "요약해주고"):
        assert _sentence_style(t) == "casual", t
    assert user_style_of("오늘 도착 예정인 택배 있냐고") == "casual"
    assert user_style_of("택배 있나요?") == "formal"
    assert user_style_of("확인해 주세요.") == "formal"
    assert user_style_of("OK") is None


def test_filler_style_from_user_when_no_evidence(monkeypatch):
    async def go():
        s = VoiceSession(FakeWS(), FakeSTT(), FakeTTS(), FakeHermes())
        s.session_id = "sid"
        monkeypatch.setattr(config, "FILLER_STYLE", "auto")
        assert s._filler_style("오늘 도착 예정인 택배 있냐고") == "casual"
        assert s._filler_style("오늘 택배 있나요?") == "formal"
        assert s._filler_style("") == "formal"
        s._remember_reply("네, 확인했어요. 내일 도착해요.")   # 답 근거가 있으면 그쪽
        assert s._filler_style("택배 있냐고") == "formal"
        monkeypatch.setattr(config, "FILLER_STYLE", "formal")  # 강제 설정은 그대로
        s._replies = {"sid": None, "texts": []}
        assert s._filler_style("택배 있냐고") == "formal"
    run(go())


# ---------- 6. 답 낱말을 되짚는 사용자 질문은 에코로 버리지 않음 ----------

def _echo_session():
    s = VoiceSession(FakeWS(), FakeSTT(), FakeTTS(), FakeHermes())
    now = [1000.0]
    s.clock = lambda: now[0]
    s.echo.clock = lambda: now[0]
    for t in ("내일 아침 아홉 시에 개표 결과가 나와요.", "선호 순위 투표는 후보에 순위를 매기는 방식이에요.",
              "확인해 볼게요, 잠깐만요."):
        s.echo.add(t)
    return s, now


def test_user_question_repeating_answer_not_echo():
    s, now = _echo_session()
    # 재생 중 시작된 발화(echo_check): 퍼지 0.75였던 실기 사례
    assert s._echo_filter("선호 순위 투표가 뭐야", True) is None
    assert s._echo_filter("내일 아침 9시에 개표 결과가 나오냐고", True) is None
    # 재생과 무관한 발화: 전역 bigram 0.87이었던 실기 사례
    s._last_play, s._speech_at, s._aec_on = now[0] - 10, now[0], True
    assert s._echo_filter("내일 아침 9시에 개표 결과가 나오냐고", False) is None


def test_real_echo_still_filtered():
    s, now = _echo_session()
    assert s._echo_filter("확인해 볼게요.", True) is not None
    assert s._echo_filter("선호 순위 투표는 후보에 순위를 매기는", True) is not None
    # 재생 끝 직후(창 안) 시작된 긴 문장 그대로는 전역 규칙으로 버림
    s._aec_on = True
    s._last_play, s._speech_at = now[0] - 0.5, now[0]
    assert s._echo_filter("선호 순위 투표는 후보에 순위를 매기는 방식이에요", False)[0] == "에코 무시(전역)"
    # 재생 끝나고 한참 뒤면 전역 규칙을 쓰지 않음(사용자가 답을 따라 말함)
    s._last_play = now[0] - (config.ECHO_GLOBAL_WINDOW_S + 5)
    assert s._echo_filter("선호 순위 투표는 후보에 순위를 매기는 방식이에요", False) is None


# ---------- 7. 스피커 태스크는 문장 처리 예외에도 살아 있음 ----------

class FlakyTTS(FakeTTS):
    def __init__(self):
        self.n = 0

    async def synth(self, text, is_current=lambda: True):
        self.n += 1
        if self.n == 1:
            return ("잘못된 값",)   # 언패킹 ValueError
        return await super().synth(text, is_current)


def test_speaker_survives_exception(caplog):
    async def go():
        ws = FakeWS()
        s = VoiceSession(ws, FakeSTT(), FlakyTTS(), FakeHermes())
        s.speaker_task = asyncio.create_task(s._speaker())
        g = s.generation
        s.queue.put_nowait((g, "첫 문장이에요."))
        s.queue.put_nowait((g, "둘째 문장이에요."))
        await settle()
        assert not s.speaker_task.done()
        assert [m["text"] for m in ws.of("speak_start")] == ["둘째 문장이에요."]
        s.speaker_task.cancel()
    with caplog.at_level(logging.ERROR):
        run(go())
    assert "스피커: 항목 처리 실패" in caplog.text


# ---------- 8. 새 턴이 앞 턴을 끊고, 앞 턴 finally가 새 턴 채움 말을 끄지 않음 ----------

def test_start_turn_cancels_previous_and_keeps_new_filler(monkeypatch):
    monkeypatch.setattr(config, "FILLER_ENABLED", True)
    monkeypatch.setattr(config, "FILLER_FIRST_S", 30.0)

    async def go():
        s = VoiceSession(FakeWS(), FakeSTT(), FakeTTS(), WaitHermes())
        s.style_note = ""
        s._start_turn(text="첫 질문")
        await settle(3)
        t1 = s.turn_task
        s._start_turn(text="둘째 질문")
        await settle(3)
        assert t1.cancelled()
        assert s._filler_task is not None and not s._filler_task.done()
        await s._cancel_turn()
    run(go())


# ---------- 9. 헤르메스 전체 시한 ----------

def test_hermes_turn_deadline():
    async def go():
        h = HermesClient(mock=True)

        async def noop(*a, **k):
            pass
        t = time.perf_counter()
        with pytest.raises(HermesError):
            await h.chat_stream("안녕", "sid", "voice-fast", noop, noop, max_s=0.1)
        assert time.perf_counter() - t < 0.5
        await h.aclose()
    run(go())


# ---------- 1. netpath: TERM 전달, 시간 초과 때 프로세스 정리, 첫 실패 로그 ----------

def test_netpath_passes_term(tmp_path, monkeypatch):
    exe = tmp_path / "ts"
    exe.write_text('#!/bin/sh\n[ -n "$TERM" ] || { echo "GUI failed" >&2; exit 3; }\n'
                   'echo \'{"Peer": {"x": {"TailscaleIPs": ["100.1.2.3"], "CurAddr": "1.2.3.4:41641"}}}\'\n')
    exe.chmod(0o755)
    monkeypatch.setattr(netpath, "_binary", lambda: str(exe))
    monkeypatch.delenv("TERM", raising=False)
    netpath._cache.clear()
    assert run(netpath.peer_path("100.1.2.3")) == "direct 1.2.3.4:41641"


def test_netpath_timeout_kills_and_logs_once(tmp_path, monkeypatch, caplog):
    exe = tmp_path / "ts"
    exe.write_text("#!/bin/sh\nexec sleep 5\n")
    exe.chmod(0o755)
    monkeypatch.setattr(netpath, "_binary", lambda: str(exe))
    monkeypatch.setattr(netpath, "_warned", set())
    netpath._cache.clear()
    with caplog.at_level(logging.INFO, logger="netpath"):
        t = time.perf_counter()
        assert run(netpath.peer_path("100.9.9.9", timeout_s=0.2)) is None
        assert run(netpath.peer_path("100.9.9.9", timeout_s=0.2)) is None
        assert time.perf_counter() - t < 2.0
    assert sum("tailscale status 실패" in r.getMessage() for r in caplog.records) == 1


# ---------- 11. 끊긴 답 노트: 실제로 재생이 시작된 문장만 ----------

def test_heard_note_uses_played_position():
    s = VoiceSession(FakeWS(), FakeSTT(), FakeTTS(), FakeHermes())
    g = s.generation
    s._aec_on = True
    s._answer = {"gen": g, "heard": ["첫 문장이에요.", "둘째 문장이에요."], "starts": [0.0, 1500.0],
                 "done": False, "thinking": True, "first_tok": True}
    s.timeline.report(g, 400.0, s.clock())
    s.timeline.stop(s.clock(), 500.0)
    s._note_cut_answer()
    assert "첫 문장이에요." in s._interrupt_note and "둘째" not in s._interrupt_note
    # 재생 보고가 없는 답은 아직 안 들린 것
    s._interrupt_note = None
    s.timeline.reset(g - 1)
    s._note_cut_answer()
    assert s._interrupt_note == "(앞 질문의 답을 듣기 전에 사용자가 끊었음)"
