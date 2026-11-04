"""server/history.py 저장소와 session의 hello 복원·log 처리."""
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.history import HistoryStore  # noqa: E402
from server.session import VoiceSession  # noqa: E402
from tests.test_session_hold import FakeHermes, FakeSTT, FakeTTS, FakeWS  # noqa: E402


def it(t, r="user", at=1):
    return {"r": r, "t": t, "at": at}


def test_append_validates_and_persists(tmp_path):
    p = tmp_path / "h.json"
    s = HistoryStore(p)
    assert s.last_session() is None and s.items("a") == []
    assert s.append("a", it("안녕"))
    assert not s.append("a", it(""))                 # 빈 글 무시
    assert not s.append("a", {"r": "x", "t": "?"})   # 모르는 역할
    assert not s.append("a", {"r": "bot"})           # t 없음
    assert not s.append(None, it("x"))
    s.set_last("a")
    s2 = HistoryStore(p)
    assert s2.last_session() == "a"
    assert s2.items("a") == [{"r": "user", "t": "안녕", "at": 1}]


def test_item_cap(tmp_path):
    s = HistoryStore(tmp_path / "h.json", max_items=300)
    for i in range(305):
        s.append("a", it(f"m{i}", at=i))
    items = HistoryStore(tmp_path / "h.json").items("a")
    assert len(items) == 300 and items[0]["t"] == "m5" and items[-1]["t"] == "m304"


def test_keeps_recent_five_sessions(tmp_path, monkeypatch):
    import server.history as h
    clock = [1000.0]
    monkeypatch.setattr(h.time, "time", lambda: clock[0])
    s = HistoryStore(tmp_path / "h.json")
    s.set_last("s0")
    for i in range(1, 7):
        clock[0] += 1
        s.append(f"s{i}", it("x"))
    # s0은 가장 오래됐지만 마지막 세션이라 남고, 그다음 오래된 s1·s2가 빠진다
    left = set(HistoryStore(tmp_path / "h.json")._data["sessions"])
    assert left == {"s0", "s3", "s4", "s5", "s6"}
    clock[0] += 1
    s.set_last("s6")
    s.touch("s3")
    clock[0] += 1
    s.append("s7", it("y"))
    assert set(s._data["sessions"]) == {"s3", "s4", "s5", "s6", "s7"}


def test_corrupt_file_starts_empty(tmp_path, caplog):
    p = tmp_path / "h.json"
    p.write_text("{깨짐", encoding="utf-8")
    s = HistoryStore(p)
    assert s.last_session() is None and s.items("a") == []
    assert any("깨져" in r.getMessage() for r in caplog.records)
    s.append("a", it("다시"))
    assert json.loads(p.read_text(encoding="utf-8"))["sessions"]["a"]["items"][0]["t"] == "다시"


def test_atomic_save(tmp_path, monkeypatch):
    p = tmp_path / "h.json"
    s = HistoryStore(p)
    s.append("a", it("첫"))
    before = p.read_text(encoding="utf-8")

    def boom(src, dst):
        raise OSError("디스크 오류")
    monkeypatch.setattr(os, "replace", boom)
    s.append("a", it("둘"))                    # 저장 실패는 로그만, 예외 안 냄
    assert p.read_text(encoding="utf-8") == before  # 기존 파일은 온전
    assert [f.name for f in tmp_path.iterdir()] == ["h.json"]  # 임시 파일도 남지 않음


# ---------- session ----------

def _session(store):
    ws = FakeWS()
    s = VoiceSession(ws, FakeSTT(), FakeTTS(), FakeHermes(), store)
    s.style_note = ""
    return s, ws


def test_hello_without_resume_restores_last_session(tmp_path):
    store = HistoryStore(tmp_path / "h.json")
    store.append("old-session-1234", it("지난번 질문"))
    store.append("old-session-1234", it("지난번 답", r="bot"))
    store.set_last("old-session-1234")

    async def go():
        s, ws = _session(store)
        await s._on_text(json.dumps({"type": "hello", "resume": None}))
        assert s.session_id == "old-session-1234"
        types = ws.types()
        assert types.index("ready") + 1 == types.index("history")
        h = ws.of("history")[0]
        assert h["session_id"] == "old-session-1234"
        assert [x["t"] for x in h["items"]] == ["지난번 질문", "지난번 답"]
        # 새 대화: 새 id가 마지막 세션이 된다
        await s._on_text(json.dumps({"type": "new_session"}))
        assert s.session_id != "old-session-1234" and store.last_session() == s.session_id
    asyncio.run(go())


def test_log_message_appends_to_current_session(tmp_path):
    store = HistoryStore(tmp_path / "h.json")

    async def go():
        s, ws = _session(store)
        await s._on_text(json.dumps({"type": "hello", "resume": "sid-abc"}))
        assert store.last_session() == "sid-abc"
        assert ws.of("history") == [{"type": "history", "session_id": "sid-abc", "items": []}]
        n = len(ws.msgs)
        await s._on_text(json.dumps({"type": "log", "item": it("안녕하세요", at=5)}))
        await s._on_text(json.dumps({"type": "log", "item": {"r": "bot", "t": ""}}))
        assert len(ws.msgs) == n  # 응답 없음
        store.flush()  # 루프 안 저장은 디바운스(스레드)라 바로 씀
        assert HistoryStore(tmp_path / "h.json").items("sid-abc") == [{"r": "user", "t": "안녕하세요", "at": 5}]
    asyncio.run(go())


def test_debounced_save_in_loop(tmp_path):
    """이벤트 루프 안에서는 저장을 모아 스레드로 쓴다(루프를 fsync로 막지 않음)."""
    import server.history as h
    p = tmp_path / "h.json"
    store = HistoryStore(p)

    async def go():
        for i in range(5):
            store.append("a", it(f"m{i}", at=i))
        assert not p.exists()  # 아직 디바운스 중
        await asyncio.sleep(h.SAVE_DEBOUNCE_S + 0.3)
        assert [x["t"] for x in HistoryStore(p).items("a")] == [f"m{i}" for i in range(5)]
        store.append("a", it("마지막", at=9))
    asyncio.run(go())
    store.flush()  # 루프가 끝난 뒤 남은 변경도 flush로 씀
    assert HistoryStore(p).items("a")[-1]["t"] == "마지막"
    store.append("a", it("루프 밖", at=10))  # 루프 밖은 바로 동기 저장
    assert HistoryStore(p).items("a")[-1]["t"] == "루프 밖"
