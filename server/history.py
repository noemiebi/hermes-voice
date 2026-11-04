"""화면 대화 기록과 '마지막 세션'을 서버에 보존(docs/프로토콜.md 'log'·'history').

iOS가 홈화면 앱의 localStorage를 지우는 일이 있어(2026-11-03), 클라이언트 기록의 정본을 서버 JSON 파일 하나에 둔다.
사용자 1명 전용이라 단일 파일·단일 프로세스 가정. 형식:
{"last_session": str|null, "sessions": {sid: {"items": [{"r","t","at"}], "updated": float}}}
"""
from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import tempfile
import threading
import time
import weakref
from pathlib import Path
from typing import Optional

log = logging.getLogger("history")

ROLES = ("user", "bot", "sys")
MAX_ITEMS = 300      # 세션당 최대 항목 수(오래된 것부터 버림)
MAX_SESSIONS = 5     # 최근 세션만 유지
SAVE_DEBOUNCE_S = 0.5  # 이벤트 루프 안에서는 이 시간 안의 변경을 모아 한 번에 저장(쓰기·fsync는 스레드에서)


def _flush_at_exit(ref) -> None:
    store = ref()
    if store is not None:
        store.flush()


class HistoryStore:
    def __init__(self, path: str | Path, max_items: int = MAX_ITEMS, max_sessions: int = MAX_SESSIONS):
        self.path = Path(path)
        self.max_items, self.max_sessions = max_items, max_sessions
        self._lock = threading.Lock()
        self._wlock = threading.Lock()   # 파일 쓰기 직렬화(스레드 저장과 flush가 겹치지 않게)
        self._ver = 0                    # 변경 번호
        self._saved_ver = 0              # 파일에 쓴 마지막 변경 번호
        self._timer: Optional[asyncio.TimerHandle] = None
        self._timer_loop = None
        self._data = self._load()
        atexit.register(_flush_at_exit, weakref.ref(self))  # 종료 때 미뤄 둔 저장을 마저 씀

    # ---------- 파일 ----------
    def _empty(self) -> dict:
        return {"last_session": None, "sessions": {}}

    def _load(self) -> dict:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return self._empty()
        except OSError as e:
            log.warning("기록 파일을 못 읽음(빈 상태로 시작): %s", e)
            return self._empty()
        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or not isinstance(data.get("sessions"), dict):
                raise ValueError("형식이 다름")
            last = data.get("last_session")
            sessions = {}
            for sid, s in data["sessions"].items():
                if isinstance(sid, str) and isinstance(s, dict) and isinstance(s.get("items"), list):
                    items = [it for it in s["items"] if self._valid(it)]
                    upd = s.get("updated")
                    sessions[sid] = {"items": items, "updated": float(upd) if isinstance(upd, (int, float)) else 0.0}
            return {"last_session": last if isinstance(last, str) and last else None, "sessions": sessions}
        except (ValueError, TypeError) as e:
            log.warning("기록 파일이 깨져 있음(빈 상태로 시작): %s: %s", self.path, e)
            return self._empty()

    def _write_text(self, text: str) -> None:
        """임시 파일에 쓴 뒤 os.replace로 바꿔치기(중간에 죽어도 이전 파일은 온전)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".history-", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _snapshot(self) -> tuple[int, str]:
        with self._lock:
            return self._ver, json.dumps(self._data, ensure_ascii=False)

    def _write_snapshot(self, ver: int, text: str) -> None:
        """스냅숏 하나를 파일에. 이미 더 새 변경을 썼으면 건너뜀(늦게 끝난 옛 스냅숏이 덮어쓰지 않게)."""
        with self._wlock:
            if ver <= self._saved_ver:
                return
            try:
                self._write_text(text)
            except OSError:
                log.exception("기록 저장 실패: %s", self.path)
                return
            self._saved_ver = ver

    def _save(self) -> None:
        """지금 상태를 바로(동기) 저장."""
        self._write_snapshot(*self._snapshot())

    def flush(self) -> None:
        """미뤄 둔 저장이 있으면 지금 동기로 씀(종료 때·테스트용)."""
        t, self._timer = self._timer, None
        if t is not None:
            t.cancel()
        if self._ver > self._saved_ver:
            self._save()

    def _kick(self) -> None:
        """디바운스 타이머 만료: 스냅숏은 루프에서 뜨고 쓰기·fsync는 스레드에서."""
        self._timer = None
        if self._ver <= self._saved_ver:
            return
        ver, text = self._snapshot()
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, self._write_snapshot, ver, text)
        fut.add_done_callback(lambda f: f.cancelled() or f.exception() is None
                              or log.error("기록 저장 실패: %r", f.exception()))

    def _save_safe(self) -> None:
        """변경 표시 뒤 저장. 이벤트 루프 안이면 SAVE_DEBOUNCE_S 모아서 스레드로, 밖이면 바로 동기 저장.
        (호출자가 self._lock을 쥔 상태에서 부름)"""
        self._ver += 1
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            ver, text = self._ver, json.dumps(self._data, ensure_ascii=False)
            self._write_snapshot(ver, text)
            return
        if self._timer is not None and self._timer_loop is not loop:
            self._timer.cancel()  # 다른(끝난) 루프에 걸린 타이머는 버리고 다시 건다
            self._timer = None
        if self._timer is None:
            self._timer_loop = loop
            self._timer = loop.call_later(SAVE_DEBOUNCE_S, self._kick)

    @staticmethod
    def _valid(item) -> bool:
        return (isinstance(item, dict) and item.get("r") in ROLES and isinstance(item.get("t"), str)
                and item["t"] != "")

    def _session(self, sid: str) -> dict:
        s = self._data["sessions"].get(sid)
        if s is None:
            s = self._data["sessions"][sid] = {"items": [], "updated": time.time()}
        return s

    def _prune(self) -> None:
        sessions = self._data["sessions"]
        if len(sessions) <= self.max_sessions:
            return
        keep_last = self._data.get("last_session")
        order = sorted(sessions, key=lambda k: sessions[k]["updated"], reverse=True)
        keep = set(order[: self.max_sessions])
        if keep_last in sessions and keep_last not in keep:  # 마지막 세션은 항상 남긴다
            keep.discard(order[self.max_sessions - 1])
            keep.add(keep_last)
        for k in list(sessions):
            if k not in keep:
                del sessions[k]

    # ---------- 공개 ----------
    def last_session(self) -> Optional[str]:
        return self._data.get("last_session")

    def set_last(self, sid: Optional[str]) -> None:
        if not isinstance(sid, str) or not sid:
            return
        with self._lock:
            self._data["last_session"] = sid
            self._session(sid)["updated"] = time.time()
            self._prune()
            self._save_safe()

    def append(self, sid: Optional[str], item, replace_last: bool = False) -> bool:
        """항목 하나 추가. r/t가 잘못됐거나 t가 비면 무시하고 False.
        replace_last: 마지막 항목이 같은 r이면 그걸 이 항목으로 바꾼다(합쳐진 발화 전사)."""
        if not isinstance(sid, str) or not sid or not self._valid(item):
            return False
        at = item.get("at")
        if not isinstance(at, (int, float)) or isinstance(at, bool):
            at = int(time.time() * 1000)
        with self._lock:
            s = self._session(sid)
            if replace_last and s["items"] and s["items"][-1].get("r") == item["r"]:
                s["items"].pop()
            s["items"].append({"r": item["r"], "t": item["t"], "at": int(at)})
            if len(s["items"]) > self.max_items:
                del s["items"][: len(s["items"]) - self.max_items]
            s["updated"] = time.time()
            self._prune()
            self._save_safe()
        return True

    def items(self, sid: Optional[str]) -> list[dict]:
        s = self._data["sessions"].get(sid or "")
        return [dict(it) for it in s["items"]] if s else []

    def touch(self, sid: Optional[str]) -> None:
        if not isinstance(sid, str) or not sid:
            return
        with self._lock:
            self._session(sid)["updated"] = time.time()
            self._prune()
            self._save_safe()
