"""화면 대화 기록과 '마지막 세션'을 서버에 보존(docs/프로토콜.md 'log'·'history').

iOS가 홈화면 앱의 localStorage를 지우는 일이 있어(2026-11-03), 클라이언트 기록의 정본을 서버 JSON 파일 하나에 둔다.
사용자 1명 전용이라 단일 파일·단일 프로세스 가정. 형식:
{"last_session": str|null, "sessions": {sid: {"items": [{"r","t","at"}], "updated": float}}}
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("history")

ROLES = ("user", "bot", "sys")
MAX_ITEMS = 300      # 세션당 최대 항목 수(오래된 것부터 버림)
MAX_SESSIONS = 5     # 최근 세션만 유지


class HistoryStore:
    def __init__(self, path: str | Path, max_items: int = MAX_ITEMS, max_sessions: int = MAX_SESSIONS):
        self.path = Path(path)
        self.max_items, self.max_sessions = max_items, max_sessions
        self._lock = threading.Lock()
        self._data = self._load()

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

    def _save(self) -> None:
        """임시 파일에 쓴 뒤 os.replace로 바꿔치기(중간에 죽어도 이전 파일은 온전)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".history-", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _save_safe(self) -> None:
        try:
            self._save()
        except OSError:
            log.exception("기록 저장 실패: %s", self.path)

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

    def append(self, sid: Optional[str], item) -> bool:
        """항목 하나 추가. r/t가 잘못됐거나 t가 비면 무시하고 False."""
        if not isinstance(sid, str) or not sid or not self._valid(item):
            return False
        at = item.get("at")
        if not isinstance(at, (int, float)) or isinstance(at, bool):
            at = int(time.time() * 1000)
        with self._lock:
            s = self._session(sid)
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
