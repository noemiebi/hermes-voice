"""HERMES_REASONING 옵션 변환과 유휴 세션 판단."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import session  # noqa: E402
from server.hermes import reasoning_options  # noqa: E402


def test_reasoning_options():
    assert reasoning_options("off") == {"reasoning": {"enabled": False}}
    assert reasoning_options("low") == {"reasoning": {"enabled": True, "effort": "low"}}
    assert reasoning_options("default") is None
    assert reasoning_options("") is None


def test_session_idle():
    session._LAST_TURN["s1"] = 1000.0
    assert not session.session_is_idle("s1", now=1000.0 + 5 * 3600, idle_hours=6)
    assert session.session_is_idle("s1", now=1000.0 + 6 * 3600, idle_hours=6)
    assert not session.session_is_idle("s1", now=1000.0 + 99 * 3600, idle_hours=0)
    assert not session.session_is_idle("unknown", now=1e12, idle_hours=6)
    assert not session.session_is_idle(None, idle_hours=6)
