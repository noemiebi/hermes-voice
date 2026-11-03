"""선택적 접속 토큰(VOICE_TOKEN): 가짜 WS로 VoiceSession.run()을 구동해 hello 인증을 시험."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import config  # noqa: E402
from server.session import VoiceSession  # noqa: E402


class FakeClient:
    host = "100.64.0.9"


class FakeWS:
    """receive()는 미리 넣은 메시지를 차례로 주고, 다 떨어지면 disconnect."""

    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.msgs, self.audio = [], []
        self.closed = None
        self.client = FakeClient()

    async def receive(self):
        await asyncio.sleep(0)
        if self.closed is not None or not self.incoming:
            return {"type": "websocket.disconnect"}
        m = self.incoming.pop(0)
        if isinstance(m, bytes):
            return {"type": "websocket.receive", "bytes": m}
        return {"type": "websocket.receive", "text": m if isinstance(m, str) else json.dumps(m)}

    async def send_text(self, t):
        self.msgs.append(json.loads(t))

    async def send_bytes(self, b):
        self.audio.append(b)

    async def close(self, code=1000):
        self.closed = code

    def types(self):
        return [m["type"] for m in self.msgs]


class Dummy:
    loaded = True


def run_session(incoming):
    ws = FakeWS(incoming)
    asyncio.run(asyncio.wait_for(VoiceSession(ws, Dummy(), Dummy(), Dummy()).run(), 5))
    return ws


def assert_rejected(ws):
    assert ws.msgs == [{"type": "error", "code": "unauthorized", "message": "접속 토큰이 틀립니다"}], ws.msgs
    assert ws.closed == 4401
    assert "ready" not in ws.types()


def test_no_token_configured_allows_hello_without_token(monkeypatch):
    monkeypatch.setattr(config, "VOICE_TOKEN", "")
    ws = run_session([{"type": "hello", "resume": None}])
    assert "ready" in ws.types()
    assert ws.closed is None


def test_no_token_configured_allows_non_hello_first(monkeypatch):
    monkeypatch.setattr(config, "VOICE_TOKEN", "")
    ws = run_session([{"type": "settings", "aec": True}, {"type": "hello"}])
    assert "ready" in ws.types()
    assert ws.closed is None


def test_matching_token_ready(monkeypatch):
    monkeypatch.setattr(config, "VOICE_TOKEN", "s3cret-tok")
    ws = run_session([{"type": "hello", "resume": None, "token": "s3cret-tok"},
                      {"type": "settings", "aec": True}])
    assert ws.types()[0] == "ready", ws.types()
    assert ws.closed is None
    assert not any(m.get("code") == "unauthorized" for m in ws.msgs)


def test_wrong_token_rejected(monkeypatch, caplog):
    monkeypatch.setattr(config, "VOICE_TOKEN", "s3cret-tok")
    with caplog.at_level("WARNING", logger="session"):
        ws = run_session([{"type": "hello", "resume": None, "token": "nope"}])
    assert_rejected(ws)
    assert "hello 거부: 토큰 불일치 (100.64.0.9)" in caplog.text


def test_missing_token_rejected(monkeypatch):
    monkeypatch.setattr(config, "VOICE_TOKEN", "s3cret-tok")
    assert_rejected(run_session([{"type": "hello", "resume": None}]))


def test_non_string_token_rejected(monkeypatch):
    monkeypatch.setattr(config, "VOICE_TOKEN", "s3cret-tok")
    assert_rejected(run_session([{"type": "hello", "token": 12345}]))


def test_first_message_not_hello_rejected(monkeypatch):
    monkeypatch.setattr(config, "VOICE_TOKEN", "s3cret-tok")
    # 토큰을 실어도 hello가 아니면 거부, 뒤따르는 hello도 처리하지 않는다
    ws = run_session([{"type": "text", "text": "안녕", "token": "s3cret-tok"},
                      {"type": "hello", "token": "s3cret-tok"}])
    assert_rejected(ws)


def test_binary_or_garbage_first_rejected(monkeypatch):
    monkeypatch.setattr(config, "VOICE_TOKEN", "s3cret-tok")
    assert_rejected(run_session([b"\0\0" * 160]))
    assert_rejected(run_session(["not json"]))
