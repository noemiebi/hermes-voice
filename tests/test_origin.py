"""/ws Origin 검사(server/app.py _origin_allowed): 같은 기기 브라우저의 다른 웹페이지가 붙지 못하게."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import app as app_module  # noqa: E402  (엔진은 생성만, 모델 로드는 lifespan: with 없이 쓰면 안 돈다)
from server import config  # noqa: E402


class DummySession:
    """실제 VoiceSession 대신: 연결되면 알리고 바로 끝낸다."""

    def __init__(self, ws, *a):
        self.ws = ws

    async def run(self):
        await self.ws.send_text('{"type":"ok"}')
        await self.ws.close()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "VoiceSession", DummySession)
    monkeypatch.setattr(config, "VOICE_ALLOWED_ORIGINS", ())
    return TestClient(app_module.app)  # with 없이: lifespan(모델 로드) 안 돎


def connects(client, headers=None):
    try:
        with client.websocket_connect("/ws", headers=headers or {}) as ws:
            return ws.receive_json() == {"type": "ok"}
    except WebSocketDisconnect as e:
        assert e.code == 4403, e.code  # 실서버(uvicorn)에서는 핸드셰이크 HTTP 403
        return False


def test_no_origin_allowed(client):
    assert connects(client)


def test_same_origin_allowed(client):
    # TestClient의 Host는 "testserver"
    assert connects(client, {"origin": "http://testserver"})
    assert connects(client, {"origin": "https://h.ts.net:8464", "host": "h.ts.net:8464"})
    assert connects(client, {"origin": "https://h.ts.net", "host": "h.ts.net:443"})  # 기본 포트


def test_forwarded_host_allowed(client):
    assert connects(client, {"origin": "https://h.ts.net:8464", "host": "127.0.0.1:8480",
                             "x-forwarded-host": "h.ts.net:8464"})


def test_other_origin_rejected(client, caplog):
    assert not connects(client, {"origin": "http://evil.example"})
    assert "ws 거부: origin http://evil.example" in caplog.text
    assert not connects(client, {"origin": "null"})
    assert not connects(client, {"origin": "http://testserver:9999"})  # 포트 다름
    assert not connects(client, {"origin": "http://evil.example", "x-forwarded-host": "other.example"})


def test_allowlist(client, monkeypatch):
    monkeypatch.setattr(config, "VOICE_ALLOWED_ORIGINS", ("http://evil.example",))
    assert connects(client, {"origin": "http://evil.example"})
    assert not connects(client, {"origin": "http://other.example"})


def test_wildcard(client, monkeypatch):
    monkeypatch.setattr(config, "VOICE_ALLOWED_ORIGINS", ("*",))
    assert connects(client, {"origin": "http://evil.example"})
