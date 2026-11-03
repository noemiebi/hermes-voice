"""FastAPI 앱: / 정적(web/), /ws 음성 WebSocket, /health."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import mimetypes
from urllib.parse import urlsplit

mimetypes.add_type("audio/mp4", ".m4a")  # 기본값 audio/mp4a-latm은 iOS Safari가 재생하지 못할 수 있음

from . import config
from .hermes import HermesClient
from .history import HistoryStore
from . import pitch
from .session import VoiceSession
from .stt import STTEngine
from .tts import TTSEngine
from .tts_qwen import QwenTTSEngine, TTSRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("app")

stt = STTEngine(save_clips_dir=config.STT_SAVE_CLIPS_DIR)
supertonic = TTSEngine()
# TTS_ENGINE=qwen이면 두 엔진을 모두 올리고 문장마다 Qwen 우선, 안 되면 Supertonic(server/tts_qwen.py)
tts = TTSRouter(supertonic, QwenTTSEngine() if config.TTS_ENGINE == "qwen" else None, engine=config.TTS_ENGINE)
hermes = HermesClient()
history = HistoryStore(config.HISTORY_PATH)  # 화면 대화 기록·마지막 세션(앱 전체에 하나)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 두 모델은 서로 다른 워커 스레드에서 동시에 로드
    loads = asyncio.gather(stt.load(), tts.load())
    app.state.loading = loads
    pstore = pitch.get_store()
    pstore.profile.active_tts = [k for k, _ in pitch.tts_keys(tts)]
    log.info("음역 프로필: %s", pstore.profile.summary())

    async def learn_pitch():
        # TTS가 올라오면 현재 음색으로 문장 몇 개를 합성해 TTS 음역을 잰다(키별로 한 번, 음색이 바뀌면 다시)
        try:
            await asyncio.shield(loads)
        except Exception:  # noqa: BLE001 로드 실패는 lifespan 쪽 로그로 충분
            pass
        if not config.PITCH_GATE:
            return
        try:
            if await pitch.learn_tts(tts, pstore):
                log.info("음역 프로필: %s", pstore.profile.summary())
        except Exception:  # noqa: BLE001
            log.exception("음역 학습(TTS) 실패")

    pitch_task = asyncio.create_task(learn_pitch())
    if config.HERMES_MOCK:
        log.warning("HERMES_MOCK=1: 가짜 헤르메스 스트림 사용")
    yield
    pitch_task.cancel()
    pstore.flush()
    await tts.aclose()
    await hermes.aclose()


app = FastAPI(title="hermes-voice", lifespan=lifespan)


@app.get("/health")
async def health():
    if config.VOICE_TOKEN:  # 공개 노출 가능성: 내부 상태는 숨기고 살아 있는지만
        return {"ok": stt.loaded and tts.loaded}
    return {
        "ok": stt.loaded and tts.loaded,
        "stt": {"loaded": stt.loaded, "model": stt.model, "load_ms": stt.load_ms, "error": stt.error},
        "tts": {"loaded": tts.loaded, "engine": tts.engine, "provider": tts.provider, "load_ms": tts.load_ms,
                "error": tts.error, "fallbacks": tts.fallbacks, "fallbacks_10min": tts.recent_fallbacks(),
                "last_fallback": tts.last_fallback, "fallback_turns": tts.fallback_turns,
                "qwen": {"enabled": False, "ready": False} if tts.qwen is None else {"enabled": True,
                    "ready": tts.qwen.ready, "model": config.QWEN_TTS_MODEL, "voice": tts.qwen.voice,
                    "load_ms": tts.qwen.load_ms, "error": tts.qwen.error, "last": tts.qwen.last}},
        "hermes": {"mock": hermes.mock, "reachable": await hermes.health()},
    }


def _norm_host(netloc: str, scheme: str = "") -> str:
    """host[:port] 소문자화, 기본 포트(http 80, https 443)는 뗀다(scheme 모르면 둘 다)."""
    h = netloc.strip().lower()
    defaults = {"http": (":80",), "https": (":443",), "ws": (":80",), "wss": (":443",)}.get(scheme, (":80", ":443"))
    for d in defaults:
        if h.endswith(d):
            return h[: -len(d)]
    return h


def _origin_allowed(origin: str | None, host: str, forwarded_host: str = "") -> bool:
    """Origin 없음(비브라우저) / 같은 출처(Origin host == Host 또는 X-Forwarded-Host) / VOICE_ALLOWED_ORIGINS(정확 일치 또는 *)."""
    if origin is None:
        return True
    allowed = config.VOICE_ALLOWED_ORIGINS
    if "*" in allowed or origin.strip().rstrip("/") in allowed:
        return True
    u = urlsplit(origin.strip())
    if not u.netloc:
        return False  # "null" 등
    o = _norm_host(u.netloc, u.scheme.lower())
    hosts = [host] + [h for h in forwarded_host.split(",") if h.strip()]
    return any(h.strip() and o == _norm_host(h) for h in hosts)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    # WebSocket에는 CORS가 없다: 같은 기기 브라우저의 아무 웹페이지가 붙지 못하게 accept 전에 Origin을 본다.
    origin = ws.headers.get("origin")
    host = ws.headers.get("host", "")
    if not _origin_allowed(origin, host, ws.headers.get("x-forwarded-host", "")):
        log.warning("ws 거부: origin %s (host %s)", origin, host)
        await ws.close(code=4403)  # accept 전 close -> 핸드셰이크 403
        return
    await ws.accept()
    await VoiceSession(ws, stt, tts, hermes, history).run()


# 정적 파일은 매번 재검증(no-cache): 서비스 워커·앱 파일이 브라우저 HTTP 캐시에 묶여 갱신이 안 되던 문제 방지.
@app.middleware("http")
async def _no_cache_static(request, call_next):
    response = await call_next(request)
    if request.method == "GET" and not request.url.path.startswith(("/ws", "/api", "/health")):
        response.headers["Cache-Control"] = "no-cache"
    return response


# web/가 아직 없어도 기동되도록 check_dir=False (폴더는 만들지 않음)
app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True, check_dir=False), name="web")
