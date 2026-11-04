"""헤르메스 API(/v1/chat/completions, SSE) 비동기 클라이언트 + 가짜 스트림(HERMES_MOCK=1)."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import httpx

from . import config

log = logging.getLogger("hermes")

VOICE_SYSTEM_NOTE = (
    "[음성 대화 모드] 지금 사용자와 한국어로 말로 대화하고 있다. 사용자 메시지는 음성 인식 결과라서 "
    "잘못 알아들은 단어나 말 더듬, 말 바꾸기가 섞여 있을 수 있다. 앞뒤 맥락으로 가장 그럴듯한 뜻을 잡고, "
    "정말 애매할 때만 짧게 되물어라. 답은 소리 내어 읽힌다: 구어체의 짧은 문장 1~3개로 답하고, "
    "사용자가 자세히 원할 때만 길게 말한다. 마크다운, 목록, 표, 코드 블록, URL, 이모지는 쓰지 마라. "
    "숫자와 단위는 읽기 쉽게 써라. 도구는 평소처럼 쓰되, 도구 결과를 확인하기 전에는 성공했다고 단정하지 마라."
)

DeltaCb = Callable[[str], Awaitable[None]]
ToolCb = Callable[[str, str, str], Awaitable[None]]  # (name, state started|completed, label)


class HermesError(Exception):
    pass


@dataclass
class TurnResult:
    text: str
    session_id: str
    finish_reason: Optional[str] = None


REASONING_MODELS = ("voice-fast",)  # HERMES_REASONING을 붙이는 모델


def reasoning_options(mode: str) -> Optional[dict]:
    """HERMES_REASONING 값 -> 헤르메스 body의 model_options(없으면 None = 헤르메스 설정값)."""
    mode = (mode or "").strip().lower()
    if mode in ("off", "none", "false", "0"):
        return {"reasoning": {"enabled": False}}
    if mode in ("minimal", "low", "medium", "high", "xhigh"):
        return {"reasoning": {"enabled": True, "effort": mode}}
    return None


def new_session_id() -> str:
    return f"voice-{uuid.uuid4().hex[:16]}"


class HermesClient:
    def __init__(self, base_url: str = config.HERMES_API_URL, api_key: str = config.HERMES_API_KEY,
                 mock: bool = config.HERMES_MOCK, reasoning: str = config.HERMES_REASONING):
        self.base_url, self.api_key, self.mock = base_url, api_key, mock
        self.reasoning = reasoning
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=300.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        if self.mock:
            return True
        try:
            r = await self._client.get(f"{self.base_url}/health", timeout=3.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def chat_stream(self, text: str, session_id: str, model: str,
                          on_delta: DeltaCb, on_tool: ToolCb, max_s: Optional[float] = None) -> TurnResult:
        """한 턴 응답 스트림. 전체가 max_s(기본 HERMES_TURN_MAX_S)를 넘으면 HermesError(읽기 타임아웃은 토큰 사이 간격만 봄)."""
        limit = config.HERMES_TURN_MAX_S if max_s is None else max_s
        try:
            async with asyncio.timeout(limit if limit and limit > 0 else None):
                if self.mock:
                    return await self._mock_stream(text, session_id, on_delta, on_tool)
                return await self._chat_stream(text, session_id, model, on_delta, on_tool)
        except TimeoutError:
            log.warning("헤르메스 응답 %.0f초 초과 -> 턴 종료", limit)
            raise HermesError(f"답이 너무 오래 걸려서 멈췄어요({limit:.0f}초 초과)") from None

    async def _chat_stream(self, text: str, session_id: str, model: str,
                           on_delta: DeltaCb, on_tool: ToolCb) -> TurnResult:
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "text/event-stream",
                   "X-Hermes-Session-Id": session_id}
        body = {"model": model, "stream": True,
                "messages": [{"role": "system", "content": VOICE_SYSTEM_NOTE},
                             {"role": "user", "content": text}]}
        # 헤르메스는 추론 옵션을 최상위 reasoning이 아니라 model_options.reasoning에서 읽는다.
        opts = reasoning_options(self.reasoning) if model in REASONING_MODELS else None
        if opts:
            body["model_options"] = opts
        parts: list[str] = []
        finish = None
        async with self._client.stream("POST", f"{self.base_url}/v1/chat/completions",
                                       headers=headers, json=body) as resp:
            if resp.status_code != 200:
                detail = (await resp.aread()).decode("utf-8", "replace")[:300]
                raise HermesError(f"헤르메스 API {resp.status_code}: {detail}")
            sid = resp.headers.get("X-Hermes-Session-Id") or session_id
            event = None
            data_lines: list[str] = []
            async for line in resp.aiter_lines():
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event = line[6:].strip()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                    continue
                if line.strip():
                    continue
                # 빈 줄 = 프레임 끝
                if not data_lines:
                    event = None
                    continue
                data = "\n".join(data_lines)
                ev, event, data_lines = event, None, []
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ev == "hermes.tool.progress":
                    status = obj.get("status")
                    state = "started" if status == "running" else "completed" if status == "completed" else None
                    if state:
                        await on_tool(obj.get("tool") or "", state, obj.get("label") or "", obj.get("toolCallId"))
                    continue
                if ev:  # hermes.status, approval.request 등은 무시
                    log.debug("event %s %s", ev, data[:200])
                    continue
                if "error" in obj and not obj.get("choices"):
                    raise HermesError(str(obj["error"].get("message") if isinstance(obj["error"], dict) else obj["error"]))
                for ch in obj.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content")
                    if delta:
                        parts.append(delta)
                        await on_delta(delta)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                if finish == "error" or (finish and finish not in ("stop", "length") and obj.get("error")):
                    err = obj.get("error") or {}
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    raise HermesError(f"헤르메스 응답 오류: {msg or finish}")
        return TurnResult("".join(parts), sid, finish)

    async def _mock_stream(self, text: str, session_id: str, on_delta: DeltaCb, on_tool: ToolCb) -> TurnResult:
        await asyncio.sleep(0.3)
        await on_tool("web_search", "started", "검색: " + text[:20])
        await asyncio.sleep(1.2)
        await on_tool("web_search", "completed", "")
        answer = (f"말씀하신 건 \"{text}\" 맞죠? 이건 가짜 헤르메스 응답이에요. "
                  "실제 API가 뜨면 진짜 답으로 바뀝니다. 오늘도 좋은 하루 보내세요!")
        toks = []
        i = 0
        while i < len(answer):
            n = random.randint(1, 4)
            toks.append(answer[i:i + n])
            i += n
        for tk in toks:
            await asyncio.sleep(0.03)
            await on_delta(tk)
        return TurnResult(answer, session_id, "stop")
