"""WebSocket 연결 하나의 상태 기계와 턴 파이프라인."""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import random
import struct
import time
from typing import Optional

import httpx
import numpy as np
from fastapi import WebSocket

from . import config
from .aec_gate import EchoGate, PlaybackTimeline, ReferenceBuffer
from .echo import EchoMemory
from .hermes import HermesClient, HermesError, new_session_id
from .history import HistoryStore
from .sentences import SentenceSplitter
from .stt import STTEngine
from .tts import SAMPLE_RATE as OUT_RATE, TTSEngine
from .tts_text import normalize_for_tts
from .vad import Endpointer, VADEvent, VADParams

log = logging.getLogger("session")

MODELS = ("voice-fast", "voice-deep")
MODEL_DISPLAY = config.MODEL_LABELS  # 환경변수 MODEL_LABEL_FAST/DEEP(비우면 노트 생략)
FILLERS = ("잠깐 확인해 볼게요.", "잠시만요, 찾아볼게요.", "확인해 볼게요, 잠깐만요.")
CHUNK_SAMPLES = OUT_RATE // 2          # 약 0.5초씩 전송
MAX_UTTERANCE_BYTES = 16000 * 2 * 90   # 90초
_END = object()                        # 턴 끝 표시
HEARD_NOTE_CHARS = 120                 # 끊긴 답 노트에 넣는 "들은 부분" 최대 길이
THINK_GRACE_S = 0.6                    # 재생 끝 뒤 다음 문장이 이만큼 안 오면 speaking -> thinking
ECHO_GATE_FRAC = 0.7                   # 발화 음성 창 중 에코 우세 비율이 이 이상이면 받아쓰기 없이 에코로 봄
ECHO_GATE_MIN_JUDGED = 5               # 그 비율을 믿으려면 판정된 음성 창이 이만큼은 있어야 함
DUCK_EARLY_WINDOWS = 3                 # 덕킹 조기 판정: 연속 창 수(96 ms)
DUCK_KEEP_DB = 3.0                     # 덕킹 조기 판정: pre보다 이만큼도 안 내려간 창이 이어지면 사용자 말
RESUME_LEAD_S = 0.06                   # 클라이언트가 resume 뒤 재생을 다시 시작하기까지(now + 0.06)
PLAYBACK_GRACE_S = 3.0                 # playback idle을 못 받아도 추정 재생 끝 + 이 시간 뒤엔 재생 아님으로 봄
# session_id -> 마지막 턴이 끝난 시각(time.time). 재연결(resume)해도 유휴 판단이 이어지게 모듈 단위로 둔다.
_LAST_TURN: dict[str, float] = {}


def session_is_idle(session_id: Optional[str], now: Optional[float] = None,
                    idle_hours: Optional[float] = None) -> bool:
    """마지막 턴 뒤 idle_hours(기본 VOICE_SESSION_IDLE_HOURS)가 지났으면 True. 기록 없거나 0이면 False."""
    hours = config.VOICE_SESSION_IDLE_HOURS if idle_hours is None else idle_hours
    last = _LAST_TURN.get(session_id or "")
    if hours <= 0 or last is None:
        return False
    return ((time.time() if now is None else now) - last) >= hours * 3600


class VoiceSession:
    def __init__(self, ws: WebSocket, stt: STTEngine, tts: TTSEngine, hermes: HermesClient,
                 store: Optional[HistoryStore] = None):
        self.ws, self.stt, self.tts, self.hermes = ws, stt, tts, hermes
        self.store = store                # 화면 대화 기록·마지막 세션(server/history.py). None이면 기록 안 함
        self.session_id: Optional[str] = None
        self.model = "voice-fast"
        self.state = "idle"
        self.generation = 0
        self.listening = False
        self.audio = bytearray()
        self.turn_task: Optional[asyncio.Task] = None
        self.queue: asyncio.Queue = asyncio.Queue()
        self.speaker_task: Optional[asyncio.Task] = None
        self._send_lock = asyncio.Lock()
        self._closed = False
        # 핸즈프리
        self.conversation = False
        self.playback = "idle"            # 클라이언트 스피커 상태
        self.playback_gen = 0
        self.half_duplex = False
        self.aec = True
        self.vad_params = VADParams.from_env()
        self.endpointer: Optional[Endpointer] = None
        self._carry = b""                 # 받아쓰는 중 말이 이어지면 앞 발화를 붙여 다시 처리
        self._turn_pcm = b""
        self._speech_wall: Optional[float] = None
        # 끼어들기 뒤 헤르메스에 알릴 노트
        self._answer = {"gen": 0, "heard": [], "done": False}
        self._interrupt_note: Optional[str] = None
        # 스피커 에코 대책
        self.clock = time.monotonic
        self.echo = EchoMemory(config.ECHO_MEMORY_S, clock=lambda: self.clock())
        self.hold_confirm_ms = config.HOLD_CONFIRM_MS
        self.echo_tail_s = config.ECHO_TAIL_MS / 1000
        self.holding = False              # 잠정 끼어들기(hold) 중: 재생을 멈춰 두고 진짜 말인지 판정
        self._hold_gen = 0
        self._hold_event = asyncio.Event()
        self._hold_event.set()
        self._hold_task: Optional[asyncio.Task] = None
        self._hold_pcm = b""
        self._hold_base_ms = 0
        self._hold_at = 0.0
        self._speaking_now = False        # 스피커 태스크가 문장을 합성·전송하는 중
        self._play_until = 0.0            # 보낸 오디오 길이로 추정한 클라이언트 재생 끝(clock). pos_ms 없는 옛 클라이언트용
        self._last_report = -1e9          # 마지막 playback speaking(pos_ms) 보고 시각
        self._sched: list[tuple[float, float]] = []  # 이 generation 문장별 (서버 기준 예상 시작, 길이 s): 재생 지연 진단
        self._sched_gen = -1
        self._lag_logged = -1e9
        self._tail_until = 0.0            # 이 시각까지 에코 꼬리 가드
        self._echo_check = False          # 지금 발화는 STT 뒤 에코 비교 대상
        self._turn_echo = False
        self._inlog = [0.0, 0, 0.0]       # VOICE_INPUT_LOG: 제곱합, 샘플 수, VAD 최대
        # 에코 대조 감지기(server/aec_gate.py): 받아쓰기 전에 에코를 가림
        self.aec_ref = ReferenceBuffer()  # generation별 TTS 참조(16 kHz)
        self.timeline = PlaybackTimeline()  # 클라이언트 pos_ms로 만든 재생 시간축
        self.gate = EchoGate()
        self._aec_on = False              # pos_ms를 보내는 클라이언트일 때만 감지기 사용
        self._gatelog = [0, 0, 0, -1.0, 0.0, 0]  # VOICE_INPUT_LOG: 창 수, 판정 수, 에코 수, 최대 상관, expected 합, 개수
        self._gate_recent: list = []      # 최근 창 판정(hold 로그용: 비에코 근거 집계)
        # hold 덕킹 능동 검사(HOLD_MODE=duck)
        self.hold_mode = config.HOLD_MODE if config.HOLD_MODE in ("duck", "pause") else "duck"
        self._hold_mode = "pause"         # 지금 hold의 방식
        self._mic_hist: list[tuple[float, float]] = []  # (시각, 창 평균 제곱) 최근 3초
        self._duck: Optional[dict] = None  # 진행 중인 덕킹 검사 {"at", "pre_db"}
        self._duck_skip: Optional[dict] = None  # 확실한 에코로 덕킹을 생략한 발화 {"at", "nonecho"}
        # 덕킹 검사가 에코로 확정돼 재생 복귀한 발화 {"at", "gen", "pre_db", "rechecked", "run", "gap", "best"}
        # (run/best: 복귀 뒤 pre + DUCK_ECHO_RISE_DB 이상인 창 연속 수/최대, gap: 연속 중 빈 창 수)
        self._duck_echo: Optional[dict] = None
        self._stale_play = -1e9           # 옛 generation(이미 끝낸 턴)의 speaking 보고 마지막 시각: 클라이언트가 아직 재생 중
        self._stale_floor = 0             # 이 generation 미만 보고는 무시(interrupt로 클라이언트가 이미 버린 재생)
        self._echo_task: Optional[asyncio.Task] = None  # 재생 중 끝난 에코 검사 대상 발화의 받아쓰기 판정
        self._echo_pcm = b""
        self.style_note = config.VOICE_STYLE_NOTE

    # ---------- 송신 ----------
    async def send_json(self, obj: dict) -> None:
        if self._closed:
            return
        async with self._send_lock:
            try:
                await self.ws.send_text(json.dumps(obj, ensure_ascii=False))
            except Exception:  # noqa: BLE001 연결 끊김
                self._closed = True

    async def send_audio(self, gen: int, pcm: bytes) -> None:
        if self._closed:
            return
        async with self._send_lock:
            try:
                await self.ws.send_bytes(struct.pack("<I", gen & 0xFFFFFFFF) + pcm)
            except Exception:  # noqa: BLE001
                self._closed = True

    async def set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            await self.send_json({"type": "status", "state": state})

    def rest_state(self) -> str:
        if not self.conversation:
            return "idle"
        if self.endpointer is not None and self.endpointer.in_speech:
            return "listening"
        return "waiting"

    async def send_ready(self) -> None:
        await self.send_json({"type": "ready", "session_id": self.session_id, "model": self.model,
                              "out_sample_rate": OUT_RATE, "agent_name": config.AGENT_NAME,
                              "model_labels": config.MODEL_LABELS})

    # ---------- 수명 ----------
    async def _authenticate(self) -> bool:
        """VOICE_TOKEN이 설정돼 있으면 첫 메시지가 token이 맞는 hello여야 한다.
        통과하면 그 hello를 평소대로 처리하고 True, 아니면 unauthorized 에러 뒤 4401로 닫고 False."""
        msg = await self.ws.receive()
        if msg.get("type") == "websocket.disconnect":
            return False
        raw = msg.get("text")
        hello = None
        if raw is not None:
            try:
                hello = json.loads(raw)
            except json.JSONDecodeError:
                hello = None
        token = hello.get("token") if isinstance(hello, dict) and hello.get("type") == "hello" else None
        if isinstance(token, str) and hmac.compare_digest(token.encode(), config.VOICE_TOKEN.encode()):
            await self._on_text(raw)
            return True
        client = getattr(self.ws, "client", None)
        client_ip = getattr(client, "host", None) or "?"
        log.warning("hello 거부: 토큰 불일치 (%s)", client_ip)
        await self.send_json({"type": "error", "code": "unauthorized", "message": "접속 토큰이 틀립니다"})
        self._closed = True
        try:
            await self.ws.close(code=4401)
        except Exception:  # noqa: BLE001 이미 끊김
            pass
        return False

    async def run(self) -> None:
        self.speaker_task = asyncio.create_task(self._speaker())
        try:
            if config.VOICE_TOKEN and not await self._authenticate():
                return
            while True:
                msg = await self.ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await self._on_audio(msg["bytes"])
                elif msg.get("text") is not None:
                    await self._on_text(msg["text"])
        finally:
            self._closed = True
            self._cancel_echo_task()
            await self._cancel_turn()
            if self.speaker_task:
                self.speaker_task.cancel()

    async def _on_audio(self, data: bytes) -> None:
        if self.listening:  # 누르고 말하기가 우선
            if len(self.audio) < MAX_UTTERANCE_BYTES:
                self.audio += data
            return
        if self.conversation and self.endpointer is not None:
            self._record_mic(data)
            for ev in self.endpointer.feed(data):
                await self._on_vad(ev)
            if self.holding and self._duck is not None:
                await self._check_duck()
            await self._check_duck_echo()
            if (self.holding and self._hold_task is None and self.endpointer.in_speech
                    and self._confirm_ms() - self._hold_base_ms >= self.hold_confirm_ms):
                await self._confirm_hold()
            await self._check_duck_skip()
            if config.VOICE_INPUT_LOG:
                self._log_input(data)

    # ---------- 덕킹 능동 검사 ----------
    def _record_mic(self, data: bytes) -> None:
        """마이크 32 ms 창별 평균 제곱을 도착 시각 기준으로 기록(덕킹 검사용, 최근 3초)."""
        a = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float64) / 32768.0
        if a.size == 0:
            return
        now = self.clock()
        n = (a.size + 511) // 512
        for k in range(n):
            w = a[k * 512:(k + 1) * 512]
            t, pw = now - (n - 1 - k) * 0.032, float(np.mean(w * w))
            self._mic_hist.append((t, pw))
            self._track_duck_echo(t, pw)
        cut = now - 3.0
        i = 0
        while i < len(self._mic_hist) and self._mic_hist[i][0] < cut:
            i += 1
        if i:
            del self._mic_hist[:i]

    def _track_duck_echo(self, t: float, pw: float) -> None:
        """덕킹 에코 확정 뒤 마이크가 pre + DUCK_ECHO_RISE_DB 이상인 창의 최장 연속(빈 창 1개 허용)을 잰다."""
        d = self._duck_echo
        if d is None or t < d["at"]:
            return
        if 10 * np.log10(max(pw, 1e-18)) >= d["pre_db"] + config.DUCK_ECHO_RISE_DB:
            d["run"] += 1 + d["gap"]
            d["gap"] = 0
            d["best"] = max(d["best"], d["run"])
        elif d["run"] and d["gap"] == 0:
            d["gap"] = 1
        else:
            d["run"], d["gap"] = 0, 0

    def _duck_echo_rise_ms(self) -> float:
        d = self._duck_echo
        return 0.0 if d is None else d["best"] * 32.0

    async def _check_duck_echo(self) -> None:
        """덕킹 에코로 복귀한 발화가 복귀 뒤 HOLD_CONFIRM_MS 넘게 이어지면 덕킹 검사를 한 번 더(_hold).
        사용자 말이면 끼어들기 확정, 에코면 표시 유지."""
        d = self._duck_echo
        if d is None or d["rechecked"] or self.holding:
            return
        ep = self.endpointer
        if ep is None or not ep.in_speech or d["gen"] != self.generation or not self._playing():
            return
        if self.clock() - d["at"] >= self.hold_confirm_ms / 1000:
            d["rechecked"] = True
            log.info("덕킹 에코 뒤 말이 %d ms 넘게 이어짐 (복귀 뒤 상승 %.0f ms) -> 덕킹 재검사",
                     self.hold_confirm_ms, self._duck_echo_rise_ms())
            await self._hold()
            await self.set_state("listening")

    def _mic_db_between(self, t0: float, t1: float) -> Optional[float]:
        xs = [p for t, p in self._mic_hist if t0 <= t <= t1]
        if not xs:
            return None
        return float(10 * np.log10(max(sum(xs) / len(xs), 1e-18)))

    async def _check_duck(self) -> None:
        """hold(duck) 뒤 DUCK_TEST_DELAY_MS부터 DUCK_TEST_MS 동안의 마이크 평균이 hold 직전 320 ms보다
        DUCK_DROP_DB 넘게 내려가면 에코(내 소리를 줄이자 따라 내려감) -> resume, 아니면 사용자 말 -> interrupt."""
        d = self._duck
        if d is None or not self.holding:
            self._duck = None
            return
        if self._hold_task is not None:  # 검사 전에 말이 끝나 STT 판정으로 넘어감
            self._duck = None
            return
        t0 = d["at"] + config.DUCK_TEST_DELAY_MS / 1000
        t1 = t0 + config.DUCK_TEST_MS / 1000
        pre = d["pre_db"]
        # 조기 판정: 검사 구간 창을 차례로 보며 연속 DUCK_EARLY_WINDOWS창이 pre - DUCK_DROP_DB 이하면 에코,
        # pre - DUCK_KEEP_DB 초과면 사용자 말
        low = high = 0
        verdict = None
        for t, pw in self._mic_hist:
            if t < t0 or t > t1:
                continue
            db = 10 * np.log10(max(pw, 1e-18))
            low = low + 1 if db <= pre - config.DUCK_DROP_DB else 0
            high = high + 1 if db > pre - DUCK_KEEP_DB else 0
            if low >= DUCK_EARLY_WINDOWS:
                verdict, at = "echo", t
                break
            if high >= DUCK_EARLY_WINDOWS:
                verdict, at = "user", t
                break
        if verdict is None:
            if not self._mic_hist or self._mic_hist[-1][0] < t1:
                return
            post = self._mic_db_between(t0, t1)
            if post is None:
                self._duck = None
                log.info("덕킹 검사: 검사 구간 입력 없음 -> 기존 판정으로")
                return
            verdict = "echo" if pre - post >= config.DUCK_DROP_DB else "user"
            how, at, val = "평균", t1, post
        else:
            how, val = f"연속 {DUCK_EARLY_WINDOWS}창", db
        self._duck = None
        took = (at - d["at"]) * 1000
        if verdict == "echo":
            log.info("덕킹 검사(%s): pre %.1f dB -> %.1f dB (%.1f dB 하강, hold 뒤 %.0f ms) -> 에코, 재생 복귀",
                     how, pre, val, pre - val, took)
            prev = self._duck_echo
            gen = self._hold_gen
            await self._resume()
            # 이 발화의 나머지는 에코로 표시: 말 끝에서 복귀 뒤 레벨 상승이 없으면 받아쓰지 않고 버린다
            self._duck_echo = {"at": self.clock(), "gen": gen, "pre_db": pre,
                               "rechecked": bool(prev and prev.get("rechecked")), "run": 0, "gap": 0, "best": 0}
        else:
            log.info("덕킹 검사(%s): pre %.1f dB -> %.1f dB (%.1f dB 하강, hold 뒤 %.0f ms) -> 사용자 말, 끼어들기 확정",
                     how, pre, val, pre - val, took)
            await self.interrupt(announce=False, notify=True)

    def _confirm_ms(self) -> int:
        """hold 확정 카운터: 말소리 창 시간에서 에코 우세 창을 뺀 것."""
        ep = self.endpointer
        if ep is None:
            return 0
        return int(getattr(ep, "confirm_ms", ep.speech_ms))

    def _echo_hint(self, w: bytes, back: int) -> Optional[bool]:
        """Endpointer가 창마다 부름. True 에코 우세 / False 아님 / None 판정 보류."""
        now = self.clock()
        t = now - back * 0.032
        active = self._aec_on and (self.timeline.active(now) or self._playing())
        win = np.frombuffer(w, dtype="<i2").astype(np.float32) / 32768.0
        r = self.gate.process(win, t, self.aec_ref, self.timeline, active=active)
        self._gate_recent.append(r)
        del self._gate_recent[:-16]
        if self._duck_skip is not None and r.judged and not r.echo:
            self._duck_skip["nonecho"] += 1
        if config.VOICE_INPUT_LOG and active:
            g = self._gatelog
            g[0] += 1
            if r.judged:
                g[1] += 1
                g[2] += int(r.echo)
                g[3] = max(g[3], r.corr)
            if r.expected_db == r.expected_db:  # NaN 아님
                g[4] += r.expected_db
                g[5] += 1
            if g[0] >= 31:
                log.info("에코 대조: 판정 %d/%d창, 에코 %d, 최대 corr %.2f, lag %.0f ms(추정 %.0f), 마이크 %.1f dB, "
                         "expected 평균 %s, 바닥 %.1f dB, 이득 %.1f dB",
                         g[1], g[0], g[2], g[3], r.lag_ms, self.gate.lag_est_ms, r.mic_db,
                         f"{g[4] / g[5]:.1f} dB" if g[5] else "-", r.floor_db, self.gate.gain_db)
                self._gatelog = [0, 0, 0, -1.0, 0.0, 0]
        if not r.judged:
            return None
        return r.echo

    def _sure_echo(self) -> bool:
        """덕킹 검사를 생략할 만큼 확실한 에코인가: 최근 DUCK_SKIP_WINDOWS창 중 판정된 창이 절반 이상이고
        그 corr 중앙값 ≥ DUCK_SKIP_CORR, mic_db 중앙값 ≤ expected_db 중앙값 + DUCK_SKIP_LEVEL_DB.
        expected_db가 없는(NaN) 창은 레벨 비교에서 빠지고, 남는 창이 없으면 생략하지 않는다.
        시작보호(onset) 창은 게이트가 여유(+ONSET_EXTRA_DB)를 더 주지만 여기서는 더 주지 않는다(보수적)."""
        n = config.DUCK_SKIP_WINDOWS
        if config.DUCK_SKIP_CORR <= 0 or n <= 0 or not self._aec_on:
            return False
        if not self.timeline.active(self.clock()):
            return False
        rs = [r for r in self._gate_recent[-n:] if r.judged]
        if len(rs) < max(1, (n + 1) // 2):
            return False
        lv = [r for r in rs if r.expected_db == r.expected_db]
        if len(lv) < max(1, (n + 1) // 2):
            return False
        corr = float(np.median([r.corr for r in rs]))
        mic = float(np.median([r.mic_db for r in lv]))
        exp = float(np.median([r.expected_db for r in lv]))
        return corr >= config.DUCK_SKIP_CORR and mic <= exp + config.DUCK_SKIP_LEVEL_DB

    async def _check_duck_skip(self) -> None:
        """덕킹을 생략한 발화가 HOLD_CONFIRM_MS 넘게 이어지고 비에코 창이 DUCK_SKIP_REVOKE_WINDOWS개 이상이면
        생략을 취소하고 hold(덕킹 검사)로 간다."""
        d = self._duck_skip
        if d is None:
            return
        ep = self.endpointer
        if self.holding or ep is None or not ep.in_speech or not self._playing():
            self._duck_skip = None
            return
        if (self.clock() - d["at"] >= self.hold_confirm_ms / 1000
                and d["nonecho"] >= config.DUCK_SKIP_REVOKE_WINDOWS):
            self._duck_skip = None
            log.info("덕킹 생략 취소: 말이 %d ms 넘게 이어지고 비에코 %d창 -> hold", self.hold_confirm_ms, d["nonecho"])
            await self._hold()
            await self.set_state("listening")

    def _gate_echo(self, ev: VADEvent) -> bool:
        """발화의 음성 창 중 에코 우세 비율이 ECHO_GATE_FRAC 이상이면 True(받아쓰기 없이 에코)."""
        return ev.echo_judged >= ECHO_GATE_MIN_JUDGED and ev.echo_frac >= ECHO_GATE_FRAC

    def _gate_desc(self, ev: Optional[VADEvent] = None) -> str:
        r = self.gate.last
        s = (f"corr={r.corr:.2f} lag={r.lag_ms:.0f}ms mic_db={r.mic_db:.1f} floor_db={r.floor_db:.1f} "
             f"expected_db={r.expected_db:.1f}{' 시작보호' if r.onset else ''} 규칙={r.rule}")
        if ev is None and self._gate_recent:
            # hold 직전 창들의 판정 근거 집계(비에코로 센 근거: 레벨초과/상관낮음/해제/보류 등)
            min_db = self.vad_params.barge_min_db
            cnt: dict[str, int] = {}
            for x in self._gate_recent[-13:]:
                k = x.rule if x.mic_db >= min_db else f"최소음량미달({x.rule})"
                cnt[k] = cnt.get(k, 0) + 1
            s += " 최근13창=" + ",".join(f"{k}:{v}" for k, v in cnt.items())
        if ev is not None:
            s += f" 에코 비율={ev.echo_frac:.2f}({ev.echo_judged}창)"
        return s

    def _log_input(self, data: bytes) -> None:
        """대화 중 waiting/listening이면 1초마다 수신 RMS와 VAD 최대 확률을 로그."""
        acc = self._inlog
        if self.state not in ("waiting", "listening"):
            acc[:] = [0.0, 0, 0.0]
            return
        a = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float64) / 32768.0
        acc[0] += float(np.sum(a * a))
        acc[1] += a.size
        if self.endpointer is not None and self.endpointer.last_probs:
            acc[2] = max(acc[2], max(self.endpointer.last_probs))
        if acc[1] >= 16000:
            rms = (acc[0] / acc[1]) ** 0.5
            log.info("입력 %.1f dBFS, VAD 최대 %.2f (%s%s)", 20 * np.log10(max(rms, 1e-9)), acc[2], self.state,
                     ", 꼬리" if self._tail_active() else "")
            acc[:] = [0.0, 0, 0.0]

    # ---------- 핸즈프리 ----------
    def _turn_running(self) -> bool:
        return bool(self.turn_task and not self.turn_task.done())

    def _playing(self) -> bool:
        """지금 내 목소리가 스피커에서 나오는 중(또는 곧 나올 예정)인가.
        pos_ms를 보내는 클라이언트는 클라이언트 보고만 믿는다: speaking이고 마지막 보고가
        PLAYBACK_REPORT_TIMEOUT_S 안이면 재생 중(idle 보고면 즉시 아님). 실기(19:56)에서 실제 재생이 서버의
        전송 길이 추정보다 15초 넘게 늦어 끼어들기를 놓쳤다.
        옛 클라이언트는 보낸 오디오 길이로 추정한 끝 + PLAYBACK_GRACE_S까지."""
        if self.holding or self._speaking_now:
            return True
        if self.playback != "speaking":
            return False
        if self._aec_on:
            return self.clock() - self._last_report < config.PLAYBACK_REPORT_TIMEOUT_S
        return self.clock() < self._play_until + PLAYBACK_GRACE_S

    def _stale_playing(self) -> bool:
        """옛 generation 오디오가 클라이언트에서 아직 재생 중(최근 PLAYBACK_REPORT_TIMEOUT_S 안 speaking 보고)."""
        return self.clock() - self._stale_play < config.PLAYBACK_REPORT_TIMEOUT_S

    def _est_play_end(self, gen: int) -> float:
        """방금 참조에 붙인 문장까지 다 들리는 추정 시각. pos_ms 시간축이 있으면 그걸로, 없으면 _play_until."""
        now = self.clock()
        if self._aec_on and self.timeline.gen == gen and self.aec_ref.gen == gen and self.timeline.running:
            pos = float(self.timeline.pos_at(np.array([now]))[0])
            if pos == pos:  # NaN 아님
                return now + max(0.0, (self.aec_ref.duration_ms - pos) / 1000.0)
        return max(self._play_until, now)

    def _check_lag(self, gen: int, pos: float) -> None:
        """진단: 클라이언트 재생 위치가 서버 전송 기준 예상 위치보다 2초 넘게 뒤처지면 5초에 한 번 로그."""
        if gen != self._sched_gen or not self._sched:
            return
        now = self.clock()
        exp = sum(min(max(now - st, 0.0), dur) for st, dur in self._sched) * 1000.0
        lag = (exp - pos) / 1000.0
        if lag > 2.0 and now - self._lag_logged >= 5.0:
            self._lag_logged = now
            log.info("재생 지연: 클라이언트 %.1f s 뒤처짐 (pos %.0f ms, 예상 %.0f ms)", lag, pos, exp)

    def _barge_mode(self) -> bool:
        """끼어들기 기준(엄격)을 쓸 상황: 턴이 생각·말하는 중이거나, 보낼 문장이 남았거나, 스피커 재생 중.
        stale한 state 값은 보지 않는다. hold 중에는 재생이 멈춰 있으니 엄격하지 않게."""
        if self.holding:
            return False
        return ((self._turn_running() and self.state != "transcribing") or not self.queue.empty()
                or self._playing())

    def _tail_active(self) -> bool:
        """재생이 막 끝난 뒤 ECHO_TAIL_MS 동안(에코 꼬리 가드)."""
        return not self._playing() and self.clock() < self._tail_until

    async def _set_conversation(self, on: bool) -> None:
        if on:
            if self.endpointer is None:
                self.endpointer = Endpointer(self.vad_params, strict=self._barge_mode, tail=self._tail_active,
                                             echo_hint=self._echo_hint)
            self.endpointer.reset()
            self._carry = b""
            self.conversation = True
            self.listening = False
            await self.set_state(self.rest_state())
            log.info("conversation on")
        else:
            if not self.conversation:
                return
            self.conversation = False
            if self.endpointer is not None:
                self.endpointer.reset()
            self._carry = b""
            busy = self._turn_running() or self.state in ("thinking", "speaking") or self._playing() \
                or self.holding or not self.queue.empty()
            if busy:
                await self.interrupt(announce=False, notify=True)
            await self.set_state("idle")
            log.info("conversation off")

    async def _on_vad(self, ev: VADEvent) -> None:
        if ev.kind == "start":
            self._speech_wall = time.perf_counter()
            # 재생 중이거나 재생이 막 끝난 뒤 시작된 말은 받아쓴 뒤 에코인지 본다
            self._echo_check = self._playing() or self.clock() < self._tail_until or self._stale_playing()
            self._duck_skip = None
            self._duck_echo = None
            if self._echo_task is not None:
                # 에코 판정(받아쓰기) 중 말이 다시 시작: 그 발화를 새 발화 앞에 붙이고 에코 검사 유지
                task, self._echo_task = self._echo_task, None
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass
                    self._carry = self._echo_pcm
                    self._echo_check = True
                    log.info("에코 판정 중 말 이어짐: 앞 발화 %.1f s 붙임", len(self._carry) / 32000)
            if self.holding:
                # 잠정 끼어들기 판정 중 말이 다시 시작: 받아쓰던 발화를 붙이고 hold 유지
                if self._hold_task and not self._hold_task.done():
                    task, self._hold_task = self._hold_task, None
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass
                    self._carry = self._hold_pcm
                    log.info("hold 중 말 이어짐: 앞 발화 %.1f s 붙임", len(self._carry) / 32000)
                self._hold_task = None
                self._hold_base_ms = 0
                self._echo_check = True
            elif self.state == "transcribing" and self._turn_running():
                # 끝 침묵 뒤 말이 이어짐: 받아쓰던 발화를 새 발화 앞에 붙인다
                self._carry = self._turn_pcm
                self._echo_check = self._echo_check or self._turn_echo
                self.generation += 1
                await self._cancel_turn()
                log.info("말 이어짐: 앞 발화 %.1f s 붙임", len(self._carry) / 32000)
            elif self._playing() or (ev.barge and self.clock() < self._tail_until):
                # 스피커 소리(또는 그 꼬리)일 수 있음: 재생만 멈추고 진짜 말인지 판정(generation 유지).
                # 경계 시점에 일반 모드로 시작이 났어도 실제 재생 중이면 반드시 hold로 보낸다.
                if not ev.barge:
                    log.info("일반 모드 시작이지만 재생 중 -> hold로 판정")
                if self._sure_echo():
                    # 확실한 에코: 음량을 흔들지 않고 재생 유지, 말 끝에서 대조(_gate_echo)로 판정
                    log.info("확실한 에코 -> 덕킹 생략 (%s)", self._gate_desc())
                    self._echo_check = True
                    self._duck_skip = {"at": self.clock(), "nonecho": 0}
                else:
                    await self._hold()
            elif ev.barge:
                t = time.perf_counter()
                await self.interrupt(announce=False, notify=True)
                log.info("끼어들기 판정 -> interrupt gen %d (%.0f ms 처리)", self.generation,
                         (time.perf_counter() - t) * 1000)
            elif ev.tail:
                log.info("꼬리 가드 구간에서 말 시작(에코 검사 예정)")
            await self.send_json({"type": "vad", "speech": True})
            if self._duck_skip is None:  # 덕킹 생략이면 재생이 이어지니 speaking 유지
                await self.set_state("listening")
        elif ev.kind in ("end", "discard"):
            self._duck_skip = None
            duck_echo, self._duck_echo = self._duck_echo, None
            await self.send_json({"type": "vad", "speech": False})
            carry, self._carry = self._carry, b""
            pcm = carry + ev.pcm
            if self.holding:
                if ev.kind == "discard" and not carry:
                    log.info("hold: 짧은 소리 (%d ms) -> 재생 재개", ev.speech_ms)
                    await self._resume()
                    return
                if self._gate_echo(ev):
                    log.info("에코 무시(대조): %s (음성 %d ms) -> 재생 재개", self._gate_desc(ev), ev.speech_ms)
                    await self._resume()
                    return
                log.info("hold: 말 끝 (음성 %d ms, 발화 %.2f s) -> 받아써서 판정", ev.speech_ms, len(pcm) / 32000)
                self._hold_pcm = pcm
                self._hold_task = asyncio.create_task(self._resolve_hold(self._hold_gen, pcm))
                return
            if ev.kind == "discard" and not carry:
                log.info("짧은 발화 버림 (%d ms)", ev.speech_ms)
                await self.set_state("speaking" if self._playing() else self.rest_state())
                return
            log.info("말 끝 (음성 %d ms, 끝 침묵 %d ms, 발화 %.2f s%s)", ev.speech_ms, ev.end_lag_ms,
                     len(pcm) / 32000, f", smart-turn {ev.smart}" if ev.smart else "")
            echo_check, self._echo_check = self._echo_check, False
            if duck_echo is not None and duck_echo["gen"] == self.generation:
                rise = duck_echo["best"] * 32.0
                if rise < config.DUCK_ECHO_RISE_MS:
                    log.info("에코 무시(덕킹 확정): 음성 %d ms (복귀 뒤 pre %.1f dB + %.0f dB 넘는 구간 %.0f ms)",
                             ev.speech_ms, duck_echo["pre_db"], config.DUCK_ECHO_RISE_DB, rise)
                    await self.set_state("speaking" if self._playing() else self.rest_state())
                    return
                log.info("덕킹 에코 뒤 레벨 상승 %.0f ms (pre %.1f dB + %.0f dB 초과) -> 받아써서 텍스트 에코 필터로",
                         rise, duck_echo["pre_db"], config.DUCK_ECHO_RISE_DB)
                echo_check = True
            if echo_check and self._gate_echo(ev):
                log.info("에코 무시(대조): %s (음성 %d ms)", self._gate_desc(ev), ev.speech_ms)
                await self.set_state("speaking" if self._playing() else self.rest_state())
                return
            short_echo = None
            if echo_check and ev.speech_ms <= config.SHORT_ECHO_MS:
                short_echo = ev.echo_frac if ev.echo_judged else 0.0
            if echo_check and (self._playing() or self._stale_playing() or not self.queue.empty()
                               or self._turn_running()):
                # 답이 진행 중(재생·옛 재생·보낼 문장·생각 중): STT 판정 전에는 generation·큐를 건드리지 않는다.
                # 실기(17:46): 에코 발화가 새 턴을 열어 답이 끊겼고, 클라이언트는 옛 답을 계속 재생했다.
                log.info("에코 검사 대상 발화 -> 재생 유지하고 받아써서 판정 (음성 %d ms)", ev.speech_ms)
                self._cancel_echo_task()
                self._echo_pcm = pcm
                self._echo_task = asyncio.create_task(
                    self._resolve_pending("echo", self.generation, pcm, short_echo))
                await self.set_state("speaking" if (self._playing() or not self.queue.empty())
                                     else ("thinking" if self._turn_running() else self.rest_state()))
                return
            self._start_turn(pcm=pcm, echo_check=echo_check, short_echo=short_echo)

    # ---------- 잠정 끼어들기(hold/resume) ----------
    async def _hold(self) -> None:
        self.holding = True
        self._hold_gen = self.generation
        self._hold_task = None
        self._hold_pcm = b""
        self._hold_base_ms = self._confirm_ms()
        self._hold_at = self.clock()
        self._hold_mode = self.hold_mode
        self._duck = None
        if self._hold_mode == "duck":
            # 재생은 계속(소리만 낮춤): 서버도 오디오를 계속 보내고 시간축도 이어 간다
            pre = self._mic_db_between(self._hold_at - 0.32, self._hold_at)
            if pre is not None:
                self._duck = {"at": self._hold_at, "pre_db": pre}
            await self.send_json({"type": "hold", "generation": self.generation, "mode": "duck",
                                  "gain": config.DUCK_GAIN})
        else:
            self._hold_event.clear()
            self.timeline.stop(self._hold_at)  # 클라이언트가 재생을 멈춤(위치 고정)
            await self.send_json({"type": "hold", "generation": self.generation})
        log.info("잠정 끼어들기 -> hold(%s) gen %d (%s%s%s)", self._hold_mode, self.generation, self._gate_desc(),
                 "" if self._aec_on else ", 대조 꺼짐: pos_ms 없음",
                 f", pre {self._duck['pre_db']:.1f} dB" if self._duck else "")

    def _clear_hold(self) -> None:
        if not self.holding and self._hold_task is None:
            return
        self.holding = False
        self._duck = None
        task, self._hold_task = self._hold_task, None
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()
        self._hold_event.set()

    async def _resume(self) -> None:
        if not self.holding:
            return
        gen = self._hold_gen
        self._clear_hold()
        if gen != self.generation:
            return
        if self._hold_mode == "pause":
            if self.playback == "speaking":  # 멈춰 있던 시간만큼 재생 끝 추정을 미룬다
                self._play_until = max(self._play_until, self._hold_at) + (self.clock() - self._hold_at)
            if self.timeline.gen == gen:
                self.timeline.resume(self.clock() + RESUME_LEAD_S)
        await self.send_json({"type": "resume", "generation": gen})
        log.info("resume gen %d", gen)
        if self._playing() or not self.queue.empty():
            await self.set_state("speaking")
        elif self._turn_running():
            await self.set_state("thinking")
        else:
            await self.set_state(self.rest_state())

    async def _confirm_hold(self) -> None:
        log.info("hold: 말이 %d ms 넘게 이어짐 -> 끼어들기 확정", self.hold_confirm_ms)
        await self.interrupt(announce=False, notify=True)

    def _echo_filter(self, text: str, echo_check: bool, short_echo: Optional[float] = None
                     ) -> Optional[tuple[str, str]]:
        """받아쓴 문장 텍스트 에코 필터. 에코면 (로그 머리, 근거), 아니면 None.
        echo_check: 최근 문장 비교(echo.reason). short_echo: 짧은 조각 완화 기준(게이트 에코 비율, None이면 안 씀).
        전역 규칙(길고 거의 같은 문장)은 항상."""
        if not text:
            return None
        if echo_check:
            reason = self.echo.reason(text)
            if reason:
                return "에코 무시", reason
        if short_echo is not None:
            big = config.SHORT_ECHO_BIGRAM if short_echo >= config.SHORT_ECHO_FRAC else None
            lreason = self.echo.lenient_reason(text, bigram=big)
            if lreason:
                return "에코 무시(짧은 조각)", f"{lreason}, 게이트 에코 비율 {short_echo:.2f}"
        greason = self.echo.global_reason(text)
        if greason:  # 재생 꼬리와 무관: 길고 거의 같은 문장
            return "에코 무시(전역)", greason
        return None

    async def _resolve_pending(self, kind: str, gen: int, pcm: bytes, short_echo: Optional[float] = None) -> None:
        """재생을 그대로 둔 채 끝난 발화를 받아써서 판정(공용). 에코·빈 결과면 재생 계속, 진짜 말이면
        그때 interrupt(옛 답 끊기·끊긴 답 노트) + 그 텍스트로 새 턴.
        kind="hold": hold 중 끝난 발화(에코면 resume). kind="echo": 재생 중 끝난 에코 검사 대상 발화
        (generation·큐를 건드리지 않고 판정, 에코면 재생 계속)."""
        hold = kind == "hold"

        def valid() -> bool:
            return gen == self.generation and (self.holding if hold else self._echo_task is not None)

        async def keep_playing() -> None:
            if hold:
                self._hold_task = None
                await self._resume()
            else:
                self._echo_task = None
                await self.set_state("speaking" if (self._playing() or not self.queue.empty())
                                     else ("thinking" if self._turn_running() else self.rest_state()))
        try:
            if hold:
                await self.set_state("transcribing")
            text, ms = ("", 0)
            if self.stt.loaded:
                text, ms = await self.stt.transcribe(pcm)
            if not valid():
                return
            filt = self._echo_filter(text, True, short_echo)
            if not text or filt:
                if text:
                    log.info("%s: %r (%s, STT %d ms)%s", filt[0], text, filt[1], ms,
                             "" if hold else " -> 재생 계속")
                else:
                    log.info("%s: 받아쓰기 비어 재생 계속 (STT %d ms)", kind, ms)
                await keep_playing()
                return
            log.info("%s: 진짜 말 %r (STT %d ms) -> 끼어들기", kind, text, ms)
            if hold:
                self._hold_task = None
            else:
                self._echo_task = None
            await self.interrupt(announce=False, notify=True)
            await self.send_json({"type": "transcript", "text": text, "ms": ms})
            self._start_turn(text=text)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("%s 판정 실패 -> 재생 계속", kind)
            await keep_playing()

    async def _resolve_hold(self, gen: int, pcm: bytes) -> None:
        """hold 중 끝난 발화를 받아써서 에코·빈 결과면 resume, 진짜 말이면 끼어들기 + 새 턴."""
        await self._resolve_pending("hold", gen, pcm)

    def _cancel_echo_task(self) -> None:
        task, self._echo_task = self._echo_task, None
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _on_text(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self.send_json({"type": "error", "message": "JSON이 아니에요"})
            return
        t = msg.get("type")
        if t != "hello" and self.session_id is None:
            self.session_id = new_session_id()
        if t == "hello":
            if msg.get("vp") is not None:
                log.info("client viewport: %s", json.dumps(msg.get("vp"), ensure_ascii=False))
            resume = msg.get("resume")
            resume = resume if isinstance(resume, str) and resume.strip() else None
            # resume이 없으면(iOS가 localStorage를 지운 경우 등) 서버가 기억하는 마지막 세션을 잇는다.
            last = self.store.last_session() if self.store else None
            self.session_id = resume or last or new_session_id()
            await self._rotate_if_idle(announce=False)
            if self.store:
                self.store.set_last(self.session_id)
            await self.send_ready()
            if self.store:
                items = self.store.items(self.session_id)
                log.info("hello resume=%s -> %s (history %d개)", resume[:8] if resume else "없음",
                         self.session_id[:8], len(items))
                await self.send_json({"type": "history", "session_id": self.session_id, "items": items})
            await self.send_json({"type": "status", "state": self.state})
        elif t == "start_utterance":
            if self.turn_task and not self.turn_task.done() or self.state == "speaking":
                await self.interrupt(announce=False)
            self.audio = bytearray()
            self.listening = True
            await self.set_state("listening")
        elif t == "end_utterance":
            if not self.listening:
                return
            self.listening = False
            pcm, self.audio = bytes(self.audio), bytearray()
            self._start_turn(pcm=pcm)
        elif t == "text":
            text = (msg.get("text") or "").strip()
            if text:
                await self.interrupt(announce=False)
                self._start_turn(text=text)
        elif t == "interrupt":
            await self.interrupt(announce=not self.conversation)
            if self.conversation:
                await self.set_state(self.rest_state())
        elif t == "conversation":
            on = bool(msg.get("on"))
            if on and not self.conversation:
                await self._rotate_if_idle()  # 세션 교체는 대화를 '시작'할 때만. 대화 중에는 절대 바꾸지 않는다.
            await self._set_conversation(on)
        elif t == "playback":
            st = msg.get("state")
            g = msg.get("generation")
            if not isinstance(g, int):
                g = self.generation
            pos = msg.get("pos_ms")
            pos = float(pos) if isinstance(pos, (int, float)) and not isinstance(pos, bool) else None
            lat = msg.get("out_latency_ms")
            lat = float(lat) if isinstance(lat, (int, float)) and not isinstance(lat, bool) and 0 <= lat < 1000 else 0.0
            if pos is not None:
                pos -= lat  # 스피커에서 실제로 나오는 위치(출력 지연만큼 뒤)
            if st == "speaking" and self._stale_floor <= g < self.generation:
                # 끼어들기 없이 턴이 바뀌었는데(말 끝 -> 새 턴) 클라이언트가 옛 답을 아직 재생 중: 에코 검사 대상
                self._stale_play = self.clock()
            elif st == "idle" and g < self.generation and self._stale_playing():
                self._stale_play = -1e9
                self._tail_until = max(self._tail_until, self.clock() + self.echo_tail_s)
            if st == "speaking" and g == self.generation:
                self.playback, self.playback_gen = "speaking", g
                if pos is not None:
                    self._aec_on = True
                    self._last_report = self.clock()
                    self._check_lag(g, pos)
                    if not (self.holding and self._hold_mode == "pause"):  # 일시정지 hold 중 늦은 보고는 무시
                        self.timeline.report(g, pos, self.clock())
            elif st == "idle" and g in (self.generation, self.playback_gen):
                if self.playback == "speaking" and g == self.generation:
                    self._tail_until = self.clock() + self.echo_tail_s  # 에코 꼬리 가드 시작
                if g == self.timeline.gen:
                    self.timeline.stop(self.clock(), pos)
                self.playback = "idle"  # 옛 generation의 늦은 idle은 무시
        elif t == "settings":
            if "aec" in msg:
                self.aec = bool(msg["aec"])
            if "half_duplex" in msg:
                self.half_duplex = bool(msg["half_duplex"])
            log.info("settings aec=%s half_duplex=%s", self.aec, self.half_duplex)
        elif t == "set_model":
            if msg.get("model") in MODELS:
                self.model = msg["model"]
                await self.send_ready()
            else:
                await self.send_json({"type": "error", "message": f"모르는 모델: {msg.get('model')}"})
        elif t == "diag":
            # 클라이언트 진단 줄(아이폰 콘솔 대신 서버 로그로). 응답 없음.
            text = msg.get("text")
            if isinstance(text, str):
                log.info("client: %s", text[:400])
        elif t == "log":
            # 화면 대화 기록 한 줄(클라이언트 logItem). 응답 없음.
            if self.store:
                self.store.append(self.session_id, msg.get("item"))
        elif t == "new_session":
            await self.interrupt(announce=False)
            self._interrupt_note = None
            self.session_id = new_session_id()
            if self.store:
                self.store.set_last(self.session_id)
            await self.send_ready()
            await self.set_state(self.rest_state())
        else:
            await self.send_json({"type": "error", "message": f"모르는 메시지: {t}"})

    # ---------- 끼어들기 ----------
    async def _cancel_turn(self) -> None:
        task, self.turn_task = self.turn_task, None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    def _drain_queue(self) -> None:
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _note_cut_answer(self) -> None:
        """답을 끝까지 못 들은 채 끊겼으면 다음 헤르메스 요청에 붙일 노트를 만든다.

        헤르메스는 SSE 연결이 끊겨도 이미 진행 중인 LLM 호출은 끝까지 받아 세션에 전체 답을 저장한다
        (2026-11-02 실측). 그래서 사용자가 실제로 들은 부분을 알려 줘야 맥락이 맞는다."""
        a = self._answer
        if a["gen"] != self.generation:
            return
        turn_running = self._turn_running()
        playing = self._playing() or not self.queue.empty()
        if not (turn_running or playing or (a["heard"] and not a["done"])):
            return
        heard = " ".join(a["heard"]).strip()
        if heard:
            if len(heard) > HEARD_NOTE_CHARS:
                heard = "…" + heard[-HEARD_NOTE_CHARS:]
            self._interrupt_note = f"(앞 답변은 사용자가 중간에 끊었음. 사용자가 들은 부분: \"{heard}\")"
        elif a.get("thinking"):
            self._interrupt_note = "(앞 질문의 답을 듣기 전에 사용자가 끊었음)"

    async def interrupt(self, announce: bool = True, notify: bool = False) -> None:
        self._note_cut_answer()
        self._clear_hold()
        self._cancel_echo_task()
        self._duck_echo = None
        self._stale_play = -1e9  # 클라이언트에 interrupt를 보내 옛 재생을 버림
        self.generation += 1
        self._stale_floor = self.generation
        self._drain_queue()
        await self._cancel_turn()  # 헤르메스 스트림 연결이 닫힘 -> 서버가 run 중단
        self.listening = False
        self.playback = "idle"
        self.timeline.stop(self.clock())  # 클라이언트가 재생을 버림
        if notify:
            await self.send_json({"type": "interrupt", "generation": self.generation})
        if announce:
            await self.set_state("idle")

    # ---------- 턴 ----------
    def _start_turn(self, pcm: Optional[bytes] = None, text: Optional[str] = None,
                    echo_check: bool = False, short_echo: Optional[float] = None) -> None:
        self._clear_hold()
        self._cancel_echo_task()
        self._duck_echo = None
        self.generation += 1
        self._drain_queue()
        self._turn_pcm = pcm or b""
        self._turn_echo = echo_check
        self.turn_task = asyncio.create_task(self._turn(self.generation, pcm, text, echo_check, short_echo))

    async def _turn(self, gen: int, pcm: Optional[bytes], text: Optional[str], echo_check: bool = False,
                    short_echo: Optional[float] = None) -> None:
        """short_echo: 재생 중 시작된 짧은 발화(음성 ≤ SHORT_ECHO_MS)의 게이트 에코 비율. None이 아니면 STT 결과를
        완화 기준(echo.lenient_reason)으로 한 번 더 거른다: 숫자·단위만 있는 전사는 항상, bigram 완화는 비율 ≥
        SHORT_ECHO_FRAC일 때만."""
        t0 = time.perf_counter()
        try:
            if text is None:
                await self.set_state("transcribing")
                if not self.stt.loaded:
                    await self.send_json({"type": "error", "message": "STT가 아직 준비되지 않았어요"})
                    await self.set_state(self.rest_state())
                    return
                text, ms = await self.stt.transcribe(pcm or b"")
                if gen != self.generation:
                    return
                filt = self._echo_filter(text, echo_check, short_echo)
                if filt:
                    log.info("%s: %r (%s, STT %d ms)", filt[0], text, filt[1], ms)
                    await self.set_state(self.rest_state())
                    return
                await self.send_json({"type": "transcript", "text": text, "ms": ms})
                log.info("STT %d ms: %r", ms, text)
                if not text:
                    await self.set_state(self.rest_state())
                    return
            await self.set_state("thinking")
            await self._think(gen, text, t0)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("turn failed")
            if gen == self.generation:
                await self.send_json({"type": "error", "message": f"처리 중 오류: {e}"})
                self.queue.put_nowait((gen, _END))

    async def _rotate_if_idle(self, announce: bool = True) -> None:
        """오래 쉬었으면 새 세션으로 시작(첫 턴 프롬프트·기록이 짧아져 첫 토큰이 빨라짐).
        대화를 시작하는 순간(hello 복원, conversation on)에만 부른다. 턴 중간에는 부르지 않는다(사용자 결정 2026-11-02)."""
        if self.conversation or not session_is_idle(self.session_id):
            return
        old_sid = self.session_id
        self.session_id = new_session_id()
        self._interrupt_note = None
        if self.store:
            self.store.set_last(self.session_id)
        log.info("유휴 %.1f시간 넘음: 새 세션 %s (이전 %s)", config.VOICE_SESSION_IDLE_HOURS, self.session_id, old_sid)
        if announce:
            await self.send_ready()

    async def _think(self, gen: int, text: str, t0: float) -> None:
        if self.session_id:  # 끊기거나 실패한 턴도 활동으로 친다(끝나면 다시 갱신)
            _LAST_TURN[self.session_id] = time.time()
        splitter = SentenceSplitter()
        if self.style_note:
            # 모델은 자기가 무슨 모델인지 모르므로(빠름 모드에서 Opus라고 주장한 실기 사례) 실제 라우팅 모델을 알려 준다.
            note = self.style_note
            model_name = MODEL_DISPLAY.get(self.model)
            if model_name:
                note = f"{note[:-1]} 지금 답하는 모델은 {model_name}.)" if note.endswith(")") else f"{note} (지금 답하는 모델: {model_name})"
            text = f"{note}\n{text}"
        if self._interrupt_note:
            text = f"{self._interrupt_note}\n{text}"
            log.info("끊긴 답 노트 붙임: %s", self._interrupt_note)
            self._interrupt_note = None
        self._answer = {"gen": gen, "heard": [], "done": False, "thinking": True}
        spoke = {"any": False, "filler": False}
        first = {"tok": None}

        def enqueue(sentence: str) -> None:
            spoke["any"] = True
            self.queue.put_nowait((gen, sentence))

        async def on_delta(delta: str) -> None:
            if gen != self.generation:
                return
            if first["tok"] is None:
                first["tok"] = time.perf_counter()
                log.info("첫 토큰 %.0f ms (턴 시작부터)", (first["tok"] - t0) * 1000)
            await self.send_json({"type": "assistant_delta", "text": delta})
            for s in splitter.feed(delta):
                enqueue(s)

        async def on_tool(name: str, state: str, label: str, call_id: Optional[str] = None) -> None:
            if gen != self.generation:
                return
            msg = {"type": "tool", "name": name, "state": state}
            if call_id:
                msg["toolCallId"] = call_id  # 클라이언트가 같은 도구의 여러 호출을 짝지어 칩 하나로 합치는 데 씀
            await self.send_json(msg)
            if state == "started" and not spoke["any"] and not spoke["filler"]:
                spoke["filler"] = True
                enqueue(random.choice(FILLERS))

        try:
            result = await self.hermes.chat_stream(text, self.session_id, self.model, on_delta, on_tool)
        except (HermesError, httpx.HTTPError) as e:
            msg = str(e) or type(e).__name__
            if isinstance(e, httpx.ConnectError):
                msg = "헤르메스 API에 연결할 수 없어요"
            await self.send_json({"type": "error", "message": msg})
            self.queue.put_nowait((gen, _END))
            return
        if gen != self.generation:
            return
        for s in splitter.flush():
            enqueue(s)
        if result.session_id and result.session_id != self.session_id:
            self.session_id = result.session_id
            if self.store:
                self.store.set_last(self.session_id)
            await self.send_ready()
        if self.session_id:
            _LAST_TURN[self.session_id] = time.time()
        self._answer["done"] = True
        await self.send_json({"type": "assistant_done", "text": result.text})
        self.queue.put_nowait((gen, _END))

    # ---------- 말하기 ----------
    async def _wait_hold(self, gen: int) -> None:
        """hold 중이면 풀릴 때까지(또는 generation이 바뀔 때까지) 기다린다."""
        while self.holding and self._hold_mode == "pause" and gen == self.generation:
            await self._hold_event.wait()

    async def _next_item(self):
        """큐에서 다음 항목. 말하던 중 다음 문장이 (추정 재생 끝 + 600 ms) 안에 안 오면 thinking으로."""
        if self.state == "speaking" and self.queue.empty():
            wait = max(THINK_GRACE_S, self._play_until - self.clock() + THINK_GRACE_S)
            try:
                return await asyncio.wait_for(self.queue.get(), wait)
            except asyncio.TimeoutError:
                if self.state == "speaking" and self._turn_running() and not self.holding:
                    await self.set_state("thinking")
        return await self.queue.get()

    def _play_lead_s(self, gen: int) -> float:
        """지금부터 이 generation의 이미 보낸 오디오가 클라이언트에서 다 재생되기까지 남은 초(추정).
        _est_play_end(pos_ms 시간축 또는 보낸 길이 기준 _play_until)로 구한다. TTS 라우터가 이만큼은
        Qwen 첫 청크를 더 기다려도 소리가 끊기지 않는다고 본다."""
        return max(0.0, self._est_play_end(gen) - self.clock())

    async def _speak_stream(self, gen: int, item: str, spoken: str, lead_s: float = 0.0) -> bool:
        """문장 하나를 스트리밍 엔진으로 합성하며 청크가 오는 대로 보낸다. 오디오를 보냈으면 True.

        문장 전체 pcm을 받아 보내던 경로와 같은 결과가 되게 청크마다 참조 버퍼(aec_ref)·_play_until·
        재생 지연 진단(_sched)을 점진적으로 늘린다. speak_start는 첫 청크 직전. generation이 바뀌면
        (끼어들기·interrupt·대화 끔) 스트림을 닫아 엔진이 워커에 cancel을 보낸다. pause hold 중에는
        청크마다 _wait_hold로 멈추고, duck hold는 계속 보낸다."""
        t0 = time.perf_counter()
        started = False
        first_ms: Optional[int] = None
        total = 0
        st0 = 0.0
        stream = self.tts.synth_stream(spoken, is_current=lambda g=gen: g == self.generation,
                                       lead_s=lead_s, turn_id=gen)
        try:
            async for pcm in stream:
                if gen != self.generation:
                    break
                if pcm.size == 0:
                    continue
                if not started:
                    first_ms = int((time.perf_counter() - t0) * 1000)
                    await self._wait_hold(gen)
                    if gen != self.generation:
                        break
                    if not self.holding:  # 덕킹 hold 중에는 판정 상태(listening)를 유지
                        await self.set_state("speaking")
                    await self.send_json({"type": "speak_start", "generation": gen, "text": item})
                    if self._answer["gen"] == gen and item not in FILLERS:
                        self._answer["heard"].append(item)
                    if self._sched_gen != gen:
                        self._sched_gen, self._sched = gen, []
                    st0 = max(self._sched[-1][0] + self._sched[-1][1], self.clock()) if self._sched else self.clock()
                    self._sched.append((st0, 0.0))
                    started = True
                self.aec_ref.add(gen, pcm)
                total += len(pcm)
                if self._sched_gen == gen and self._sched:
                    self._sched[-1] = (st0, total / OUT_RATE)
                self._play_until = max(self._play_until, self.clock()) + len(pcm) / OUT_RATE
                if total == len(pcm):
                    self.echo.add(item, end=self._est_play_end(gen))  # 보관은 추정 재생 종료부터
                for i in range(0, len(pcm), CHUNK_SAMPLES):
                    await self._wait_hold(gen)
                    if gen != self.generation:
                        break
                    await self.send_audio(gen, pcm[i:i + CHUNK_SAMPLES].tobytes())
        except Exception as e:  # noqa: BLE001
            log.exception("TTS failed")
            if gen == self.generation:
                await self.send_json({"type": "error", "message": f"TTS 오류: {e}"})
        finally:
            await stream.aclose()
        if started:
            log.info("TTS(%s) 첫 청크 %d ms, 총 %d ms, %.1f s 음성: %s", getattr(self.tts, "last_engine", "?"),
                     first_ms or 0, int((time.perf_counter() - t0) * 1000), total / OUT_RATE, item)
        return started

    async def _speaker(self) -> None:
        sent_gen = None  # 오디오를 보낸 generation
        while True:
            gen, item = await self._next_item()
            if gen != self.generation:
                continue
            if item is _END:
                await self._wait_hold(gen)
                if gen != self.generation:
                    continue
                if sent_gen == gen:
                    await self.send_json({"type": "speak_end", "generation": gen})
                    if not self._aec_on:  # pos_ms 클라이언트는 실제 idle 보고 때 꼬리 가드를 시작
                        self._tail_until = max(self._tail_until, max(self._play_until, self.clock()) + self.echo_tail_s)
                await self.set_state(self.rest_state())
                continue
            if not self.tts.loaded:
                await self.send_json({"type": "error", "message": "TTS가 아직 준비되지 않았어요"})
                continue
            try:  # 독음 정규화는 합성에만 쓴다(자막 speak_start·heard·에코 메모리는 원문 item)
                spoken = normalize_for_tts(item)
            except Exception:  # noqa: BLE001
                log.exception("독음 정규화 실패, 원문으로 합성")
                spoken = item
            if spoken != item:
                log.debug("독음: %s -> %s", item, spoken)
            if not spoken:
                continue  # 기호만 있던 문장은 읽지 않는다
            self._speaking_now = True
            try:
                if hasattr(self.tts, "synth_stream"):  # 스트리밍 엔진(TTSRouter: Qwen 또는 Supertonic 폴백)
                    # 이 답에서 이미 보낸 오디오가 있을 때만 재생 여유를 센다(_play_until은 이전 답의 값이 남을 수 있음)
                    lead = self._play_lead_s(gen) if sent_gen == gen else 0.0
                    if await self._speak_stream(gen, item, spoken, lead_s=lead):
                        sent_gen = gen
                    continue
                try:
                    res = await self.tts.synth(spoken, is_current=lambda g=gen: g == self.generation)
                except Exception as e:  # noqa: BLE001
                    log.exception("TTS failed")
                    await self.send_json({"type": "error", "message": f"TTS 오류: {e}"})
                    continue
                if res is None or gen != self.generation:
                    continue
                pcm, ms = res
                log.info("TTS %d ms, %.1f s 음성: %s", ms, len(pcm) / OUT_RATE, item)
                await self._wait_hold(gen)
                if gen != self.generation:
                    continue
                if not self.holding:  # 덕킹 hold 중에는 판정 상태(listening)를 유지
                    await self.set_state("speaking")
                await self.send_json({"type": "speak_start", "generation": gen, "text": item})
                self.aec_ref.add(gen, pcm)  # 에코 대조용 참조(이 generation 위치 0부터 이어 붙임)
                if self._answer["gen"] == gen and item not in FILLERS:
                    self._answer["heard"].append(item)
                sent_gen = gen
                if self._sched_gen != gen:
                    self._sched_gen, self._sched = gen, []
                st0 = max(self._sched[-1][0] + self._sched[-1][1], self.clock()) if self._sched else self.clock()
                self._sched.append((st0, len(pcm) / OUT_RATE))
                self._play_until = max(self._play_until, self.clock()) + len(pcm) / OUT_RATE
                self.echo.add(item, end=self._est_play_end(gen))  # 보관은 추정 재생 종료부터
                for i in range(0, len(pcm), CHUNK_SAMPLES):
                    await self._wait_hold(gen)
                    if gen != self.generation:
                        break
                    await self.send_audio(gen, pcm[i:i + CHUNK_SAMPLES].tobytes())
            finally:
                self._speaking_now = False
