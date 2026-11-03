"""Qwen3-TTS 스트리밍 엔진(두 번째 TTS)과 엔진 선택·폴백 라우터.

Qwen3-TTS는 mlx-audio가 필요해 별도 venv(.venv-tts)의 서브프로세스(server/qwen_tts_worker.py)로 돌린다.
이 모듈은 asyncio로 워커를 띄우고(죽으면 자동 재시작, 최대 3회/10분) JSON lines로 문장을 보내
24 kHz 청크를 받아 44.1 kHz int16으로 바꿔 흘려 준다.

신호 처리(청크 단위, 경계 연속):
  앞뒤 무음 정리(SilenceTrimmer, 24 kHz 원신호 기준 -45 dBFS: 앞 60 ms·뒤 120 ms만 남김)
  -> 말 속도 QWEN_TTS_TEMPO배(TempoStretcher, WSOLA 40 ms 조각으로 음높이 유지, 24 kHz)
  -> 상태 있는 리샘플(StreamResampler, resample_poly 147/80과 같은 결과)
  -> 고정 이득(QWEN_TTS_GAIN_DB) -> 소프트 리미터(-1 dBFS 넘지 않음, 하드 클립 없음) -> int16.
텍스트는 session.py가 normalize_for_tts로 만든 결과를 그대로 받는다(여기서 다시 하지 않음).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import time
from collections import deque
from fractions import Fraction
from typing import AsyncIterator, Callable, Optional

import numpy as np

from . import config

log = logging.getLogger("tts_qwen")

OUT_RATE = 44100
WORKER = config.ROOT / "server" / "qwen_tts_worker.py"


class QwenTTSError(RuntimeError):
    pass


# ---------------------------------------------------------------- 신호 처리
class StreamResampler:
    """청크로 들어오는 신호를 scipy resample_poly(x, up, down) 한 번에 한 것과 같게 리샘플한다.

    출력 k는 입력 시각 k*down/up에 대응한다. 입력을 down의 배수 위치에서 자른 구간을 다시 리샘플하면
    출력 격자가 전체 신호의 격자와 맞는다. 필터 반길이(입력 기준 약 10샘플)보다 넉넉한 MARGIN만큼
    오른쪽 끝(아직 미래 샘플이 필요한 부분)은 다음 push까지 미루고, 왼쪽은 MARGIN 이상 문맥을 남긴다."""

    MARGIN = 64  # 입력 샘플

    def __init__(self, sr_in: int, sr_out: int = OUT_RATE):
        f = Fraction(sr_out, sr_in)
        self.up, self.down = f.numerator, f.denominator
        self.x = np.zeros(0, dtype=np.float64)
        self.x0 = 0          # self.x[0]의 절대 입력 위치(down의 배수)
        self.n_in = 0        # 지금까지 받은 입력 샘플 수
        self.k = 0           # 다음에 낼 출력 샘플 번호

    def _resample(self, x: np.ndarray) -> np.ndarray:
        if self.up == self.down:
            return x.copy()
        from scipy.signal import resample_poly
        return resample_poly(x, self.up, self.down)

    def _emit(self, k_end: int) -> np.ndarray:
        if k_end <= self.k or self.x.size == 0:
            return np.zeros(0, dtype=np.float32)
        y = self._resample(self.x)
        base = self.x0 * self.up // self.down          # y[0]의 절대 출력 번호(x0이 down의 배수라 정수)
        out = y[self.k - base:k_end - base]
        self.k = k_end
        # 다음 출력에 필요한 왼쪽 문맥(MARGIN)만 남기고 버림(down의 배수 위치에서)
        keep_from = (self.k * self.down // self.up) - self.MARGIN
        keep_from = max(self.x0, keep_from // self.down * self.down)
        if keep_from > self.x0:
            self.x = self.x[keep_from - self.x0:]
            self.x0 = keep_from
        return out.astype(np.float32)

    def push(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        if a.size:
            self.x = np.concatenate([self.x, a])
            self.n_in += a.size
        safe = self.n_in - self.MARGIN
        if safe <= 0:
            return np.zeros(0, dtype=np.float32)
        k_end = -(-safe * self.up // self.down)          # ceil: 입력 시각 < safe인 출력까지
        return self._emit(k_end)

    def flush(self) -> np.ndarray:
        total = -(-self.n_in * self.up // self.down)     # resample_poly 출력 길이 ceil(n*up/down)
        return self._emit(total)


class SilenceTrimmer:
    """스트림 앞 무음(thresh_db 미만)은 lead_s만 남기고 버리고, 끝 무음은 tail_s만 남긴다.
    중간 쉼은 그대로 둔다(뒤에 소리가 이어지면 미뤄 둔 무음을 함께 내보냄)."""

    def __init__(self, sr: int, thresh_db: float = -45.0, lead_s: float = 0.06, tail_s: float = 0.12,
                 frame_s: float = 0.032, max_hold_s: float = 1.5):
        self.F = max(1, int(sr * frame_s))
        self.lead, self.tail = int(sr * lead_s), int(sr * tail_s)
        self.max_hold = int(sr * max_hold_s)
        self.th = 10 ** (thresh_db / 20)
        self.buf = np.zeros(0, dtype=np.float32)
        self.started = False

    def _loud(self, buf: np.ndarray) -> np.ndarray:
        n = buf.size // self.F
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        fr = buf[:n * self.F].reshape(n, self.F).astype(np.float64)
        rms = np.sqrt(np.mean(fr * fr, axis=1))
        return np.flatnonzero(rms > self.th)

    def push(self, a: np.ndarray) -> np.ndarray:
        self.buf = np.concatenate([self.buf, np.asarray(a, dtype=np.float32).reshape(-1)])
        loud = self._loud(self.buf)
        if not self.started:
            if loud.size == 0:
                n = self.buf.size // self.F * self.F
                drop = max(0, n - self.lead)                 # 완성된 무음 프레임은 lead만 남기고 버림
                self.buf = self.buf[drop:]
                return np.zeros(0, dtype=np.float32)
            self.started = True
            self.buf = self.buf[max(0, loud[0] * self.F - self.lead):]
            loud = self._loud(self.buf)
        if loud.size:
            cut = (loud[-1] + 1) * self.F
            out, self.buf = self.buf[:cut], self.buf[cut:]
            return out
        if self.buf.size > self.max_hold:                   # 긴 쉼: 끝 tail만 남기고 내보냄
            cut = self.buf.size - self.tail
            out, self.buf = self.buf[:cut], self.buf[cut:]
            return out
        return np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        if not self.started:
            return np.zeros(0, dtype=np.float32)
        out, self.buf = self.buf[:self.tail], np.zeros(0, dtype=np.float32)
        return out


class TempoStretcher:
    """음높이를 유지한 채 말 속도를 tempo배로(WSOLA, audiotsm). 조각 QWEN_TTS_TSM_FRAME_MS(기본 40 ms)·겹침 50%·탐색 ±QWEN_TTS_TSM_TOL_MS(기본 15 ms).
    한 객체가 청크 사이 상태를 들고 있어 청크로 나눠 넣어도 한 번에 넣은 것과 같은 결과(이음매 없음).
    tempo == 1.0이면 그대로 통과."""

    def __init__(self, sr: int, tempo: float = 1.0, frame_ms: float = config.QWEN_TTS_TSM_FRAME_MS,
                 tol_ms: float = config.QWEN_TTS_TSM_TOL_MS):
        self.tempo = tempo
        self.tsm = None
        if abs(tempo - 1.0) > 1e-6:
            from audiotsm import wsola
            fl = max(2, int(sr * frame_ms / 1000))
            self.tsm = wsola(1, speed=tempo, frame_length=fl, synthesis_hop=fl // 2,
                             tolerance=int(sr * tol_ms / 1000))

    def push(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if self.tsm is None or a.size == 0:
            return a
        from audiotsm.io.array import ArrayReader, ArrayWriter
        r, w = ArrayReader(a[None, :]), ArrayWriter(1)
        while not r.empty:
            self.tsm.read_from(r)
            while not self.tsm.write_to(w)[1]:
                pass
        return w.data[0].astype(np.float32)

    def flush(self) -> np.ndarray:
        if self.tsm is None:
            return np.zeros(0, dtype=np.float32)
        from audiotsm.io.array import ArrayWriter
        w = ArrayWriter(1)
        while not self.tsm.flush_to(w)[1]:
            pass
        self.tsm.clear()
        return w.data[0].astype(np.float32)


def soft_limit(x: np.ndarray, ceiling_db: float = -1.0, knee_db: float = -4.0) -> np.ndarray:
    """|x| <= knee는 그대로, 그 위는 tanh로 눌러 ceiling을 넘지 않게(무기억 소프트 리미터, 1차 미분 연속)."""
    c, t = 10 ** (ceiling_db / 20), 10 ** (knee_db / 20)
    a = np.abs(x)
    over = a > t
    if not over.any():
        return x
    y = x.copy()
    y[over] = np.sign(x[over]) * (t + (c - t) * np.tanh((a[over] - t) / (c - t)))
    return y


class QwenPostProcessor:
    """워커 24 kHz float 청크 -> 44.1 kHz int16 청크(무음 정리·속도(WSOLA)·리샘플·이득·리미터).
    속도 신축은 리샘플 전 24 kHz 단계에서 한다(연산 절약)."""

    def __init__(self, sr_in: int, gain_db: float, peak_db: float = -1.0, tempo: float = 1.0):
        self.trim = SilenceTrimmer(sr_in)
        self.tempo = TempoStretcher(sr_in, tempo)
        self.rs = StreamResampler(sr_in, OUT_RATE)
        self.gain = 10 ** (gain_db / 20)
        self.peak_db = peak_db

    def _finish(self, y: np.ndarray) -> np.ndarray:
        if y.size == 0:
            return np.zeros(0, dtype=np.int16)
        y = soft_limit(y * self.gain, self.peak_db)
        return (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)

    def push(self, a: np.ndarray) -> np.ndarray:
        t = self.tempo.push(self.trim.push(a))
        return self._finish(self.rs.push(t) if t.size else np.zeros(0))

    def flush(self) -> np.ndarray:
        t = self.tempo.push(self.trim.flush())
        t = np.concatenate([t, self.tempo.flush()])
        y = self.rs.push(t) if t.size else np.zeros(0)
        return self._finish(np.concatenate([y, self.rs.flush()]))


# ---------------------------------------------------------------- 워커 엔진
class QwenTTSEngine:
    MAX_RESTARTS = 3
    RESTART_WINDOW_S = 600.0

    def __init__(self, cmd: Optional[list[str]] = None, env: Optional[dict] = None,
                 voice: str = config.QWEN_TTS_VOICE, gain_db: float = config.QWEN_TTS_GAIN_DB,
                 ready_timeout_s: float = 300.0, tempo: float = config.QWEN_TTS_TEMPO):
        self.voice = voice
        self.tempo = tempo
        self.gain_db = gain_db
        self.cmd = cmd or [config.QWEN_TTS_PYTHON, "-I", str(WORKER),
                           "--model", config.QWEN_TTS_MODEL, "--voice", voice,
                           "--interval", str(config.QWEN_TTS_STREAM_INTERVAL)] + (
            ["--instruct", config.QWEN_TTS_INSTRUCT] if config.QWEN_TTS_INSTRUCT else [])
        self.env = env
        self.ready_timeout_s = ready_timeout_s
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.ready = False
        self.error: Optional[str] = None
        self.load_ms: Optional[int] = None
        self.info: dict = {}
        self._ready_evt = asyncio.Event()
        self._streams: dict[int, asyncio.Queue] = {}
        self._next_id = 0
        self._reader: Optional[asyncio.Task] = None
        self._restarts: deque = deque()
        self._closing = False
        self._write_lock = asyncio.Lock()
        self.last: dict = {}   # 마지막 합성 통계(done 메시지)

    @property
    def loaded(self) -> bool:
        return self.ready

    def _child_env(self) -> dict:
        if self.env is not None:
            return self.env
        env = dict(os.environ)
        env["QWEN_TTS_HF_HOME"] = config.QWEN_TTS_HF_HOME
        env.pop("HF_HUB_CACHE", None)   # 메인 서버(whisper) 캐시 위치를 물려주지 않음: 워커가 HF_HOME/hub로 정함
        env["HF_HUB_OFFLINE"] = env.get("HF_HUB_OFFLINE", "1")
        env["PYTHONUNBUFFERED"] = "1"
        return env

    async def start(self) -> None:
        """워커를 띄우고 ready를 기다린다(실패해도 예외 없이 error에 남김)."""
        self._ready_evt = asyncio.Event()
        self.ready = False
        t = time.perf_counter()
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *self.cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=None, env=self._child_env(), limit=16 * 1024 * 1024)
        except Exception as e:  # noqa: BLE001
            self.error = f"워커 실행 실패: {e!r}"
            log.error("Qwen TTS %s", self.error)
            return
        self._reader = asyncio.create_task(self._read_loop(self.proc))
        try:
            await asyncio.wait_for(self._ready_evt.wait(), self.ready_timeout_s)
        except asyncio.TimeoutError:
            self.error = "워커 준비 시간 초과"
            log.error("Qwen TTS %s", self.error)
            return
        if self.ready:
            self.load_ms = int((time.perf_counter() - t) * 1000)
            self.error = None
            log.info("Qwen TTS ready in %d ms (%s)", self.load_ms, self.info.get("model", "?"))

    async def load(self) -> None:
        await self.start()

    async def _read_loop(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        while True:
            try:
                line = await proc.stdout.readline()
            except Exception as e:  # noqa: BLE001 너무 긴 줄 등
                log.error("Qwen 워커 출력 읽기 실패: %r", e)
                break
            if not line:
                break
            try:
                msg = json.loads(line)
            except ValueError:
                log.debug("Qwen 워커 비프로토콜 출력: %r", line[:120])
                continue
            if "ready" in msg:
                self.ready = bool(msg["ready"])
                self.info = msg
                if not self.ready:
                    self.error = str(msg.get("error", "로드 실패"))
                self._ready_evt.set()
                continue
            q = self._streams.get(msg.get("id"))
            if q is not None:
                q.put_nowait(msg)
        await self._on_exit(proc)

    async def _on_exit(self, proc: asyncio.subprocess.Process) -> None:
        try:
            rc = await asyncio.wait_for(proc.wait(), 5)
        except asyncio.TimeoutError:
            rc = None
        if proc is not self.proc:
            return
        self.ready = False
        self._ready_evt.set()
        for q in list(self._streams.values()):
            q.put_nowait({"error": f"워커 종료(rc={rc})"})
        if self._closing:
            return
        self.error = f"워커 종료(rc={rc})"
        now = time.monotonic()
        while self._restarts and now - self._restarts[0] > self.RESTART_WINDOW_S:
            self._restarts.popleft()
        if len(self._restarts) >= self.MAX_RESTARTS:
            log.error("Qwen TTS 워커가 10분 안에 %d번 넘게 죽어 재시작 중단(rc=%s)", self.MAX_RESTARTS, rc)
            return
        self._restarts.append(now)
        log.warning("Qwen TTS 워커 종료(rc=%s) -> 재시작 %d/%d", rc, len(self._restarts), self.MAX_RESTARTS)
        asyncio.get_running_loop().create_task(self.start())

    async def _write(self, obj: dict) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None or proc.returncode is not None:
            raise QwenTTSError("워커가 없음")
        async with self._write_lock:
            proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
            await proc.stdin.drain()

    async def cancel(self, rid: int) -> None:
        try:
            await self._write({"cancel": rid})
        except Exception:  # noqa: BLE001 워커가 이미 없음
            pass

    async def synth_stream(self, text: str, is_current: Callable[[], bool] = lambda: True,
                           seed: Optional[int] = None) -> AsyncIterator[np.ndarray]:
        """문장 -> int16 44.1 kHz 청크들. is_current()가 거짓이 되면(약 50 ms 안에) 워커에 cancel 보내고 끝.
        소비자가 중간에 그만두면(aclose) 역시 cancel. 워커 오류는 QwenTTSError."""
        if not self.ready:
            raise QwenTTSError(self.error or "준비 안 됨")
        self._next_id += 1
        rid = self._next_id
        q: asyncio.Queue = asyncio.Queue()
        self._streams[rid] = q
        finished = False
        post: Optional[QwenPostProcessor] = None
        t0 = time.perf_counter()
        first_ms = None
        try:
            req = {"id": rid, "text": text, "voice": self.voice}
            if seed is not None:
                req["seed"] = seed
            await self._write(req)
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), 0.05)
                except asyncio.TimeoutError:
                    if not is_current():
                        return
                    continue
                if not is_current():
                    return
                if "pcm" in msg:
                    sr = int(msg.get("sr", 24000))
                    if post is None:
                        post = QwenPostProcessor(sr, self.gain_db, tempo=self.tempo)
                    raw = np.frombuffer(base64.b64decode(msg["pcm"]), dtype="<i2").astype(np.float32) / 32768.0
                    out = post.push(raw)
                    if out.size:
                        if first_ms is None:
                            first_ms = int((time.perf_counter() - t0) * 1000)
                        yield out
                elif msg.get("done"):
                    finished = True
                    self.last = {"ms": msg.get("ms"), "worker_first_ms": msg.get("first_ms"),
                                 "first_ms": first_ms, "samples": msg.get("samples")}
                    if post is not None:
                        out = post.flush()
                        if out.size and is_current():
                            yield out
                    return
                elif msg.get("cancelled"):
                    finished = True
                    return
                elif "error" in msg:
                    finished = True
                    raise QwenTTSError(str(msg["error"]))
        finally:
            self._streams.pop(rid, None)
            if not finished:
                # 소비자가 그만뒀거나(끼어들기·폴백 시간 초과) 취소됨: 워커가 다음 청크 경계에서 멈추게
                asyncio.get_running_loop().create_task(self.cancel(rid))

    async def aclose(self) -> None:
        self._closing = True
        proc = self.proc
        if proc and proc.returncode is None:
            try:
                if proc.stdin:
                    proc.stdin.close()
                await asyncio.wait_for(proc.wait(), 3)
            except Exception:  # noqa: BLE001
                proc.kill()


# ---------------------------------------------------------------- 엔진 선택·폴백
class TTSRouter:
    """TTS_ENGINE에 따라 문장을 Qwen 스트리밍 또는 Supertonic으로 합성.

    qwen: Qwen 워커가 준비 안 됐거나, 첫 청크가 first_chunk_timeout_s 안에 안 오거나, 첫 청크 전에 오류가 나면
    그 문장은 Supertonic으로 폴백(WARNING). 첫 청크 뒤 오류는 그 문장을 거기서 끝낸다(이미 일부를 보냄).
    Supertonic이 로드되지 않았으면(모델 없음 등) 폴백할 곳이 없으므로 첫 청크를 한도 없이 기다린다
    (문장을 버리지 않음). Qwen이 준비 안 됐거나 첫 청크 전 오류가 나면 그 문장은 소리 없이 건너뛴다.

    첫 청크 대기 한도 = max(first_chunk_timeout_s, min(lead_s - lead_margin_s, first_chunk_max_s)).
    lead_s는 호출자가 추정한 '앞 문장 오디오가 클라이언트에서 다 재생되기까지 남은 초'(모르면 0):
    그동안은 더 기다려도 소리가 끊기지 않는다. 한 답(turn_id) 안에서 한 번 폴백하면 그 답의 남은 문장은
    Qwen을 시도하지 않고 Supertonic으로 간다(답 중간에 목소리가 오가지 않게). 다음 답부터 다시 Qwen."""

    def __init__(self, supertonic, qwen: Optional[QwenTTSEngine] = None, engine: str = "supertonic",
                 first_chunk_timeout_s: float = config.QWEN_TTS_FIRST_CHUNK_TIMEOUT_S,
                 lead_margin_s: float = config.QWEN_TTS_LEAD_MARGIN_S,
                 first_chunk_max_s: float = config.QWEN_TTS_FIRST_CHUNK_MAX_S):
        self.supertonic = supertonic
        self.qwen = qwen
        self.engine = engine if (engine == "qwen" and qwen is not None) else "supertonic"
        self.first_chunk_timeout_s = first_chunk_timeout_s
        self.lead_margin_s = lead_margin_s
        self.first_chunk_max_s = first_chunk_max_s
        self.fallbacks = 0
        self.fallback_turns = 0               # 폴백이 난 답 수(turn_id 기준)
        self._fallback_turn = None            # 마지막으로 폴백한 답의 turn_id
        self._fallback_times: deque = deque(maxlen=1000)
        self.last_fallback: Optional[str] = None
        self.last_engine: Optional[str] = None
        self.last_ms: Optional[int] = None

    # Supertonic 호환 속성(세션·health가 씀)
    @property
    def loaded(self) -> bool:
        return bool(self.supertonic.loaded or (self.engine == "qwen" and self.qwen and self.qwen.ready))

    @property
    def provider(self):
        return getattr(self.supertonic, "provider", None)

    @property
    def load_ms(self):
        return getattr(self.supertonic, "load_ms", None)

    @property
    def error(self):
        return getattr(self.supertonic, "error", None)

    async def load(self) -> None:
        jobs = [self.supertonic.load()]
        if self.engine == "qwen" and self.qwen is not None:
            jobs.append(self.qwen.start())
        results = await asyncio.gather(*jobs, return_exceptions=True)
        if isinstance(results[0], BaseException):   # Supertonic 로드 실패는 기동을 막지 않는다
            if getattr(self.supertonic, "error", None) is None:
                try:
                    self.supertonic.error = repr(results[0])
                except Exception:  # noqa: BLE001
                    pass
        if self.engine == "qwen" and not self.supertonic.loaded:
            log.warning("Supertonic 로드 실패(폴백 없이 Qwen만 사용): %s", getattr(self.supertonic, "error", None))
        for r in results[1:]:
            if isinstance(r, BaseException):
                raise r
        if self.engine != "qwen" and isinstance(results[0], BaseException):
            raise results[0]

    async def synth(self, text: str, is_current: Callable[[], bool] = lambda: True,
                    lead_s: float = 0.0, turn_id=None):
        return await self.supertonic.synth(text, is_current)

    def first_chunk_limit(self, lead_s: float = 0.0) -> float:
        """첫 청크 대기 한도(초). 앞 문장 재생이 남았으면 그만큼(여유 lead_margin_s를 빼고) 더 기다린다."""
        extra = min(float(lead_s or 0.0) - self.lead_margin_s, self.first_chunk_max_s)
        return max(self.first_chunk_timeout_s, extra)

    def recent_fallbacks(self, window_s: float = 600.0) -> int:
        now = time.monotonic()
        return sum(1 for t in self._fallback_times if now - t <= window_s)

    def _note_fallback(self, reason: str, text: str, turn_id=None) -> None:
        if turn_id is not None and turn_id != self._fallback_turn:
            self._fallback_turn = turn_id
            self.fallback_turns += 1
        self.fallbacks += 1
        self._fallback_times.append(time.monotonic())
        self.last_fallback = reason
        log.warning("Qwen TTS 폴백 -> Supertonic (%s): %s", reason, text[:40])

    async def synth_stream(self, text: str, is_current: Callable[[], bool] = lambda: True,
                           lead_s: float = 0.0, turn_id=None) -> AsyncIterator[np.ndarray]:
        t0 = time.perf_counter()
        if self.engine == "qwen" and turn_id is not None and turn_id == self._fallback_turn:
            log.info("같은 답이라 Supertonic 유지: %s", text[:40])
        elif self.engine == "qwen":
            reason = None
            can_fallback = bool(self.supertonic.loaded)
            limit = self.first_chunk_limit(lead_s) if can_fallback else None
            if self.qwen is None or not self.qwen.ready:
                reason = f"준비 안 됨: {self.qwen.error if self.qwen else '없음'}"
            else:
                if not can_fallback:
                    log.info("Supertonic 없음: Qwen 대기: %s", text[:40])
                agen = self.qwen.synth_stream(text, is_current)
                try:
                    first = await asyncio.wait_for(agen.__anext__(), limit)
                except StopAsyncIteration:
                    first = None
                    if is_current():
                        reason = "빈 출력"
                except asyncio.TimeoutError:
                    first = None
                    reason = f"첫 청크 {limit:.1f}s 초과"
                except Exception as e:  # noqa: BLE001
                    first = None
                    reason = f"오류: {e}"
                if first is not None:
                    self.last_engine = "qwen"
                    self.last_ms = int((time.perf_counter() - t0) * 1000)
                    try:
                        yield first
                        async for c in agen:
                            yield c
                    except QwenTTSError as e:
                        log.warning("Qwen TTS 문장 중간 오류, 이 문장은 여기까지: %s", e)
                    finally:
                        await agen.aclose()
                    return
                await agen.aclose()
                if reason is None:      # 합성 중 이 generation이 끝남
                    return
            if not is_current():
                return
            if not can_fallback:
                log.warning("Qwen TTS 실패, Supertonic 없음 -> 이 문장 건너뜀 (%s): %s", reason, text[:40])
                return
            self._note_fallback(reason, text, turn_id)
        res = await self.supertonic.synth(text, is_current)
        if res is None:
            return
        pcm, ms = res
        self.last_engine = "supertonic"
        self.last_ms = ms
        yield pcm

    async def aclose(self) -> None:
        if self.qwen is not None:
            await self.qwen.aclose()
