"""Silero VAD(ONNX, torch 없이) 스트리밍 래퍼 + 말 시작/끝 판정기.

모델: silero-vad 패키지의 silero_vad/data/silero_vad.onnx (v5/v6, 16 kHz).
입력 input [1, 64+512] float32 (앞 64샘플은 직전 창의 꼬리 = 공식 구현의 context),
state [2,1,128] float32, sr int64 -> 확률 [1,1], 새 state.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

log = logging.getLogger("vad")

SAMPLE_RATE = 16000
WINDOW = 512                     # 32 ms
WINDOW_MS = WINDOW * 1000 // SAMPLE_RATE
CONTEXT = 64


def default_model_path() -> str:
    env = os.environ.get("SILERO_VAD_MODEL")
    if env:
        return env
    try:
        import silero_vad  # 패키지만 설치(--no-deps). import 시 torch를 부르면 실패할 수 있음
        return str(Path(silero_vad.__file__).parent / "data" / "silero_vad.onnx")
    except Exception:  # noqa: BLE001
        pass
    import site
    for d in site.getsitepackages():
        p = Path(d) / "silero_vad" / "data" / "silero_vad.onnx"
        if p.exists():
            return str(p)
    raise FileNotFoundError("silero_vad.onnx를 찾지 못함 (SILERO_VAD_MODEL로 지정)")


_session_lock = threading.Lock()
_session: Optional[ort.InferenceSession] = None


def _get_session(path: Optional[str] = None) -> ort.InferenceSession:
    """모든 연결이 세션 하나를 공유(상태는 연결별로 따로 들고 다님)."""
    global _session
    with _session_lock:
        if _session is None:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.log_severity_level = 3
            _session = ort.InferenceSession(path or default_model_path(), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        return _session


class SileroVAD:
    """32 ms 창 단위 스트리밍. feed(pcm int16 bytes) -> 완성된 창마다 확률 리스트."""

    def __init__(self, model_path: Optional[str] = None):
        self.sess = _get_session(model_path)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)

    def prob(self, window: np.ndarray) -> float:
        """float32 [-1,1] 512샘플 창 하나 -> 말소리 확률."""
        x = np.concatenate([self._context, window.reshape(1, WINDOW).astype(np.float32)], axis=1)
        out, self._state = self.sess.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -CONTEXT:]
        return float(out[0, 0])

    def feed(self, pcm16: bytes) -> list[float]:
        a = np.frombuffer(pcm16[: len(pcm16) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, a]) if self._buf.size else a
        probs = []
        while self._buf.size >= WINDOW:
            w, self._buf = self._buf[:WINDOW], self._buf[WINDOW:]
            probs.append(self.prob(w))
        return probs


@dataclass
class VADParams:
    threshold: float = 0.5        # 말 시작 확률
    start_ms: int = 160           # 연속 이만큼 넘으면 말 시작
    end_threshold: float = 0.35   # 이 아래가
    end_silence_ms: int = 700     # 연속 이만큼이면 말 끝
    barge_threshold: float = 0.7  # 끼어들기(재생·생각 중) 시작 확률
    barge_ms: int = 400
    barge_min_db: float = -52.0   # 끼어들기 시작·hold 확정 카운터는 이 음량(dBFS) 이상인 창만 셈
    tail_threshold: float = 0.6   # 재생 직후 에코 꼬리 구간의 시작 확률(시간은 start_ms 그대로)
    preroll_ms: int = 300
    min_speech_ms: int = 300      # 이보다 짧은 발화는 버림
    max_utterance_ms: int = 60000
    turn_detector: str = "silence"   # silence | smart-turn
    smart_turn_silence_ms: int = 300  # smart-turn: 이만큼 조용하면 모델로 말 끝인지 물어봄

    @classmethod
    def from_env(cls) -> "VADParams":
        e = os.environ.get
        return cls(
            threshold=float(e("VAD_THRESHOLD", "0.5")),
            start_ms=int(e("VAD_START_MS", "160")),
            end_threshold=float(e("VAD_END_THRESHOLD", "0.35")),
            end_silence_ms=int(e("VAD_END_SILENCE_MS", "700")),
            barge_threshold=float(e("BARGE_THRESHOLD", "0.7")),
            barge_ms=int(e("BARGE_MS", "400")),
            barge_min_db=float(e("BARGE_MIN_DB", "-52")),
            tail_threshold=float(e("VAD_TAIL_THRESHOLD", "0.6")),
            preroll_ms=int(e("VAD_PREROLL_MS", "300")),
            min_speech_ms=int(e("VAD_MIN_SPEECH_MS", "300")),
            max_utterance_ms=int(e("VAD_MAX_UTTERANCE_MS", "60000")),
            turn_detector=e("TURN_DETECTOR", "silence").strip().lower() or "silence",
            smart_turn_silence_ms=int(e("SMART_TURN_SILENCE_MS", "300")),
        )


@dataclass
class VADEvent:
    kind: str                      # "start" | "end" | "discard"
    pcm: bytes = b""               # end일 때 프리롤 포함 발화 PCM
    speech_ms: int = 0
    barge: bool = False            # start가 끼어들기 기준으로 판정됐는지
    tail: bool = False             # start가 에코 꼬리 가드 기준으로 판정됐는지
    end_lag_ms: int = 0            # end: 마지막 말소리 창 이후 판정까지 걸린 오디오 시간
    echo_frac: float = 0.0         # end/discard: 발화의 음성 창 중 에코 우세 비율(판정된 창 기준)
    echo_judged: int = 0           # end/discard: 에코 대조로 판정된 음성 창 수
    smart: Optional[tuple] = None  # smart-turn 판정 (완료?, 확률, ms)


class Endpointer:
    """VAD 확률로 말 시작/끝을 판정. 오디오 시간 기준(창 수)으로만 동작해 결정적이다.

    strict(): 끼어들기 상황이면 True를 돌려주는 콜백. 그때는 barge 기준으로 시작을 판정.
    tail(): 재생이 막 끝난 에코 꼬리 구간이면 True. 그때는 시작 확률만 tail_threshold로 올린다
            (시간은 start_ms; strict가 우선).
    echo_hint(window_bytes, back): 창마다 한 번 불림(back = 이 feed에서 이 창 뒤에 남은 창 수).
            True = 에코 우세, False = 에코 아님, None = 판정 보류. strict 모드의 시작 연속 카운터에서
            에코 우세 창은 말소리로 세지 않고 연속을 끊는다. 말하는 중에는 에코 우세 창을
            confirm_ms(hold 확정 카운터)에서 뺀다. 일반 모드의 시작 판정과 발화 PCM 수집은 그대로.
    barge_min_db: strict 시작 카운터와 confirm_ms는 창 RMS가 이 값 이상인 창만 센다. 그 아래 창은
            세지 않되 연속을 끊지도 않는다(작게 말하다 커지는 경우).
    """

    def __init__(self, params: VADParams, vad: Optional[SileroVAD] = None, strict=None, turn_model=None,
                 tail=None, echo_hint=None):
        self.p = params
        self.vad = vad or SileroVAD()
        self.turn_model = turn_model
        if self.turn_model is None and params.turn_detector == "smart-turn":
            try:
                from .smart_turn import SmartTurn
                self.turn_model = SmartTurn()
            except Exception as e:  # noqa: BLE001
                log.warning("Smart Turn을 못 써서 침묵 기준만 사용: %s", e)
        self.strict = strict or (lambda: False)
        self.tail = tail or (lambda: False)
        self.echo_hint = echo_hint
        self.preroll_windows = max(0, params.preroll_ms // WINDOW_MS)
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self.in_speech = False
        self._run = 0                 # 시작 후보 연속 창 수
        self._silence = 0             # 말 중 연속 무음 창 수
        self._speech_windows = 0
        self._ring: list[bytes] = []  # 말 시작 전 최근 창(프리롤 + 시작 후보)
        self._ring_flags: list[tuple] = []  # _ring과 나란히 (에코 플래그, 확률)
        self._echo_windows = 0        # 말 중 에코 우세 음성 창 수
        self._span = 0                # 시작 후보 구간 창 수(세지 않은 작은 창 포함)
        self._confirm_windows = 0     # hold 확정 카운터: 에코 아님 + barge_min_db 이상인 음성 창
        self._judged_windows = 0      # 말 중 판정된 음성 창 수
        self._utt = bytearray()
        self._pending = b""
        self._smart_checked = False
        self._smart = None
        self.last_prob = 0.0
        self.last_probs: list[float] = []   # 마지막 feed()의 창별 확률(진단 로그용)

    def set_tail(self, tail) -> None:
        self.tail = tail or (lambda: False)

    @property
    def speech_ms(self) -> int:
        """말하는 중이면 지금까지의 음성 창 시간(시작 판정 창 포함), 아니면 0."""
        return self._speech_windows * WINDOW_MS if self.in_speech else 0

    @property
    def confirm_ms(self) -> int:
        """speech_ms에서 에코 우세 창을 뺀 것(hold 확정 카운터용)."""
        return self._confirm_windows * WINDOW_MS if self.in_speech else 0

    def feed(self, pcm16: bytes) -> list[VADEvent]:
        data = self._pending + pcm16
        n = len(data) // (WINDOW * 2) * (WINDOW * 2)
        self._pending = data[n:]
        events: list[VADEvent] = []
        self.last_probs = []
        total = n // (WINDOW * 2)
        for k, i in enumerate(range(0, n, WINDOW * 2)):
            w = data[i:i + WINDOW * 2]
            p = self.vad.prob(np.frombuffer(w, dtype="<i2").astype(np.float32) / 32768.0)
            self.last_prob = p
            self.last_probs.append(p)
            echo = None
            if self.echo_hint is not None:
                try:
                    echo = self.echo_hint(w, total - 1 - k)
                except Exception:  # noqa: BLE001 감지기 오류는 판정 보류로
                    log.exception("echo_hint 실패")
            ev = self._step(w, p, echo)
            if ev:
                events.append(ev)
        return events

    def _count_echo(self, echo, p: float) -> None:
        if p >= self.p.end_threshold and echo is not None:
            self._judged_windows += 1
            if echo:
                self._echo_windows += 1

    @staticmethod
    def _db(w: bytes) -> float:
        a = np.frombuffer(w, dtype="<i2").astype(np.float32) / 32768.0
        return float(20 * np.log10(max(float(np.sqrt(np.mean(a * a))), 1e-9)))

    def _step(self, w: bytes, p: float, echo=None) -> Optional[VADEvent]:
        P = self.p
        loud = self._db(w) >= P.barge_min_db
        if not self.in_speech:
            strict = bool(self.strict())
            tail = not strict and bool(self.tail())
            if strict:
                thr, need_ms = P.barge_threshold, P.barge_ms
            elif tail:
                thr, need_ms = max(P.threshold, P.tail_threshold), P.start_ms
            else:
                thr, need_ms = P.threshold, P.start_ms
            self._ring.append(w)
            self._ring_flags.append((echo, p))
            speech = p >= thr and not (strict and echo)   # 끼어들기 판정에서 에코 우세 창은 말소리 아님
            if speech and (loud or not strict):
                self._run += 1
                self._span += 1
            elif speech and self._run:
                self._span += 1    # 작은 창: 세지 않되 연속은 유지
            else:
                self._run = self._span = 0
            keep = self.preroll_windows + max(self._span, 1)
            if len(self._ring) > keep:
                del self._ring[: len(self._ring) - keep]
                del self._ring_flags[: len(self._ring_flags) - keep]
            if self._run * WINDOW_MS >= need_ms:
                self.in_speech = True
                self._utt = bytearray(b"".join(self._ring))
                self._echo_windows = self._judged_windows = 0
                for e, pp in self._ring_flags[-self._span:]:
                    self._count_echo(e, pp)
                self._confirm_windows = self._run
                self._ring = []
                self._ring_flags = []
                self._speech_windows = self._span
                self._silence = 0
                self._run = self._span = 0
                return VADEvent("start", barge=strict, tail=tail)
            return None
        self._utt += w
        self._count_echo(echo, p)
        if p >= P.end_threshold:
            if not echo and loud:
                self._confirm_windows += 1
            self._speech_windows += 1
            self._silence = 0
            self._smart_checked = False
        else:
            self._silence += 1
        too_long = len(self._utt) // 2 * 1000 // SAMPLE_RATE >= P.max_utterance_ms
        smart_end = False
        if (self.turn_model is not None and not self._smart_checked
                and self._silence * WINDOW_MS >= P.smart_turn_silence_ms
                and self._silence * WINDOW_MS < P.end_silence_ms):
            self._smart_checked = True  # 쉼 한 번에 한 번만
            try:
                self._smart = self.turn_model.is_complete(bytes(self._utt))
                smart_end = self._smart[0]
                log.info("smart-turn 침묵 %d ms: 완료=%s p=%.2f (%d ms)", self._silence * WINDOW_MS, *self._smart)
            except Exception:  # noqa: BLE001
                log.exception("smart-turn 실패")
        if smart_end or self._silence * WINDOW_MS >= P.end_silence_ms or too_long:
            speech_ms = self._speech_windows * WINDOW_MS
            pcm = bytes(self._utt)
            lag = self._silence * WINDOW_MS
            # 끝 침묵은 STT에 필요 없으니 200 ms만 남기고 자름
            trim = max(0, self._silence * WINDOW - int(0.2 * SAMPLE_RATE)) * 2
            if trim:
                pcm = pcm[: len(pcm) - trim]
            self.in_speech = False
            self._utt = bytearray()
            self._silence = 0
            self._speech_windows = 0
            self._ring = []
            self._ring_flags = []
            judged, echo_w = self._judged_windows, self._echo_windows
            self._echo_windows = self._judged_windows = self._confirm_windows = 0
            kind = "end" if speech_ms >= P.min_speech_ms else "discard"
            smart, self._smart, self._smart_checked = self._smart, None, False
            return VADEvent(kind, pcm=pcm, speech_ms=speech_ms, end_lag_ms=lag, smart=smart,
                            echo_frac=(echo_w / judged) if judged else 0.0, echo_judged=judged)
        return None
