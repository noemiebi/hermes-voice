"""에코 대조 감지기: 받아쓰기 전에 마이크 소리가 내 TTS의 스피커 에코인지 가린다(순수 numpy).

원리(docs/프로토콜.md '에코 대조 감지기'):
- 서버가 보낸 TTS PCM(44.1 kHz)을 generation별로 16 kHz로 내려 참조 버퍼로 둔다(위치 0 = 첫 샘플).
- 클라이언트가 250 ms마다 보내는 재생 위치(`playback` pos_ms)로 "언제 스피커에서 참조의 어디가 나왔나"
  시간축(PlaybackTimeline)을 만든다. hold·idle이면 그 뒤는 스피커 무음.
- 마이크 32 ms 창마다 최근 W창(640 ms)의 RMS 포락선(dB)과, 같은 시각들에 스피커에서 나온 참조 포락선을
  lag ∈ [-100, +600] ms(16 ms 간격)만큼 밀어 가며 피어슨 상관을 구한다. 최대 상관 ≥ ECHO_CORR이면 에코 우세.
- 에코 바닥: 에코 우세 창들에서 (마이크 dB - 참조 dB)를 느린 EMA로 추적해 '이 참조 레벨이면 에코는 이 정도'를
  추정한다(floor_db = 참조 dB(근방 최대) + 에코 이득). 마이크가 floor + ECHO_MARGIN_DB 이상이면
  사용자 말이 섞인 것으로 보고 에코 우세를 푼다.
- 예측 레벨 게이트(상관과 독립, 2026-11-02 실기 보강): 창마다 예상 에코 레벨 expected_db =
  참조 dB(pos(t - lag_est) ±64 ms 최대) + gain_est를 계산해 마이크 ≤ expected + ECHO_MARGIN_DB(문장 시작
  보호 구간 ONSET_GUARD_MS 안에서는 +4)이면 상관과 무관하게 에코 우세. 실기(19:55): 문장 중간에도 상관이
  낮게(0.45) 나와 -51 dBFS 에코에 hold가 걸렸고, 실제 끼어들기는 -30 dBFS로 에코(-44~-51)보다 12 dB 넘게 컸다.
  레벨로 못 가르는 경우(사용자 말이 에코 수준으로 눌려 들어옴)는 hold 덕킹 능동 검사(session.py)가 맡는다.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

REF_RATE = 16000
WIN = 512                 # 32 ms @16 kHz (마이크 창, VAD 창과 같음)
WIN_MS = 32.0
HOP = 256                 # 참조 포락선 격자 16 ms
HOP_MS = 16.0
DB_FLOOR = -70.0          # 포락선 dB 하한(무음·잡음 바닥을 같은 값으로 묶음)
REF_SILENT_DB = -50.0     # 참조 포락선 최대가 이보다 작으면 판정 보류
MAX_REF_S = 300.0         # generation 하나의 참조 상한(5분)
LAG_MIN_MS, LAG_MAX_MS, LAG_STEP_MS = -100.0, 600.0, 16.0
HISTORY = 20              # W = 20창 = 640 ms
GAIN_ALPHA = 0.03         # 에코 이득 EMA(느리게)
NOISE_RISE_DB = 0.03      # 잡음 바닥 추정이 창마다 올라가는 최대치(약 1 dB/s)
ACTIVE_AFTER_S = 1.0      # 재생이 멈춘 뒤 이만큼은 계속 대조

ECHO_CORR = float(os.environ.get("ECHO_CORR", "0.6"))
ECHO_MARGIN_DB = float(os.environ.get("ECHO_MARGIN_DB", "8"))
ONSET_GUARD_MS = float(os.environ.get("ONSET_GUARD_MS", "400"))   # 참조 무음->유음 뒤 여유를 넓히는 시간
ONSET_EXTRA_DB = 4.0      # 그동안 더하는 여유(AEC 재수렴 구간)
ONSET_SILENCE_MS = 160.0  # 이만큼 이어진 참조 무음 뒤의 유음을 '문장 시작'으로 봄
LAG_DEFAULT_MS = 90.0     # lag 추정 전 기본값(실기 76~92 ms)
GAIN_DEFAULT_DB = -22.0   # 에코 이득 초기값(실기 -25~-30보다 크게: 초기엔 에코로 기울게)
EXPECT_NEAR_MS = 64.0     # 예상 레벨에 쓰는 참조 근방 ±
REF_MIN_STD_DB = 3.0      # 상관에 쓰는 참조 포락선의 최소 표준편차(dB)
LAG_EST_ALPHA = 0.1       # 상관으로 확인된 창의 lag로 lag_est를 따라가는 속도
# 학습 상태 저장(aec_state.json): 상관으로 에코가 확인돼 lag·이득을 갱신한 창이 이만큼 쌓이면 '학습됨'
PRIOR_MIN_LEARNED = int(os.environ.get("AEC_PRIOR_MIN_LEARNED", "40"))
PRIOR_SCORE = 0.3         # prior lag에 심어 두는 lag 점수(EMA라 실제 상관이 쌓이면 곧 덮인다)
PRIOR_GAIN_RANGE = (-60.0, 0.0)
SAVE_MIN_INTERVAL_S = 60.0

log = logging.getLogger("hermes-voice.aec")


def _db(x):
    return 20.0 * np.log10(np.maximum(x, 1e-9))


def resample_44k_to_16k(pcm: np.ndarray) -> np.ndarray:
    """int16(또는 float) 44.1 kHz -> float32 16 kHz [-1,1]."""
    a = np.asarray(pcm)
    a = a.astype(np.float32) / 32768.0 if a.dtype == np.int16 else a.astype(np.float32)
    if a.size == 0:
        return a
    try:
        from scipy.signal import resample_poly
        return resample_poly(a, 160, 441).astype(np.float32)
    except Exception:  # noqa: BLE001 scipy 없으면 이동 평균 저역 통과 + 선형 보간 데시메이션
        k = 3
        lp = np.convolve(a, np.ones(k, dtype=np.float32) / k, mode="same")
        n = int(round(a.size * REF_RATE / 44100))
        x = np.arange(n) * (44100 / REF_RATE)
        return np.interp(x, np.arange(a.size), lp).astype(np.float32)


class ReferenceBuffer:
    """generation별 참조 신호(16 kHz)와 16 ms 격자 RMS 포락선(dB, 창 32 ms).

    env[i] = 샘플 [i*HOP, i*HOP+WIN)의 RMS = 참조 위치 (16i, 16i+32] ms 구간. '위치 p에서 끝나는 32 ms'는
    i = (p - 32) / 16."""

    def __init__(self, max_s: float = MAX_REF_S):
        self.max_samples = int(max_s * REF_RATE)
        self.gen: Optional[int] = None
        self.reset()

    def reset(self, gen: Optional[int] = None) -> None:
        self.gen = gen
        self._parts: list[np.ndarray] = []
        self._tail = np.zeros(0, dtype=np.float32)   # 아직 창이 안 찬 끝부분(다음 창 계산용)
        self._tail_start = 0                          # _tail[0]의 절대 샘플 위치
        self.samples = 0
        self.env = np.zeros(0, dtype=np.float32)

    @property
    def duration_ms(self) -> float:
        return self.samples * 1000.0 / REF_RATE

    def add(self, gen: int, pcm44k) -> None:
        if gen != self.gen:
            self.reset(gen)
        room = self.max_samples - self.samples
        if room <= 0:
            return
        a = resample_44k_to_16k(pcm44k)[:room]
        if a.size == 0:
            return
        self._parts.append(a)
        self.samples += a.size
        self.add_16k_env(a)

    def add_16k_env(self, a: np.ndarray) -> None:
        buf = np.concatenate([self._tail, a]) if self._tail.size else a
        n_new = 0 if buf.size < WIN else (buf.size - WIN) // HOP + 1
        if n_new:
            idx = np.arange(n_new)[:, None] * HOP + np.arange(WIN)[None, :]
            rms = np.sqrt(np.mean(buf[idx] ** 2, axis=1))
            self.env = np.concatenate([self.env, np.maximum(_db(rms), DB_FLOOR).astype(np.float32)])
        consumed = n_new * HOP
        self._tail = buf[consumed:].copy()
        self._tail_start += consumed

    def audio(self) -> np.ndarray:
        return np.concatenate(self._parts) if self._parts else np.zeros(0, dtype=np.float32)

    def env_at(self, pos_ms: np.ndarray) -> np.ndarray:
        """참조 위치(그 위치에서 끝나는 32 ms) 포락선 dB. NaN·범위 밖 = 무음(DB_FLOOR)."""
        out = np.full(pos_ms.shape, DB_FLOOR, dtype=np.float32)
        ok = np.isfinite(pos_ms)
        if not ok.any() or self.env.size == 0:
            return out
        i = np.rint((pos_ms[ok] - WIN_MS) / HOP_MS).astype(np.int64)
        good = (i >= 0) & (i < self.env.size)
        vals = np.full(i.shape, DB_FLOOR, dtype=np.float32)
        vals[good] = self.env[i[good]]
        out[ok] = vals
        return out


class PlaybackTimeline:
    """클라이언트 보고로 만든 '시각 -> 스피커에서 나오는 참조 위치' 추정. 구간마다 pos = p0 + (t - t0)*1000.
    구간 밖(재생 전, hold·idle 뒤)은 NaN(스피커 무음)."""

    KEEP_S = 5.0

    def __init__(self):
        self.reset()

    def reset(self, gen: Optional[int] = None) -> None:
        self.gen = gen
        self._segs: list[list[float]] = []   # [t0, t1(inf=진행 중), p0]
        self.stopped_at: Optional[float] = None
        self.last_pos: Optional[float] = None

    @property
    def running(self) -> bool:
        return bool(self._segs) and self._segs[-1][1] == float("inf")

    def _close(self, t: float) -> None:
        if self.running:
            s = self._segs[-1]
            s[1] = max(t, s[0])
            self.last_pos = s[2] + (s[1] - s[0]) * 1000.0

    def report(self, gen: int, pos_ms: float, t: float) -> None:
        """speaking 보고: 지금(t) 스피커에서 pos_ms가 나오는 중."""
        if gen != self.gen:
            self.reset(gen)
        self._close(t)
        self._segs.append([t, float("inf"), float(pos_ms)])
        self.stopped_at = None
        self.last_pos = float(pos_ms)
        self._prune(t)

    def stop(self, t: float, pos_ms: Optional[float] = None) -> None:
        """hold·idle: 스피커가 멈춤. pos_ms가 오면 그 위치에서 멈춘 것으로 맞춘다."""
        if not self.running:
            return
        s = self._segs[-1]
        if pos_ms is not None and t > s[0]:
            # 마지막 구간 끝을 보고 위치에 맞게(1배속 가정이므로 끝 시각으로 반영)
            s[1] = max(s[0], min(t, s[0] + (float(pos_ms) - s[2]) / 1000.0))
            self.last_pos = s[2] + (s[1] - s[0]) * 1000.0
        else:
            self._close(t)
        self.stopped_at = t

    def resume(self, t: float) -> None:
        """hold 해제: 멈춘 위치부터 t에 다시 나온다고 가정(다음 보고가 바로잡음)."""
        if self.running or self.last_pos is None:
            return
        self._segs.append([t, float("inf"), self.last_pos])
        self.stopped_at = None

    def _prune(self, now: float) -> None:
        cut = now - self.KEEP_S
        while len(self._segs) > 1 and self._segs[0][1] < cut:
            self._segs.pop(0)

    def pos_at(self, ts: np.ndarray) -> np.ndarray:
        out = np.full(ts.shape, np.nan, dtype=np.float64)
        for t0, t1, p0 in self._segs:
            m = (ts >= t0) & (ts < t1)
            if m.any():
                out[m] = p0 + (ts[m] - t0) * 1000.0
        return out

    def active(self, now: float, after_s: float = ACTIVE_AFTER_S) -> bool:
        """재생 중이거나 멈춘 지 after_s 안."""
        if self.running:
            return True
        return self.stopped_at is not None and now - self.stopped_at < after_s


@dataclass
class GateResult:
    echo: bool = False
    judged: bool = False           # 레벨·상관 중 하나라도 판정했는가(False면 판정 보류)
    corr: float = 0.0
    lag_ms: float = 0.0
    mic_db: float = DB_FLOOR
    floor_db: float = float("nan")  # 상관 쪽 추정 에코 레벨(dB, 상관 lag 기준), 모르면 NaN
    expected_db: float = float("nan")  # 예측 레벨 게이트의 예상 에코 레벨(dB), 참조 무음이면 NaN
    onset: bool = False            # 문장 시작 보호 구간(여유 +ONSET_EXTRA_DB)
    released: bool = False         # 바닥 + margin을 넘는 창이 이어져 '사용자 말 섞임'으로 해제
    rule: str = "보류"             # 판정 근거: 에코면 레벨/상관, 아니면 해제/레벨초과·상관낮음/이력부족 등


class EchoGate:
    """마이크 32 ms 창마다 에코 우세 여부 판정. 두 규칙 중 하나라도 맞으면 에코 우세.

    1) 예측 레벨: expected_db = 참조 dB(pos(t - lag_est) ±EXPECT_NEAR_MS 최대) + gain_est.
       mic_db ≤ max(expected_db, 잡음 바닥) + margin(문장 시작 ONSET_GUARD_MS 안이면 + ONSET_EXTRA_DB)이면 에코.
       참조가 무음(-50 dBFS 미만)이면 이 규칙은 보류.
    2) 상관: 창별 최대 상관(lag 44개 중 최대)은 W=20창으로는 우연히도 높게 나온다(독립 음성도 중앙값 0.75,
       합성 실측). 그래서 lag별 상관을 느린 EMA(LAG_ALPHA)로 쌓아 그 최대 lag(±1칸)에서의 상관만 쓴다.
       상관 ≥ corr_thr이면 에코. lag_est·gain_est는 이 규칙으로 에코가 확인된 창에서만 느리게 학습.
    해제: 최근 RELEASE_LOOK창 중 마이크가 바닥(+margin) 이상인 창이 RELEASE_COUNT개 이상이면 '사용자 말 섞임'으로
    보고 RELEASE_HOLD_MS 동안 두 규칙 모두 에코 우세를 내지 않는다(음절 사이에 마이크가 에코 수준으로
    떨어져도 끼어들기 연속이 끊기지 않게).
    """

    LAG_ALPHA = 0.08
    NEAR = 1                   # 상관 쪽 추정 에코 레벨에 쓰는 참조 근방(±창)
    RELEASE_LOOK = 10          # 320 ms
    RELEASE_COUNT = 2
    RELEASE_HOLD_MS = 480.0

    def __init__(self, corr_thr: float = ECHO_CORR, margin_db: float = ECHO_MARGIN_DB,
                 history: int = HISTORY, lag_min_ms: float = LAG_MIN_MS, lag_max_ms: float = LAG_MAX_MS,
                 lag_step_ms: float = LAG_STEP_MS, onset_guard_ms: float = ONSET_GUARD_MS):
        self.corr_thr, self.margin_db, self.W = corr_thr, margin_db, history
        self.onset_guard_ms = onset_guard_ms
        self.lags = np.arange(lag_min_ms, lag_max_ms + 0.5, lag_step_ms) / 1000.0  # 초
        self.reset()

    def reset(self) -> None:
        self._t: list[float] = []
        self._db: list[float] = []
        self._over: list[bool] = []            # 창별: 마이크가 예상 에코 + margin 초과
        self.gain_db: float = GAIN_DEFAULT_DB  # 에코 이득(마이크 dB - 참조 dB) EMA
        self.lag_est_ms: float = LAG_DEFAULT_MS
        self._lag_score: Optional[np.ndarray] = None
        self._release_until = -1e9
        self.noise_db: Optional[float] = None  # 마이크 잡음 바닥(빨리 내려가고 천천히 올라감)
        self.learned = 0                       # lag·이득을 갱신한(상관으로 에코 확인된) 창 수
        self.prior: Optional[dict] = None      # set_prior로 넣은 사전값
        self.last = GateResult()

    def set_prior(self, lag_ms: float, gain_db: float) -> bool:
        """저장해 둔 lag·이득을 사전값으로. 학습 전(learned == 0)일 때만 넣는다. 넣었으면 True.

        lag_est·gain_est를 그 값에서 시작하고(이후 학습은 평소처럼 EMA로 갱신), 상관 lag 점수(_lag_score)를
        prior lag 근처에 약하게(PRIOR_SCORE) 심어 첫 창들의 최대 상관 lag가 범위 끝(-100 ms)으로 튀지 않게 한다.
        learned는 0에서 시작(신뢰도 낮음): 저장 조건은 이번 세션의 학습 창 수로만 센다."""
        if self.learned > 0:
            return False
        try:
            lag, gain = float(lag_ms), float(gain_db)
        except (TypeError, ValueError):
            return False
        if not (np.isfinite(lag) and np.isfinite(gain)):
            return False
        lo, hi = float(self.lags[0] * 1000.0), float(self.lags[-1] * 1000.0)
        lag = min(max(lag, lo), hi)
        gain = min(max(gain, PRIOR_GAIN_RANGE[0]), PRIOR_GAIN_RANGE[1])
        self.lag_est_ms, self.gain_db = lag, gain
        score = np.zeros(len(self.lags))
        k = int(np.argmin(np.abs(self.lags * 1000.0 - lag)))
        score[k] = PRIOR_SCORE
        for j in (k - 1, k + 1):
            if 0 <= j < len(score):
                score[j] = PRIOR_SCORE / 2
        self._lag_score = score
        self.prior = {"lag_ms": lag, "gain_db": gain}
        return True

    @property
    def confident(self) -> bool:
        return self.learned >= PRIOR_MIN_LEARNED

    def state(self) -> dict:
        return {"lag_ms": round(float(self.lag_est_ms), 1), "gain_db": round(float(self.gain_db), 2),
                "learned": int(self.learned)}

    def _level(self, t: float, mic_db: float, ref: ReferenceBuffer, timeline: PlaybackTimeline, res: GateResult):
        """예측 레벨 규칙. 반환: None(참조 무음, 보류) | (에코?, 초과?)."""
        lag = self.lag_est_ms / 1000.0
        near = np.arange(-EXPECT_NEAR_MS, EXPECT_NEAR_MS + 0.5, HOP_MS) / 1000.0
        env = ref.env_at(timeline.pos_at(t - lag + near))
        ref_db = float(env.max())
        if ref_db < REF_SILENT_DB:
            return None
        expected = max(ref_db + self.gain_db, self.noise_db if self.noise_db is not None else DB_FLOOR)
        # 문장 시작 보호: 지금 참조 위치 앞 ONSET_GUARD_MS 안에 ONSET_SILENCE_MS 넘는 참조 무음이 있었나
        g = int(self.onset_guard_ms // HOP_MS)
        sl = int(ONSET_SILENCE_MS // HOP_MS)
        back = np.arange(0, g + sl + 1) * HOP_MS / 1000.0
        hist = ref.env_at(timeline.pos_at(t - lag - back)) < REF_SILENT_DB   # [0]=지금, 뒤로 갈수록 과거
        onset = False
        if g > 0 and hist.any():
            run = 0
            for k in range(1, len(hist)):
                run = run + 1 if hist[k] else 0
                if run >= sl and k - sl + 1 <= g:
                    onset = True
                    break
        res.expected_db, res.onset = expected, onset
        margin = self.margin_db + (ONSET_EXTRA_DB if onset else 0.0)
        return mic_db <= expected + margin, mic_db > expected + margin

    def process(self, window: np.ndarray, t: float, ref: Optional[ReferenceBuffer],
                timeline: Optional[PlaybackTimeline], active: bool = True) -> GateResult:
        """window: float32 [-1,1] 512샘플(이 창은 시각 t에 끝남)."""
        w = np.asarray(window, dtype=np.float32)
        mic_db = float(max(_db(float(np.sqrt(np.mean(w * w)))) if w.size else DB_FLOOR, DB_FLOOR))
        if self.noise_db is None or mic_db < self.noise_db:
            self.noise_db = mic_db
        else:
            self.noise_db += min(NOISE_RISE_DB, mic_db - self.noise_db)
        self._t.append(t)
        self._db.append(mic_db)
        if len(self._t) > self.W:
            del self._t[0], self._db[0]
        res = GateResult(mic_db=mic_db)
        if (not active or ref is None or timeline is None or ref.env.size == 0
                or ref.gen != timeline.gen):
            self._over.append(False)
            del self._over[:-self.RELEASE_LOOK]
            self.last = res
            return res
        # 1) 예측 레벨
        lv = self._level(t, mic_db, ref, timeline, res)
        level_echo = bool(lv and lv[0])
        self._over.append(bool(lv and lv[1]))
        del self._over[:-self.RELEASE_LOOK]
        # 2) 상관(이력이 찼을 때만)
        corr_state = "이력부족"
        corr_echo = False
        if len(self._t) >= self.W:
            corr_state, corr_echo = self._corr(t, mic_db, ref, timeline, res)
        # 해제: 예상 레벨을 넘는 창이 이어지면 사용자 말이 섞인 것
        if sum(self._over) >= self.RELEASE_COUNT:
            self._release_until = t + self.RELEASE_HOLD_MS / 1000.0
        released = t < self._release_until
        res.judged = lv is not None or corr_state not in ("이력부족", "보류")
        if released and (level_echo or corr_echo):
            res.released = True
        if not released and level_echo:
            res.echo, res.rule = True, "레벨"
        elif not released and corr_echo:
            res.echo, res.rule = True, "상관"
        elif res.judged:
            parts = ["해제"] if res.released else []
            if lv is not None and not level_echo:
                parts.append("레벨초과")
            parts.append(corr_state if not corr_echo else "상관")
            res.rule = "+".join(parts)
        # lag·이득 학습: 상관으로 에코가 확인된 창에서만
        if corr_echo and not released and res.lag_ms is not None:
            self.lag_est_ms += LAG_EST_ALPHA * (res.lag_ms - self.lag_est_ms)
            self.learned += 1
            ref_now = getattr(self, "_ref_now", None)
            if ref_now is not None and ref_now > REF_SILENT_DB and mic_db > DB_FLOOR + 3:
                self.gain_db += GAIN_ALPHA * ((mic_db - ref_now) - self.gain_db)
        self.last = res
        return res

    def _corr(self, t, mic_db, ref, timeline, res):
        """상관 규칙. 반환 (상태 문자열, 에코?)."""
        self._ref_now = None
        ts = np.asarray(self._t)
        mic_all = np.asarray(self._db, dtype=np.float64)
        # [lag, 창] 시각에 스피커에서 나오던 참조 위치 -> 포락선
        tt = ts[None, :] - self.lags[:, None]
        envs_all = ref.env_at(timeline.pos_at(tt)).astype(np.float64)
        loud = envs_all.max(axis=1) >= REF_SILENT_DB
        if not loud.any():
            return "보류", False
        # 디지털 무음 창(마이크 dB 하한, iOS가 재생 중 입력을 0으로 보냄: 실기 17:46 입력 -180 dBFS)은 상관에서 뺀다.
        # 섞어 두면 무음->에코 계단이 아무 lag와나 상관을 만들어 lag 점수가 엉뚱한 lag(412, -100 ms)로 튀었다.
        live = mic_all > DB_FLOOR + 0.5
        if int(live.sum()) < self.W // 2:
            return "보류", False
        mic, envs = mic_all[live], envs_all[:, live]
        mc = mic - mic.mean()
        mn = float(np.sqrt(np.sum(mc * mc)))
        ec = envs - envs.mean(axis=1, keepdims=True)
        en = np.sqrt(np.sum(ec * ec, axis=1))
        corr = np.zeros(len(self.lags))
        # 참조 포락선이 거의 평평하면(표준편차 < REF_MIN_STD_DB) 상관이 무의미: 그 lag는 판정에서 뺀다
        ok = loud & (en / np.sqrt(envs.shape[1]) >= REF_MIN_STD_DB)
        if mn > 1e-6 and ok.any():
            corr[ok] = (ec[ok] @ mc) / (en[ok] * mn)
        if self._lag_score is None:
            self._lag_score = corr.copy()
        else:
            self._lag_score += self.LAG_ALPHA * (corr - self._lag_score)
        k0 = int(np.argmax(self._lag_score))
        cand = [k for k in (k0 - 1, k0, k0 + 1) if 0 <= k < len(self.lags) and ok[k]]
        if not cand:
            return "보류", False
        k = max(cand, key=lambda i: corr[i])
        res.corr = float(corr[k])
        res.lag_ms = float(self.lags[k] * 1000.0)
        ref_row = envs_all[k]
        mic = mic_all
        r = self.NEAR
        pad = np.concatenate([np.full(r, DB_FLOOR), ref_row, np.full(r, DB_FLOOR)])
        near = np.max(np.stack([pad[i:i + len(ref_row)] for i in range(2 * r + 1)]), axis=0)
        floor = np.maximum(near + self.gain_db, self.noise_db)
        res.floor_db = float(floor[-1])
        # 상관 lag 기준으로도 바닥 + margin 초과가 이어지면 해제
        look = slice(-self.RELEASE_LOOK, None)
        if int(np.sum(mic[look] >= floor[look] + self.margin_db)) >= self.RELEASE_COUNT:
            self._release_until = t + self.RELEASE_HOLD_MS / 1000.0
        self._ref_now = float(ref_row[-1])
        if res.corr >= self.corr_thr:
            return "상관", True
        return "상관낮음", False


class AecStateStore:
    """에코 대조 게이트 학습 상태(lag·이득)를 클라이언트·설정별로 저장(VOICE_DATA_DIR/aec_state.json).

    형식: {"version": 1, "keys": {key: {"lag_ms", "gain_db", "learned", "updated"}}}. 임시 파일 뒤
    os.replace로 원자적 저장, 키마다 SAVE_MIN_INTERVAL_S에 한 번 이하. 파일이 없거나 깨졌으면 무시(빈 상태)."""

    def __init__(self, path, min_interval_s: float = SAVE_MIN_INTERVAL_S, clock=time.monotonic):
        self.path = Path(path)
        self.min_interval_s = min_interval_s
        self.clock = clock
        self._last_save: dict[str, float] = {}

    @staticmethod
    def make_key(client, vp, aec: bool, half_duplex: bool) -> str:
        """클라이언트 종류 + 화면 해상도(screen w x h @dpr) + 설정(aec, 반이중)."""
        c = client if isinstance(client, str) and client else "?"
        res = "?"
        if isinstance(vp, dict):
            sc, dpr = vp.get("screen"), vp.get("dpr")
            if isinstance(sc, list) and len(sc) >= 2 and all(isinstance(x, (int, float)) for x in sc[:2]):
                res = f"{int(sc[0])}x{int(sc[1])}"
                if isinstance(dpr, (int, float)) and not isinstance(dpr, bool):
                    res += f"@{float(dpr):g}"
        return f"{c[:20]}|{res}|aec={int(bool(aec))}|hd={int(bool(half_duplex))}"

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception as e:  # noqa: BLE001
            log.warning("aec_state 읽기 실패, 무시: %s", e)
            return {}
        keys = data.get("keys") if isinstance(data, dict) else None
        return keys if isinstance(keys, dict) else {}

    def load(self, key: str) -> Optional[dict]:
        v = self._read().get(key)
        if not isinstance(v, dict):
            return None
        try:
            lag, gain = float(v["lag_ms"]), float(v["gain_db"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (np.isfinite(lag) and np.isfinite(gain)):
            return None
        return {"lag_ms": lag, "gain_db": gain, "learned": int(v.get("learned") or 0)}

    def save(self, key: str, state: dict, force: bool = False) -> bool:
        """throttle(키마다 min_interval_s)을 지키며 저장. 저장했으면 True."""
        now = self.clock()
        last = self._last_save.get(key)
        if not force and last is not None and now - last < self.min_interval_s:
            return False
        self._last_save[key] = now
        keys = self._read()
        keys[key] = {**state, "updated": int(time.time())}
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps({"version": 1, "keys": keys}, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception as e:  # noqa: BLE001
            log.warning("aec_state 저장 실패: %s", e)
            return False
        return True
