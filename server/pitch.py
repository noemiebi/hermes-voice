"""음높이(F0)로 '내 TTS 에코'와 '사용자 말'을 가른다(numpy만 사용).

실측(2026-11-03): 사용자 발화 F0 중앙값 94~130 Hz(중앙 109), Supertonic F1 198~259 Hz(중앙 218). 거의 한 옥타브 차이라
재생 중 말 시작 앞부분(PITCH_ONSET_MS)의 F0 중앙값이 어느 음역에 드는지로 덕킹 검사 없이 판정할 수 있다.

- f0_track: 40 ms 창·10 ms hop 정규화 자기상관(70~400 Hz), 옥타브 오류 보정, 3창 중앙값 필터.
- VoiceProfile: user / tts(엔진:음색 키별) 분포. 발화마다 (로그 F0 중앙값, 블록 중앙값 MAD)를 최근 200개까지 보관.
- VoiceProfileStore: VOICE_DATA_DIR/voice_profile.json 원자적 저장, 60초에 한 번 이하.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from . import config

log = logging.getLogger("pitch")

SR = 16000
MAX_UTTS = 200          # 분포마다 보관하는 최근 발화 수
VERSION = 1


# ---------- F0 추정 ----------
def to_float(x) -> np.ndarray:
    """int16 배열·bytes·float 배열을 float64 [-1, 1]로."""
    if isinstance(x, (bytes, bytearray, memoryview)):
        x = np.frombuffer(bytes(x)[: len(x) // 2 * 2], dtype="<i2")
    a = np.asarray(x)
    if a.dtype.kind in "iu":
        return a.astype(np.float64) / 32768.0
    return a.astype(np.float64).reshape(-1)


def resample(x: np.ndarray, sr: int, to: int = SR) -> np.ndarray:
    """FFT 방식 리샘플(대역 제한 포함). 44.1 kHz TTS 출력을 16 kHz로 내릴 때 쓴다."""
    x = to_float(x)
    if sr == to or x.size == 0:
        return x
    n_out = max(1, int(round(x.size * to / sr)))
    X = np.fft.rfft(x)
    k = n_out // 2 + 1
    if X.size < k:
        X = np.concatenate([X, np.zeros(k - X.size, dtype=X.dtype)])
    return np.fft.irfft(X[:k], n_out) * (n_out / x.size)


def _frames(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if x.size < win:
        return np.zeros((0, win))
    n = 1 + (x.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _nacf(fr: np.ndarray, kmax: int) -> np.ndarray:
    """창별 정규화 자기상관 r[k] = Σx[n]x[n+k] / sqrt(Σx[n]² Σx[n+k]²) (n = 0..N-k-1), k = 0..kmax."""
    m, n = fr.shape
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(fr, nfft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), nfft, axis=1)[:, : kmax + 1]
    sq = fr * fr
    c = np.concatenate([np.zeros((m, 1)), np.cumsum(sq, axis=1)], axis=1)  # c[:, j] = Σ_{i<j} sq
    k = np.arange(kmax + 1)
    e0 = c[:, n - k]                 # Σ_{i < n-k} x²
    e1 = c[:, n:n + 1] - c[:, k]     # Σ_{i ≥ k} x²
    return ac / np.sqrt(np.maximum(e0 * e1, 1e-20))


def _peak(r: np.ndarray, k: int, kmin: int, kmax: int) -> int:
    """k 근처(±2) 최댓값 위치."""
    lo, hi = max(kmin, k - 2), min(kmax, k + 2)
    if lo > hi:
        return k
    return lo + int(np.argmax(r[lo:hi + 1]))


def _pick(r: np.ndarray, kmin: int, kmax: int, min_corr: float, ratio: float, margin: float) -> Optional[float]:
    seg = r[kmin:kmax + 1]
    g = int(np.argmax(seg)) + kmin
    if r[g] < min_corr:
        return None
    # 1) 짧은 주기(높은 주파수) 쪽: 1/2(또는 1/3) 주기 근처가 ratio배 이상이면 그쪽(옥타브 낮게 잡는 오류 방지)
    k = g
    moved = True
    while moved:
        moved = False
        for m in (2, 3):
            if k / m < kmin:
                continue
            h = _peak(r, int(round(k / m)), kmin, kmax)
            if r[h] >= ratio * r[k] and r[h] >= min_corr:
                k, moved = h, True
                break
    # 2) 긴 주기 쪽: 두 배 주기가 margin 넘게 더 높으면 낮은 주파수(옥타브 높게 잡는 오류 방지)
    while 2 * k <= kmax:
        d = _peak(r, 2 * k, kmin, kmax)
        if r[d] > r[k] + margin:
            k = d
        else:
            break
    # 포물선 보간
    if kmin < k < kmax:
        a, b, c = r[k - 1], r[k], r[k + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den < 0 else 0.0
        return k + float(np.clip(off, -0.5, 0.5))
    return float(k)


def f0_track(x16k, sr: int = SR, win_ms: float = 40.0, hop_ms: float = 10.0, fmin: float = 70.0,
             fmax: float = 400.0, min_corr: float = 0.5, min_db: Optional[float] = None,
             ratio: float = 0.85, margin: float = 0.04) -> np.ndarray:
    """유성음 창들의 F0(Hz) 배열(창 순서). RMS가 min_db(기본 PITCH_MIN_DB, -45 dBFS) 미만인 창과
    상관이 min_corr 미만인 창은 뺀다. 이어진 창끼리 3창 중앙값 필터로 튐을 없앤다."""
    x = to_float(x16k)
    if sr != SR:
        x, sr = resample(x, sr), SR
    if min_db is None:
        min_db = config.PITCH_MIN_DB
    win, hop = int(sr * win_ms / 1000), int(sr * hop_ms / 1000)
    fr = _frames(x, win, hop)
    if fr.shape[0] == 0:
        return np.zeros(0)
    fr = fr - fr.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(fr * fr, axis=1))
    loud = 20 * np.log10(np.maximum(rms, 1e-12)) >= min_db
    kmin, kmax = int(np.floor(sr / fmax)), int(np.ceil(sr / fmin))
    kmax = min(kmax, win - 8)
    f0 = np.full(fr.shape[0], np.nan)
    idx = np.flatnonzero(loud)
    if idx.size:
        R = _nacf(fr[idx], kmax + 2)
        for j, i in enumerate(idx):
            k = _pick(R[j], kmin, kmax, min_corr, ratio, margin)
            if k is not None:
                f = sr / k
                if fmin * 0.95 <= f <= fmax * 1.05:
                    f0[i] = f
    # 3창 중앙값 필터(이웃이 무성음이면 있는 것만)
    out = []
    for i in range(f0.size):
        if f0[i] != f0[i]:
            continue
        nb = f0[max(0, i - 1):i + 2]
        nb = nb[nb == nb]
        out.append(float(np.median(nb)))
    return np.asarray(out)


def median_hz(f0: Iterable[float]) -> Optional[float]:
    a = np.asarray(list(f0), dtype=np.float64)
    return float(np.median(a)) if a.size else None


BLOCK = 30   # 퍼짐을 잴 때 쓰는 블록(유효 창 30개 ≈ 말 시작 판정 창 하나)


def utt_stats(f0: Iterable[float], onset: bool = False) -> Optional[tuple[float, float]]:
    """발화 하나의 (로그 F0 중앙값, 로그 퍼짐). 유효 창이 PITCH_MIN_FRAMES 미만이면 None.
    퍼짐: 판정은 짧은 앞부분 창들의 F0 '중앙값'으로 하므로, 창 하나하나의 MAD 대신 BLOCK창 블록(반씩 겹침)
    중앙값들이 발화 중앙값에서 벗어난 정도의 중앙값(블록이 2개 미만이면 창 MAD).
    onset=True(사용자 발화): 중앙값 자리에 첫 BLOCK 유효 창의 중앙값을 쓴다. 끼어들기 판정은 말 앞부분으로 하는데
    한국어 말 첫머리는 억양이 올라가 발화 전체보다 높다(clips 실측: 전체 109 Hz인 사람이 앞부분 140~180 Hz)."""
    a = np.asarray(list(f0), dtype=np.float64)
    a = a[(a > 0) & np.isfinite(a)]
    if a.size < max(1, config.PITCH_MIN_FRAMES):
        return None
    lg = np.log(a)
    med = float(np.median(lg))
    if onset:
        head = float(np.median(lg[:BLOCK]))
        blocks = [np.median(lg[i:i + BLOCK]) for i in range(0, lg.size - BLOCK + 1, BLOCK // 2)]
        spread = (float(np.median(np.abs(np.asarray(blocks) - med))) if len(blocks) >= 2
                  else float(np.median(np.abs(lg - med))))
        return head, spread
    blocks = [np.median(lg[i:i + BLOCK]) for i in range(0, lg.size - BLOCK + 1, BLOCK // 2)]
    if len(blocks) >= 2:
        return med, float(np.median(np.abs(np.asarray(blocks) - med)))
    return med, float(np.median(np.abs(lg - med)))


# ---------- 음역 프로필 ----------
class Dist:
    """한 목소리의 F0 분포: 발화별 (로그 중앙값, 로그 퍼짐(utt_stats)) 최근 MAX_UTTS개."""

    def __init__(self, entries: Optional[list] = None):
        self.entries: list[list[float]] = []
        for e in entries or []:
            if (isinstance(e, (list, tuple)) and len(e) == 2
                    and all(isinstance(v, (int, float)) and np.isfinite(v) for v in e)):
                self.entries.append([float(e[0]), float(e[1])])
        del self.entries[:-MAX_UTTS]

    def add(self, med: float, mad: float) -> None:
        self.entries.append([float(med), float(mad)])
        del self.entries[:-MAX_UTTS]

    @property
    def n(self) -> int:
        return len(self.entries)

    def center(self) -> float:
        return float(np.median([e[0] for e in self.entries]))

    def center_hz(self) -> float:
        return float(np.exp(self.center()))

    def scale(self) -> float:
        """로그 F0 퍼짐: 발화 안 퍼짐(중앙값)과 발화 사이 중앙값의 MAD를 합친 것(제곱합의 제곱근)."""
        meds = np.array([e[0] for e in self.entries])
        within = float(np.median([e[1] for e in self.entries]))
        between = float(np.median(np.abs(meds - np.median(meds))))
        return float(np.hypot(within, between))

    def half_width(self) -> float:
        """판정 구간 반폭(로그): max(PITCH_MARGIN × 퍼짐, log(1 + PITCH_MIN_WIDTH))."""
        return max(config.PITCH_MARGIN * self.scale(), float(np.log1p(config.PITCH_MIN_WIDTH)))

    def contains(self, lg: float) -> bool:
        return self.n > 0 and abs(lg - self.center()) <= self.half_width()

    def range_hz(self) -> tuple[float, float]:
        c, h = self.center(), self.half_width()
        return float(np.exp(c - h)), float(np.exp(c + h))


class VoiceProfile:
    """user 분포 하나와 TTS 분포(키 'supertonic:F1.json', 'qwen:Sohee' 별)."""

    def __init__(self, user: Optional[Dist] = None, tts: Optional[dict] = None):
        self.user = user or Dist()
        self.tts: dict[str, Dist] = dict(tts or {})
        self.active_tts: Optional[list[str]] = None  # 지금 서버 설정의 음색 키(None이면 저장된 전부)
        self._lock = threading.Lock()

    # 직렬화
    def to_json(self) -> dict:
        return {"version": VERSION, "user": self.user.entries,
                "tts": {k: d.entries for k, d in self.tts.items()}}

    @classmethod
    def from_json(cls, data) -> "VoiceProfile":
        if not isinstance(data, dict):
            return cls()
        tts = data.get("tts") if isinstance(data.get("tts"), dict) else {}
        return cls(Dist(data.get("user") if isinstance(data.get("user"), list) else None),
                   {str(k): Dist(v) for k, v in tts.items() if isinstance(v, list)})

    # 학습
    def add_user(self, f0) -> bool:
        st = utt_stats(f0, onset=True)
        if st is None:
            return False
        with self._lock:
            self.user.add(*st)
        return True

    def set_tts(self, key: str, f0_list: list) -> int:
        """문장별 F0 배열 목록으로 key 음색 분포를 새로 만든다. 쓴 문장 수."""
        d = Dist()
        for f0 in f0_list:
            st = utt_stats(f0)
            if st is not None:
                d.add(*st)
        if d.n:
            with self._lock:
                self.tts[key] = d
        return d.n

    # 판정
    def user_ready(self) -> bool:
        return self.user.n >= config.PITCH_USER_MIN_UTTS

    def tts_dists(self) -> list[tuple[str, Dist]]:
        keys = self.active_tts if self.active_tts is not None else list(self.tts)
        return [(k, self.tts[k]) for k in keys if k in self.tts and self.tts[k].n > 0]

    def ready(self) -> bool:
        return self.user_ready() and bool(self.tts_dists())

    def classify(self, f0_values) -> tuple[str, float]:
        """("user"|"tts"|"unknown", score). score: 판정 구간 안쪽 깊이(1 = 중심, 0 = 경계), unknown이면 0.
        유효 창이 PITCH_MIN_FRAMES 미만, 프로필 미학습, 두 구간에 다 들거나 다 안 들면 unknown."""
        a = np.asarray(list(f0_values), dtype=np.float64)
        a = a[(a > 0) & np.isfinite(a)]
        if a.size < max(1, config.PITCH_MIN_FRAMES) or not self.ready():
            return "unknown", 0.0
        lg = float(np.log(np.median(a)))
        with self._lock:
            in_user = self.user.contains(lg)
            tts_hit = [d for _, d in self.tts_dists() if d.contains(lg)]
            if in_user == bool(tts_hit):
                return "unknown", 0.0
            d = self.user if in_user else max(tts_hit, key=lambda x: -abs(lg - x.center()))
            score = 1.0 - abs(lg - d.center()) / d.half_width()
        return ("user" if in_user else "tts"), float(score)

    def summary(self) -> str:
        u = (f"user {self.user.n}발화 중앙 {self.user.center_hz():.0f} Hz" if self.user.n
             else "user 0발화")
        if self.user.n and not self.user_ready():
            u += f"(판정은 {config.PITCH_USER_MIN_UTTS}발화부터)"
        ts = [f"tts({k}) 중앙 {d.center_hz():.0f} Hz" for k, d in self.tts_dists()]
        if not ts:
            ts = [f"tts({','.join(self.active_tts or []) or '-'}) 없음"]
        return ", ".join([u] + ts)


class VoiceProfileStore:
    """voice_profile.json 원자적 저장(임시 파일 + os.replace). save()는 PITCH_SAVE_INTERVAL_S에 한 번 이하."""

    def __init__(self, path, interval_s: Optional[float] = None, clock=time.monotonic):
        self.path = Path(path) if path else None
        self.interval_s = config.PITCH_SAVE_INTERVAL_S if interval_s is None else interval_s
        self.clock = clock
        self._last = -1e18
        self.dirty = False
        self.profile = self.load()

    def load(self) -> VoiceProfile:
        if self.path is None:
            return VoiceProfile()
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return VoiceProfile()
        except OSError as e:
            log.warning("음역 프로필을 못 읽음(빈 상태로 시작): %s", e)
            return VoiceProfile()
        try:
            return VoiceProfile.from_json(json.loads(raw))
        except (ValueError, TypeError) as e:
            log.warning("음역 프로필이 깨져 있음(빈 상태로 시작): %s: %s", self.path, e)
            return VoiceProfile()

    def _write(self) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".voice_profile-", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                with self.profile._lock:
                    json.dump(self.profile.to_json(), f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def save(self, force: bool = False) -> bool:
        """바뀐 게 있고 마지막 저장 뒤 interval_s가 지났으면(force면 바로) 저장. 저장했으면 True."""
        self.dirty = True
        return self.flush(force)

    def flush(self, force: bool = True) -> bool:
        if self.path is None or not self.dirty:
            return False
        now = self.clock()
        if not force and now - self._last < self.interval_s:
            return False
        try:
            self._write()
        except OSError:
            log.exception("음역 프로필 저장 실패: %s", self.path)
            return False
        self._last = now
        self.dirty = False
        return True


_STORE: Optional[VoiceProfileStore] = None
_STORE_LOCK = threading.Lock()


def get_store() -> VoiceProfileStore:
    """앱 전체에 하나(PITCH_PROFILE_PATH, off면 저장·로드 안 함)."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            p = config.PITCH_PROFILE_PATH.strip()
            _STORE = VoiceProfileStore(None if p.lower() == "off" else p)
        return _STORE


# ---------- TTS 음역 학습 ----------
TTS_LEARN_SENTENCES = (
    "안녕하세요, 무엇을 도와드릴까요?",
    "오늘 서울은 맑고 낮 최고 기온은 이십 도 정도예요.",
    "그 일정은 내일 오후 세 시로 잡아 둘게요.",
    "더 궁금한 게 있으면 편하게 말씀해 주세요.",
)


def tts_keys(tts) -> list[tuple[str, str]]:
    """지금 서버에서 나올 수 있는 음색 키 [(키, 엔진)]. TTSRouter면 Qwen(설정 시)과 Supertonic(폴백) 둘 다."""
    out = []
    qwen = getattr(tts, "qwen", None)
    if getattr(tts, "engine", None) == "qwen" and qwen is not None:
        out.append((f"qwen:{getattr(qwen, 'voice', '?')}", "qwen"))
    st = getattr(tts, "supertonic", tts)
    voice = getattr(st, "voice", None)
    if voice:
        out.append((f"supertonic:{os.path.basename(str(voice))}", "supertonic"))
    return out


async def _synth_one(tts, engine: str, text: str) -> Optional[np.ndarray]:
    """문장 하나를 해당 엔진으로 합성해 44.1 kHz int16. Supertonic은 엔진 워커 스레드(TTSEngine.synth)."""
    if engine == "qwen":
        chunks = [np.asarray(c).reshape(-1) async for c in tts.qwen.synth_stream(text)]
        return np.concatenate(chunks) if chunks else None
    st = getattr(tts, "supertonic", tts)
    r = await st.synth(text)
    return None if r is None else r[0]


async def learn_tts(tts, store: Optional[VoiceProfileStore] = None, sentences=TTS_LEARN_SENTENCES,
                    sr: int = 44100) -> list[str]:
    """현재 음색(키별)으로 문장들을 합성해 TTS F0 분포를 잰다(이미 잰 키는 건너뜀). 잰 키 목록."""
    import asyncio
    store = store or get_store()
    prof = store.profile
    keys = tts_keys(tts)
    prof.active_tts = [k for k, _ in keys]
    done = []
    for key, engine in keys:
        if key in prof.tts and prof.tts[key].n >= max(1, len(sentences) // 2):
            continue
        if engine == "qwen" and not getattr(tts.qwen, "ready", False):
            continue
        if engine == "supertonic" and not getattr(getattr(tts, "supertonic", tts), "loaded", False):
            continue
        f0s = []
        for text in sentences:
            try:
                pcm = await _synth_one(tts, engine, text)
            except Exception as e:  # noqa: BLE001
                log.warning("음역 학습 합성 실패(%s): %s", key, e)
                continue
            if pcm is not None and len(pcm):
                f0s.append(await asyncio.to_thread(lambda p=pcm: f0_track(resample(p, sr))))
        n = prof.set_tts(key, f0s)
        if n:
            done.append(key)
            d = prof.tts[key]
            lo, hi = d.range_hz()
            log.info("음역 학습: %s 문장 %d개 중앙 %.0f Hz (판정 구간 %.0f~%.0f Hz)", key, n, d.center_hz(), lo, hi)
    if done:
        store.save()
    return done
