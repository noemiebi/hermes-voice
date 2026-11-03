"""mlx-whisper STT 상주. mlx 호출은 단일 워커 스레드에서만."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import wave
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import config

log = logging.getLogger("stt")

SAMPLE_RATE = 16000
MIN_SAMPLES = int(0.25 * SAMPLE_RATE)  # 0.25초 미만은 버림

# whisper가 무음·잡음에서 자주 지어내는 문구(짧을 때만 버림)
_HALLUCINATIONS = {
    "감사합니다", "감사합니다감사합니다", "고맙습니다", "시청해주셔서감사합니다", "시청해주셔서고맙습니다",
    "구독과좋아요부탁드립니다", "구독과좋아요", "좋아요와구독부탁드립니다", "다음영상에서만나요",
    "mbc뉴스이덕영입니다", "mbc뉴스", "자막제공", "자막by", "한글자막by", "아", "음", "어",
    "thankyou", "thanksforwatching", "you", "bye",
    "시청해주셔서정말감사합니다", "오늘도시청해주셔서감사합니다", "구독좋아요알림설정", "구독좋아요알림설정부탁드립니다",
    "다음시간에만나요", "다음에또만나요", "끝까지시청해주셔서감사합니다", "영상편집", "자막편집",
    "kbs뉴스", "sbs뉴스", "ytn뉴스", "jtbc뉴스", "mbc", "kbs", "sbs",
    "제작지원", "후원", "광고를포함하고있습니다", "유료광고포함", "이영상은유료광고를포함하고있습니다",
    "자막은설정에서선택하실수있습니다", "amaraorg커뮤니티에서제공한자막",
}
_HALLU_PATTERNS = [re.compile(p) for p in (
    r"시청.*감사", r"구독.*좋아요", r"좋아요.*구독", r"자막\s*(제공|by|편집)", r"(MBC|KBS|SBS|YTN|JTBC)\s*뉴스", r"^\W*$",
    r"유료\s*광고", r"(amara|Amara)", r"다음\s*(영상|시간)에서?\s*만나",
)]


def _norm(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text).lower()


def is_hallucination(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    n = _norm(t)
    if not n or n in _HALLUCINATIONS:
        return True
    if len(n) <= 20 and any(p.search(t) for p in _HALLU_PATTERNS):
        return True
    # 같은 글자/어절 반복(예: "아아아아아아")
    if len(n) >= 6 and len(set(n)) <= 2:
        return True
    return False


class Corrections:
    """잘못<TAB>올바름 표. 파일이 바뀌면 다시 읽음. 긴 것부터 치환."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._mtime: Optional[float] = None
        self.pairs: list[tuple[str, str]] = []

    def _reload(self) -> None:
        try:
            m = self.path.stat().st_mtime
        except OSError:
            self.pairs, self._mtime = [], None
            return
        if m == self._mtime:
            return
        pairs = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#") or "\t" not in line:
                continue
            wrong, right = line.split("\t", 1)
            if wrong.strip():
                pairs.append((wrong.strip(), right.strip()))
        pairs.sort(key=lambda p: -len(p[0]))
        self.pairs, self._mtime = pairs, m

    def apply(self, text: str) -> str:
        self._reload()
        for wrong, right in self.pairs:
            text = text.replace(wrong, right)
        return text


def _prompt_terms(prompt: str) -> list[str]:
    return [n for n in (_norm(t) for t in re.split(r"[,，、\n]", prompt or "")) if n]


def is_prompt_echo(text: str, prompt: str) -> bool:
    """whisper가 initial_prompt를 그대로(또는 거의) 뱉은 결과인지.
    어휘 하나만 말한 경우("카카오톡")는 살린다: 프롬프트 어휘 3개 이상이 나오고 그 밖의 글자가 적을 때,
    또는 프롬프트 전체와 매우 비슷할 때만 버림."""
    terms = _prompt_terms(prompt)
    if not terms:
        return False
    n = _norm(text)
    if not n:
        return False
    hits = [t for t in terms if t in n]
    if len(hits) >= 3:
        rest = n
        for t in sorted(hits, key=len, reverse=True):
            rest = rest.replace(t, "")
        if len(rest) <= 0.3 * len(n):
            return True
    whole = "".join(terms)
    return len(n) >= 8 and SequenceMatcher(None, n, whole).ratio() >= 0.7


def too_dense(text: str, seconds: float, max_cps: float = config.STT_MAX_CHARS_PER_SEC) -> bool:
    """오디오 길이에 비해 글자가 비정상적으로 많음(환각 반복 등). 짧은 클립은 1초로 본다."""
    return max_cps > 0 and len(_norm(text)) > max_cps * max(seconds, 1.0)


class ClipStore:
    """받아쓴 발화를 dir/YYYYmmdd-HHMMSS-mmm.wav(16 kHz mono int16) + .txt로 저장, 최근 keep개만 유지."""

    def __init__(self, path: str, keep: int = 200):
        self.dir = Path(path) if path else None
        self.keep = keep

    def save(self, pcm16: bytes, info: dict[str, Any]) -> Optional[Path]:
        if self.dir is None:
            return None
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            now = time.time()
            stem = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"
            wav = self.dir / f"{stem}.wav"
            with wave.open(str(wav), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(pcm16[: len(pcm16) // 2 * 2])
            lines = [f"{k}\t{v}" for k, v in info.items()]
            (self.dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            self._prune()
            return wav
        except Exception:  # noqa: BLE001
            log.exception("STT 클립 저장 실패")
            return None

    def _prune(self) -> None:
        if self.keep <= 0:
            return
        wavs = sorted(self.dir.glob("*.wav"))
        for old in wavs[: max(0, len(wavs) - self.keep)]:
            old.unlink(missing_ok=True)
            old.with_suffix(".txt").unlink(missing_ok=True)


class STTEngine:
    def __init__(self, model: str = config.STT_MODEL, corrections: str = config.STT_CORRECTIONS,
                 prompt: Optional[str] = None, temperatures: Optional[tuple] = None,
                 save_clips_dir: Optional[str] = None, save_clips_max: int = config.STT_SAVE_CLIPS_MAX,
                 decode_opts: Optional[dict] = None):
        self.model = model
        self.corrections = Corrections(corrections)
        self.prompt = (config.STT_PROMPT if prompt is None else prompt).strip()
        temps = tuple(config.STT_TEMPERATURES if temperatures is None else temperatures) or (0.0,)
        self.temperature: Any = temps[0] if len(temps) == 1 else tuple(temps)
        self.decode_opts = dict(decode_opts or {})
        # 기본은 저장 안 함. 서버(app.py)만 config 경로를 넘겨 켠다 — 벤치·스모크가 시험 문장을 클립에 섞던 문제 방지.
        self.clips = ClipStore(save_clips_dir or "", save_clips_max)
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._mlx = None
        self.loaded = False
        self.load_ms: Optional[int] = None
        self.error: Optional[str] = None

    def load_sync(self) -> None:
        t = time.perf_counter()
        try:
            os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")
            import mlx_whisper
            self._mlx = mlx_whisper
            mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32),
                                   path_or_hf_repo=self.model, language="ko")
            self.loaded = True
            self.load_ms = int((time.perf_counter() - t) * 1000)
            log.info("STT loaded %s in %d ms (prompt=%s, temperature=%s)",
                     self.model, self.load_ms, bool(self.prompt), self.temperature)
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            log.exception("STT load failed")

    async def load(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._exec, self.load_sync)

    def _keep_segment(self, s: dict) -> bool:
        if s.get("no_speech_prob", 0) > config.STT_NO_SPEECH_PROB and s.get("avg_logprob", 0) < config.STT_LOGPROB_MIN:
            return False
        if config.STT_COMPRESSION_MAX > 0 and s.get("compression_ratio", 0) > config.STT_COMPRESSION_MAX:
            return False
        return True

    def transcribe_detail(self, pcm16: bytes, save: bool = True) -> dict[str, Any]:
        """int16 LE 16k mono -> {"text", "raw", "ms", "reason"}. 버린 결과면 text는 "", reason에 이유."""
        t = time.perf_counter()
        audio = np.frombuffer(pcm16[: len(pcm16) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        secs = audio.size / SAMPLE_RATE
        if audio.size < MIN_SAMPLES or float(np.abs(audio).max(initial=0.0)) < 0.003:
            return {"text": "", "raw": "", "ms": int((time.perf_counter() - t) * 1000), "reason": "short/silent"}
        kw = dict(path_or_hf_repo=self.model, language="ko", condition_on_previous_text=False,
                  temperature=self.temperature)
        if self.prompt:
            kw["initial_prompt"] = self.prompt
        kw.update(self.decode_opts)
        r = self._mlx.transcribe(audio, **kw)
        segs = r.get("segments") or []
        raw = (r.get("text") or "").strip()
        if segs:
            text = "".join(s.get("text", "") for s in segs if self._keep_segment(s)).strip()
        else:
            text = raw
        reason = ""
        if segs and not text and raw:
            reason = "segments"
        text = self.corrections.apply(text)
        if text and is_hallucination(text):
            reason, text = "hallucination", ""
        elif text and self.prompt and is_prompt_echo(text, self.prompt):
            reason, text = "prompt-echo", ""
        elif text and too_dense(text, secs):
            reason, text = "too-dense", ""
        ms = int((time.perf_counter() - t) * 1000)
        if reason:
            log.info("STT 버림(%s): %r", reason, raw)
        out = {"text": text, "raw": raw, "ms": ms, "reason": reason}
        if save:
            self.clips.save(pcm16, {"text": text, "raw": raw, "ms": ms, "reason": reason or "-",
                                    "seconds": f"{secs:.2f}", "model": self.model})
        return out

    def transcribe_sync(self, pcm16: bytes) -> tuple[str, int]:
        """int16 LE 16k mono -> (교정·필터 거친 텍스트, ms). 버릴 결과면 텍스트는 ""."""
        d = self.transcribe_detail(pcm16)
        return d["text"], d["ms"]

    async def transcribe(self, pcm16: bytes) -> tuple[str, int]:
        return await asyncio.get_running_loop().run_in_executor(self._exec, self.transcribe_sync, pcm16)
