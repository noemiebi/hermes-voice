"""Supertonic 3 TTS 상주. 단일 워커 스레드에서 문장 -> int16 44.1kHz PCM."""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np
import onnxruntime as ort

from . import config
from .supertonic_helper import (
    TextToSpeech,
    load_cfgs,
    load_onnx_all,
    load_text_processor,
    load_voice_style,
)

log = logging.getLogger("tts")

SAMPLE_RATE = 44100
FRAME = SAMPLE_RATE * 32 // 1000   # 32 ms 프레임(라우드니스 측정)
VOICED_DB = -40.0                  # 이보다 큰 프레임만 음성 구간으로 측정


def _db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-9))


def speech_rms_db(wav: np.ndarray) -> Optional[float]:
    """float [-1,1] 신호의 음성 구간(32 ms 프레임 RMS > -40 dBFS) RMS dBFS. 음성 구간이 없으면 None."""
    n = len(wav) // FRAME * FRAME
    if n == 0:
        return None
    fr = wav[:n].reshape(-1, FRAME).astype(np.float64)
    ms = np.mean(fr * fr, axis=1)
    voiced = ms > 10 ** (VOICED_DB / 10)
    if not voiced.any():
        return None
    return _db(float(np.sqrt(np.mean(ms[voiced]))))


def trim_silence(wav: np.ndarray, thresh_db: float = -45.0, lead_s: float = 0.06, tail_s: float = 0.12) -> np.ndarray:
    """앞뒤 무음을 잘라 내되 앞 lead_s, 뒤 tail_s만큼은 남긴다(2026-11-02 벤치: 앞뒤 약 0.4초 무음)."""
    if len(wav) < FRAME * 2:
        return wav
    n = len(wav) // FRAME
    frames = wav[: n * FRAME].reshape(n, FRAME)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-12
    loud = np.flatnonzero(20 * np.log10(rms) > thresh_db)
    if len(loud) == 0:
        return wav
    start = max(0, loud[0] * FRAME - int(lead_s * SAMPLE_RATE))
    end = min(len(wav), (loud[-1] + 1) * FRAME + int(tail_s * SAMPLE_RATE))
    return wav[start:end]


def normalize_loudness(wav: np.ndarray, target_db: float = config.TTS_TARGET_RMS_DB,
                       peak_db: float = config.TTS_PEAK_DB) -> tuple[np.ndarray, float]:
    """음성 구간 RMS를 target_db로 맞추는 이득을 적용(피크는 peak_db 이하로 제한). (신호, 이득 dB)."""
    rms = speech_rms_db(wav)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if rms is None or peak <= 0:
        return wav, 0.0
    gain = 10 ** ((target_db - rms) / 20)
    gain = min(gain, 10 ** (peak_db / 20) / peak)
    return (wav * gain).astype(np.float32), _db(gain)


def _providers(name: str) -> list:
    if name == "coreml":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def build_tts(model_dir: str, provider: str = "cpu") -> TextToSpeech:
    """supertonic_helper.load_text_to_speech와 같되 provider를 고를 수 있게."""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    cfgs = load_cfgs(model_dir)
    sessions = load_onnx_all(model_dir, opts, _providers(provider))
    return TextToSpeech(cfgs, load_text_processor(model_dir), *sessions)


class TTSEngine:
    def __init__(self, model_dir: str = config.SUPERTONIC_MODEL_DIR,
                 voice: str = config.SUPERTONIC_VOICE, speed: float = config.TTS_SPEED,
                 steps: int = config.TTS_STEPS, provider: str = config.TTS_PROVIDER):
        self.model_dir, self.voice, self.speed, self.steps, self.provider = (
            model_dir, voice, speed, steps, provider)
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self.tts: Optional[TextToSpeech] = None
        self.style = None
        self.loaded = False
        self.load_ms: Optional[int] = None
        self.error: Optional[str] = None

    def load_sync(self) -> None:
        t = time.perf_counter()
        try:
            self.tts = build_tts(self.model_dir, self.provider)
            self.style = load_voice_style([self.voice])
            self.tts("안녕하세요.", "ko", self.style, self.steps, self.speed, 0.1)  # 예열
            self.loaded = True
            self.load_ms = int((time.perf_counter() - t) * 1000)
            log.info("TTS loaded (%s) in %d ms", self.provider, self.load_ms)
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            log.exception("TTS load failed")

    async def load(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._exec, self.load_sync)

    def synth_sync(self, text: str) -> np.ndarray:
        assert self.tts is not None, "TTS not loaded"
        wav, _dur = self.tts(text, "ko", self.style, self.steps, self.speed, 0.2)
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        wav = trim_silence(wav)  # Supertonic은 앞뒤로 약 0.4초씩 무음을 붙인다 -> 체감 지연·문장 사이 공백 줄이기
        wav, gain_db = normalize_loudness(wav)
        if log.isEnabledFor(logging.DEBUG):
            before = speech_rms_db(wav / (10 ** (gain_db / 20)))
            log.debug("TTS 레벨: 음성 RMS %.1f -> %.1f dBFS, 이득 %+.1f dB, 피크 %.1f dBFS",
                      before or -99, speech_rms_db(wav) or -99, gain_db,
                      _db(float(np.max(np.abs(wav))) if len(wav) else 0.0))
        return (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)

    async def synth(self, text: str, is_current: Callable[[], bool] = lambda: True
                    ) -> Optional[tuple[np.ndarray, int]]:
        """(pcm int16, ms). 워커에서 차례가 왔을 때 is_current()가 거짓이면 건너뛰고 None."""
        def job():
            if not is_current():
                return None
            t = time.perf_counter()
            pcm = self.synth_sync(text)
            return pcm, int((time.perf_counter() - t) * 1000)
        return await asyncio.get_running_loop().run_in_executor(self._exec, job)
