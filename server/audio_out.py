"""클라이언트로 보내는 TTS 오디오 부호화(순수 numpy).

바이너리 프레임 = `<I generation`(4바이트 LE) + 본문. 본문은 ready.out_codec에 따라
- pcm16: int16 LE, ready.out_sample_rate Hz 모노(샘플당 2바이트)
- mulaw: G.711 μ-law 8비트(샘플당 1바이트). 클라이언트는 web/playcore.js hvMulawDecode로 푼다.
"""
from __future__ import annotations

import numpy as np

CODECS = ("pcm16", "mulaw")
_BIAS = 0x84
_CLIP = 32635


def mulaw_encode(pcm) -> np.ndarray:
    """int16 PCM -> uint8 G.711 μ-law(벡터화)."""
    x = np.asarray(pcm, dtype=np.int16).reshape(-1).astype(np.int32)
    sign = (x < 0).astype(np.int32)
    mag = np.minimum(np.abs(x), _CLIP) + _BIAS              # 132..32767
    exp = np.frexp(mag.astype(np.float64))[1].astype(np.int32) - 8   # 최상위 비트 위치 - 7 (0..7)
    mant = (mag >> (exp + 3)) & 0x0F
    return (~((sign << 7) | (exp << 4) | mant) & 0xFF).astype(np.uint8)


def _decode_table() -> np.ndarray:
    u = (~np.arange(256, dtype=np.int32)) & 0xFF
    exp = (u >> 4) & 0x07
    mant = u & 0x0F
    mag = (((mant << 3) + _BIAS) << exp) - _BIAS
    return np.where(u & 0x80, -mag, mag).astype(np.int16)


MULAW_TABLE = _decode_table()


def mulaw_decode(data) -> np.ndarray:
    """uint8 μ-law(bytes 또는 배열) -> int16 PCM."""
    b = np.frombuffer(bytes(data), dtype=np.uint8) if isinstance(data, (bytes, bytearray, memoryview)) \
        else np.asarray(data, dtype=np.uint8)
    return MULAW_TABLE[b]


def encode(pcm, codec: str) -> bytes:
    """int16 PCM -> 프레임 본문 바이트."""
    if codec == "mulaw":
        return mulaw_encode(pcm).tobytes()
    return np.asarray(pcm, dtype="<i2").tobytes()
