"""스피커 에코 판별: 최근 읽어 준 문장과 받아쓴 문장이 같은 말인지.

아이폰 스피커 재생은 iOS AEC를 뚫고 마이크로 다시 들어와 STT가 그대로 받아쓴다(2026-11-02 실기).
그래서 '재생 중/직후에 시작된 발화'의 받아쓰기는 최근 speak_start 문장과 비교해 에코면 버린다.
"""
from __future__ import annotations

import re
import time
from collections import deque
from difflib import SequenceMatcher
from typing import Iterable, Optional

ECHO_RATIO = 0.75      # SequenceMatcher 비율
ECHO_BIGRAM = 0.7      # 받아쓴 문장의 문자 bigram 중 최근 문장에 있는 비율
MIN_LEN = 2            # 정규화 뒤 이보다 짧으면("네") 에코로 보지 않음
GLOBAL_MIN_LEN = 12    # 전역 규칙(재생 꼬리와 무관): 정규화 길이가 이 이상이고
GLOBAL_SIM = 0.85      # 최근 문장과의 유사도(비율 또는 bigram 포함률)가 이 이상이면 에코


def normalize(text: str) -> str:
    """공백·문장부호 제거, 소문자."""
    return re.sub(r"[\W_]+", "", text or "").lower()


def _bigrams(s: str) -> list[str]:
    return [s[i:i + 2] for i in range(len(s) - 1)]


def echo_reason(text: str, recent: Iterable[str]) -> Optional[str]:
    """에코면 판정 근거 문자열, 아니면 None."""
    n = normalize(text)
    if len(n) < MIN_LEN:
        return None
    sents = [normalize(s) for s in recent]
    sents = [s for s in sents if s]
    if not sents:
        return None
    joined = "".join(sents)
    if n in joined:
        return "부분 문자열"
    for s in sents:
        r = SequenceMatcher(None, n, s).ratio()
        if r >= ECHO_RATIO:
            return f"유사도 {r:.2f}"
    bg = _bigrams(n)
    if bg:
        have = set(_bigrams(joined))
        # 문장 경계를 넘는 bigram은 joined에만 있으니 문장별로도 모아 둔다
        for s in sents:
            have.update(_bigrams(s))
        frac = sum(b in have for b in bg) / len(bg)
        if frac >= ECHO_BIGRAM:
            return f"bigram {frac:.2f}"
    return None


def global_echo_reason(text: str, recent: Iterable[str], min_len: int = GLOBAL_MIN_LEN,
                       sim: float = GLOBAL_SIM) -> Optional[str]:
    """재생 꼬리와 무관하게 모든 발화에 쓰는 엄격한 규칙. 길고(≥ min_len) 최근 문장과 거의 같으면 에코.
    실기(19:56): 재생 끝 추정 13초 뒤 받아쓴 5.8초 발화가 방금 읽은 문장 그대로였다."""
    n = normalize(text)
    if len(n) < min_len:
        return None
    sents = [s for s in (normalize(x) for x in recent) if s]
    if not sents:
        return None
    joined = "".join(sents)
    if n in joined:
        return "부분 문자열"
    for s in sents:
        r = SequenceMatcher(None, n, s).ratio()
        if r >= sim:
            return f"유사도 {r:.2f}"
    bg = _bigrams(n)
    have = set(_bigrams(joined))
    for s in sents:
        have.update(_bigrams(s))
    frac = sum(b in have for b in bg) / len(bg)
    if frac >= sim:
        return f"bigram {frac:.2f}"
    return None


_NUM_ONLY = re.compile(r"(?:\d+(?:년|월|일|번|시|분|초|개|명|회|살|세|주|차|층|호|원|도|퍼센트)?)+")


def _digit_bigrams(tok: str) -> list[str]:
    return [tok[i:i + 2] for i in range(len(tok) - 1)]


def lenient_echo_reason(text: str, recent: Iterable[str], last_n: int = 2, bigram: Optional[float] = 0.5) -> Optional[str]:
    """짧은 에코 조각용 완화 기준(재생 중 시작된 짧은 발화에만 쓴다). 최근 last_n문장과 비교.

    1) 숫자·단위만 있는 전사('2020년', '3번 1일'): 숫자 토큰이 모두 최근 문장 숫자에 있거나, 숫자 bigram의
       절반 이상이 최근 문장 숫자 bigram에 있으면 에코. 실기(17:46): 답의 '2025년'/'20일'을 '2020년'으로 받아씀.
    2) bigram이 None이 아니면: 최근 문장 부분 문자열이거나 문자 bigram 포함률 ≥ bigram이면 에코."""
    n = normalize(text)
    if not n:
        return None
    sents = [s for s in (normalize(x) for x in recent) if s][-last_n:] if last_n > 0 else []
    if not sents:
        return None
    joined = "".join(sents)
    if _NUM_ONLY.fullmatch(n):
        toks = re.findall(r"\d+", n)
        have = re.findall(r"\d+", joined)
        if toks and have:
            if all(t in have for t in toks):
                return "숫자 일치(완화)"
            bg = [b for t in toks for b in _digit_bigrams(t)]
            hb = {b for t in have for b in _digit_bigrams(t)}
            if bg and sum(b in hb for b in bg) / len(bg) >= 0.5:
                return f"숫자 bigram {sum(b in hb for b in bg) / len(bg):.2f}(완화)"
    if bigram is None or len(n) < MIN_LEN:
        return None
    if n in joined:
        return "부분 문자열(완화)"
    bg = _bigrams(n)
    if bg:
        have = set(_bigrams(joined))
        for s in sents:
            have.update(_bigrams(s))
        frac = sum(b in have for b in bg) / len(bg)
        if frac >= bigram:
            return f"bigram {frac:.2f}(완화)"
    return None


def is_echo(text: str, recent: Iterable[str]) -> bool:
    return echo_reason(text, recent) is not None


class EchoMemory:
    """speak_start로 보낸 문장을 그 문장의 (추정) 재생 종료 시각부터 memory_s초 동안 기억."""

    def __init__(self, memory_s: float = 20.0, clock=time.monotonic):
        self.memory_s = memory_s
        self.clock = clock
        self._items: deque[tuple[float, str]] = deque()

    def add(self, text: str, end: Optional[float] = None) -> None:
        """end: 이 문장의 추정 재생 종료 시각(clock). 없으면 지금."""
        self._items.append((self.clock() if end is None else end, text))
        self._prune()

    def _prune(self) -> None:
        cut = self.clock() - self.memory_s
        # 종료 시각이 순서대로가 아닐 수 있으니(시간축 보정) 전체를 걸러 낸다
        if any(t < cut for t, _ in self._items):
            self._items = deque((t, x) for t, x in self._items if t >= cut)

    def recent(self) -> list[str]:
        self._prune()
        return [t for _, t in self._items]

    def reason(self, text: str) -> Optional[str]:
        return echo_reason(text, self.recent())

    def lenient_reason(self, text: str, bigram: Optional[float] = 0.5, last_n: int = 2) -> Optional[str]:
        return lenient_echo_reason(text, self.recent(), last_n=last_n, bigram=bigram)

    def global_reason(self, text: str) -> Optional[str]:
        return global_echo_reason(text, self.recent())

    def is_echo(self, text: str) -> bool:
        return self.reason(text) is not None
