"""스피커 에코 판별: 최근 읽어 준 문장과 받아쓴 문장이 같은 말인지.

아이폰 스피커 재생은 iOS AEC를 뚫고 마이크로 다시 들어와 STT가 그대로 받아쓴다(2026-11-02 실기).
그래서 '재생 중/직후에 시작된 발화'의 받아쓰기는 최근 speak_start 문장과 비교해 에코면 버린다.
"""
from __future__ import annotations

import os
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
# 퍼지 규칙(에코 검사 대상 발화에만): 받아쓰기 오류('십일월'->'십일올', 조사 붙음)를 견디게 최근 문장 독음의
# 아무 위치에서 길이 L±FUZZY_SLACK 부분 문자열과의 SequenceMatcher 비율이 이 이상이면 에코
ECHO_FUZZY_RATIO = float(os.environ.get("ECHO_FUZZY_RATIO", "0.75"))
FUZZY_MIN_LEN = 6      # 정규화(공백·문장부호 제거) 길이가 이보다 짧으면 퍼지 규칙을 쓰지 않음
FUZZY_SLACK = 2


def normalize(text: str) -> str:
    """공백·문장부호 제거, 소문자."""
    return re.sub(r"[\W_]+", "", text or "").lower()


_DIGITS = "영일이삼사오육칠팔구"
_SMALL = ["", "십", "백", "천"]
_BIG = ["", "만", "억", "조"]


def _sino(n: int) -> str:
    """정수를 한자어 수사 독음으로(11 -> 십일, 2025 -> 이천이십오, 10000 -> 만)."""
    if n == 0:
        return "영"
    out = []
    for gi, big in enumerate(_BIG):
        g = (n // 10 ** (4 * gi)) % 10000
        if not g:
            continue
        part = ""
        for k in range(3, -1, -1):
            d = (g // 10 ** k) % 10
            if d:
                part += ("" if d == 1 and k > 0 else _DIGITS[d]) + _SMALL[k]
        if big == "만" and g == 1:
            part = ""
        out.append(part + big)
    return "".join(reversed(out))


def read_digits(text: str) -> str:
    """남은 아라비아 숫자를 한자어 독음으로(TTS가 소리 내는 대로: '11월' -> '십일월', 6월 -> 유월).
    0으로 시작하거나 16자리 넘는 숫자는 한 자리씩 읽는다."""
    def rep(m: re.Match) -> str:
        num, after = m.group(0), text[m.end():m.end() + 1]
        if num.startswith("0") and len(num) > 1 or len(num) > 16:
            return "".join("공" if c == "0" else _DIGITS[int(c)] for c in num)
        n = int(num)
        if after == "월" and n in (6, 10):
            return "유" if n == 6 else "시"
        return _sino(n)
    return re.sub(r"\d+", rep, text or "")


def spoken_form(text: str) -> str:
    """TTS가 실제로 읽는 독음(normalize_for_tts + 남은 숫자 한자어 독음). 정규화 실패면 숫자 독음만."""
    try:
        from .tts_text import normalize_for_tts
        t = normalize_for_tts(text)
    except Exception:  # noqa: BLE001
        t = text or ""
    return read_digits(t)


def _variants(text: str) -> list[str]:
    """받아쓴 문장의 비교 형태: 원형과 독음형(다르면)."""
    out = [text]
    sp = spoken_form(text)
    if normalize(sp) and normalize(sp) != normalize(text):
        out.append(sp)
    return out


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


def fuzzy_ratio(text: str, recent: Iterable[str], last_n: int = 2, slack: int = FUZZY_SLACK) -> float:
    """받아쓴 문장(공백 제거 길이 L)과 최근 last_n문장의 아무 위치 길이 L±slack 부분 문자열 사이 최대
    SequenceMatcher 비율. L < FUZZY_MIN_LEN이면 0."""
    n = normalize(text)
    L = len(n)
    if L < FUZZY_MIN_LEN:
        return 0.0
    sents = [s for s in (normalize(x) for x in recent) if s][-last_n:] if last_n > 0 else []
    best = 0.0
    sm = SequenceMatcher(None, autojunk=False)
    sm.set_seq2(n)
    for s in sents:
        for ln in range(max(1, L - slack), L + slack + 1):
            if ln > len(s):
                ln = len(s)
            for i in range(0, len(s) - ln + 1):
                sm.set_seq1(s[i:i + ln])
                if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
                    continue
                r = sm.ratio()
                if r > best:
                    best = r
            if ln == len(s):
                break
    return best


def is_echo(text: str, recent: Iterable[str]) -> bool:
    return echo_reason(text, recent) is not None


class EchoMemory:
    """speak_start로 보낸 문장을 그 문장의 (추정) 재생 종료 시각부터 memory_s초 동안 기억.

    문장마다 원문과 독음(TTS가 읽은 소리: normalize_for_tts + 숫자 한자어 독음)을 함께 두고, 모든 비교를
    원문·독음 둘 다에 대해 한다. 받아쓴 문장도 원형과 독음형(숫자를 독음으로) 둘 다로 비교한다.
    어느 조합이든 걸리면 에코. 실기(18:33): 답 원문 '11월이라'를 TTS가 '십일월이라'로 읽었고 STT는
    '십일올이라나'로 받아써 원문 비교로는 못 잡았다."""

    def __init__(self, memory_s: float = 20.0, clock=time.monotonic):
        self.memory_s = memory_s
        self.clock = clock
        self._items: deque[tuple[float, str, str]] = deque()

    def add(self, text: str, spoken: Optional[str] = None, end: Optional[float] = None) -> None:
        """spoken: TTS에 넘긴 독음 문장(없으면 원문으로 만든다). end: 이 문장의 추정 재생 종료 시각(clock).
        없으면 지금."""
        sp = read_digits(spoken) if spoken is not None else spoken_form(text)
        self._items.append((self.clock() if end is None else end, text, sp))
        self._prune()

    def _prune(self) -> None:
        cut = self.clock() - self.memory_s
        # 종료 시각이 순서대로가 아닐 수 있으니(시간축 보정) 전체를 걸러 낸다
        if any(it[0] < cut for it in self._items):
            self._items = deque(it for it in self._items if it[0] >= cut)

    def recent(self) -> list[str]:
        self._prune()
        return [it[1] for it in self._items]

    def recent_spoken(self) -> list[str]:
        self._prune()
        return [it[2] for it in self._items]

    def _lists(self) -> list[list[str]]:
        orig, sp = self.recent(), self.recent_spoken()
        return [orig] if sp == orig else [orig, sp]

    def _any(self, text: str, fn) -> Optional[str]:
        for v in _variants(text):
            for i, lst in enumerate(self._lists()):
                r = fn(v, lst)
                if r:
                    return r if i == 0 else f"{r}, 독음"
        return None

    def reason(self, text: str) -> Optional[str]:
        return self._any(text, echo_reason)

    def lenient_reason(self, text: str, bigram: Optional[float] = 0.5, last_n: int = 2) -> Optional[str]:
        return self._any(text, lambda v, lst: lenient_echo_reason(v, lst, last_n=last_n, bigram=bigram))

    def global_reason(self, text: str) -> Optional[str]:
        return self._any(text, global_echo_reason)

    def fuzzy(self, text: str, last_n: int = 2) -> float:
        """퍼지 비율 최대(받아쓴 문장 원형·독음형 x 최근 last_n문장 원문·독음)."""
        best = 0.0
        for v in _variants(text):
            for lst in self._lists():
                best = max(best, fuzzy_ratio(v, lst, last_n=last_n))
        return best

    def fuzzy_reason(self, text: str, ratio: Optional[float] = None, last_n: int = 2) -> Optional[float]:
        """에코 검사 대상 발화에만 쓰는 퍼지 규칙. 에코면 비율, 아니면 None."""
        thr = ECHO_FUZZY_RATIO if ratio is None else ratio
        r = self.fuzzy(text, last_n=last_n)
        return r if r >= thr else None

    def is_echo(self, text: str) -> bool:
        return self.reason(text) is not None
