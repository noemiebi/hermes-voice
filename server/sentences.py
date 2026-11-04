"""스트리밍 문장 분할기 + 음성용 텍스트 정리."""
from __future__ import annotations

import re

# 마크다운·기호 정리
_RE_CODE_FENCE = re.compile(r"```[^\n]*\n?")
_RE_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
# 번호 목록은 1~2자리 + . 또는 ) + 공백 + 숫자 아닌 글자일 때만(2026. 11. 4. 같은 날짜는 그대로)
_RE_BULLET = re.compile(r"(?m)^\s*(?:[-*+•]|\d{1,2}[.)](?=\s+\D))\s+")
_RE_INLINE_ITEM = re.compile(r"(?<=\s)\d{1,2}\)\s+(?=[^\d\s])")   # 문장 안 "1) 사과 2) 배"의 2)
_RE_HEADING = re.compile(r"(?m)^\s*#{1,6}\s*")
_RE_QUOTE = re.compile(r"(?m)^\s*>\s?")
# ~ < > _ 는 지우지 않는다(3~5도·10:00~12:00·my_file 등은 tts_text가 읽는 법을 정한다)
_RE_SYMBOLS = re.compile(r"[*`#|\[\]{}\\]")
_RE_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U00002B00-\U00002BFF️‍]")
_RE_SPACES = re.compile(r"[ \t]+")


def clean_for_speech(text: str) -> str:
    t = _RE_CODE_FENCE.sub("", text)
    t = _RE_LINK.sub(r"\1", t)
    t = _RE_URL.sub("", t)
    t = _RE_HEADING.sub("", t)
    t = _RE_QUOTE.sub("", t)
    t = _RE_BULLET.sub("", t)
    t = _RE_INLINE_ITEM.sub("", t)
    t = _RE_SYMBOLS.sub("", t)
    t = _RE_EMOJI.sub("", t)
    t = _RE_SPACES.sub(" ", t)
    return t.strip()


# 문장 경계: 종결부호(.?! 및 전각·말줄임) 뒤 공백/끝, 또는 줄바꿈.
# 숫자 사이 마침표(3.5)는 경계가 아님.
_BOUNDARY = re.compile(r"(?:(?<!\d)[.](?!\d)|[?!。？！…])+[\"'”’)\]]*(?=\s)|\n+")


class SentenceSplitter:
    """토큰을 feed()로 넣으면 완성된 문장 리스트를 돌려준다. 끝에 flush()."""

    def __init__(self, min_chars: int = 6, max_chars: int = 120):
        self.buf = ""
        self.pending = ""  # 너무 짧아 다음 문장과 합칠 조각
        self.min_chars = min_chars
        self.max_chars = max_chars

    def _emit(self, piece: str) -> list[str]:
        piece = clean_for_speech(piece)
        if not piece:
            return []
        joined = f"{self.pending} {piece}".strip() if self.pending else piece
        if len(re.sub(r"\W", "", joined)) < self.min_chars:
            self.pending = joined
            return []
        self.pending = ""
        return [joined]

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while True:
            m = _BOUNDARY.search(self.buf)
            if not m:
                break
            piece, self.buf = self.buf[: m.end()], self.buf[m.end():]
            out += self._emit(piece)
        # 부호 없이 너무 길어지면 쉼표·공백에서 자름
        while len(self.buf) > self.max_chars:
            cut = self.buf.rfind(", ", 0, self.max_chars)   # 쉼표가 충분히 뒤쪽에 있으면 쉼표 우선
            if cut < self.max_chars // 2:
                cut = self.buf.rfind(" ", 0, self.max_chars)
            if cut < self.max_chars // 2:
                cut = self.max_chars
            piece, self.buf = self.buf[: cut + 1], self.buf[cut + 1:]
            out += self._emit(piece)
        return out

    def flush(self) -> list[str]:
        rest = clean_for_speech(self.buf)
        self.buf = ""
        tail = f"{self.pending} {rest}".strip() if self.pending else rest
        self.pending = ""
        return [tail] if tail else []
