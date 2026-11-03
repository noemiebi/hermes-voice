"""TTS 직전 문장 정규화(독음).

Supertonic 3를 한국어("ko")로 고정해 쓰므로 영어 단어·약어·기호·숫자 표기를 그대로 넘기면 독음이 어색하다.
`normalize_for_tts(text)`는 한 문장을 받아 한글 독음 위주의 문자열로 바꾼다(순수 함수, 사전은 첫 호출 때 한 번 읽어 캐시).
화면 자막·에코 비교는 원문을 쓰고, 합성에만 이 결과를 쓴다.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

LEXICON_PATH = Path(__file__).with_name("tts_lexicon.tsv")
USER_LEXICON_DEFAULT = str(Path.home() / ".hermes" / "voice" / "tts-lexicon.tsv")

LETTERS = {
    "A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프", "G": "지", "H": "에이치",
    "I": "아이", "J": "제이", "K": "케이", "L": "엘", "M": "엠", "N": "엔", "O": "오", "P": "피",
    "Q": "큐", "R": "알", "S": "에스", "T": "티", "U": "유", "V": "브이", "W": "더블유", "X": "엑스",
    "Y": "와이", "Z": "지",
}
EXTENSIONS = {
    "py": "파이", "md": "엠디", "json": "제이슨", "wav": "웨이브", "txt": "텍스트", "js": "제이에스",
    "ts": "티에스", "html": "에이치티엠엘", "css": "씨에스에스", "pdf": "피디에프", "png": "피엔지",
    "jpg": "제이펙", "jpeg": "제이펙", "mp3": "엠피쓰리", "mp4": "엠피포", "csv": "씨에스브이",
    "tsv": "티에스브이", "yaml": "야믈", "yml": "야믈", "sh": "에스에이치", "zip": "집", "log": "로그",
    "toml": "톰엘", "swift": "스위프트", "docx": "독스", "xlsx": "엑셀", "pptx": "피피티", "hwp": "한글",
}
TLDS = {"com": "컴", "net": "넷", "org": "오알지", "kr": "케이알", "co": "씨오", "io": "아이오", "ai": "에이아이"}
UNITS = {
    "km": "킬로미터", "kg": "킬로그램", "cm": "센티미터", "mm": "밀리미터", "ml": "밀리리터", "mL": "밀리리터",
    "ms": "밀리초", "KB": "킬로바이트", "MB": "메가바이트", "GB": "기가바이트", "TB": "테라바이트",
    "kHz": "킬로헤르츠", "Hz": "헤르츠", "dB": "데시벨", "mg": "밀리그램",
}

_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF\U0001F1E6-\U0001F1FF"
    "\U00002300-\U000023FF️︎‍⃣]"
)
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"'()\[\]]+", re.I)
_EMAIL = re.compile(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)")
_EXT = "|".join(sorted(EXTENSIONS, key=len, reverse=True))
_PATH = re.compile(r"(?:~|\.{1,2})?/?(?:[\w.-]+/)+[\w.-]+\.(?:%s)(?![A-Za-z0-9])" % _EXT, re.I)
_FILE_EXT = re.compile(r"(?<=[\w)])\.(%s)(?![A-Za-z0-9])" % _EXT, re.I)
_NB = r"(?<![A-Za-z0-9])"   # 영문·숫자 단어 경계(한글 조사가 바로 붙어도 경계로 본다)
_NA = r"(?![A-Za-z0-9])"

_lexicon_cache: Optional[tuple[dict[str, str], re.Pattern]] = None


# ---------- 사전 ----------
def _read_tsv(path: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip() or line.lstrip().startswith("#") or "\t" not in line:
                    continue
                k, v = line.split("\t", 1)
                if k.strip():
                    out[k.strip().lower()] = v.strip()
    except OSError:
        pass  # 없으면 조용히 무시
    return out


def load_lexicon(base: str | Path | None = None, user: str | Path | None = None) -> dict[str, str]:
    """기본 사전 + 사용자 사전(같은 영문이면 사용자 우선). 키는 소문자."""
    if user is None:
        user = os.environ.get("TTS_LEXICON_USER", USER_LEXICON_DEFAULT)
    lex = _read_tsv(base or LEXICON_PATH)
    if user:
        lex.update(_read_tsv(user))
    return lex


def _compile(lex: dict[str, str]) -> re.Pattern:
    keys = sorted(lex, key=len, reverse=True)
    alts = [r"\s+".join(re.escape(p) for p in k.split()) for k in keys]
    return re.compile(_NB + "(" + "|".join(alts or ["(?!x)x"]) + ")" + _NA, re.I)


def _default_lexicon() -> tuple[dict[str, str], re.Pattern]:
    global _lexicon_cache
    if _lexicon_cache is None:
        lex = load_lexicon()
        _lexicon_cache = (lex, _compile(lex))
    return _lexicon_cache


def reload_lexicon() -> None:
    global _lexicon_cache
    _lexicon_cache = None


# ---------- 단계 ----------
def _spell(s: str) -> str:
    return "".join(LETTERS.get(c.upper(), c) for c in s)


def _markdown(t: str) -> str:
    t = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", t)               # [글](주소), ![대체](주소)
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", t)                    # 머리
    t = re.sub(r"(?m)^\s*(?:[-*+•]|>)\s+", "", t)                  # 글머리·인용
    t = re.sub(r"\|?\s*:?-{3,}:?\s*(?=\||$)", " ", t)              # 표 구분 줄
    t = t.replace("**", "").replace("__", "").replace("`", "")
    t = t.replace("|", ", ")
    return _EMOJI.sub("", t)


def _url_email(t: str) -> str:
    def url(m: re.Match) -> str:
        s = m.group(0)
        tail = len(s) - len(s.rstrip(".,!?;:"))
        return "링크" + (s[len(s) - tail:] if tail else "")
    t = _URL.sub(url, t)

    def email(m: re.Match) -> str:
        local = m.group(1).replace(".", " 점 ")
        parts = [TLDS.get(p.lower(), p) for p in m.group(2).split(".")]
        return f"{local} 골뱅이 " + " 점 ".join(parts)
    return _EMAIL.sub(email, t)


def _files(t: str) -> str:
    t = _PATH.sub(lambda m: m.group(0).replace("/", " ").strip(), t)
    return _FILE_EXT.sub(lambda m: " 점 " + EXTENSIONS[m.group(1).lower()], t)


def _numbers(t: str) -> str:
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)                                   # 3,990 -> 3990
    t = re.sub(r"(?<![\d-])(0\d{1,2})-(\d{3,4})-(\d{4})(?![\d-])", r"\1 \2 \3", t)   # 전화번호

    def ymd(m: re.Match) -> str:
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y}년 {mo}월 {d}일"
        return m.group(0)
    t = re.sub(r"(?<![\d.])((?:19|20)\d{2})[-./](\d{1,2})[-./](\d{1,2})(?![\d])", ymd, t)

    def ym(m: re.Match) -> str:
        mo = int(m.group(2))
        return f"{m.group(1)}년 {mo}월" if 1 <= mo <= 12 else m.group(0)
    t = re.sub(r"(?<![\d.])((?:19|20)\d{2})[-.](\d{1,2})(?![\d.-])", ym, t)

    def hm(m: re.Match) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        return f"{h}시" if mi == 0 else f"{h}시 {mi}분"
    t = re.sub(r"(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])", hm, t)
    t = re.sub(r"(?<=\d)\s*:\s*(?=\d)", " 대 ", t)                                   # 16:9

    def slash(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        ia, ib = int(a), int(b)
        # 한 자리/한 자리이고 분자 < 분모면 분수(1/2, 2/3), 아니면 유효한 월/일이면 날짜(11/2), 그 밖은 분수
        if not (len(a) == 1 and len(b) == 1 and ia < ib) and 1 <= ia <= 12 and 1 <= ib <= 31:
            return f"{ia}월 {ib}일"
        return f"{b}분의 {a}"
    t = re.sub(r"(?<![\d/.])(\d{1,3})\s*/\s*(\d{1,3})(?![\d/])", slash, t)

    t = re.sub(r"(?<=\d)\.(?=\d)", "점 ", t)                                          # 3.5 -> 3점 5
    t = re.sub(r"(?<=\d)\s*[~～∼–—-]\s*(?=\d)", "에서 ", t)                            # 3~5, 10-20
    t = re.sub(r"(?<=\S)\s*[~～∼]\s*(?=\d)", "에서 ", t)                               # 3시~5시
    t = re.sub(r"(?:(?<=^)|(?<=[\s(]))[-−](?=\d)", "마이너스 ", t)                       # -5도
    t = re.sub(r"\$\s?(\d+(?:점 \d+)?)", r"\1달러", t)
    t = re.sub(r"[₩￦]\s?(\d+(?:점 \d+)?)", r"\1원", t)
    t = re.sub(r"\s?(?:°\s?[CcＣ]|℃)" + _NA, "도", t).replace("°", "도")
    units = "|".join(sorted(UNITS, key=len, reverse=True))
    t = re.sub(r"(?<=\d)\s?(%s)(?![A-Za-z])" % units, lambda m: UNITS[m.group(1)], t)
    return t


def _symbols(t: str) -> str:
    t = re.sub(r"\s*[%％]", "퍼센트", t)
    t = re.sub(r"\s*&\s*", " 앤드 ", t)
    t = re.sub(r"(?<=\d)\s*\+\s*(?=\d)", " 플러스 ", t)
    t = t.replace("+", " ")
    t = re.sub(r"\s*=\s*", "는 ", t)
    t = re.sub(r"\s*[→⇒➔➜⟶]\s*", ", ", t)            # 화살표는 쉼표(쉬어 읽기)로
    t = re.sub(r"\s*[←↔⇔]\s*", ", ", t)
    t = re.sub(r"\s*[·•ㆍ]\s*", ", ", t)
    t = re.sub(r"\s+[-–—]+\s+|\s*[–—]+\s*", ", ", t)  # 대시
    t = re.sub(r"(?<=[A-Za-z가-힣])-(?=[A-Za-z가-힣0-9])", " ", t)
    t = re.sub(r"(?<=[A-Za-z0-9가-힣])\s*/\s*(?=[A-Za-z가-힣])", " 또는 ", t)
    t = t.replace("/", " ")
    t = re.sub(r"[\"'“”‘’`]", "", t)                    # 따옴표는 붙은 조사와 떨어지지 않게 그냥 지운다
    t = re.sub(r"[「」『』《》〈〉<>\[\]{}()（）*_^\\#@~]", " ", t)
    return t


def _english(t: str, lex: dict[str, str], pat: re.Pattern, mode: str) -> str:
    t = re.sub(r"(?<=[A-Za-z])['’]s" + _NA, "의", t)                    # Claude's -> Claude의
    t = pat.sub(lambda m: lex[re.sub(r"\s+", " ", m.group(1).lower())], t)
    # 사전에 없는 2~6글자 대문자 약어(숫자 섞여도 됨): 글자별로
    t = re.sub(_NB + r"(?=[A-Z0-9]*[A-Z])[A-Z0-9]{2,6}" + _NA, lambda m: _spell(m.group(0)), t)
    t = re.sub(_NB + r"[A-Z]" + _NA, lambda m: LETTERS[m.group(0)], t)  # 한 글자 대문자
    if mode == "spell":
        t = re.sub(r"[A-Za-z]{2,}", lambda m: _spell(m.group(0)), t)
    return t


def _tidy(t: str) -> str:
    t = re.sub(r"[ \t\r\n ]+", " ", t)
    t = re.sub(r"\s+([.,!?;:…])", r"\1", t)
    t = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", t)     # 겹친 쉼표
    t = re.sub(r"^[\s,.;:!?…]+", "", t)
    t = re.sub(r",\s*([.!?…])", r"\1", t)
    t = re.sub(r"([,;:])(?=[^\s\d])", r"\1 ", t)
    return t.strip().rstrip(",;:").strip()


def normalize_for_tts(text: str, *, lexicon: Optional[dict[str, str]] = None,
                      english_mode: Optional[str] = None) -> str:
    """한 문장을 TTS용 독음 문자열로. 읽을 게 없으면 빈 문자열."""
    if not text:
        return ""
    if lexicon is None:
        lex, pat = _default_lexicon()
    else:
        lex = {k.lower(): v for k, v in lexicon.items()}
        pat = _compile(lex)
    if english_mode is None:
        english_mode = os.environ.get("TTS_ENGLISH_MODE", "keep").strip().lower()
    t = _markdown(text)
    t = _url_email(t)
    t = _files(t)
    t = _numbers(t)
    t = _english(t, lex, pat, english_mode)  # Wi-Fi 같은 붙임표 항목이 기호 처리 전에 잡히도록
    t = _symbols(t)
    t = _tidy(t)
    return t if re.search(r"\w", t) else ""
