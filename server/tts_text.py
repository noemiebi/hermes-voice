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
    # 띄어 쓴 날짜(2026. 11. 4.): 뒤에 말이 이어지면 끝 마침표도 먹는다
    t = re.sub(r"(?<![\d.])((?:19|20)\d{2})\.\s(\d{1,2})\.\s(\d{1,2})(?:\.(?=\s+\S))?(?![\d])", ymd, t)

    def ym(m: re.Match) -> str:
        mo = int(m.group(2))
        return f"{m.group(1)}년 {mo}월" if 1 <= mo <= 12 else m.group(0)
    t = re.sub(r"(?<![\d.])((?:19|20)\d{2})[-.](\d{1,2})(?![\d.-])", ym, t)

    # 범위(3~5, 10-20, 10:00~12:00, 3시~5시)는 시각(hm)보다 먼저 "에서"로 바꾼다(시각을 바꾼 뒤엔 숫자가 아니라 못 잡음)
    t = re.sub(r"(?<=\d)\s*[~～∼–—-]\s*(?=\d)", "에서 ", t)
    t = re.sub(r"(?<=\S)\s*[~～∼]\s*(?=\d)", "에서 ", t)

    def hm(m: re.Match) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        # 1~12시만 고유어(열 시). 13~24시는 숫자 그대로 두어 한자어로(십삼 시) 읽게 한다
        hh = f"{native(h)} 시" if 1 <= h <= 12 else f"{h}시"
        return hh if mi == 0 else f"{hh} {mi}분"
    t = re.sub(r"(?<![\d:])([01]?\d|2[0-4]):([0-5]\d)(?![\d:])", hm, t)
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
    t = re.sub(r"(?:(?<=^)|(?<=[\s(]))[-−](?=\d)", "마이너스 ", t)                       # -5도
    t = re.sub(r"\$\s?(\d+(?:점 \d+)?)", r"\1달러", t)
    t = re.sub(r"[₩￦]\s?(\d+(?:점 \d+)?)", r"\1원", t)
    t = re.sub(r"\s?(?:°\s?[CcＣ]|℃)" + _NA, "도", t).replace("°", "도")
    units = "|".join(sorted(UNITS, key=len, reverse=True))
    t = re.sub(r"(?<=\d)\s?(%s)(?![A-Za-z])" % units, lambda m: UNITS[m.group(1)], t)
    return _native_counters(t)


# ---------- 고유어 수사(네 번, 한 개, 열 시) ----------
_NATIVE_ONES = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
_NATIVE_TENS = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]


def native(n: int) -> str:
    """1~99를 단위 앞 고유어 관형형으로(4 -> 네, 20 -> 스무, 21 -> 스물한)."""
    if n == 20:
        return "스무"
    return _NATIVE_TENS[n // 10] + _NATIVE_ONES[n % 10]


# 긴 것부터(번째·시간이 번·시보다 먼저). 쪽(페이지)은 한자어라 제외.
_COUNTERS = ["번째", "켤레", "송이", "그루", "가지", "군데", "사람", "자루", "포기", "조각", "시간", "마리",
             "번", "개", "명", "살", "시", "잔", "장", "권", "대", "병", "벌", "채", "척", "통", "곳",
             "달", "판", "끼", "알", "줄"]
# 단위 뒤에 오면 한자어(번호·장절·월·국가 등)로 보고 그대로 둔다
_NO_AFTER = {
    "번": r"지|길|[:.)]"                                                    # 5번지, 12번길, 1번:
          r"|\s?(?:버스|출구|도로|게이트|홀|라인|트랙|타자|플랫폼|문제|문항|항목|선택지|보기|후보|고객|손님|환자"
          r"|방(?!문)|채널|정답|답(?![했해하])|줄|테이블|좌석|창구|승강장|타석)"   # 3번 방(3번 방문은 횟수)
          r"|[이은가을는]\s(?:정답|맞|틀|답)",                                    # 2번이 정답, 3번은 틀렸어
    "개": r"월|국|년",
    "장": r"\s?\d+\s?절",
    "대": r"\s?(?:사건|선거|대통령|강국|보험|은행|도시|기업|주주|그룹|명절|원칙)|\s?\d",
    "달": r"러",
    "병": r"원|동|실",
    "판": r"\s?\d+\s?쇄",
}
# "N시"는 이 말이 뒤따를 때만 시각으로 본다(보수적)
_TIME_AFTER = re.compile(r"(?:에|까지|부터|쯤|경|께|\s?반|\s?정각|분|\d{1,2}\s?분|\s|[.,!?…]|$)")
_COUNTER_RE = re.compile(r"(?<![\dA-Za-z.#])(\d{1,2})(\s?)(%s)" % "|".join(_COUNTERS))


def _counter_word(num: str, unit: str, before: str, after: str) -> Optional[str]:
    """숫자+단위 하나를 고유어로 읽은 말. 한자어로 둘 자리(번호·장절·연령대·13시 이후 등)면 None."""
    n = int(num)
    if num.startswith("0") or not 1 <= n <= 99:
        return None
    if re.search(r"(?:제|No\.\s?|#\s?|점 )$", before):   # 제4번, No. 3, #2, 소수(3점 5개)
        return None
    if unit in _NO_AFTER and re.match(_NO_AFTER[unit], after):
        return None
    if unit == "대" and n % 10 == 0:                     # 20대(연령대)
        return None
    if unit == "시" and (n > 12 or not _TIME_AFTER.match(after)):   # 13시 이후는 숫자 그대로(십삼 시)
        return None
    if unit == "번째" and n == 1:
        return "첫 번째"
    return f"{native(n)} {unit}"


_COUNTER_RANGE_RE = re.compile(r"(?<![\dA-Za-z.#])(\d{1,2})에서 (\d{1,2})(\s?)(%s)" % "|".join(_COUNTERS))


def _native_counters(t: str) -> str:
    """숫자 + 고유어 단위(번·개·명·살·시·시간…)를 고유어 수사로. 1~99만, 예외는 그대로."""
    def rng(m: re.Match) -> str:
        # 3에서 5개 -> 세 개에서 다섯 개(앞 숫자에도 단위를 붙여 읽는다)
        a, b, unit = m.group(1), m.group(2), m.group(4)
        before, after = t[:m.start()], t[m.end():]
        wb = _counter_word(b, unit, before + a + "에서 ", after)
        if wb is None:
            return m.group(0)
        wa = _counter_word(a, unit, before, "에서")
        return f"{wa if wa else a}에서 {wb}" + (" " if after[:1].isdigit() else "")
    t = _COUNTER_RANGE_RE.sub(rng, t)

    def rep(m: re.Match) -> str:
        w = _counter_word(m.group(1), m.group(3), t[:m.start()], t[m.end():])
        if w is None:
            return m.group(0)
        return w + (" " if t[m.end():m.end() + 1].isdigit() else "")   # 7시30분 -> 일곱 시 30분
    return _COUNTER_RE.sub(rep, t)


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
    t = re.sub(r"[^\u0000-\uFFFF]", "", t)   # U+FFFF 초과(국기 태그·CJK 확장 B 등)는 Supertonic이 못 받는다
    t = _tidy(t)
    return t if re.search(r"\w", t) else ""
