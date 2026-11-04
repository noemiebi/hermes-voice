import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.sentences import SentenceSplitter, clean_for_speech  # noqa: E402


def stream(text, step=2, **kw):
    sp = SentenceSplitter(**kw)
    out = []
    for i in range(0, len(text), step):
        out += sp.feed(text[i:i + step])
    return out + sp.flush()


def test_basic_korean_endings():
    assert stream("오늘 날씨가 맑습니다. 내일은 비가 올 거예요. 우산 챙기세요!") == [
        "오늘 날씨가 맑습니다.", "내일은 비가 올 거예요.", "우산 챙기세요!"]


def test_question_and_newline():
    assert stream("지금 몇 시인지 궁금하세요?\n오후 세 시 반이에요") == [
        "지금 몇 시인지 궁금하세요?", "오후 세 시 반이에요"]


def test_short_fragment_merged_with_next():
    assert stream("네. 알겠어요. 바로 확인해 볼게요.") == ["네. 알겠어요. 바로 확인해 볼게요."]


def test_short_tail_flushed():
    assert stream("이 문장은 충분히 길어요. 네.") == ["이 문장은 충분히 길어요.", "네."]


def test_decimal_not_split():
    assert stream("기온은 3.5도 정도예요. 꽤 춥네요, 조심하세요.") == [
        "기온은 3.5도 정도예요.", "꽤 춥네요, 조심하세요."]


def test_emits_before_stream_end():
    sp = SentenceSplitter()
    assert sp.feed("첫 번째 문장입니다.") == []
    assert sp.feed(" 두 번") == ["첫 번째 문장입니다."]
    assert sp.flush() == ["두 번"]


def test_markdown_removed():
    text = "## 요약\n- **첫째** 항목은 [링크](https://x.com)예요.\n1. 둘째 `코드` 입니다 😀\n"
    assert stream(text, step=3) == ["요약 첫째 항목은 링크예요.", "둘째 코드 입니다"]


def test_clean_url():
    assert clean_for_speech("자세한 건 https://example.com/a?b=1 에서 보세요.") == "자세한 건 에서 보세요."


def test_long_without_punctuation_is_cut():
    text = "가나다라 " * 40
    out = stream(text, step=5, max_chars=60)
    assert len(out) >= 3 and all(len(s) <= 61 for s in out)


def test_empty():
    assert stream("") == []
    assert stream("**  **\n\n") == []


def test_symbols_kept_for_tts():
    # ~ < > _ 는 지우지 않는다(3~5도가 35도로 붙지 않게)
    assert clean_for_speech("기온은 3~5도예요.") == "기온은 3~5도예요."
    assert clean_for_speech("10:00~12:00에 열려요.") == "10:00~12:00에 열려요."
    assert clean_for_speech("파일은 my_file.py 입니다.") == "파일은 my_file.py 입니다."


def test_list_marker_keeps_dates():
    assert clean_for_speech("2026. 11. 4. 입니다.") == "2026. 11. 4. 입니다."
    assert clean_for_speech("1) 사과 2) 배") == "사과 배"
    assert clean_for_speech("1. 첫째") == "첫째"
    assert clean_for_speech("12) 열둘째") == "열둘째"


def test_end_to_end_range():
    from server.tts_text import load_lexicon, normalize_for_tts
    lex = load_lexicon(user="")
    out = [normalize_for_tts(s, lexicon=lex) for s in stream("기온은 3~5도예요. 10:00~12:00에 열려요.")]
    assert out == ["기온은 3에서 5도예요.", "열 시에서 열두 시에 열려요."]


def test_long_cut_prefers_comma():
    sp = SentenceSplitter(max_chars=60)
    text = "가" * 40 + ", " + "나다 " * 10   # 쉼표가 절반(30)보다 뒤에 있음
    out = sp.feed(text)
    assert out and out[0].endswith(",")
