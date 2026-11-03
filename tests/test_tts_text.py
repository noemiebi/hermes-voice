"""server/tts_text.normalize_for_tts 독음 규칙."""
import pytest

from server import tts_text
from server.tts_text import load_lexicon, normalize_for_tts

BASE = load_lexicon(user="")  # 사용자 사전 없이 기본 사전만(테스트가 실제 홈 폴더에 기대지 않게)


def n(s, **kw):
    kw.setdefault("lexicon", BASE)
    kw.setdefault("english_mode", "keep")
    return normalize_for_tts(s, **kw)


def test_base_lexicon_size():
    assert len(BASE) >= 80


@pytest.mark.parametrize("src,want", [
    # 1. 정리
    ("**중요**한 `설정`이에요.", "중요한 설정이에요."),
    ("## 오늘 할 일", "오늘 할 일"),
    ("- 첫째 항목", "첫째 항목"),
    ("* 둘째 항목", "둘째 항목"),
    ("| 이름 | 값 |", "이름, 값"),
    ("좋아요 🎉👍", "좋아요"),
    ("자세한 건 https://example.com/a?b=1 을 보세요.", "자세한 건 링크 을 보세요."),
    ("www.naver.com 접속.", "링크 접속."),
    ("[문서](https://x.y/z) 참고", "문서 참고"),
    ("a@b.c 로 보내", "a 골뱅이 b 점 c 로 보내"),
    ("hong@gmail.com 으로", "hong 골뱅이 지메일 점 컴 으로"),
    ("config.json 열어 봐.", "config 점 제이슨 열어 봐."),
    ("녹음은 out.wav 에 있어.", "녹음은 out 점 웨이브 에 있어."),
    ("README.md 고쳤어.", "리드미 점 엠디 고쳤어."),
    ("NASDAQ 지수", "엔에이에스디에이큐 지수"),  # 사전에 없는 6글자 대문자는 글자별
    # 2. 사전
    ("Hermes가 Claude Opus로 답해요.", "헤르메스가 클로드 오푸스로 답해요."),
    ("nous research 모델", "누스 리서치 모델"),
    ("Wi-Fi가 끊겼어요.", "와이파이가 끊겼어요."),
    ("GitHub와 Git", "깃허브와 깃"),
    ("ChatGPT랑 GPT", "챗지피티랑 지피티"),
    ("iPhone과 macOS, iOS", "아이폰과 맥오에스, 아이오에스"),
    ("Reminders에 넣었어요", "미리 알림에 넣었어요"),
    ("OK, Thanks!", "오케이, 땡스!"),
    ("Claude's 답", "클로드의 답"),
    # 3. 약어
    ("KTX 표", "케이티엑스 표"),
    ("A4 용지", "에이4 용지"),
    ("ABCDEFG 그대로", "ABCDEFG 그대로"),  # 7글자 넘으면 손대지 않음
    # 4. 영어 단어·한 글자
    ("hello world 예제", "헬로 world 예제"),
    ("플랜 B로 가요", "플랜 비로 가요"),
    # 5. 기호
    ("습도 45%", "습도 45퍼센트"),
    ("기온 23℃, 체감 21°C.", "기온 23도, 체감 21도."),
    ("R&D 예산", "알 앤드 디 예산"),
    ("1+1=2", "1 플러스 1는 2"),
    ("C++ 공부", "씨 공부"),
    ("3~5개", "3에서 다섯 개"),
    ("10-20분", "10에서 20분"),
    ("3시~5시", "세 시에서 다섯 시"),
    ("서울 → 부산", "서울, 부산"),       # 화살표는 쉼표로(선택한 규칙)
    ("사과·배·감", "사과, 배, 감"),
    ("커피/차 중에", "커피 또는 차 중에"),
    ("“따옴표”와 [대괄호] <꺾쇠> {중괄호}", "따옴표와 대괄호 꺾쇠 중괄호"),
    ("*별표* 강조(괄호 안)", "별표 강조 괄호 안"),
    # 6. 숫자
    ("3,990원", "3990원"),
    ("1,234,567명", "1234567명"),
    ("11:10에", "열한 시 10분에"),
    ("11:00에 만나", "열한 시에 만나"),
    ("09:05 출발", "아홉 시 5분 출발"),
    ("2026-11-02 회의", "2026년 11월 2일 회의"),
    ("2026.11.02.", "2026년 11월 2일."),
    ("11/2 일정", "11월 2일 일정"),
    ("2/3 정도", "3분의 2 정도"),      # 한 자리/한 자리이고 분자<분모면 분수
    ("3.5배", "3점 5배"),
    ("010-1234-5678로 전화", "010 1234 5678로 전화"),
    ("-5도", "마이너스 5도"),
    ("16:9 화면", "16 대 9 화면"),
    ("10GB 남음", "10기가바이트 남음"),
])
def test_rules(src, want):
    assert n(src) == want


@pytest.mark.parametrize("src,want", [
    # 7. 고유어 수사(횟수·개수 단위)
    ("4번 말했어", "네 번 말했어"),
    ("1번만", "한 번만"),
    ("20번", "스무 번"),
    ("21개", "스물한 개"),
    ("3번째", "세 번째"),
    ("1번째", "첫 번째"),
    ("11시 10분", "열한 시 10분"),
    ("10:30", "열 시 30분"),
    ("13시까지", "열세 시까지"),
    ("5명", "다섯 명"),
    ("2시간", "두 시간"),
    ("12살", "열두 살"),
    ("99마리", "아흔아홉 마리"),
    ("하루 3번씩", "하루 세 번씩"),
    ("차 3 대", "차 세 대"),
    # 그대로(한자어·번호)
    ("4번 버스", "4번 버스"),
    ("제4번", "제4번"),
    ("1번: 사과", "1번: 사과"),
    ("100번", "100번"),
    ("3장 1절", "3장 1절"),
    ("2분", "2분"),
    ("3개월", "3개월"),
    ("2대 사건", "2대 사건"),
    ("20대 여성", "20대 여성"),
    ("3점 5개", "3점 5개"),
    ("5달러", "5달러"),
    ("11월 2일", "11월 2일"),
    ("3층 5호", "3층 5호"),
    ("12번길", "12번길"),
])
def test_native_counters(src, want):
    assert n(src) == want


def test_mixed_sentence():
    src = "내일 10:30에 Gmail로 PDF 3,990원 영수증 보내 줘(약 2~3분)."
    assert n(src) == "내일 열 시 30분에 지메일로 피디에프 3990원 영수증 보내 줘 약 2에서 3분."


def test_spell_mode():
    assert n("hello world", english_mode="spell") == "헬로 더블유오알엘디"


@pytest.mark.parametrize("src", ["", "**!!**", "😀👍", "---", "( )", "→"])
def test_empty(src):
    assert n(src) == ""


def test_korean_untouched():
    s = "오늘 서울은 맑고 낮 최고 기온은 열여덟 도예요."
    assert n(s) == s


def test_user_lexicon_overrides(tmp_path):
    user = tmp_path / "tts-lexicon.tsv"
    user.write_text("# 내 사전\nClaude\t클로우드\nHana\t하나\n", encoding="utf-8")
    lex = load_lexicon(user=user)
    assert lex["claude"] == "클로우드" and lex["hermes"] == "헤르메스"
    assert n("Claude랑 Hana랑 Hermes", lexicon=lex) == "클로우드랑 하나랑 헤르메스"


def test_user_lexicon_missing_is_silent(tmp_path):
    missing = tmp_path / "nope" / "tts-lexicon.tsv"
    assert load_lexicon(user=missing) == BASE
    assert not missing.parent.exists()  # 폴더를 만들지 않는다


def test_default_lexicon_from_env(tmp_path, monkeypatch):
    user = tmp_path / "u.tsv"
    user.write_text("Gmail\t지 메일\n", encoding="utf-8")
    monkeypatch.setenv("TTS_LEXICON_USER", str(user))
    monkeypatch.setenv("TTS_ENGLISH_MODE", "keep")
    tts_text.reload_lexicon()
    try:
        assert normalize_for_tts("Gmail 확인") == "지 메일 확인"
    finally:
        tts_text.reload_lexicon()


def test_style_note_compatible():
    from server import config
    note = config.VOICE_STYLE_NOTE
    # 모델 이름은 끝 ')' 앞에 붙는다(session._think). 독음 지시는 화면 글자를 한글로 바꿔 버려서 넣지 않음.
    assert note.startswith("(") and note.endswith(")") and "독음" not in note


def test_session_synth_uses_normalized_but_caption_keeps_original():
    """합성에는 정규화 결과, speak_start·heard·에코 메모리는 원문. 빈 결과 생략은 test_empty와 session의 continue."""
    import asyncio
    import json

    import numpy as np

    from server.hermes import TurnResult
    from server.session import VoiceSession
    from tests.test_session_hold import FakeEP, FakeSTT, FakeWS, settle

    said = []

    class TTS:
        loaded = True

        async def synth(self, text, is_current=lambda: True):
            said.append(text)
            return np.zeros(4410, dtype=np.int16), 1

    class Hermes:
        def __init__(self):
            self.calls = []

        async def chat_stream(self, text, sid, model, on_delta, on_tool):
            self.calls.append(text)
            await on_delta("**!!** Gmail로 3,990원 보냈어요. ")
            return TurnResult(text="", session_id=sid)

    async def go():
        ws = FakeWS()
        s = VoiceSession(ws, FakeSTT(), TTS(), Hermes())
        s.endpointer = FakeEP()
        s.style_note = ""
        s.speaker_task = asyncio.create_task(s._speaker())
        await s._on_text(json.dumps({"type": "hello", "resume": None}))
        await s._on_text(json.dumps({"type": "text", "text": "보냈어?"}))
        await settle(30)
        s.speaker_task.cancel()
        return s, ws

    s, ws = asyncio.run(go())
    texts = [m["text"] for m in ws.of("speak_start")]
    assert "지메일로 3990원 보냈어요." in said
    assert all("Gmail" not in t for t in said) and all(t for t in said)
    assert any("Gmail로 3,990원" in t for t in texts)
    assert all("지메일" not in t for t in texts)
    assert any("Gmail로 3,990원" in h for h in s._answer["heard"])
