"""에코 필터 is_echo 사례."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.echo import EchoMemory, is_echo, normalize  # noqa: E402
from server.session import FILLERS  # noqa: E402

RECENT = [FILLERS[0], "오늘 서울은 맑고 낮 기온은 십오 도 정도예요.", "저녁에는 조금 쌀쌀해질 수 있어요."]


def test_normalize():
    assert normalize(" 잠깐, 확인해 볼게요! ") == "잠깐확인해볼게요"
    assert normalize("OK Google.") == "okgoogle"


def test_same_sentence_is_echo():
    assert is_echo("잠깐 확인해 볼게요.", RECENT)
    assert is_echo("오늘 서울은 맑고 낮 기온은 15도 정도예요", RECENT)


def test_partial_transcript_is_echo():
    assert is_echo("확인해 볼게요", RECENT)
    assert is_echo("저녁에는 조금 쌀쌀해질", RECENT)
    # 받아쓰기 오류가 섞여도
    assert is_echo("저녁에는 좀 쌀쌀해 질 수 있어요", RECENT)


def test_similar_user_speech_passes():
    assert not is_echo("내일 부산 날씨는 어때?", RECENT)
    assert not is_echo("서울 말고 대전 날씨 알려 줘", RECENT)
    assert not is_echo("그만하고 다른 얘기 하자", RECENT)


def test_short_reply_not_echo():
    assert not is_echo("네", RECENT)
    assert not is_echo("네.", ["네, 알겠어요."])
    assert not is_echo("", RECENT)


def test_memory_expires():
    now = [0.0]
    m = EchoMemory(memory_s=20, clock=lambda: now[0])
    m.add("잠깐 확인해 볼게요.")
    assert m.is_echo("확인해 볼게요")
    now[0] = 21.0
    assert not m.is_echo("확인해 볼게요")
    assert m.recent() == []


def test_memory_keeps_from_play_end():
    """보관은 문장 전송 시각이 아니라 추정 재생 종료 시각부터."""
    now = [0.0]
    m = EchoMemory(memory_s=20, clock=lambda: now[0])
    m.add("잠깐 확인해 볼게요.", end=30.0)   # 재생이 30초에 끝날 예정
    now[0] = 45.0
    assert m.is_echo("확인해 볼게요")
    now[0] = 51.0
    assert not m.recent()


def test_global_rule_long_similar_only():
    from server.echo import global_echo_reason
    recent = ["직접 다루는 데 초점이 있어요.", "그런 점이 궁금하시면 더 들어보시겠어요?"]
    # 길고 거의 같은 문장(두 문장에 걸침, 약간 틀림)
    assert global_echo_reason("직접 다루는데 초점이 있어요. 그런 점이 궁금하시면 더 들어보시겠어요", recent)
    # 12자 미만은 전역 규칙에서 제외
    assert global_echo_reason("더 들어보시겠어요", recent) is None
    # 길지만 다른 말
    assert global_echo_reason("아니 그건 됐고 내일 아침 일정 좀 알려 줄래", recent) is None


def test_lenient_numeric_fragments():
    from server.echo import lenient_echo_reason
    recent = ["당직은 총 9개예요.", "6월에 3번(1일, 18일, 25일), 7월에 3번(2일, 14일, 20일)이에요.",
              "2025년 1월부터 올해 12월까지 확인한 결과고, 9월 이후로는 당직이 없어요."]
    assert lenient_echo_reason("2020년", recent)
    assert lenient_echo_reason("3번", recent)
    assert lenient_echo_reason("2025년 1월", recent)
    assert not lenient_echo_reason("1999년", recent)
    assert not lenient_echo_reason("잠깐만", recent)
    assert not lenient_echo_reason("9개", recent[:1] + ["다른 말이에요.", "숫자 없는 문장이에요."])  # 최근 2문장만
    assert lenient_echo_reason("올해 12월 결과가 이후로", recent, bigram=0.5)
    assert not lenient_echo_reason("올해 12월 결과가 이후로", recent, bigram=None)
