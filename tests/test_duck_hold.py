"""hold 덕킹 능동 검사(HOLD_MODE=duck)와 음성 스타일 노트 시험(가짜 세션)."""
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_session_hold import FakeEP, FakeHermes, FakeSTT, FakeTTS, FakeWS, settle  # noqa: E402

from server import config  # noqa: E402
from server.session import VoiceSession  # noqa: E402
from server.vad import VADEvent  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def chunk(db: float, ms: int = 96, seed: int = 0) -> bytes:
    n = 16 * ms
    a = np.random.default_rng(seed).standard_normal(n) * 10 ** (db / 20)
    return (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()


async def setup_duck():
    ws, stt, hermes = FakeWS(), FakeSTT(), FakeHermes()
    s = VoiceSession(ws, stt, FakeTTS(), hermes)
    s.clock = Clock()
    s.endpointer = FakeEP()
    s.hold_mode = "duck"
    s.speaker_task = asyncio.create_task(s._speaker())
    await s._on_text(json.dumps({"type": "hello", "resume": None}))
    await s._on_text(json.dumps({"type": "conversation", "on": True}))
    await s._on_text(json.dumps({"type": "text", "text": "서울 날씨 어때?"}))
    await settle()
    gen = s.generation
    await s._on_text(json.dumps({"type": "playback", "state": "speaking", "generation": gen}))
    return s, ws, stt, hermes, gen


async def feed(s, db, ms_total, seed=0):
    """96 ms 조각으로 마이크 입력을 보내며 시계를 진행."""
    for i in range(ms_total // 96):
        s.clock.t += 0.096
        await s._on_audio(chunk(db, seed=seed + i))


async def hold_now(s):
    await feed(s, -44, 480)                       # hold 직전 입력(-44 dBFS)
    s.endpointer.in_speech, s.endpointer.speech_ms = True, 416
    await s._on_vad(VADEvent("start", barge=True))


def test_duck_echo_drop_resumes():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await hold_now(s)
        h = ws.of("hold")
        assert h == [{"type": "hold", "generation": gen, "mode": "duck", "gain": config.DUCK_GAIN, "ramp_ms": config.DUCK_RAMP_DOWN_MS}]
        assert s.holding and s._duck is not None
        await feed(s, -52, 576, seed=50)          # 덕킹 뒤 8 dB 하강 -> 에코
        assert ws.of("resume") == [{"type": "resume", "generation": gen, "ramp_ms": 250}]
        assert not ws.of("interrupt") and not s.holding and stt.calls == 0
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_no_drop_interrupts():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await hold_now(s)
        await feed(s, -43, 576, seed=60)          # 줄여도 그대로 -> 사용자 말 쪽: 바로 끊지 않고 일시정지
        assert pauses(ws) == [{"type": "hold", "generation": gen}]
        assert not ws.of("interrupt") and not ws.of("resume") and s.holding and s._hold_mode == "pause"
        # 말 끝 STT가 진짜 말 -> 그때 끼어들기 + 새 턴
        stt.text = "잠깐 그만"
        s.endpointer.in_speech = False
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=900))
        await settle()
        assert ws.of("interrupt") and not ws.of("resume")
        assert stt.calls == 1 and ws.of("transcript")
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_end_before_test_uses_stt():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await hold_now(s)
        await feed(s, -44, 192, seed=70)          # 검사 완료(480 ms) 전에
        stt.text = "오늘 서울은 맑고 낮 기온은"   # 에코를 받아씀
        s.endpointer.in_speech = False
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=600))
        await settle()
        assert stt.calls == 1
        assert ws.of("resume") == [{"type": "resume", "generation": gen, "ramp_ms": 250}] and not ws.of("interrupt")
        await feed(s, -60, 576, seed=80)          # 늦게 온 입력이 검사를 다시 돌리지 않음
        assert len(ws.of("resume")) == 1 and not ws.of("interrupt")
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_hold_keeps_sending_audio():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        n = len(ws.of("speak_start"))
        await hold_now(s)
        hermes.more = ["저녁에는 조금 쌀쌀해질 수 있어요."]
        hermes.release.set()
        await settle()
        assert len(ws.of("speak_start")) == n + 1   # 덕킹은 재생을 멈추지 않음
        assert s.state == "listening"
        s.speaker_task.cancel()
    asyncio.run(go())


def test_voice_style_note_prepended():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        assert hermes.calls[0].startswith("(음성 대화. 네 이름은 Herbot이야. 3문장 이내로")
        assert hermes.calls[0].endswith("\n서울 날씨 어때?")
        s.style_note = ""
        await s._on_text(json.dumps({"type": "text", "text": "고마워"}))
        await settle()
        assert hermes.calls[-1].endswith("\n고마워") and "음성 대화" not in hermes.calls[-1]
        s.speaker_task.cancel()
    asyncio.run(go())


def chunk_levels(dbs, seed=0):
    """창(32 ms)별 레벨 목록으로 조각."""
    rng = np.random.default_rng(seed)
    a = np.concatenate([rng.standard_normal(512) * 10 ** (d / 20) for d in dbs])
    return (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()


def pauses(ws):
    """덕킹 hold에서 바뀐 일시정지 hold 메시지(mode 없음)."""
    return [m for m in ws.of("hold") if "mode" not in m]


async def feed_until(s, ws, dbs_per_chunk, kind, limit_chunks=10):
    """조각을 보내며 kind 메시지가 나올 때까지 걸린 hold 뒤 시간(s)."""
    at = s._hold_at
    for i in range(limit_chunks):
        s.clock.t += 0.096
        await s._on_audio(chunk_levels(dbs_per_chunk, seed=100 + i))
        if (pauses(ws) if kind == "pause" else ws.of(kind)):
            return s.clock.t - at
    return None


def test_duck_early_resume_within_300ms():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await hold_now(s)
        took = await feed_until(s, ws, [-53, -53, -53], "resume")
        assert took is not None and took <= 0.31, took
        assert not ws.of("interrupt")
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_early_interrupt():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await hold_now(s)
        took = await feed_until(s, ws, [-43, -43, -43], "pause")
        assert took is not None and took <= 0.31, took
        assert not ws.of("resume") and not ws.of("interrupt")
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_ambiguous_falls_back_to_average():
    """창마다 오락가락(-60/-48, pre -44)하면 조기 판정 없이 DUCK_TEST_MS 끝 평균(약 7 dB 하강)으로 에코."""
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await hold_now(s)
        took = await feed_until(s, ws, [-60, -48, -60], "resume")
        assert took is not None and took >= 0.47, took
        assert not ws.of("interrupt")
        s.speaker_task.cancel()
    asyncio.run(go())


# ---------- v22: 재생 판단은 클라이언트 보고 기준 ----------

def test_playing_follows_client_reports_not_send_estimate():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        # 서버는 다 보냈다고 보고(전송 길이 추정 끝이 10초 전), 클라이언트는 뒤처져 계속 재생 보고
        s._play_until = s.clock.t - 10.0
        for i in range(8):
            s.clock.t += 0.25
            await s._on_text(json.dumps({"type": "playback", "state": "speaking", "generation": gen,
                                         "pos_ms": 2000 + i * 250}))
            assert s._playing()
        # 경계 시점의 일반 모드 시작도 재생 중이면 hold
        s.endpointer.in_speech, s.endpointer.speech_ms = True, 200
        await s._on_vad(VADEvent("start", barge=False))
        assert ws.of("hold") and s.holding
        await s._resume()
        # 보고가 끊기면(1.5초) 재생 아님
        s.clock.t += 1.6
        assert not s._playing()
        await s._on_text(json.dumps({"type": "playback", "state": "speaking", "generation": gen, "pos_ms": 4500}))
        assert s._playing()
        # idle 보고면 즉시 아님
        await s._on_text(json.dumps({"type": "playback", "state": "idle", "generation": gen, "pos_ms": 4600}))
        assert not s._playing()
        s.speaker_task.cancel()
    asyncio.run(go())


def test_global_echo_filter_long_after_tail():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await s._on_text(json.dumps({"type": "playback", "state": "idle", "generation": gen, "pos_ms": 1000}))
        s.clock.t += 13.0                 # 꼬리(1.5 s) 한참 밖
        n_tr = len(ws.of("transcript"))
        stt.text = "오늘 서울은 맑고 낮 기온은 십오 도 정도예요"   # 방금 읽은 답 그대로(12자 이상)
        s.endpointer.in_speech = False
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=900))
        await settle()
        assert len(ws.of("transcript")) == n_tr and len(hermes.calls) == 1
        stt.text = "오늘 서울은"           # 12자 미만은 꼬리 밖이면 통과
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=900))
        await settle()
        assert len(ws.of("transcript")) == n_tr + 1
        s.speaker_task.cancel()
    asyncio.run(go())


# ---------- 확실한 에코면 덕킹 생략(DUCK_SKIP_*) ----------
from server.aec_gate import GateResult  # noqa: E402


async def setup_skip():
    s, ws, stt, hermes, gen = await setup_duck()
    await s._on_text(json.dumps({"type": "playback", "state": "speaking", "generation": gen, "pos_ms": 1000}))
    assert s._aec_on and s.timeline.active(s.clock())
    return s, ws, stt, hermes, gen


def fill_recent(s, corr, mic_db, expected_db, n=6, echo=True):
    s._gate_recent = [GateResult(echo=echo, judged=True, corr=corr, mic_db=mic_db, expected_db=expected_db,
                                 rule="상관" if echo else "레벨초과+상관낮음") for _ in range(n)]


async def start_now(s):
    s.endpointer.in_speech, s.endpointer.speech_ms = True, 416
    await s._on_vad(VADEvent("start", barge=True))


def test_sure_echo_skips_duck():
    async def go():
        s, ws, stt, hermes, gen = await setup_skip()
        fill_recent(s, 0.9, -46.0, -45.0)
        await start_now(s)
        assert not ws.of("hold") and not s.holding and s._echo_check
        assert s._duck_skip is not None and s._playing()
        # 말 끝: 대조로 에코 -> 받아쓰기 없이 무시, 재생 그대로
        s.endpointer.in_speech = False
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=600, echo_frac=0.9, echo_judged=12))
        await settle()
        assert stt.calls == 0 and not ws.of("interrupt") and s.generation == gen
        assert s.state == "speaking"
        s.speaker_task.cancel()
    asyncio.run(go())


def test_sure_echo_but_loud_mic_holds():
    async def go():
        s, ws, stt, hermes, gen = await setup_skip()
        fill_recent(s, 0.9, -35.0, -45.0)          # expected + 10 dB
        await start_now(s)
        assert ws.of("hold") and s.holding and s._duck_skip is None
        s.speaker_task.cancel()
    asyncio.run(go())


def test_low_corr_holds():
    async def go():
        s, ws, stt, hermes, gen = await setup_skip()
        fill_recent(s, 0.3, -46.0, -45.0)
        await start_now(s)
        assert ws.of("hold") and s.holding
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_skip_disabled(monkeypatch):
    monkeypatch.setattr(config, "DUCK_SKIP_CORR", 0.0)

    async def go():
        s, ws, stt, hermes, gen = await setup_skip()
        fill_recent(s, 0.95, -46.0, -45.0)
        await start_now(s)
        assert ws.of("hold") and s.holding
        s.speaker_task.cancel()
    asyncio.run(go())


def test_nan_expected_holds():
    async def go():
        s, ws, stt, hermes, gen = await setup_skip()
        fill_recent(s, 0.95, -46.0, float("nan"))
        await start_now(s)
        assert ws.of("hold") and s.holding
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_skip_revoked_by_long_non_echo_speech():
    async def go():
        s, ws, stt, hermes, gen = await setup_skip()
        fill_recent(s, 0.9, -46.0, -45.0)
        await start_now(s)
        assert not s.holding
        # 비에코 창이 이어져도 HOLD_CONFIRM_MS 전에는 그대로
        s._duck_skip["nonecho"] = 6
        await s._on_text(json.dumps({"type": "playback", "state": "speaking", "generation": gen, "pos_ms": 1300}))
        await feed(s, -40, 480, seed=70)
        assert not s.holding
        await s._on_text(json.dumps({"type": "playback", "state": "speaking", "generation": gen, "pos_ms": 1800}))
        await feed(s, -40, 480, seed=80)            # 960 ms 경과 -> hold(덕킹 검사)
        assert ws.of("hold") and s.holding and s._duck_skip is None
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_skip_not_revoked_without_non_echo():
    async def go():
        s, ws, stt, hermes, gen = await setup_skip()
        fill_recent(s, 0.9, -46.0, -45.0)
        await start_now(s)
        s._duck_skip["nonecho"] = 2
        for k in range(3):
            await s._on_text(json.dumps({"type": "playback", "state": "speaking", "generation": gen,
                                         "pos_ms": 1300 + k * 500}))
            await feed(s, -46, 480, seed=90 + k)
        assert not ws.of("hold") and not s.holding
        s.speaker_task.cancel()
    asyncio.run(go())


# ---------- 덕킹 에코 확정 뒤 발화의 나머지(DUCK_ECHO_RISE_*) ----------

async def feed_playing(s, gen, db, ms_total, seed=0):
    """재생 보고(pos_ms)를 곁들여 입력을 보낸다(재생 중 유지)."""
    for i in range(ms_total // 96):
        s.clock.t += 0.096
        if i % 3 == 0:
            await s._on_text(json.dumps({"type": "playback", "state": "speaking", "generation": gen,
                                         "pos_ms": 1000 + (s.clock.t - 100) * 1000}))
        await s._on_audio(chunk(db, seed=seed + i))


async def duck_echo_resumed(s, ws, gen):
    await hold_now(s)
    await feed_playing(s, gen, -53, 288, seed=200)      # 덕킹 뒤 9 dB 하강 -> 에코, 재생 복귀
    assert ws.of("resume") and not ws.of("interrupt") and not s.holding
    assert s._duck_echo is not None and s._duck_echo["gen"] == gen


def test_duck_echo_rest_dropped_without_stt():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await duck_echo_resumed(s, ws, gen)
        await feed_playing(s, gen, -44, 384, seed=210)  # 복귀 뒤 에코가 pre 레벨로 이어짐
        stt.text = "11월 9일에 치고"
        s.endpointer.in_speech = False
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=1216))
        await settle()
        assert stt.calls == 0 and not ws.of("transcript") and s.generation == gen
        assert len(hermes.calls) == 1
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_echo_rest_with_rise_goes_to_stt():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await duck_echo_resumed(s, ws, gen)
        await feed_playing(s, gen, -30, 384, seed=220)  # pre(-44)보다 14 dB 높게 384 ms -> 사용자가 뒤이어 말함
        stt.text = "잠깐 그거 말고 내일 일정 알려 줘"
        s.endpointer.in_speech = False
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=1500))
        await settle()
        assert stt.calls == 1 and ws.of("transcript")[-1]["text"] == stt.text
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_echo_rest_with_rise_still_text_filtered():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await duck_echo_resumed(s, ws, gen)
        await feed_playing(s, gen, -30, 384, seed=230)
        stt.text = "오늘 서울은 맑고 낮 기온은"      # 받아쓴 게 방금 읽은 답 -> 텍스트 에코 필터
        s.endpointer.in_speech = False
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=1500))
        await settle()
        assert stt.calls == 1 and not ws.of("transcript")
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_echo_long_rest_rechecks():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await duck_echo_resumed(s, ws, gen)
        assert len(ws.of("hold")) == 1
        await feed_playing(s, gen, -44, 960, seed=240)  # 복귀 뒤 900 ms 넘게 이어짐 -> 덕킹 재검사
        assert len(ws.of("hold")) == 2 and s.holding
        await feed_playing(s, gen, -43, 384, seed=250)  # 줄여도 그대로 -> 사용자 말 쪽: 일시정지
        assert pauses(ws) and s.holding and s._hold_mode == "pause"
        s.endpointer.speech_ms += 1000                   # 멈춘 뒤에도 말이 900 ms 넘게 이어짐 -> 끼어들기 확정
        await feed(s, -30, 192, seed=255)
        assert ws.of("interrupt") and not s.holding and s._duck_echo is None
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_echo_recheck_echo_keeps_mark_once():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await duck_echo_resumed(s, ws, gen)
        await feed_playing(s, gen, -44, 960, seed=260)
        assert len(ws.of("hold")) == 2
        await feed_playing(s, gen, -56, 288, seed=270)  # 다시 에코 -> 복귀, 표시 유지
        assert len(ws.of("resume")) == 2 and not ws.of("interrupt")
        assert s._duck_echo is not None and s._duck_echo["rechecked"]
        await feed_playing(s, gen, -44, 1152, seed=280)  # 재검사는 한 번만
        assert len(ws.of("hold")) == 2
        s.endpointer.in_speech = False
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=2400))
        await settle()
        assert stt.calls == 0 and not ws.of("transcript")
        s.speaker_task.cancel()
    asyncio.run(go())


# ---------- 짧은 에코 조각 완화 필터·옛 재생 보고 ----------

def test_short_numeric_fragment_after_stale_playback():
    """실기 17:46: 에코 발화가 새 턴을 열어 generation이 바뀐 뒤에도 클라이언트는 옛 답을 계속 재생했다.
    옛 generation 재생 보고로 에코 검사를 켜고, 짧은 숫자 조각('2020년')은 완화 필터로 버린다."""
    async def go():
        s, ws, stt, hermes, gen = await setup_skip()
        s.echo.add("6월에 3번(1일, 18일, 25일), 7월에 3번(2일, 14일, 20일)이에요.")
        # interrupt 없이 generation만 바뀐 상황(예: 판정 전 경로로 열린 턴이 버려짐)
        s.generation += 1
        n_tr = len(ws.of("transcript"))
        # 클라이언트는 옛 답(gen)을 계속 재생 중(서버는 옛 generation 보고라 재생 아님으로 봄)
        for i in range(8):
            s.clock.t += 0.25
            await s._on_text(json.dumps({"type": "playback", "state": "speaking", "generation": gen,
                                         "pos_ms": 5000 + i * 250}))
        assert not s._playing() and s._stale_playing()
        s.endpointer.in_speech = True
        await s._on_vad(VADEvent("start"))
        assert not ws.of("hold")
        stt.text = "2020년"
        s.endpointer.in_speech = False
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=800))
        await settle()
        assert len(ws.of("transcript")) == n_tr
        # 옛 재생이 끝나면(idle) 꼬리 가드, 그 뒤 한참 지나면 숫자도 통과
        await s._on_text(json.dumps({"type": "playback", "state": "idle", "generation": gen, "pos_ms": 9000}))
        assert not s._stale_playing() and s._tail_active()
        s.clock.t += 5.0
        await s._on_vad(VADEvent("start"))
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=800))
        await settle()
        assert ws.of("transcript")[-1]["text"] == "2020년"
        s.speaker_task.cancel()
    asyncio.run(go())


def test_short_fragment_lenient_bigram_needs_gate_fraction():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        s.echo.add("2025년 1월부터 올해 12월까지 확인한 결과고, 9월 이후로는 당직이 없어요.")
        n_tr = len(ws.of("transcript"))
        stt.text = "올해 12월 결과는 이후로 없어"   # bigram 0.67: 평소(0.7) 못 잡음
        s.endpointer.in_speech = False
        s._echo_check = True
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=900, echo_frac=0.4, echo_judged=10))
        await settle()
        assert len(ws.of("transcript")) == n_tr
        # 게이트 에코 비율이 낮으면 완화 bigram은 안 씀
        s._echo_check = True
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=900, echo_frac=0.1, echo_judged=10))
        await settle()
        assert len(ws.of("transcript")) == n_tr + 1
        # 길면(> SHORT_ECHO_MS) 완화 필터 대상 아님
        stt.text = "1일 9월"                        # 짧으면 숫자 일치(완화)로 버려질 전사
        s._echo_check = True
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=900, echo_frac=0.0, echo_judged=10))
        await settle()
        assert len(ws.of("transcript")) == n_tr + 1
        s._echo_check = True
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=1500, echo_frac=0.4, echo_judged=10))
        await settle()
        assert ws.of("transcript")[-1]["text"] == "1일 9월"
        s.speaker_task.cancel()
    asyncio.run(go())


# ---------- 재생 중 끝난 에코 검사 대상 발화: STT 판정 전에는 generation·큐를 건드리지 않음 ----------

def test_echo_fragment_during_playback_keeps_answer():
    async def go():
        s, ws, stt, hermes, gen = await setup_skip()
        s.echo.add("2025년 1월부터 올해 12월까지 확인한 결과예요.")
        n_tr, n_ss = len(ws.of("transcript")), len(ws.of("speak_start"))
        s.queue.put_nowait((gen, "다음 문장이에요."))   # 아직 읽을 문장
        stt.text = "2020년"
        s.endpointer.in_speech = False
        s._echo_check = True
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=800))
        assert s.generation == gen and s._echo_task is not None
        await settle()
        assert s.generation == gen and s._echo_task is None
        assert stt.calls == 1 and len(ws.of("transcript")) == n_tr and not ws.of("interrupt")
        assert len(hermes.calls) == 1
        # 큐를 비우지 않았으니 다음 문장을 계속 읽는다
        assert any(m["text"] == "다음 문장이에요." and m["generation"] == gen for m in ws.of("speak_start")[n_ss:])
        s.speaker_task.cancel()
    asyncio.run(go())


def test_real_speech_during_playback_interrupts_then_turn():
    async def go():
        s, ws, stt, hermes, gen = await setup_skip()
        stt.text = "잠깐만 내일 일정은 어때"
        s.endpointer.in_speech = False
        s._echo_check = True
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=1400))
        assert s.generation == gen                     # 판정 전에는 그대로
        await settle()
        assert ws.of("interrupt") == [{"type": "interrupt", "generation": gen + 1}]
        assert ws.of("transcript")[-1]["text"] == stt.text
        assert s.generation == gen + 2 and len(hermes.calls) == 2 and hermes.calls[-1].endswith(stt.text)
        s.speaker_task.cancel()
    asyncio.run(go())


def test_speech_resumes_while_echo_judging_carries():
    async def go():
        s, ws, stt, hermes, gen = await setup_skip()
        stt.text = "오늘 서울은"
        s.endpointer.in_speech = False
        s._echo_check = True
        await s._on_vad(VADEvent("end", pcm=b"\1\0" * 8000, speech_ms=500))
        assert s._echo_task is not None
        s.endpointer.in_speech = True
        await s._on_vad(VADEvent("start", barge=True))   # 판정 중 말이 이어짐
        assert s._echo_task is None and len(s._carry) == 16000 and s._echo_check
        s.speaker_task.cancel()
    asyncio.run(go())


# ---------- 덕킹 검사: 디지털 무음 창 제외와 에코 예상 레벨 조건(2026-11-03 실기 18:21) ----------

async def hold_with(s, pre_dbs, expected_db=float("nan")):
    """hold 직전 창 레벨(조각당 3창)을 넣고 게이트 expected_db를 정한 뒤 hold."""
    for i in range(5):
        s.clock.t += 0.096
        await s._on_audio(chunk_levels(pre_dbs, seed=200 + i))
    s.gate.last = GateResult(expected_db=expected_db)
    s.endpointer.in_speech, s.endpointer.speech_ms = True, 416
    await s._on_vad(VADEvent("start", barge=True))


def test_duck_silent_pre_resumes_with_echo_check():
    """pre 구간이 전부 디지털 무음(-180 dBFS)이면 끼어들기 확정 없이 재생 복귀하고 말 끝 STT 필터로."""
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await hold_with(s, [-180, -180, -180], expected_db=-33.1)
        assert s.holding and s._duck is not None and s._duck["pre_db"] is None
        await feed_until(s, ws, [-38, -38, -38], "resume")
        assert ws.of("resume") == [{"type": "resume", "generation": gen, "ramp_ms": 250}]
        assert not ws.of("interrupt") and not s.holding and s._echo_check
        await feed(s, -38, 576, seed=300)          # 이어지는 에코가 끼어들기로 바뀌지 않음
        assert not ws.of("interrupt")
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_no_drop_below_expected_is_echo():
    """pre 유효, 하강은 없지만 post가 expected + DUCK_USER_MARGIN_DB 이하면 사용자 말로 확정하지 않는다."""
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await hold_with(s, [-38, -38, -38], expected_db=-33.1)
        await feed(s, -38, 576, seed=310)
        assert ws.of("resume") == [{"type": "resume", "generation": gen, "ramp_ms": 250}]
        assert not ws.of("interrupt") and s._echo_check
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_loud_user_over_expected_interrupts():
    """진짜 사용자 말(pre -16 -> post -9, expected -37)은 그대로 끼어들기 확정."""
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await hold_with(s, [-16, -16, -16], expected_db=-37.0)
        await feed(s, -9, 576, seed=320)
        assert pauses(ws) == [{"type": "hold", "generation": gen}]
        assert not ws.of("resume")
        s.endpointer.speech_ms += 1000             # 멈춘 뒤에도 계속 말함 -> 끼어들기 확정
        await feed(s, -9, 192, seed=325)
        assert ws.of("interrupt") == [{"type": "interrupt", "generation": gen + 1}]
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_post_silent_windows_excluded():
    """post에 무음 창이 섞여도 유효 창 평균/연속으로 판정: 무음 빼면 하강 없음 -> 사용자 말."""
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await hold_with(s, [-44, -44, -44])
        took = await feed_until(s, ws, [-43, -180, -43], "pause")
        assert took is not None and not ws.of("resume")
        s.speaker_task.cancel()
    asyncio.run(go())


def test_mic_db_between_skips_silence():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        t0 = s.clock.t
        for i in range(4):
            s.clock.t += 0.096
            await s._on_audio(chunk_levels([-30, -180, -30], seed=330 + i))
        v = s._mic_db_between(t0, s.clock.t)
        assert v is not None and abs(v + 30) < 1.0
        assert s._mic_db_between(t0, s.clock.t, min_windows=9) is None   # 유효 8창
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_post_silent_extends_once_then_judges():
    """post가 무음뿐이면 DUCK_TEST_MS만큼 한 번 연장, 그 사이 유효 입력이 오면 그걸로 판정."""
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await hold_with(s, [-44, -44, -44])
        await feed(s, -180, 480, seed=340)         # 원래 검사 구간(180~480 ms)은 무음
        assert s.holding and s._duck is not None and s._duck.get("ext")
        await feed(s, -53, 384, seed=350)          # 연장 구간에 에코 하강
        assert ws.of("resume") and not ws.of("interrupt")
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_post_silent_after_extension_resumes():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await hold_with(s, [-44, -44, -44])
        await feed(s, -180, 960, seed=360)
        assert ws.of("resume") and not ws.of("interrupt") and s._echo_check and not s.holding
        s.speaker_task.cancel()
    asyncio.run(go())


# ---------- 독음·퍼지 에코 규칙(실기 18:33)·게이트 사전값 ----------

def test_hold_fuzzy_echo_resumes():
    """hold 중 끝난 발화를 받아썼더니 답의 독음을 잘못 받아쓴 것('11월이라' -> '십일올이라나'): 퍼지 규칙으로 에코."""
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        s.echo.add("쌀쌀한 11월이라 뜨끈한 국물 어떠세요?", "쌀쌀한 11월이라 뜨끈한 국물 어떠세요?")
        await hold_now(s)
        await feed(s, -44, 192, seed=70)
        stt.text = "쌀쌀한 십일올이라나"
        s.endpointer.in_speech = False
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=1536))
        await settle()
        assert stt.calls == 1
        assert ws.of("resume") == [{"type": "resume", "generation": gen, "ramp_ms": 250}] and not ws.of("interrupt")
        s.speaker_task.cancel()
    asyncio.run(go())


def test_fuzzy_not_applied_without_playback():
    """재생이 전혀 없을 때의 일반 발화는 퍼지 규칙을 쓰지 않는다(사용자가 답을 따라 말해도 통과)."""
    async def go():
        ws, stt, hermes = FakeWS(), FakeSTT(), FakeHermes()
        s = VoiceSession(ws, stt, FakeTTS(), hermes)
        s.clock = Clock()
        s.endpointer = FakeEP()
        s.speaker_task = asyncio.create_task(s._speaker())
        await s._on_text(json.dumps({"type": "hello", "resume": None}))
        await s._on_text(json.dumps({"type": "conversation", "on": True}))
        s.echo.add("쌀쌀한 11월이라 뜨끈한 국물 어떠세요?", end=s.clock())
        s.clock.t += 10.0                               # 꼬리 가드도 지남
        assert s._echo_filter("쌀쌀한 십일올이라나", False) is None
        assert s._echo_filter("쌀쌀한 십일올이라나", True)[0].startswith("에코 무시(퍼지")
        s.endpointer.in_speech = True
        await s._on_vad(VADEvent("start"))
        stt.text = "쌀쌀한 십일올이라나"
        s.endpointer.in_speech = False
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=900))
        await settle()
        assert ws.of("transcript") and ws.of("transcript")[-1]["text"] == "쌀쌀한 십일올이라나"
        s.speaker_task.cancel()
    asyncio.run(go())


def test_aec_prior_loaded_on_hello_and_saved(tmp_path):
    from server.aec_gate import AecStateStore, PRIOR_MIN_LEARNED
    async def go():
        store = AecStateStore(tmp_path / "aec_state.json")
        vp = {"screen": [393, 852, 852], "dpr": 3}
        store.save(AecStateStore.make_key("pwa", vp, True, False), {"lag_ms": 180.0, "gain_db": -26.0, "learned": 80})
        store.save(AecStateStore.make_key("pwa", vp, False, False), {"lag_ms": 120.0, "gain_db": -15.0, "learned": 80})
        ws, stt, hermes = FakeWS(), FakeSTT(), FakeHermes()
        s = VoiceSession(ws, stt, FakeTTS(), hermes)
        s.aec_store = AecStateStore(tmp_path / "aec_state.json", clock=lambda: s.clock())
        s.clock = Clock()
        await s._on_text(json.dumps({"type": "hello", "resume": None, "client": "pwa", "vp": vp}))
        assert s.gate.lag_est_ms == 180.0 and s.gate.gain_db == -26.0 and s.gate.learned == 0
        await s._on_text(json.dumps({"type": "settings", "aec": True, "half_duplex": False}))   # 같은 키: 그대로
        assert s.gate.lag_est_ms == 180.0
        await s._on_text(json.dumps({"type": "settings", "aec": False, "half_duplex": False}))  # AEC 끔: 다른 키
        assert s.gate.lag_est_ms == 120.0 and s.gate.gain_db == -15.0
        await s._on_text(json.dumps({"type": "settings", "aec": False, "half_duplex": True}))   # 저장값 없음: 기본값
        assert s.gate.prior is None and s.gate.lag_est_ms != 120.0
        # 학습이 충분해지면 저장(1분에 한 번 이하)
        s.gate.lag_est_ms, s.gate.gain_db = 150.0, -20.0
        s.gate.learned = PRIOR_MIN_LEARNED - 1
        s._maybe_save_aec()
        key = AecStateStore.make_key("pwa", vp, False, True)
        assert s.aec_store.load(key) is None
        s.gate.learned = PRIOR_MIN_LEARNED
        s._maybe_save_aec()
        assert s.aec_store.load(key)["lag_ms"] == 150.0
        s.gate.lag_est_ms, s.gate.learned = 160.0, PRIOR_MIN_LEARNED + 5
        s.clock.t += 30.0
        s._maybe_save_aec()
        assert s.aec_store.load(key)["lag_ms"] == 150.0      # throttle
        s.clock.t += 31.0
        s._maybe_save_aec()
        assert s.aec_store.load(key)["lag_ms"] == 160.0
    asyncio.run(go())


def test_duck_user_verdict_echo_after_pause_resumes():
    """덕킹 검사가 '사용자 말 쪽'이어도 바로 끊지 않는다: 일시정지 뒤 말 끝 STT가 에코면 보관분을 이어 재생."""
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        s.echo.add("잠깐 확인해 볼게요.")
        await hold_now(s)
        await feed(s, -43, 576, seed=360)
        assert pauses(ws) and s._hold_mode == "pause"
        stt.text = "잠깐 확인해 볼게요."
        s.endpointer.in_speech = False
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=700))
        await settle()
        assert ws.of("resume") and not ws.of("interrupt") and not s.holding
        s.speaker_task.cancel()
    asyncio.run(go())
