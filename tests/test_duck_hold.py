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
        assert h == [{"type": "hold", "generation": gen, "mode": "duck", "gain": config.DUCK_GAIN}]
        assert s.holding and s._duck is not None
        await feed(s, -52, 576, seed=50)          # 덕킹 뒤 8 dB 하강 -> 에코
        assert ws.of("resume") == [{"type": "resume", "generation": gen}]
        assert not ws.of("interrupt") and not s.holding and stt.calls == 0
        s.speaker_task.cancel()
    asyncio.run(go())


def test_duck_no_drop_interrupts():
    async def go():
        s, ws, stt, hermes, gen = await setup_duck()
        await hold_now(s)
        await feed(s, -43, 576, seed=60)          # 줄여도 그대로 -> 사용자 말
        assert ws.of("interrupt") == [{"type": "interrupt", "generation": gen + 1}]
        assert not ws.of("resume") and not s.holding
        # 발화는 계속 수집되다가 끝에서 평소처럼 턴
        stt.text = "잠깐 그만"
        s.endpointer.in_speech = False
        await s._on_vad(VADEvent("end", pcm=b"\0\0" * 16000, speech_ms=900))
        await settle()
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
        assert ws.of("resume") == [{"type": "resume", "generation": gen}] and not ws.of("interrupt")
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


async def feed_until(s, ws, dbs_per_chunk, kind, limit_chunks=10):
    """조각을 보내며 kind 메시지가 나올 때까지 걸린 hold 뒤 시간(s)."""
    at = s._hold_at
    for i in range(limit_chunks):
        s.clock.t += 0.096
        await s._on_audio(chunk_levels(dbs_per_chunk, seed=100 + i))
        if ws.of(kind):
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
        took = await feed_until(s, ws, [-43, -43, -43], "interrupt")
        assert took is not None and took <= 0.31, took
        assert not ws.of("resume")
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
