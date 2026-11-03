"""Qwen3-TTS 스트리밍 엔진(server/tts_qwen.py)과 세션 스트리밍 경로 시험. 실제 모델 대신 tests/fake_qwen_worker.py."""
import asyncio
import os
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_session_hold import FakeHermes, FakeSTT, FakeWS  # noqa: E402

from server.session import VoiceSession  # noqa: E402
from server.tts_qwen import (  # noqa: E402
    QwenPostProcessor, QwenTTSEngine, StreamResampler, TempoStretcher, TTSRouter, soft_limit)

FAKE = str(Path(__file__).resolve().parent / "fake_qwen_worker.py")


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


def make_engine(tmp_path, **env_extra):
    log = tmp_path / "fake.log"
    env = dict(os.environ, FAKE_QWEN_LOG=str(log), **env_extra)
    return QwenTTSEngine(cmd=[sys.executable, FAKE], env=env, gain_db=0.0, ready_timeout_s=10, tempo=1.0), log


def cancels(log: Path) -> list[str]:
    return [ln for ln in log.read_text().splitlines() if ln.startswith("cancel")] if log.exists() else []


class FakeSupertonic:
    loaded = True
    provider = "cpu"
    load_ms = 1
    error = None

    def __init__(self):
        self.calls = []

    async def load(self):
        pass

    async def synth(self, text, is_current=lambda: True):
        self.calls.append(text)
        return (np.full(22050, 1000, dtype=np.int16), 3) if is_current() else None


# ---------------------------------------------------------------- 신호 처리
def test_stream_resampler_matches_whole_and_is_continuous():
    sr = 24000
    t = np.arange(int(sr * 1.3)) / sr
    x = 0.5 * np.sin(2 * np.pi * 220 * t)
    whole = resample_poly(x, 147, 80)
    rng = np.random.default_rng(1)
    cuts = np.sort(rng.choice(np.arange(1, x.size), 25, replace=False))
    rs = StreamResampler(sr, 44100)
    out = [rs.push(c) for c in np.split(x, cuts)] + [rs.flush()]
    y = np.concatenate(out)
    assert y.size == whole.size
    assert np.max(np.abs(y - whole)) < 1e-6
    # 이어 붙인 곳에서 튐 없음: 표본 간 차이가 사인파 최대 기울기를 넘지 않음
    max_step = 0.5 * 2 * np.pi * 220 / 44100
    assert np.max(np.abs(np.diff(y[200:-200]))) < max_step * 1.05


def test_post_processor_trims_lead_and_limits_peak():
    sr = 24000
    sil = np.zeros(int(sr * 0.5), dtype=np.float32)
    tone = (0.99 * np.sin(2 * np.pi * 300 * np.arange(sr) / sr)).astype(np.float32)
    x = np.concatenate([sil, tone, sil])
    pp = QwenPostProcessor(sr, gain_db=6.0)
    parts = [pp.push(c) for c in np.array_split(x, 13)] + [pp.flush()]
    y = np.concatenate(parts).astype(np.float32) / 32767
    lead = np.flatnonzero(np.abs(y) > 0.01)[0] / 44100
    assert 0.03 <= lead <= 0.1, lead                 # 앞 무음은 약 60 ms만
    assert y.size / 44100 < 1.0 + 0.1 + 0.15 + 0.05  # 뒤 무음도 120 ms 정도만
    assert np.max(np.abs(y)) <= 10 ** (-1 / 20) + 1e-3   # -1 dBFS 넘지 않음
    assert np.max(np.abs(y)) > 0.85                  # 하드 클립이 아니라 눌렀을 뿐


def _stretch(x, parts, tempo=1.2):
    ts = TempoStretcher(24000, tempo)
    chunks = np.array_split(x, parts) if parts > 1 else [x]
    return np.concatenate([ts.push(c) for c in chunks] + [ts.flush()])


def test_tempo_stream_matches_whole_sine_and_chirp():
    sr = 24000
    t = np.arange(int(sr * 2.0)) / sr
    for x, step in ((0.5 * np.sin(2 * np.pi * 300 * t), 0.5 * 2 * np.pi * 300 / sr),
                    (0.5 * np.sin(2 * np.pi * (150 + 200 * t) * t), 0.5 * 2 * np.pi * 950 / sr)):
        x = x.astype(np.float32)
        whole = _stretch(x, 1)
        for parts in (7, 23):
            y = _stretch(x, parts)
            assert abs(y.size - whole.size) <= 0.01 * whole.size
            assert abs(y.size / x.size - 1 / 1.2) < 0.03
            # 청크 경계 불연속 없음: 인접 표본 차가 신호 최대 기울기 수준(여유 1.5배)을 넘지 않음
            assert np.max(np.abs(np.diff(y))) < step * 1.5


def test_tempo_one_is_bypass():
    x = np.random.default_rng(0).standard_normal(5000).astype(np.float32)
    assert np.array_equal(_stretch(x, 5, tempo=1.0), x)


def test_soft_limit_monotonic_and_bounded():
    x = np.linspace(-3, 3, 10001)
    y = soft_limit(x)
    assert np.all(np.diff(y) >= 0)
    assert np.max(np.abs(y)) <= 10 ** (-1 / 20)
    small = np.abs(x) < 0.5
    assert np.allclose(y[small], x[small])


# ---------------------------------------------------------------- 엔진(가짜 워커)
def test_engine_streams_three_chunks(tmp_path):
    async def main():
        eng, _ = make_engine(tmp_path)
        await eng.start()
        assert eng.ready
        chunks = [c async for c in eng.synth_stream("안녕하세요")]
        await eng.aclose()
        return chunks
    chunks = run(main())
    assert len(chunks) >= 3
    assert all(c.dtype == np.int16 for c in chunks)
    total = sum(c.size for c in chunks)
    assert abs(total - int(0.6 * 44100)) < 44100 * 0.1


def test_engine_cancel_on_close_and_not_current(tmp_path):
    async def main():
        eng, log = make_engine(tmp_path)
        await eng.start()
        agen = eng.synth_stream("LONG 문장")
        await agen.__anext__()
        await agen.aclose()                              # 소비자가 그만둠 -> cancel
        flag = {"cur": True}
        agen2 = eng.synth_stream("LONG 두 번째", is_current=lambda: flag["cur"])
        got = [await agen2.__anext__()]
        flag["cur"] = False                              # generation 바뀜 -> cancel
        got += [c async for c in agen2]
        await asyncio.sleep(0.3)
        # 다음 요청도 정상
        ok = [c async for c in eng.synth_stream("짧은 문장")]
        await eng.aclose()
        return log, ok
    log, ok = run(main())
    assert cancels(log) == ["cancel 1", "cancel 2"]
    assert len(ok) >= 3


def test_engine_error_and_restart(tmp_path):
    async def main():
        eng, _ = make_engine(tmp_path)
        await eng.start()
        err = None
        try:
            async for _ in eng.synth_stream("ERR"):
                pass
        except Exception as e:  # noqa: BLE001
            err = e
        try:
            async for _ in eng.synth_stream("DIE"):
                pass
        except Exception:  # noqa: BLE001
            pass
        for _ in range(100):
            await asyncio.sleep(0.05)
            if eng.ready:
                break
        restarted = eng.ready and len(eng._restarts) == 1
        await eng.aclose()
        return err, restarted
    err, restarted = run(main())
    assert err is not None and "fake error" in str(err)
    assert restarted


# ---------------------------------------------------------------- 라우터 폴백
def test_router_fallbacks(tmp_path):
    async def main():
        eng, _ = make_engine(tmp_path)
        st = FakeSupertonic()
        r = TTSRouter(st, eng, engine="qwen", first_chunk_timeout_s=0.5)
        a = [c async for c in r.synth_stream("SLOW")]       # 첫 청크 시간 초과
        b = [c async for c in r.synth_stream("오류 ERR")]   # 첫 청크 전 오류
        await r.load()
        c = [c async for c in r.synth_stream("정상")]
        d = [c async for c in r.synth_stream("SLOW 늦음")]
        e = [c async for c in r.synth_stream("ERR")]
        await r.aclose()
        return r, st, a, b, c, d, e
    r, st, a, b, c, d, e = run(main())
    # 준비 전 두 문장은 Supertonic, 준비 뒤 정상은 Qwen, SLOW·ERR은 다시 Supertonic
    assert len(a) == 1 and len(b) == 1 and len(d) == 1 and len(e) == 1
    assert len(c) >= 3
    assert st.calls == ["SLOW", "오류 ERR", "SLOW 늦음", "ERR"]
    assert r.fallbacks == 4 and r.recent_fallbacks() == 4


def test_router_supertonic_engine_ignores_qwen():
    async def main():
        st = FakeSupertonic()
        r = TTSRouter(st, None, engine="qwen")   # qwen 엔진 객체가 없으면 supertonic으로
        out = [c async for c in r.synth_stream("문장")]
        return r, out
    r, out = run(main())
    assert r.engine == "supertonic" and len(out) == 1 and r.fallbacks == 0


class MissingSupertonic(FakeSupertonic):
    """모델 폴더가 없어 로드 실패한 Supertonic(server/tts.py TTSEngine.load_sync와 같은 상태)."""
    loaded = False
    error = "Supertonic 모델 폴더 없음: /nonexistent"

    async def synth(self, text, is_current=lambda: True):
        self.calls.append(text)
        raise AssertionError("TTS not loaded")


def test_router_without_supertonic_waits_for_qwen(tmp_path):
    async def main():
        eng, _ = make_engine(tmp_path)
        st = MissingSupertonic()
        r = TTSRouter(st, eng, engine="qwen", first_chunk_timeout_s=0.3, first_chunk_max_s=0.3)
        await r.load()
        loaded = r.loaded
        slow = [c async for c in r.synth_stream("SLOW 느린 문장", turn_id=1)]   # 첫 청크 2초 > 한도 0.3초
        err = [c async for c in r.synth_stream("ERR", turn_id=1)]              # 폴백 불가 -> 건너뜀
        ok = [c async for c in r.synth_stream("정상", turn_id=1)]               # 같은 답이어도 Qwen 계속
        await r.aclose()
        return r, st, loaded, slow, err, ok
    r, st, loaded, slow, err, ok = run(main())
    assert loaded                       # Supertonic 없어도 Qwen ready면 /health ok
    assert len(slow) >= 3 and r.last_engine == "qwen"
    assert err == [] and len(ok) >= 3
    assert st.calls == [] and r.fallbacks == 0


def test_tts_engine_missing_model_dir_does_not_raise(tmp_path):
    from server.tts import TTSEngine
    async def main():
        e = TTSEngine(model_dir=str(tmp_path / "없음"))
        await e.load()
        return e
    e = run(main())
    assert not e.loaded and "모델 폴더 없음" in e.error


# ---------------------------------------------------------------- 세션 스트리밍 경로
async def make_session(tmp_path, timeout=2.5):
    eng, log = make_engine(tmp_path)
    await eng.start()
    st = FakeSupertonic()
    router = TTSRouter(st, eng, engine="qwen", first_chunk_timeout_s=timeout)
    ws = FakeWS()
    s = VoiceSession(ws, FakeSTT(), router, FakeHermes())
    s.style_note = ""
    s.speaker_task = asyncio.create_task(s._speaker())
    return s, ws, router, st, log


def audio_samples(ws, gen):
    return sum((len(b) - 4) // 2 for b in ws.audio if struct.unpack("<I", b[:4])[0] == gen)


def test_speaker_streams_progressively(tmp_path):
    async def main():
        s, ws, router, st, _ = await make_session(tmp_path)
        s.generation = gen = 1
        s.queue.put_nowait((gen, "첫 문장이에요."))
        # 첫 청크가 오자마자 speak_start와 오디오가 나가야 함(문장 전체를 기다리지 않음)
        seen = []
        for _ in range(200):
            await asyncio.sleep(0.005)
            seen.append(len(ws.audio))
            if s.aec_ref.samples and not s._speaking_now:
                break
        n = audio_samples(ws, gen)
        ref_ms = s.aec_ref.duration_ms
        sched = list(s._sched)
        types = ws.types()
        s.speaker_task.cancel()
        await router.aclose()
        return n, ref_ms, sched, types, seen, st
    n, ref_ms, sched, types, seen, st = run(main())
    assert types.index("speak_start") == 0 or "speak_start" in types
    assert st.calls == []                                 # 폴백 없음
    assert len([x for x in seen if x > 0]) > 0
    assert len(set(seen)) >= 3                            # 오디오 프레임이 여러 번에 걸쳐 늘어남
    assert abs(ref_ms - n / 44.1) < 5                     # 참조 버퍼 길이 = 보낸 오디오 길이
    assert len(sched) == 1 and abs(sched[0][1] - n / 44100) < 1e-6


def test_speaker_interrupt_cancels_worker(tmp_path):
    async def main():
        s, ws, router, st, log = await make_session(tmp_path)
        s.generation = gen = 1
        s.queue.put_nowait((gen, "길게 이어지는 문장이에요."))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if ws.audio:
                break
        await s.interrupt(announce=False, notify=True)
        await asyncio.sleep(0.4)
        sent_after = audio_samples(ws, gen)
        await asyncio.sleep(0.3)
        same = audio_samples(ws, gen) == sent_after
        s.speaker_task.cancel()
        await router.aclose()
        return log, same, sent_after
    log, same, sent = run(main())
    assert cancels(log) == ["cancel 1"]
    assert same and sent < 40 * 0.2 * 44100               # 끝까지 보내지 않음


def test_speaker_falls_back_to_supertonic(tmp_path):
    async def main():
        s, ws, router, st, _ = await make_session(tmp_path, timeout=0.3)
        s.generation = gen = 1
        s.queue.put_nowait((gen, "느림 문장"))
        for _ in range(300):
            await asyncio.sleep(0.01)
            if ws.audio:
                break
        await asyncio.sleep(0.05)
        n = audio_samples(ws, gen)
        s.speaker_task.cancel()
        await router.aclose()
        return st, router, n, ws.types()
    st, router, n, types = run(main())
    assert st.calls == ["느림 문장"] and router.fallbacks == 1
    assert n == 22050 and "speak_start" in types


def test_speaker_pause_hold_waits_duck_continues(tmp_path):
    async def main():
        s, ws, router, st, _ = await make_session(tmp_path)
        s.generation = gen = 1
        s.holding, s._hold_mode, s._hold_gen = True, "pause", gen
        s._hold_event.clear()
        s.queue.put_nowait((gen, "멈춤 시험 문장"))
        await asyncio.sleep(0.4)
        paused = len(ws.audio)
        s._clear_hold()
        await asyncio.sleep(0.4)
        resumed = audio_samples(ws, gen)
        # duck: hold 중에도 계속 보냄
        s.holding, s._hold_mode = True, "duck"
        s.queue.put_nowait((gen, "덕킹 시험 문장"))
        await asyncio.sleep(0.4)
        ducked = audio_samples(ws, gen) - resumed
        s.speaker_task.cancel()
        await router.aclose()
        return paused, resumed, ducked
    paused, resumed, ducked = run(main())
    assert paused == 0 and resumed > 0 and ducked > 0


# ---------------------------------------------------------------- 재생 여유 대기·답 단위 목소리 고정
def requests(log: Path) -> list[str]:
    return [ln.split(" ", 2)[2] for ln in log.read_text().splitlines() if ln.startswith("req")] if log.exists() else []


def test_router_first_chunk_limit():
    r = TTSRouter(FakeSupertonic(), None, first_chunk_timeout_s=2.5, lead_margin_s=0.5, first_chunk_max_s=12)
    assert r.first_chunk_limit(0) == 2.5            # 첫 문장(재생 중인 것 없음)
    assert r.first_chunk_limit(2.0) == 2.5          # 여유가 기본보다 짧으면 기본
    assert r.first_chunk_limit(5.0) == 4.5          # 남은 재생 - 여유
    assert r.first_chunk_limit(60.0) == 12          # 상한


def test_router_waits_for_playback_lead(tmp_path):
    # 축척: 기본 0.5 s, 가짜 SLOW 첫 청크 2 s. lead 3 s면 한도 2.5 s라 폴백 안 함(실값으로 lead 5 s·첫 청크 4 s 상당)
    async def main():
        eng, _ = make_engine(tmp_path)
        st = FakeSupertonic()
        r = TTSRouter(st, eng, engine="qwen", first_chunk_timeout_s=0.5, lead_margin_s=0.5)
        await r.load()
        out = [c async for c in r.synth_stream("SLOW 여유", lead_s=3.0, turn_id=1)]
        await r.aclose()
        return r, st, out
    r, st, out = run(main())
    assert st.calls == [] and r.fallbacks == 0 and len(out) >= 3 and r.last_engine == "qwen"


def test_router_no_lead_falls_back(tmp_path):
    async def main():
        eng, _ = make_engine(tmp_path)
        st = FakeSupertonic()
        r = TTSRouter(st, eng, engine="qwen", first_chunk_timeout_s=0.5, lead_margin_s=0.5)
        await r.load()
        out = [c async for c in r.synth_stream("SLOW 첫 문장", lead_s=0.0, turn_id=1)]
        await r.aclose()
        return r, st, out
    r, st, out = run(main())
    assert st.calls == ["SLOW 첫 문장"] and r.fallbacks == 1 and len(out) == 1


def test_router_keeps_supertonic_within_turn(tmp_path):
    async def main():
        eng, log = make_engine(tmp_path)
        st = FakeSupertonic()
        r = TTSRouter(st, eng, engine="qwen", first_chunk_timeout_s=0.5, lead_margin_s=0.5)
        await r.load()
        a = [c async for c in r.synth_stream("SLOW 하나", turn_id=7)]          # 폴백
        b = [c async for c in r.synth_stream("같은 답 둘", lead_s=9.0, turn_id=7)]  # Qwen 시도 없이 Supertonic
        mid_reqs = requests(log)
        c = [c async for c in r.synth_stream("다음 답", lead_s=9.0, turn_id=8)]   # 다시 Qwen
        eng_c = r.last_engine
        await r.aclose()
        return r, st, a, b, c, mid_reqs, eng_c, requests(log)
    r, st, a, b, c, mid_reqs, eng_c, reqs = run(main())
    assert st.calls == ["SLOW 하나", "같은 답 둘"]
    assert len(a) == 1 and len(b) == 1 and len(c) >= 3 and eng_c == "qwen"
    assert "같은 답 둘" not in mid_reqs and "같은 답 둘" not in reqs and "다음 답" in reqs
    assert r.fallbacks == 1 and r.fallback_turns == 1   # 유지는 폴백으로 세지 않음


def test_speaker_passes_playback_lead(tmp_path):
    # 첫 문장(길게, 8 s 음성)이 재생 중이면 다음 문장의 느린 첫 청크(2 s)를 기다린다
    async def main():
        s, ws, router, st, _ = await make_session(tmp_path, timeout=0.3)
        s.generation = gen = 1
        s.queue.put_nowait((gen, "길게 이어지는 첫 문장"))
        s.queue.put_nowait((gen, "느림 둘째 문장"))
        for _ in range(600):
            await asyncio.sleep(0.01)
            if len(s._sched) == 2 and not s._speaking_now:
                break
        s.speaker_task.cancel()
        await router.aclose()
        return st, router, list(s._sched)
    st, router, sched = run(main())
    assert st.calls == [] and router.fallbacks == 0 and len(sched) == 2


def test_speaker_keeps_supertonic_for_rest_of_answer(tmp_path):
    async def main():
        s, ws, router, st, log = await make_session(tmp_path, timeout=0.3)
        s.generation = gen = 1
        s.queue.put_nowait((gen, "느림 첫 문장"))
        s.queue.put_nowait((gen, "둘째 문장"))
        for _ in range(300):
            await asyncio.sleep(0.01)
            if len(st.calls) == 2 and not s._speaking_now:
                break
        reqs = requests(log)
        s.speaker_task.cancel()
        await router.aclose()
        return st, router, reqs
    st, router, reqs = run(main())
    assert st.calls == ["느림 첫 문장", "둘째 문장"] and router.fallbacks == 1 and router.fallback_turns == 1
    assert not any("둘째" in q for q in reqs)
