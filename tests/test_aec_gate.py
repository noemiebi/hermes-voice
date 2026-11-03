"""에코 대조 감지기(server/aec_gate.py)와 Endpointer 연동 단위 시험(합성 신호)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.aec_gate import (EchoGate, PlaybackTimeline, ReferenceBuffer,  # noqa: E402
                             resample_44k_to_16k)
from server.vad import WINDOW, Endpointer, VADParams  # noqa: E402

SR = 16000


def speechlike(sec: float, sr: int, seed: int, level_db: float = -18.0) -> np.ndarray:
    """음절 같은 포락선(120~260 ms 소리, 40~180 ms 쉼, 음절마다 ±6 dB)으로 변조한 잡음."""
    rng = np.random.default_rng(seed)
    n = int(sec * sr)
    env = np.zeros(n, dtype=np.float32)
    i = int(rng.uniform(0.0, 0.1) * sr)
    while i < n:
        d = int(rng.uniform(0.12, 0.26) * sr)
        amp = 10 ** (rng.uniform(-6, 6) / 20)
        seg = np.hanning(d).astype(np.float32) ** 0.5 * amp
        env[i:i + d] = seg[: max(0, min(d, n - i))]
        i += d + int(rng.uniform(0.04, 0.18) * sr)
    carrier = rng.standard_normal(n).astype(np.float32)
    carrier = np.convolve(carrier, np.ones(4, dtype=np.float32) / 4, mode="same")  # 약간 저역
    x = carrier * env
    vo = env > 0.3
    rms = float(np.sqrt(np.mean(x[vo] ** 2)))
    return (x * (10 ** (level_db / 20)) / rms).astype(np.float32)


def make_ref(sec=8.0, seed=1):
    ref44 = speechlike(sec, 44100, seed)
    return (np.clip(ref44, -1, 1) * 32767).astype(np.int16)


def echo_of(ref16: np.ndarray, delay_ms=200, atten_db=-20.0) -> np.ndarray:
    d = int(delay_ms * SR / 1000)
    e = np.concatenate([np.zeros(d, dtype=np.float32), ref16])[: ref16.size]
    e = np.convolve(e, np.ones(6, dtype=np.float32) / 6, mode="same")      # 저역(스피커·AEC 잔여)
    e = e * 10 ** (atten_db / 20)
    lim = 0.5 * float(np.max(np.abs(e)))
    return np.clip(e, -lim, lim).astype(np.float32)                         # 클리핑 왜곡


def noise_floor(n, db=-65.0, seed=9):
    return (np.random.default_rng(seed).standard_normal(n) * 10 ** (db / 20)).astype(np.float32)


def run_gate(mic: np.ndarray, ref44: np.ndarray, gate=None, ref_on=True):
    ref = ReferenceBuffer()
    if ref_on:
        ref.add(3, ref44)
    else:
        ref.add(3, np.zeros_like(ref44))
    tl = PlaybackTimeline()
    tl.report(3, 0.0, 0.0)          # 시각 0에 참조 위치 0이 스피커에서 나옴
    gate = gate or EchoGate()
    out = []
    for k in range(mic.size // WINDOW):
        t = (k + 1) * WINDOW / SR
        out.append(gate.process(mic[k * WINDOW:(k + 1) * WINDOW], t, ref, tl, active=True))
    return out


def judged(rs):
    return [r for r in rs if r.judged]


def test_resample_length():
    a = np.zeros(44100, dtype=np.int16)
    assert abs(resample_44k_to_16k(a).size - 16000) <= 1


def test_echo_only_is_echo_dominant():
    ref44 = make_ref()
    ref16 = resample_44k_to_16k(ref44)
    mic = echo_of(ref16) + noise_floor(ref16.size)
    rs = run_gate(mic, ref44)
    j = judged(rs)
    assert len(j) > 150
    frac = sum(r.echo for r in j) / len(j)
    assert frac >= 0.8, frac


def test_lag_estimate_within_32ms():
    ref44 = make_ref(seed=4)
    ref16 = resample_44k_to_16k(ref44)
    for delay in (120, 200, 350):
        mic = echo_of(ref16, delay_ms=delay) + noise_floor(ref16.size)
        lags = [r.lag_ms for r in judged(run_gate(mic, ref44)) if r.echo]
        assert lags
        med = float(np.median(lags))
        assert abs(med - delay) <= 32, (delay, med)


def test_louder_independent_voice_releases_echo():
    ref44 = make_ref(sec=10.0, seed=5)
    ref16 = resample_44k_to_16k(ref44)
    echo = echo_of(ref16)
    mic = echo + noise_floor(ref16.size)
    # 4초부터 에코(음성 구간 RMS)보다 6 dB 큰 독립 음성(다른 포락선)
    fr = np.sqrt(np.mean(echo[: echo.size // WINDOW * WINDOW].reshape(-1, WINDOW) ** 2, axis=1))
    e_rms = float(np.sqrt(np.mean(fr[fr > fr.max() * 0.1] ** 2)))
    user = speechlike(6.0, SR, seed=77, level_db=20 * np.log10(e_rms) + 6)
    s = 4 * SR
    mic[s:s + user.size] += user[: mic.size - s]
    rs = run_gate(mic, ref44)
    before = [r for r in rs[40:4 * SR // WINDOW] if r.judged]
    after = [r for r in rs[(4 * SR + 16000 // 2) // WINDOW:] if r.judged]   # 섞이고 0.5초 뒤부터
    assert sum(r.echo for r in before) / len(before) >= 0.8
    frac = sum(r.echo for r in after) / len(after)
    assert frac <= 0.3, frac
    # 끼어들기 판정(400 ms = 13창 연속 비에코)이 가능해야 한다
    run = best = 0
    for r in rs[4 * SR // WINDOW:]:
        run = run + 1 if not r.echo else 0
        best = max(best, run)
    assert best >= 13, best


def test_silent_reference_is_not_echo():
    ref44 = make_ref(seed=6)
    ref16 = resample_44k_to_16k(ref44)
    mic = echo_of(ref16) + noise_floor(ref16.size)
    rs = run_gate(mic, ref44, ref_on=False)
    assert not any(r.echo for r in rs)
    assert not any(r.judged for r in rs)


def test_hold_freezes_speaker():
    """hold 뒤(스피커 멈춤)에는 참조가 무음이라 판정 보류."""
    ref44 = make_ref(seed=7)
    ref16 = resample_44k_to_16k(ref44)
    ref = ReferenceBuffer()
    ref.add(1, ref44)
    tl = PlaybackTimeline()
    tl.report(1, 0.0, 0.0)
    tl.stop(2.0)
    gate = EchoGate()
    mic = echo_of(ref16) + noise_floor(ref16.size)
    rs = [gate.process(mic[k * WINDOW:(k + 1) * WINDOW], (k + 1) * WINDOW / SR, ref, tl)
          for k in range(mic.size // WINDOW)]
    late = rs[int(3.3 * SR / WINDOW):]   # 멈춘 시각 + 640 ms 기록 + 600 ms 최대 lag 뒤
    assert not any(r.judged for r in late)
    assert tl.pos_at(np.array([1.0, 2.5]))[0] == 1000.0 and np.isnan(tl.pos_at(np.array([2.5]))[0])


# ---------- Endpointer 연동 ----------

class FakeVAD:
    """창마다 정해진 확률을 돌려주는 가짜 VAD."""

    def __init__(self, p=0.9):
        self.p = p

    def reset(self):
        pass

    def prob(self, w):
        return self.p


def _ep(echo_flag):
    return Endpointer(VADParams(), vad=FakeVAD(0.9), strict=lambda: True,
                      echo_hint=lambda w, back: echo_flag)


def _windows(n, db=-20.0, seed=0):
    """창 n개 길이의 잡음(기본 -20 dBFS, BARGE_MIN_DB보다 큼)."""
    a = np.random.default_rng(seed).standard_normal(WINDOW * n) * 10 ** (db / 20)
    return (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()


def test_strict_echo_windows_do_not_start():
    ep = _ep(True)
    evs = ep.feed(_windows(40))   # 1.28 s, 모두 에코 우세
    assert not [e for e in evs if e.kind == "start"]


def test_strict_non_echo_400ms_starts():
    ep = _ep(False)
    evs = ep.feed(_windows(13))   # 13 * 32 = 416 ms
    assert [e for e in evs if e.kind == "start"]


def test_echo_breaks_strict_run():
    flags = iter([False] * 10 + [True] + [False] * 10)
    ep = Endpointer(VADParams(), vad=FakeVAD(0.9), strict=lambda: True,
                    echo_hint=lambda w, back: next(flags))
    evs = ep.feed(_windows(21))   # 연속 비에코가 최대 10창(320 ms) -> 시작 아님
    assert not [e for e in evs if e.kind == "start"]


def test_normal_mode_ignores_echo():
    ep = Endpointer(VADParams(), vad=FakeVAD(0.9), strict=lambda: False,
                    echo_hint=lambda w, back: True)
    evs = ep.feed(_windows(6))
    assert [e for e in evs if e.kind == "start"]


def test_echo_fraction_and_confirm_count():
    seq = [False] * 13 + [True] * 30 + [None] * 5
    it = iter(seq)
    ep = Endpointer(VADParams(), vad=FakeVAD(0.9), strict=lambda: True,
                    echo_hint=lambda w, back: next(it))
    ep.feed(_windows(len(seq)))
    assert ep.in_speech
    # 시작 13창 + 판정 보류 5창은 세고 에코 30창은 세지 않음
    assert ep.confirm_ms == (13 + 5) * 32
    ep.vad.p = 0.0
    ep.echo_hint = lambda w, back: None
    evs = ep.feed(_windows(30))
    end = [e for e in evs if e.kind == "end"][0]
    assert abs(end.echo_frac - 30 / 43) < 1e-6 and end.echo_judged == 43


# ---------- 세션: playback pos_ms 처리 ----------

def test_session_playback_pos_ms():
    import asyncio
    import json
    from test_session_hold import barge_start, settle, setup

    async def go():
        s, ws, stt, hermes, gen = await setup()   # pos_ms 없는 옛 playback 보고
        assert not s._aec_on and not s.timeline.running
        assert s.aec_ref.gen == gen and s.aec_ref.samples > 15000   # FakeTTS 1초 -> 16 kHz 참조
        await s._on_text(json.dumps({"type": "playback", "state": "speaking", "generation": gen,
                                     "pos_ms": 500, "out_latency_ms": 40}))
        assert s._aec_on and s.timeline.running and s.timeline.gen == gen
        assert abs(s.timeline.last_pos - 460) < 1e-6
        await barge_start(s)
        assert s.holding and not s.timeline.running
        # hold 중 늦게 온 보고는 시간축을 다시 열지 않는다
        await s._on_text(json.dumps({"type": "playback", "state": "speaking", "generation": gen, "pos_ms": 750}))
        assert not s.timeline.running
        await s._resume()
        assert s.timeline.running
        hermes.release.set()
        await settle()
        s.speaker_task.cancel()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    asyncio.run(go())


def test_strict_quiet_windows_not_counted_but_not_breaking():
    """BARGE_MIN_DB(-52) 아래 창은 세지 않되 연속을 끊지 않는다."""
    ep = Endpointer(VADParams(), vad=FakeVAD(0.9), strict=lambda: True, echo_hint=lambda w, back: False)
    assert not [e for e in ep.feed(_windows(30, db=-58)) if e.kind == "start"]   # 작은 소리만: 시작 안 함
    ep = Endpointer(VADParams(), vad=FakeVAD(0.9), strict=lambda: True, echo_hint=lambda w, back: False)
    data = _windows(6, db=-30) + _windows(5, db=-58, seed=1) + _windows(7, db=-30, seed=2)
    evs = ep.feed(data)
    assert [e for e in evs if e.kind == "start"]   # 큰 창 6 + 7 = 13창(416 ms), 작은 창 5개는 건너뜀
    assert ep.confirm_ms == 13 * 32


# ---------- 실기 상황(2026-11-02 19:04~): 문장 시작 직후 작은 에코 ----------

def _onset_setup(user_db=None, user_at_s=1.0):
    """1초 무음 뒤 참조 문장 시작. 에코 -44 dBFS(이력은 무음), lag 85 ms. 선택: user_at_s부터 독립 음성."""
    ref44 = make_ref(sec=4.0, seed=31)
    ref16 = resample_44k_to_16k(ref44)
    lead = SR                                 # 문장 앞 1초 무음(스피커 쉼)
    echo = echo_of(ref16, delay_ms=85)
    fr = np.sqrt(np.mean(echo[: echo.size // WINDOW * WINDOW].reshape(-1, WINDOW) ** 2, axis=1))
    voiced = float(np.sqrt(np.mean(fr[fr > fr.max() * 0.1] ** 2)))
    echo = echo * (10 ** (-44 / 20) / voiced)
    mic = np.concatenate([np.zeros(lead, dtype=np.float32), echo]) + noise_floor(lead + echo.size, db=-62)
    if user_db is not None:
        s = int(user_at_s * SR)
        u = np.random.default_rng(5).standard_normal(mic.size - s).astype(np.float32) * 10 ** (user_db / 20)
        mic[s:] += u
    ref = ReferenceBuffer()
    ref.add(9, ref44)
    tl = PlaybackTimeline()
    tl.report(9, 0.0, 1.0)                    # 시각 1.0에 참조 위치 0
    gate = EchoGate()
    state = {"k": 0}

    def hint(w, back):
        k = state["k"]
        state["k"] += 1
        r = gate.process(np.frombuffer(w, dtype="<i2").astype(np.float32) / 32768.0,
                         (k + 1) * WINDOW / SR, ref, tl, active=True)
        return r.echo if r.judged else None

    ep = Endpointer(VADParams(), vad=FakeVAD(0.9), strict=lambda: True, echo_hint=hint)
    pcm16 = (np.clip(mic, -1, 1) * 32767).astype("<i2")
    starts = []
    step = SR // 10
    for i in range(0, pcm16.size, step):
        for e in ep.feed(pcm16[i:i + step].tobytes()):
            if e.kind == "start":
                starts.append((i + step) / SR)
    return starts, gate


def test_onset_small_echo_no_strict_start():
    starts, gate = _onset_setup()
    assert starts == [], starts


def test_onset_small_echo_with_loud_user_starts_quickly():
    # 문장 시작 0.2초 뒤(시작 보호 구간 안) -30 dBFS 독립 음성: 400 ms(+100 ms 전송 단위) 안에 start
    starts, _ = _onset_setup(user_db=-30, user_at_s=1.2)
    assert starts and 1.2 < starts[0] <= 1.2 + 0.416 + 0.1, starts
    # 문장 중간(1.5초 뒤)에 끼어들어도 마찬가지
    starts, _ = _onset_setup(user_db=-30, user_at_s=2.5)
    assert starts and 2.5 < starts[0] <= 2.5 + 0.416 + 0.1, starts


# ---------- v22: 레벨 규칙 전역 적용(온셋 밖) ----------

def test_level_rule_outside_onset():
    """문장 중간(온셋 보호 밖)에서 -51 dBFS 에코는 상관과 무관하게 에코, -30 dBFS 사용자 말(+에코)은 통과."""
    ref44 = make_ref(sec=6.0, seed=41)
    ref16 = resample_44k_to_16k(ref44)

    def voiced_scale(x, db):
        fr = np.sqrt(np.mean(x[: x.size // WINDOW * WINDOW].reshape(-1, WINDOW) ** 2, axis=1))
        v = float(np.sqrt(np.mean(fr[fr > fr.max() * 0.1] ** 2)))
        return x * (10 ** (db / 20) / v)

    echo = voiced_scale(echo_of(ref16, delay_ms=85), -51)

    def run(mic):
        ref = ReferenceBuffer()
        ref.add(4, ref44)
        tl = PlaybackTimeline()
        tl.report(4, 0.0, 0.0)
        gate = EchoGate()
        gate.gain_db = -30.0           # 학습된 에코 이득(실기 -25~-30)
        out = []
        for k in range(mic.size // WINDOW):
            t = (k + 1) * WINDOW / SR
            r = gate.process(mic[k * WINDOW:(k + 1) * WINDOW], t, ref, tl)
            if t > 2.0 and r.judged:   # 시작 보호 한참 뒤, 참조가 소리 날 때
                out.append(r)
        return out

    rs = run(echo + noise_floor(echo.size, db=-68))
    assert sum(r.echo for r in rs) / len(rs) >= 0.95
    user = voiced_scale(speechlike(6.0, SR, seed=99), -30)[: echo.size]
    rs = run(echo + user + noise_floor(echo.size, db=-68))
    assert sum(r.echo for r in rs) / len(rs) <= 0.2, sum(r.echo for r in rs) / len(rs)


def test_digital_silence_keeps_learned_lag():
    """iOS가 재생 중 마이크를 완전 무음(0)으로 보내는 구간(실기 17:46, 입력 -180 dBFS)이 학습한 lag 점수를 지우지 않는다.
    예전에는 무음 창마다 상관 0이 EMA로 섞여 lag 점수가 사라지고, 에코가 다시 들어올 때 lag가 412 -> -100 ms로 튀었다."""
    ref44 = make_ref(sec=10.0, seed=5)
    ref16 = resample_44k_to_16k(ref44)
    mic = echo_of(ref16, delay_ms=200) + noise_floor(ref16.size)
    mic[int(4 * SR):int(7 * SR)] = 0.0             # 3초 디지털 무음
    gate = EchoGate()
    rs = run_gate(mic, ref44, gate=gate)
    k_silence_end = int(7 * SR) // WINDOW
    after = [r for r in rs[k_silence_end:k_silence_end + 30] if r.corr > 0]
    assert after
    assert all(abs(r.lag_ms - 200) <= 48 for r in after), [r.lag_ms for r in after]
    assert abs(gate.lag_est_ms - 200) <= 60, gate.lag_est_ms


# ---------- 학습 상태 사전값(prior)·저장(aec_state.json) ----------

def _burst_ref(ms=300, db=-18.0):
    """처음 ms 동안만 소리(-18 dBFS 잡음), 뒤는 무음인 2초 참조(44.1 kHz int16)."""
    n = int(2.0 * 44100)
    a = np.zeros(n, dtype=np.float32)
    k = int(ms * 44.1)
    a[:k] = np.random.default_rng(3).standard_normal(k) * 10 ** (db / 20)
    return (np.clip(a, -1, 1) * 32767).astype(np.int16)


def _first_window(gate, t=0.5, mic_db=-55.0):
    ref = ReferenceBuffer()
    ref.add(3, _burst_ref())
    tl = PlaybackTimeline()
    tl.report(3, 0.0, 0.0)
    w = noise_floor(WINDOW, db=mic_db, seed=11)
    return gate.process(w, t, ref, tl, active=True)


def test_prior_first_window_judged():
    """기본 lag(90 ms)로는 t=0.5 s에 참조 위치가 무음 구간이라 보류. 저장된 lag 200 ms·이득 -30 dB를 사전값으로
    넣으면 첫 창부터 expected_db를 그 이득으로 계산하고 레벨 규칙으로 판정한다."""
    r0 = _first_window(EchoGate())
    assert not r0.judged and r0.rule == "보류" and r0.expected_db != r0.expected_db
    g = EchoGate()
    assert g.set_prior(200.0, -30.0)
    r = _first_window(g)
    assert r.judged and r.rule != "보류"
    assert -60.0 < r.expected_db < -44.0              # 참조(-18 dBFS 근방) + 이득 -30
    g2 = EchoGate()
    g2.set_prior(200.0, -20.0)
    assert abs(_first_window(g2).expected_db - (r.expected_db + 10.0)) < 0.5   # 이득 prior가 그대로 반영
    assert r.echo and r.rule == "레벨"
    # lag 점수도 prior lag 근처에 심어 둠(첫 상관이 범위 끝 -100 ms로 튀지 않게)
    k = int(np.argmax(g._lag_score))
    assert abs(g.lags[k] * 1000 - 200) <= 16


def test_prior_not_applied_after_learning_and_clamped():
    g = EchoGate()
    g.learned = 5
    assert not g.set_prior(200.0, -30.0) and g.lag_est_ms != 200.0
    g = EchoGate()
    assert g.set_prior(5000.0, 20.0)
    assert g.lag_est_ms <= 600.0 and g.gain_db <= 0.0
    assert not EchoGate().set_prior(float("nan"), -20.0)


def test_learning_counts_and_confident():
    ref44 = make_ref(sec=10.0)
    ref16 = resample_44k_to_16k(ref44)
    mic = echo_of(ref16) + noise_floor(ref16.size)
    g = EchoGate()
    run_gate(mic, ref44, gate=g)
    assert g.learned > 0
    st = g.state()
    assert set(st) == {"lag_ms", "gain_db", "learned"} and st["learned"] == g.learned


def test_aec_state_store_roundtrip(tmp_path):
    from server.aec_gate import AecStateStore
    now = [0.0]
    p = tmp_path / "aec_state.json"
    st = AecStateStore(p, clock=lambda: now[0])
    assert st.load("k") is None                      # 파일 없음
    vp = {"screen": [393, 852, 852], "dpr": 3}
    k1 = AecStateStore.make_key("pwa", vp, True, False)
    k2 = AecStateStore.make_key("pwa", vp, False, False)
    assert k1 == "pwa|393x852@3|aec=1|hd=0" and k1 != k2
    assert AecStateStore.make_key(None, None, True, True) == "?|?|aec=1|hd=1"
    assert st.save(k1, {"lag_ms": 150.0, "gain_db": -24.5, "learned": 60})
    assert not st.save(k1, {"lag_ms": 160.0, "gain_db": -25.0, "learned": 90})   # 1분 throttle
    assert st.save(k2, {"lag_ms": 300.0, "gain_db": -12.0, "learned": 50})      # 다른 키는 따로
    now[0] = 61.0
    assert st.save(k1, {"lag_ms": 160.0, "gain_db": -25.0, "learned": 90})
    st2 = AecStateStore(p)
    assert st2.load(k1) == {"lag_ms": 160.0, "gain_db": -25.0, "learned": 90}
    assert st2.load(k2)["lag_ms"] == 300.0
    assert not (tmp_path / "aec_state.json.tmp").exists()
    p.write_text("{깨짐", encoding="utf-8")
    assert AecStateStore(p).load(k1) is None          # 깨진 파일은 무시
    assert AecStateStore(p).save(k1, {"lag_ms": 1.0, "gain_db": -20.0, "learned": 40})
    assert AecStateStore(p).load(k1)["lag_ms"] == 1.0
