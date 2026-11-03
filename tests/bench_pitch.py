"""음높이 판별 실데이터 확인(pytest 밖). 사용: .venv/bin/python tests/bench_pitch.py [--onset-ms 400]

- 사용자 쪽: tests/clips/*.wav 전체. user 프로필은 그 클립을 뺀 나머지 전부로(leave-one-out) 학습한다.
  주의: clips에는 받아쓰기 저장 클립이 모두 들어 있어 스피커 에코·합성 시험 클립도 섞여 있다(라벨 없음).
- TTS 쪽: 현재 Supertonic 음색으로 학습 문장 4개(서버와 같은 pitch.TTS_LEARN_SENTENCES)로 tts 프로필을 만들고,
  다른 문장 5개를 합성해 판정한다.
- 판정 창: 전체 발화(full)와, 첫 소리 시작 PITCH_PREROLL_MS 앞부터 --onset-ms 길이(onset, 재생 중 말 시작 판정과 같은 길이).
"""
import argparse
import asyncio
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import config, pitch  # noqa: E402

CLIPS = Path(__file__).resolve().parent / "clips"
TEST_SENTENCES = (
    "잠깐 확인해 볼게요.",
    "내일 서울은 흐리고 낮 최고 기온은 열두 도예요.",
    "오후 두 시에 회의가 있고 저녁엔 약속이 하나 있어요.",
    "그건 제가 잘 모르겠어요, 대신 관련 메일을 찾아봤어요.",
    "메일 세 통 중에 답장이 필요한 건 하나예요.",
)


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), "<i2")
        sr = w.getframerate()
    x = a.astype(np.float64) / 32768.0
    return pitch.resample(x, sr) if sr != 16000 else x


def onset(x: np.ndarray, ms: int) -> np.ndarray:
    """10 ms 창 RMS가 처음 -40 dBFS를 넘는 곳 - PITCH_PREROLL_MS부터 ms 길이."""
    n = x.size // 160
    if n == 0:
        return x
    fr = x[: n * 160].reshape(n, 160)
    db = 20 * np.log10(np.sqrt(np.mean(fr * fr, axis=1)) + 1e-12)
    hit = np.flatnonzero(db > -40)
    i0 = max(0, (int(hit[0]) if hit.size else 0) * 160 - config.PITCH_PREROLL_MS * 16)
    return x[i0:i0 + ms * 16]


def pct(labels: list[str]) -> str:
    n = max(1, len(labels))
    return " | ".join(f"{100 * labels.count(k) / n:5.1f}%" for k in ("user", "unknown", "tts"))


async def tts_f0(texts) -> list[np.ndarray]:
    from server.tts import TTSEngine
    eng = TTSEngine()
    await eng.load()
    if not eng.loaded:
        raise SystemExit(f"Supertonic 로드 실패: {eng.error}")
    out = []
    for t in texts:
        p, _ = await eng.synth(t)
        out.append(pitch.resample(p, eng.out_rate))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onset-ms", type=int, default=config.PITCH_ONSET_MS)
    a = ap.parse_args()

    clips = sorted(CLIPS.glob("*.wav"))
    xs = [read_wav(p) for p in clips]
    full = [pitch.f0_track(x) for x in xs]
    ons = [pitch.f0_track(onset(x, a.onset_ms)) for x in xs]
    stats = [pitch.utt_stats(f, onset=True) for f in full]   # 서버 user 학습과 같은 방식(앞부분 중앙값)
    fullmed = [pitch.utt_stats(f) for f in full]

    learn = asyncio.run(tts_f0(pitch.TTS_LEARN_SENTENCES))
    test = asyncio.run(tts_f0(TEST_SENTENCES))
    key = "supertonic:" + Path(config.SUPERTONIC_VOICE).name
    tts_full = [pitch.f0_track(x) for x in test]
    tts_ons = [pitch.f0_track(onset(x, a.onset_ms)) for x in test]

    low = [s is not None and np.exp(s[0]) < 150 for s in fullmed]   # 발화 전체 F0 중앙값 < 150 Hz

    def prof(skip: int = -1, only_low: bool = False) -> pitch.VoiceProfile:
        u = pitch.Dist([list(s) for i, s in enumerate(stats)
                        if s is not None and i != skip and (low[i] or not only_low)])
        p = pitch.VoiceProfile(u)
        p.set_tts(key, [pitch.f0_track(x) for x in learn])
        return p

    base = prof()
    print(f"clips {len(clips)}개, 유효 F0 {sum(s is not None for s in stats)}개")
    meds = np.array([np.exp(s[0]) for s in fullmed if s is not None])
    print("clip F0 중앙값 분위(Hz) 0/10/25/50/75/90/100:",
          " / ".join(f"{v:.0f}" for v in np.percentile(meds, [0, 10, 25, 50, 75, 90, 100])))
    print("clip F0 중앙값 분포:", ", ".join(f"<{b}Hz {int(np.sum(meds < b))}" for b in (100, 130, 150, 170, 200)),
          f", ≥200Hz {int(np.sum(meds >= 200))}")
    print(f"프로필(전체 클립): {base.summary()}")
    lo, hi = base.user.range_hz()
    tlo, thi = base.tts[key].range_hz()
    print(f"판정 구간: user {lo:.0f}~{hi:.0f} Hz, tts {tlo:.0f}~{thi:.0f} Hz")
    print(f"TTS 테스트 문장 F0 중앙값: {', '.join(f'{np.median(f):.0f}' for f in tts_full)} Hz")

    clean = prof(only_low=True)
    lo, hi = clean.user.range_hz()
    print(f"(B) user 프로필을 F0<150 Hz 클립만으로(에코 섞이지 않은 학습 가정): {clean.summary()}, "
          f"user 구간 {lo:.0f}~{hi:.0f} Hz")

    lab = {k: [] for k in ("clip_full", "clip_onset", "clip_full_lt150", "clip_onset_lt150", "B_onset_lt150")}
    for i in range(len(clips)):
        p = prof(i)
        lf, lo_ = p.classify(full[i])[0], p.classify(ons[i])[0]
        lab["clip_full"].append(lf)
        lab["clip_onset"].append(lo_)
        if low[i]:
            lab["clip_full_lt150"].append(lf)
            lab["clip_onset_lt150"].append(lo_)
            lab["B_onset_lt150"].append(prof(i, only_low=True).classify(ons[i])[0])
    lab["tts_full"] = [base.classify(f)[0] for f in tts_full]
    lab["tts_onset"] = [base.classify(f)[0] for f in tts_ons]
    lab["B_tts_onset"] = [clean.classify(f)[0] for f in tts_ons]

    print()
    print(f"| 대상 | 창 | n | user | unknown | tts |")
    print(f"|---|---|---|---|---|---|")
    rows = [("clips 전체", "전체 발화", "clip_full"), ("clips 전체", f"앞 {a.onset_ms} ms", "clip_onset"),
            ("clips 중 F0<150 Hz", "전체 발화", "clip_full_lt150"),
            ("clips 중 F0<150 Hz", f"앞 {a.onset_ms} ms", "clip_onset_lt150"),
            ("TTS 문장 5개", "전체 발화", "tts_full"), ("TTS 문장 5개", f"앞 {a.onset_ms} ms", "tts_onset"),
            ("(B) clips 중 F0<150 Hz", f"앞 {a.onset_ms} ms", "B_onset_lt150"),
            ("(B) TTS 문장 5개", f"앞 {a.onset_ms} ms", "B_tts_onset")]
    for who, win, k in rows:
        print(f"| {who} | {win} | {len(lab[k])} | {pct(lab[k])} |")


if __name__ == "__main__":
    main()
