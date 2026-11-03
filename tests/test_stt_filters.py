"""STT 후처리(환각·프롬프트 에코·글자 밀도·세그먼트 필터)와 클립 저장. 모델 없이 가짜 mlx로."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.stt import ClipStore, STTEngine, is_hallucination, is_prompt_echo, too_dense  # noqa: E402

PROMPT = "헤르메스, Hermes, 카카오톡, 캘린더, 미리 알림, 네이버 메일, Gmail, 클로드, Claude, 오픈클로"


def test_hallucinations():
    for t in ("시청해 주셔서 감사합니다.", "감사합니다.", "구독과 좋아요 부탁드립니다!", "MBC 뉴스 이덕영입니다.",
              "KBS 뉴스", "자막 제공 및 자막 편집", "이 영상은 유료광고를 포함하고 있습니다.", "다음 영상에서 만나요",
              "아아아아아아", "", "  ...  "):
        assert is_hallucination(t), t
    for t in ("네, 감사합니다. 내일 다시 알려 줘.", "네.", "아니", "뉴스 좀 읽어 줘", "MBC 라디오 틀어 줘"):
        assert not is_hallucination(t), t


def test_prompt_echo():
    assert is_prompt_echo(PROMPT, PROMPT)
    assert is_prompt_echo("헤르메스, 카카오톡, 캘린더, 미리 알림", PROMPT)
    assert is_prompt_echo("헤르메스 Hermes 카카오톡 캘린더 미리알림 네이버메일 Gmail", PROMPT)
    for t in ("카카오톡", "헤르메스, 지금 몇 시야?", "카카오톡에 엄마한테 온 메시지 확인해 줘.",
              "클로드한테 카카오톡이랑 캘린더 정리해 달라고 해 줘, 오늘 저녁까지 꼭 끝내야 하는 일이야"):
        assert not is_prompt_echo(t, PROMPT), t
    assert not is_prompt_echo(PROMPT, "")


def test_too_dense():
    assert not too_dense("네", 0.4)
    assert not too_dense("오늘 서울 날씨 어때 우산 챙겨야 해", 2.0)
    assert too_dense("가" * 30, 0.8)        # 짧은 클립도 1초로 봐서 25자 초과면 버림
    assert too_dense("가나다라" * 20, 2.0)
    assert not too_dense("가" * 30, 0.8, max_cps=0)


class FakeMlx:
    def __init__(self, result):
        self.result, self.calls = result, []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        return self.result


def _engine(tmp_path, result, prompt="", **kw):
    e = STTEngine(model="fake", corrections=str(tmp_path / "none.tsv"), prompt=prompt,
                  save_clips_dir=str(tmp_path / "clips"), **kw)
    e._mlx, e.loaded = FakeMlx(result), True
    return e


def _speech(sec=1.0):
    t = np.arange(int(16000 * sec)) / 16000
    return (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16).tobytes()


def test_transcribe_passes_options_and_saves(tmp_path):
    r = {"text": "카카오톡 확인해 줘", "segments": [
        {"text": "카카오톡 확인해 줘", "no_speech_prob": 0.01, "avg_logprob": -0.2, "compression_ratio": 1.1}]}
    e = _engine(tmp_path, r, prompt=PROMPT, temperatures=(0.0, 0.2))
    text, ms = e.transcribe_sync(_speech())
    assert text == "카카오톡 확인해 줘" and isinstance(ms, int)
    kw = e._mlx.calls[0]
    assert kw["initial_prompt"] == PROMPT and kw["language"] == "ko"
    assert kw["temperature"] == (0.0, 0.2) and kw["condition_on_previous_text"] is False
    wavs = list((tmp_path / "clips").glob("*.wav"))
    assert len(wavs) == 1
    info = wavs[0].with_suffix(".txt").read_text(encoding="utf-8")
    assert "text\t카카오톡 확인해 줘" in info and "ms\t" in info


def test_no_prompt_not_passed(tmp_path):
    e = _engine(tmp_path, {"text": "네", "segments": []}, prompt="", temperatures=(0.0,))
    assert e.transcribe_sync(_speech())[0] == "네"
    assert "initial_prompt" not in e._mlx.calls[0] and e._mlx.calls[0]["temperature"] == 0.0


def test_filters_in_transcribe(tmp_path):
    seg_bad = {"text": " 감사합니다.", "no_speech_prob": 0.9, "avg_logprob": -1.5, "compression_ratio": 1.0}
    seg_rep = {"text": " 네 네 네 네 네 네 네 네 네 네", "no_speech_prob": 0.1, "avg_logprob": -0.3,
               "compression_ratio": 3.0}
    seg_ok = {"text": "내일 일정 알려 줘", "no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.2}
    e = _engine(tmp_path, {"text": "내일 일정 알려 줘 감사합니다.", "segments": [seg_ok, seg_bad, seg_rep]})
    d = e.transcribe_detail(_speech(2.0))
    assert d["text"] == "내일 일정 알려 줘" and d["reason"] == ""
    e = _engine(tmp_path, {"text": "시청해 주셔서 감사합니다.", "segments": []})
    assert e.transcribe_detail(_speech())["reason"] == "hallucination"
    e = _engine(tmp_path, {"text": PROMPT, "segments": []}, prompt=PROMPT)
    assert e.transcribe_detail(_speech(3.0))["reason"] == "prompt-echo"
    e = _engine(tmp_path, {"text": "내일 회의 일정 " * 6, "segments": []})
    assert e.transcribe_detail(_speech(1.0))["reason"] == "too-dense"
    e = _engine(tmp_path, {"text": "x", "segments": []})
    assert e.transcribe_detail(b"\x00\x00" * 16000)["reason"] == "short/silent"
    assert e._mlx.calls == []


def test_clipstore_prune_and_off(tmp_path):
    cs = ClipStore(str(tmp_path / "c"), keep=3)
    for i in range(5):
        p = cs.save(b"\x01\x00" * 160, {"text": f"t{i}"})
        # 같은 밀리초 충돌 방지: 이름을 순서대로 바꿔 둠
        p.rename(p.with_name(f"{i:03d}.wav"))
        p.with_suffix(".txt").rename(p.with_name(f"{i:03d}.txt"))
    cs._prune()
    names = sorted(x.name for x in (tmp_path / "c").iterdir())
    assert names == ["002.txt", "002.wav", "003.txt", "003.wav", "004.txt", "004.wav"]
    assert ClipStore("", keep=3).save(b"\x00\x00", {}) is None
