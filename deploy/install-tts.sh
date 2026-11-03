#!/bin/zsh
# Qwen3-TTS 워커 전용 venv(.venv-tts, Python 3.12 + mlx-audio) 만들기. 사용: deploy/install-tts.sh [--force]
# 메인 .venv와 의존성(transformers·mlx 버전)이 부딪치지 않게 따로 둔다. 워커는 server/qwen_tts_worker.py.
# 모델은 HF 캐시(QWEN_TTS_HF_HOME/hub, 기본 ~/.cache/huggingface/hub)에 미리 받아 둔다(서버는 오프라인으로 읽음):
#   .venv-tts/bin/hf download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
#   (다른 위치면 HF_HOME=<경로> 를 앞에 붙이고, .env 에 QWEN_TTS_HF_HOME=<경로>)
set -e
if [[ "$(uname -sm)" != "Darwin arm64" ]]; then
  echo "오류: Apple Silicon 맥(MLX)에서만 돕니다. 다른 환경이면 .env에 TTS_ENGINE=supertonic" >&2; exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "오류: uv가 필요합니다(https://docs.astral.sh/uv/, 예: brew install uv)" >&2; exit 1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV=$ROOT/.venv-tts
if [[ "$1" == "--force" ]]; then rm -rf "$VENV"; fi
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.12 "$VENV"
fi
# 2026-11-02 청취 벤치(.venv-bench)에서 확인한 버전으로 고정
uv pip install --python "$VENV/bin/python" \
  "mlx-audio==0.6.3" "mlx==0.32.4" "transformers==5.22.1" "huggingface-hub==2.4.0" "numpy" "scipy"
"$VENV/bin/python" -c "import mlx_audio, mlx.core as mx; print('mlx-audio', mlx_audio.__version__ if hasattr(mlx_audio, '__version__') else '?', 'mlx', mx.__version__)"
echo "완료: $VENV (TTS_ENGINE 기본이 qwen이라 따로 켤 필요 없음)"
echo "다음: 모델 받기  $VENV/bin/hf download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
