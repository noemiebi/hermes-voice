# 서드파티 고지 (Third-Party Notices)

hermes-voice 자체 코드는 MIT 라이선스입니다(`LICENSE`). 이 저장소는 아래 모델 가중치를 **포함하지 않으며**, 사용자가 각 배포처에서 직접 받습니다. 받는 순간 각 모델의 라이선스가 적용되니 배포처의 원문을 꼭 읽어 주세요.

라이선스는 2026-11-03에 각 배포처(Hugging Face 모델 카드 메타데이터·LICENSE 파일, GitHub 저장소 라이선스, PyPI 메타데이터)에서 확인했습니다. 이후 바뀌었을 수 있습니다.

## 모델

| 구성 요소 | 쓰는 곳 | 배포처 | 라이선스 | 비고 |
|---|---|---|---|---|
| Whisper large-v3-turbo (MLX 변환본) | STT | [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) | MIT (원본 기준) | 변환본 모델 카드에는 라이선스 표기가 없음. 원본 [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)는 MIT, [openai/whisper](https://github.com/openai/whisper) 코드·가중치도 MIT |
| Supertonic 3 (ONNX) | TTS(폴백·가벼운 대안) | [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3) (같은 내용의 보관본 [supertone-oss-archive/supertonic-3](https://huggingface.co/supertone-oss-archive/supertonic-3)) | **BigScience OpenRAIL-M** | 가중치는 MIT가 아님(예제 코드만 MIT). OpenRAIL-M은 Attachment A에 **사용 제한** 13개 항목을 둔다: 불법 행위, 동의 없는 타인 사칭(딥페이크), 해를 끼칠 목적의 허위 정보, 차별, 기계 생성임을 밝히지 않은 콘텐츠 유포, 의료 조언·의료 결과 해석 등. 파생물을 배포할 때는 같은 제한을 넘겨야 함. 모델 저장소의 `LICENSE` 원문 참조 |
| Silero VAD (ONNX) | 말 시작·끝 감지 | [snakers4/silero-vad](https://github.com/snakers4/silero-vad), PyPI `silero-vad`(패키지 안의 `silero_vad.onnx`를 씀) | MIT | |
| Smart Turn v3 (`smart-turn-v3.2-cpu.onnx`) | 말 끝 보조 판정(선택) | [pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3), 코드 [pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn) | BSD-2-Clause | |
| Qwen3-TTS 12Hz 1.7B CustomVoice (MLX 8bit) | TTS(기본) | [mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit), 원본 [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | Apache-2.0 | 변환본·원본 모델 카드 모두 Apache-2.0. 코드 [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)도 Apache-2.0 |

## 저장소에 들어 있는 서드파티 코드

| 파일 | 원본 | 라이선스 |
|---|---|---|
| `server/supertonic_helper.py` | [supertone-oss-archive/supertonic](https://github.com/supertone-oss-archive/supertonic) `py/helper.py` (Copyright (c) 2025 Supertone Inc.) | MIT, 원문 `server/SUPERTONIC_CODE_LICENSE.txt` |
| `server/smart_turn.py`의 log-mel 계산 부분 | [pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat) `src/pipecat/audio/turn/smart_turn/_whisper_features.py` (Copyright (c) 2024-2026, Daily) | BSD-2-Clause |
| 위 코드 중 Hugging Face Transformers에서 온 부분 (`WhisperFeatureExtractor`, `audio_utils`) | [huggingface/transformers](https://github.com/huggingface/transformers) (Copyright 2022-2023 The HuggingFace Inc. team) | Apache-2.0 |

각 파일 맨 위에 출처와 저작권 표기를 남겨 두었습니다.

## 파이썬 패키지 (설치 시 받음, 저장소에 포함 안 됨)

| 패키지 | 라이선스 (PyPI 메타데이터) |
|---|---|
| [mlx-whisper](https://pypi.org/project/mlx-whisper/) | MIT |
| [mlx](https://pypi.org/project/mlx/) | MIT |
| [mlx-audio](https://pypi.org/project/mlx-audio/) (선택, `.venv-tts`) | MIT |
| [silero-vad](https://pypi.org/project/silero-vad/) | MIT |
| [onnxruntime](https://pypi.org/project/onnxruntime/) | MIT |
| [FastAPI](https://pypi.org/project/fastapi/) | MIT |
| [uvicorn](https://pypi.org/project/uvicorn/) | BSD-3-Clause |
| [websockets](https://pypi.org/project/websockets/) | BSD-3-Clause |
| [httpx](https://pypi.org/project/httpx/) | BSD-3-Clause |
| [numpy](https://pypi.org/project/numpy/) | BSD-3-Clause (번들 구성 요소에 0BSD·MIT·Zlib·CC0-1.0 포함) |
| [scipy](https://pypi.org/project/scipy/) | BSD (BSD-3-Clause 문안) |
| [soundfile](https://pypi.org/project/soundfile/) | BSD-3-Clause |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | BSD-3-Clause |
| [audiotsm](https://pypi.org/project/audiotsm/) | MIT |
| transformers, huggingface-hub (`.venv-tts`, 간접 의존) | Apache-2.0 (각 저장소 기준) |

간접 의존성(전이 의존 패키지)은 여기 다 적지 않았습니다. 필요하면 `pip-licenses` 같은 도구로 설치된 환경에서 뽑아 보세요.

## 연동 대상 (포함 안 함)

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) (NousResearch, MIT): 이 앱이 HTTP로 호출하는 별도 프로그램입니다. 코드는 포함하지 않습니다. Hermes가 부르는 LLM 제공자의 약관은 각자 설정한 제공자를 따릅니다.
