# Hermes Voice

[English](README.en.md) · 한국어

아이폰 홈 화면 웹앱(PWA)으로 내 맥에서 도는 [Hermes Agent](https://github.com/NousResearch/hermes-agent)와 말로 대화하는 로컬 음성 서버입니다. 음성 인식(STT)과 음성 합성(TTS)은 모두 맥 안에서 돌고(mlx-whisper large-v3-turbo, Supertonic 3 ONNX), 외부 음성 API를 쓰지 않습니다. 답은 Hermes Agent의 API 서버(`/v1/chat/completions` SSE 스트림)가 만들며, 문장 단위로 바로 합성해 들려줍니다. 핸즈프리로 말을 주고받고, 답하는 중에 말을 걸어 끊을 수 있습니다. 대화 상대 이름은 `AGENT_NAME`(기본 `Herbot`)으로 바꿀 수 있습니다.

설계는 [`docs/설계안.md`](docs/설계안.md), 서버·클라이언트 메시지 계약은 [`docs/프로토콜.md`](docs/프로토콜.md)에 있습니다.

## 필요한 것

- Apple Silicon 맥 (mlx-whisper가 MLX를 씀)
- Python 3.12, [uv](https://docs.astral.sh/uv/)
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) 설치, API 서버(`platforms.api_server`) 켜짐
- 아이폰 Safari (홈 화면에 추가해서 씀)
- HTTPS 접속 경로. 브라우저 마이크는 HTTPS에서만 열립니다. [Tailscale](https://tailscale.com/) `serve`를 권장합니다(아래 '배포').

## 빠른 시작

### 1. 받기와 가상환경

```sh
git clone <이 저장소 주소> hermes-voice
cd hermes-voice
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r pyproject.toml --extra dev
uv pip install --python .venv/bin/python --no-deps silero-vad   # VAD ONNX만 씀 (torch 설치 안 함)
```

### 2. 모델 받기

서버는 Hugging Face를 오프라인 모드(`HF_HUB_OFFLINE=1`)로 읽으므로, 모델을 미리 받아 둬야 합니다.

**Whisper (STT)**: HF 캐시(기본 `~/.cache/huggingface/hub`)에 받습니다.

```sh
.venv/bin/hf download mlx-community/whisper-large-v3-turbo
```

**Supertonic 3 (TTS)**: [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3)을 `SUPERTONIC_DIR`(기본 `~/.hermes/voice/models/supertonic3`)에 받습니다.

```sh
.venv/bin/hf download Supertone/supertonic-3 \
  --include "onnx/*" "voice_styles/*" LICENSE config.json \
  --local-dir ~/.hermes/voice/models/supertonic3
```

받은 뒤 구조는 다음과 같아야 합니다. 기본 음색은 `voice_styles/F1.json`이고, 다른 음색(F1~F5, M1~M5)은 `SUPERTONIC_VOICE`로 고릅니다.

```
~/.hermes/voice/models/supertonic3/
├── onnx/
│   ├── duration_predictor.onnx
│   ├── text_encoder.onnx
│   ├── tts.json
│   ├── unicode_indexer.json
│   ├── vector_estimator.onnx
│   └── vocoder.onnx
└── voice_styles/
    ├── F1.json … F5.json
    └── M1.json … M5.json
```

Supertonic 3 가중치는 MIT가 아니라 **OpenRAIL-M** 라이선스입니다(사용 제한 있음). [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)를 보세요.

**(선택) Smart Turn v3**: 말 끝 판정을 돕는 8 MB 모델입니다. `TURN_DETECTOR=smart-turn`일 때만 씁니다.

```sh
mkdir -p models && curl -L -o models/smart-turn-v3.2-cpu.onnx \
  https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx
```

### 3. `.env` 채우기

```sh
cp .env.example .env
```

최소한 `HERMES_API_KEY`(Hermes API 서버 키)를 채웁니다. 나머지는 아래 '환경 변수' 표를 보세요. `.env`는 git에 올라가지 않습니다.

### 4. Hermes에 모델 라우트 두 개 만들기

앱은 Hermes에 `voice-fast`(기본, 빠른 답)와 `voice-deep`(설정에서 '깊게') 두 모델 이름으로 요청합니다. Hermes의 `config.yaml`에서 API 서버의 `model_routes`에 두 이름을 정의하세요. 모델과 제공자는 각자 Hermes에 설정해 둔 것을 쓰면 됩니다. 예: fast는 작고 빠른 모델, deep은 큰 모델.

```yaml
platforms:
  api_server:
    enabled: true
    extra:
      host: 127.0.0.1
      port: 8642
      key: "your-api-server-secret"     # = hermes-voice의 HERMES_API_KEY
      model_routes:
        voice-fast:
          model: "<작고 빠른 모델>"
          provider: "<제공자>"           # 생략하면 Hermes 전역 제공자
        voice-deep:
          model: "<큰 모델>"
          provider: "<제공자>"
```

음성 답은 첫 토큰이 빨리 와야 자연스럽습니다. `voice-fast`에는 첫 토큰이 빠른 모델을 권합니다. `voice-fast` 요청에는 추론 단계를 `HERMES_REASONING`(기본 `off`)으로 붙입니다.

### 5. 실행과 확인

```sh
.venv/bin/python -m server                    # http://127.0.0.1:8480  (/ = 웹앱, /ws, /health)
HERMES_MOCK=1 .venv/bin/python -m server      # Hermes 없이 가짜 스트림으로 시험
```

시작할 때 STT·TTS 모델을 미리 올립니다(STT 수 초, TTS 1초 안팎).

```sh
curl -s http://127.0.0.1:8480/health
```

`"ok": true`면 준비 완료입니다. `VOICE_TOKEN`이 없으면 STT·TTS 로드 상태와 `hermes.reachable`(Hermes API 서버 연결 여부)도 함께 나옵니다. `VOICE_TOKEN`을 설정하면 `ok`만 보입니다.

### 6. 아이폰에서 열기

1. 맥의 서버를 HTTPS로 엽니다. Tailscale을 쓰면 `deploy/install.sh serve`가 `https://<맥의 Tailscale 이름>:8464` → `http://127.0.0.1:8480`을 걸고 접속 주소를 출력합니다. 다른 리버스 프록시를 써도 되지만 HTTPS여야 마이크가 열립니다.
2. 아이폰 Safari로 그 주소를 열고 공유 › **홈 화면에 추가**를 누릅니다. 이후에는 홈 화면 아이콘으로 엽니다.
3. `VOICE_TOKEN`을 설정했다면 처음 한 번 `https://<주소>/?token=<값>` 링크로 엽니다. 토큰은 기기에 저장되고 주소창에서는 지워집니다(`#token=<값>`도 됩니다). 홈 화면 앱은 Safari와 저장소가 따로이므로, 홈 화면에 추가한 뒤 그 앱에서 토큰을 한 번 더 넣어야 할 수 있습니다.

## 환경 변수 (`.env`)

경로 변수는 비워 두면 기본값을 씁니다.

### 기본

| 이름 | 기본값 | 설명 |
|---|---|---|
| `HERMES_API_URL` | `http://127.0.0.1:8642` | Hermes API 서버 주소 |
| `HERMES_API_KEY` | (빈 값) | **필수.** Hermes API 서버 키(`extra.key` / `API_SERVER_KEY`). 세션 이어가기에도 필요 |
| `HERMES_MOCK` | (빈 값) | `1`이면 Hermes 대신 가짜 스트림(도구 시작 이벤트 1회 포함) |
| `HERMES_REASONING` | `off` | `voice-fast` 요청의 추론 단계: `off`·`minimal`·`low`·`medium`·`high`, `default`면 Hermes 설정값. `voice-deep`에는 안 붙임. 측정 도구 `tests/bench_hermes.py` |
| `AGENT_NAME` | `Herbot` | 화면과 말투에 쓰는 대화 상대 이름 |
| `MODEL_LABEL_FAST` / `MODEL_LABEL_DEEP` | (빈 값) | 설정 시트의 빠름·깊게 아래에 보일 실제 모델 이름(예: `Claude Sonnet 5.5`). 설정하면 매 턴 모델에게도 알려 준다. 비우면 생략 |
| `VOICE_TOKEN` | (빈 값 = 인증 없음) | 접속 토큰. Tailscale 밖(공개망)에 열면 **필수** |
| `VOICE_ALLOWED_ORIGINS` | (빈 값) | `/ws`에 붙어도 되는 다른 출처(쉼표 구분, 예: `https://a.example:8443`, 또는 `*`). 같은 출처와 Origin 없는 클라이언트는 늘 허용 |
| `VOICE_SESSION_IDLE_HOURS` | `0` (끔) | 양수면 마지막 턴 뒤 이만큼 지나 대화를 시작할 때 새 Hermes 세션. 대화 중에는 바꾸지 않음. 끄면 새 대화는 화면의 `+` 버튼으로만 |
| `VOICE_DATA_DIR` | `~/.hermes/voice` | 서버 데이터 폴더(없으면 만듦). 대화 기록·마지막 세션 `history.json`, 사용자 독음 사전, STT 교정표 |
| `PORT` / `HOST` | `8480` / `127.0.0.1` | 바인드. 외부 노출은 리버스 프록시(tailscale serve 등)로 |

### STT

| 이름 | 기본값 | 설명 |
|---|---|---|
| `STT_MODEL` | `mlx-community/whisper-large-v3-turbo` | Hugging Face 모델 id |
| `HF_HUB_CACHE` | `$HF_HOME/hub` 또는 `~/.cache/huggingface/hub` | 모델 캐시. 서버는 오프라인 모드로 캐시만 읽음 |
| `STT_CORRECTIONS` | `$VOICE_DATA_DIR/stt-corrections.tsv` | 교정표(`잘못<TAB>올바름`). 파일이 없으면 교정 안 함 |
| `STT_PROMPT` | (빈 값 = 안 씀) | whisper `initial_prompt` 어휘(쉼표 구분). 정확도가 오히려 나빠지고 잡음에서 프롬프트 반복 환각이 생길 수 있어 기본은 끔. 켜면 프롬프트를 그대로 뱉은 결과는 버림 |
| `STT_TEMPERATURES` | `0` | 그리디 1회. `0,0.2,0.4`처럼 주면 압축률·logprob 실패 때 온도 폴백 |
| `STT_NO_SPEECH_PROB` / `STT_LOGPROB_MIN` | `0.6` / `-1.0` | 무음 확률이 앞 값 초과이고 avg_logprob이 뒤 값 미만인 세그먼트는 버림 |
| `STT_COMPRESSION_MAX` | `2.4` | 압축률(반복)이 이보다 큰 세그먼트는 버림. `0`이면 끔 |
| `STT_MAX_CHARS_PER_SEC` | `25` | 오디오 1초(1초 미만은 1초로 봄)당 글자 수 상한. `0`이면 끔 |
| `STT_SAVE_CLIPS_DIR` | `tests/clips` | 받아쓴 발화를 16 kHz wav + 같은 이름 `.txt`(전사·원문·ms·버린 이유)로 저장. 빈 문자열이면 끔. git 제외 |
| `STT_SAVE_CLIPS_MAX` | `200` | 이보다 많으면 오래된 것부터 지움 |

### TTS

| 이름 | 기본값 | 설명 |
|---|---|---|
| `TTS_ENGINE` | `supertonic` | `qwen`이면 Qwen3-TTS 스트리밍 + 문장 단위 Supertonic 폴백(`.venv-tts` 필요, 아래 '선택: Qwen3-TTS'). 한 답에서 한 번 폴백하면 그 답의 남은 문장도 Supertonic(답 중간에 목소리가 바뀌지 않게) |
| `QWEN_TTS_LEAD_MARGIN_S` | `0.5` | 앞 문장이 재생 중이면 Qwen 첫 청크를 (남은 재생 시간 - 이 값)까지 더 기다림(최소 `QWEN_TTS_FIRST_CHUNK_TIMEOUT_S` 2.5초) |
| `QWEN_TTS_FIRST_CHUNK_MAX_S` | `12` | 위 연장 대기의 상한(초) |
| `SUPERTONIC_DIR` | `~/.hermes/voice/models/supertonic3` | `onnx/`, `voice_styles/`를 담은 폴더 |
| `SUPERTONIC_MODEL_DIR` | `$SUPERTONIC_DIR/onnx` | |
| `SUPERTONIC_VOICE` | `$SUPERTONIC_DIR/voice_styles/F1.json` | 음색 |
| `TTS_SPEED` | `1.3` | Supertonic 말 속도 |
| `TTS_STEPS` | `5` | Supertonic 디노이징 단계 |
| `TTS_PROVIDER` | `cpu` | `coreml`도 되지만 측정상 CPU가 더 빠름 |
| `TTS_TARGET_RMS_DB` / `TTS_PEAK_DB` | `-18` / `-1` | 합성 결과 라우드니스 정규화(음성 구간 RMS 목표, 피크 상한) |
| `TTS_ENGLISH_MODE` | `keep` | 독음 정규화(`server/tts_text.py`)에서 사전에 없는 영어 단어를 그대로 둠. `spell`이면 글자별로 읽음(약어·한 글자 대문자는 항상 글자별) |
| `TTS_LEXICON_USER` | `$VOICE_DATA_DIR/tts-lexicon.tsv` | 사용자 독음 사전(`영문<TAB>한글`, `#` 주석). 기본 사전 `server/tts_lexicon.tsv`보다 우선, 없으면 무시 |
| `QWEN_TTS_*` | | Qwen3-TTS 설정. `QWEN_TTS_HF_HOME`(기본 `$HF_HOME` 또는 `~/.cache/huggingface`) 등. 전체는 `server/config.py`와 `docs/프로토콜.md` 'TTS 엔진' |

### 핸즈프리(VAD·끼어들기)

| 이름 | 기본값 | 뜻 |
|---|---|---|
| `VAD_THRESHOLD` | `0.5` | 말 시작 확률 기준 |
| `VAD_START_MS` | `160` | 기준을 연속으로 넘어야 하는 시간(말 시작) |
| `VAD_END_THRESHOLD` | `0.35` | 이 아래면 침묵으로 셈 |
| `VAD_END_SILENCE_MS` | `700` | 끝 침묵이 이만큼 이어지면 말 끝 |
| `VAD_PREROLL_MS` | `300` | 말 시작 앞에 붙여 STT에 넣는 오디오 |
| `VAD_MIN_SPEECH_MS` | `300` | 이보다 짧은 발화는 버림 |
| `VAD_MAX_UTTERANCE_MS` | `60000` | 한 발화 최대 길이(넘으면 강제로 끊어 턴 처리) |
| `BARGE_THRESHOLD` / `BARGE_MS` | `0.7` / `400` | 생각·말하는 중(또는 클라이언트 `playback: speaking`)의 말 시작 기준 = 끼어들기 기준 |
| `TURN_DETECTOR` | `silence` | `smart-turn`이면 Smart Turn v3로 말 끝 보조 판정 |
| `SMART_TURN_SILENCE_MS` | `300` | smart-turn: 이만큼 조용하면 모델에 묻고, 완료면 바로 말 끝(아니면 `VAD_END_SILENCE_MS`까지 기다림) |
| `SMART_TURN_THRESHOLD` | `0.5` | smart-turn 완료 확률 기준 |
| `SMART_TURN_MODEL` | `models/smart-turn-v3.2-cpu.onnx` | 모델 경로(없으면 경고 후 침묵 기준만 씀) |
| `SILERO_VAD_MODEL` | 설치된 `silero_vad/data/silero_vad.onnx` | VAD 모델 경로 |
| `DUCK_SKIP_CORR` | `0.7` | 재생 중 말 시작 때 최근 대조 창 corr 중앙값이 이 이상(그리고 레벨 조건)이면 확실한 에코로 보고 덕킹 검사 없이 재생 유지. `0`이면 끔 |
| `DUCK_SKIP_LEVEL_DB` | `4` | 덕킹 생략 레벨 조건: 최근 창 mic_db 중앙값이 expected_db 중앙값 + 이 값 이하일 때만 |

스피커 에코 대책 관련 값(`HOLD_*`, `ECHO_*`, `DUCK_*` 등)은 `server/config.py`와 `docs/프로토콜.md` '스피커 에코 대책'을 보세요.

## 사용법

- **대화 시작**: 가운데 구체를 누르면 핸즈프리 대화가 시작됩니다. 그 뒤로는 그냥 말하면 됩니다. 말이 끝나면(약 0.7초 침묵) 받아써서 Hermes에 보내고, 답을 문장 단위로 읽어 줍니다.
- **끼어들기**: 답을 듣는 중에 말하면 답이 멈추고 새로 한 말을 다음 턴으로 처리합니다. 끝까지 못 들은 답은 "사용자가 들은 부분"을 다음 요청에 함께 알려 맥락이 이어집니다.
- **아래 버튼**: 마이크 끄기, 소리 끄기, 대화 끝내기. 화면 위쪽 `+` 버튼은 새 대화(새 Hermes 세션)를 시작합니다.
- **설정**:
  - 소리 크기(100%·150%·200%, 리미터 포함)
  - 추론 수준: 빠름(`voice-fast`) / 깊게(`voice-deep`)
  - 누르고 말하기: 구체를 누르고 있는 동안만 듣습니다. 끄면 핸즈프리.
  - 에코 제거: 스피커 소리가 내 말로 잡히지 않게 합니다. 끄면 소리가 수화부로 나와 작아집니다.
  - 답하는 동안 마이크 끄기: 켜면 끼어들 수 없습니다.
  - 알림음: 말을 알아들었을 때와 답이 오래 걸릴 때 짧은 소리.
  - 글 입력칸: 화면 아래에 글로 보내는 입력칸을 보입니다.

핸즈프리 흐름과 메시지 형식은 `docs/프로토콜.md` '양방향(핸즈프리) 모드'에 있습니다.

## iOS 제약 (홈 화면 웹앱, WebKit)

- 백그라운드에서는 마이크가 멈춥니다. WebKit의 의도된 동작이라 웹앱으로는 피할 수 없고 네이티브 앱만 가능합니다. 앱은 대화를 끝내지 않고 두었다가 화면으로 돌아오면 오디오·마이크·연결을 복구합니다.
- 홈 화면 앱은 열 때마다 마이크 권한을 묻습니다(WebKit 버그 327237). 그래서 실행 중에는 마이크 트랙을 멈추지 않고 `enabled=false`로만 끕니다. iOS 설정 › 앱 › Safari › 마이크 › 허용이 완화책으로 보고됐으나 확인하지 못했습니다.
- 마이크를 쓰는 동안은 통화 음량이 적용돼 소리가 작습니다(WebKit 버그 236219). 대화 중에 음량 버튼으로 조절하고, 설정의 '소리 크기'로 더 키울 수 있습니다.

## 선택: Qwen3-TTS

두 번째 TTS 엔진으로 Qwen3-TTS(1.7B CustomVoice 8bit, 기본 음색 `Sohee`, 스트리밍)를 쓸 수 있습니다. mlx-audio가 메인 `.venv`와 의존성이 부딪쳐 별도 venv에서 워커 프로세스(`server/qwen_tts_worker.py`)로 돕니다. 실패하거나 첫 청크가 늦은 문장은 자동으로 Supertonic으로 폴백합니다.

```sh
deploy/install-tts.sh                  # .venv-tts (Python 3.12 + mlx-audio), --force면 새로 만듦
.venv-tts/bin/hf download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
#   캐시를 다른 곳에 두려면 HF_HOME=<경로> 를 앞에 붙이고 .env에 QWEN_TTS_HF_HOME=<경로>
.venv/bin/python tests/smoke_tts_qwen.py   # 첫 청크 지연·RTF·메모리, wav는 tests/out/qwen-smoke/
```

켜기: `.env`에 `TTS_ENGINE=qwen`을 넣고 서버를 다시 시작합니다. 모델 라이선스는 Apache-2.0입니다.

## 구성

| 파일 | 역할 |
|---|---|
| `server/app.py` | FastAPI 앱: `/`(웹앱), `/ws`(음성 WebSocket), `/health` |
| `server/session.py` | 연결별 상태 기계, 턴 파이프라인, 끼어들기 |
| `server/hermes.py` | Hermes SSE 클라이언트, 음성용 지시, 가짜 스트림 |
| `server/stt.py` | mlx-whisper 상주(단일 스레드), 교정표·환각 필터 |
| `server/tts.py` | TTS 상주(단일 스레드), 엔진 선택·폴백·라우드니스 정규화 |
| `server/supertonic_helper.py` | Supertonic 추론 코드(MIT, `server/SUPERTONIC_CODE_LICENSE.txt`) |
| `server/tts_qwen.py`, `server/qwen_tts_worker.py` | (선택) Qwen3-TTS 워커 연결과 워커 본체 |
| `server/tts_text.py`, `server/tts_lexicon.tsv` | 합성 직전 독음 정규화와 기본 사전 |
| `server/vad.py` | Silero VAD ONNX 스트리밍 래퍼(`SileroVAD`), 말 시작·끝 판정(`Endpointer`) |
| `server/smart_turn.py` | (선택) Smart Turn v3 ONNX, numpy log-mel(pipecat에서 가져옴, BSD-2/Apache-2.0 표기) |
| `server/echo.py`, `server/aec_gate.py` | 스피커 에코 판정 |
| `server/sentences.py` | 스트리밍 문장 분할, 마크다운 제거 |
| `server/history.py` | 화면 대화 기록 저장(`history.json`) |
| `server/config.py` | 환경 변수 기본값 |
| `web/` | PWA (`index.html`, `app.js`, `app.css`, 오디오 워클릿 `worklet.js`·`playcore.js`, 서비스 워커 `sw.js`) |
| `deploy/` | launchd 설치 스크립트, Qwen3-TTS venv 스크립트 |
| `tests/` | 단위 시험, 스모크, e2e, 벤치 |

`web/`를 고치면 `web/sw.js`의 `VERSION`을 올려야 아이폰이 새 파일을 받습니다.

## 시험

```sh
.venv/bin/python -m pytest -q                 # 단위 시험 (tests/test_*.py)
.venv/bin/python tests/smoke_tts.py           # 문장 3개 합성 시간, tests/out/smoke_ko.wav 생성
.venv/bin/python tests/smoke_stt.py           # 위 wav를 STT로 왕복 확인
.venv/bin/python tests/bench_tts_provider.py  # TTS CPU 대 CoreML 비교

# e2e: 서버를 띄운 뒤
HERMES_MOCK=1 PORT=8481 .venv/bin/python -m server &
.venv/bin/python tests/e2e_ws.py ws://127.0.0.1:8481/ws

# 핸즈프리 e2e(실제 Hermes도 가능): 실시간 속도로 음성·무음을 흘려 보냄, 약 1분
PORT=8482 .venv/bin/python -m server &
.venv/bin/python tests/e2e_handsfree.py ws://127.0.0.1:8482/ws
```

- `e2e_ws.py`: hello → 음성 턴(transcript·assistant_delta·speak_start·오디오) → 끼어들기(interrupt 뒤 옛 generation 오디오가 없는지) → 다음 턴.
- `e2e_handsfree.py`: conversation on → 무음 → 질문 음성 → vad·transcript·speak_start·오디오 → 첫 오디오 뒤 다시 말해 `interrupt`와 옛 generation 오디오 0개, 끼어든 발화의 새 턴 → conversation off. 이어서 새 연결에서 무음+잡음 10초 동안 턴이 생기지 않는지 확인.

## 배포 (launchd 상주 + tailscale serve)

```sh
deploy/install.sh install   # launchd 서비스 hermes-voice 설치·시작 (기본 동작)
deploy/install.sh restart   # 재시작 (.env를 바꾼 뒤)
deploy/install.sh stop      # 중지(서비스 내림)
deploy/install.sh status    # 상태와 /health
deploy/install.sh serve     # tailscale serve: https://<Tailscale 이름>:8464 -> http://127.0.0.1:8480, 접속 주소 출력
```

- `install`은 `deploy/hermes-voice.plist.template`의 자리표시자를 이 폴더 경로로 채워 `~/Library/LaunchAgents/hermes-voice.plist`로 쓰고 시작합니다. 먼저 `.venv`가 있어야 합니다. 로그는 `~/Library/Logs/hermes-voice.log`.
- `serve`는 `tailscale` 바이너리를 PATH나 `/Applications/Tailscale.app`에서 찾습니다(`TAILSCALE=<경로>`로 지정 가능). 호스트 이름은 `tailscale status`에서 구하고, 안 되면 `TS_HOST=<이름>`으로 줍니다. 재부팅 뒤 serve 설정이 사라지면 다시 실행하세요.
- 경로는 스크립트 위치에서 구하므로 저장소를 어디 두어도 됩니다. 개인 설정은 모두 `.env`에 둡니다.

## 보안 주의

- 이 앱은 **Hermes 에이전트를 통째로** 음성으로 여는 창입니다. Hermes에 켜 둔 도구(터미널, 파일, 메일 등)를 접속한 사람이 말로 쓸 수 있습니다.
- 기본 바인드는 `127.0.0.1`입니다. Tailscale처럼 나만 들어오는 망으로만 여는 것을 권합니다. 공개망(포트 포워딩, 공개 터널 등)에 열 때는 **`VOICE_TOKEN`을 반드시** 길고 무작위한 값으로 설정하세요. 토큰 링크는 비밀번호처럼 다루세요.
- WebSocket에는 CORS가 적용되지 않아, 토큰 없이 띄우면 같은 기기 브라우저에서 연 아무 웹페이지가 `ws://127.0.0.1:8480/ws`에 붙어 Hermes에 명령을 넣을 수 있습니다. 그래서 서버는 `/ws` 연결 때 `Origin`을 검사해 같은 출처(접속한 주소와 같은 호스트·포트)와 `Origin`이 없는 비브라우저 클라이언트만 받고, 나머지는 거부합니다(HTTP 403). 다른 출처의 페이지에서 써야 하면 `VOICE_ALLOWED_ORIGINS`에 그 origin을 넣으세요.
- **1인용 설계**입니다. Hermes 세션과 대화 기록(`history.json`)이 앱 전체에 하나뿐이라, 여러 사람이 같은 서버에 접속하면 같은 대화를 공유합니다.
- `HERMES_API_KEY`와 `VOICE_TOKEN`은 `.env`에만 두고 git에 올리지 마세요(`.gitignore`에 포함). `STT_SAVE_CLIPS_DIR`(기본 `tests/clips`)에는 내 목소리 녹음이 쌓이니, 원치 않으면 빈 값으로 끄세요.

## 라이선스

MIT ([`LICENSE`](LICENSE)). 모델과 서드파티 코드의 라이선스는 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)에 있습니다. 특히 Supertonic 3 모델 가중치는 OpenRAIL-M입니다.
