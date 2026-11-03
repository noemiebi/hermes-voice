"""환경 변수 기본값 (docs/프로토콜.md)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

HERMES_API_URL = os.environ.get("HERMES_API_URL", "http://127.0.0.1:8642").rstrip("/")
HERMES_API_KEY = os.environ.get("HERMES_API_KEY", "")
# 선택적 접속 토큰: 비어 있으면 인증 없음(Tailscale 안 기본). 설정하면 /ws 첫 메시지 hello의 token이 일치해야 한다.
VOICE_TOKEN = os.environ.get("VOICE_TOKEN", "").strip()
# /ws Origin 허용 목록(쉼표 구분, 정확한 origin 문자열 예: "https://a.example:8443", 또는 "*"). 기본 빈 값:
# Origin 없음(브라우저 아닌 클라이언트)과 같은 출처(Origin host == Host/X-Forwarded-Host)만 허용(app._origin_allowed).
VOICE_ALLOWED_ORIGINS = tuple(o.strip().rstrip("/") for o in os.environ.get("VOICE_ALLOWED_ORIGINS", "").split(",") if o.strip())
HERMES_MOCK = os.environ.get("HERMES_MOCK", "") not in ("", "0", "false", "no")
# voice-fast 요청의 추론(thinking) 단계: off | minimal | low | medium | high | default(헤르메스 설정값).
# 측정상 off가 첫 토큰이 가장 빠르고 답도 멀쩡함(tests/bench_hermes.py). voice-deep에는 적용 안 함.
HERMES_REASONING = os.environ.get("HERMES_REASONING", "off").strip().lower()
# 마지막 턴 뒤 이 시간이 지나면 다음 대화 시작 때 새 헤르메스 세션(0이면 끔).
# 기본 끔: 새 대화는 '+' 버튼으로만(사용자 결정 2026-11-03).
VOICE_SESSION_IDLE_HOURS = float(os.environ.get("VOICE_SESSION_IDLE_HOURS", "0"))
# (경로 env는 빈 값이면 기본값을 쓴다: .env.example의 빈 줄 그대로 둬도 됨)
# 서버 데이터 폴더(화면 대화 기록 history.json, 사용자 독음 사전 tts-lexicon.tsv). 없으면 만든다.
VOICE_DATA_DIR = Path(os.environ.get("VOICE_DATA_DIR") or str(Path.home() / ".hermes" / "voice")).expanduser()
VOICE_DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH = VOICE_DATA_DIR / "history.json"

STT_MODEL = os.environ.get("STT_MODEL", "mlx-community/whisper-large-v3-turbo")
# Hugging Face 캐시: 기본은 HF 표준 위치(HF_HOME 또는 ~/.cache/huggingface 아래 hub/). 다른 디스크에 두려면 env로.
_HF_HOME_DEFAULT = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
HF_HUB_CACHE = os.environ["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE") or str(Path(_HF_HOME_DEFAULT) / "hub")
# 모델은 이미 캐시에 있음: 네트워크로 다시 받지 않는다.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# STT 교정표(잘못<TAB>올바름). 파일이 없으면 교정 안 함.
STT_CORRECTIONS = os.environ.get("STT_CORRECTIONS") or str(VOICE_DATA_DIR / "stt-corrections.tsv")
# whisper initial_prompt 어휘(쉼표 구분). 기본 빈 값 = 안 씀: 2026-11-02 벤치(tests/bench_stt.py)에서 깨끗한 음성 CER이
# 2.1% -> 4.7%로 나빠지고(지메일->Gmail, 세어->세워), 잡음 클립에서 "네이버 메일, 네이버 메일…" 반복 환각을 만들었다.
# 켜면 프롬프트를 그대로 뱉은 결과는 버린다(stt.is_prompt_echo). 예: "헤르메스, 카카오톡, 클로드, 오픈클로"
STT_PROMPT = os.environ.get("STT_PROMPT", "")
# 온도 폴백: "0"이면 끔(그리디 1회). 예: "0,0.2,0.4"면 압축률·logprob 기준 실패 때 다시 디코딩.
STT_TEMPERATURES = tuple(float(x) for x in os.environ.get("STT_TEMPERATURES", "0").split(",") if x.strip())
STT_NO_SPEECH_PROB = float(os.environ.get("STT_NO_SPEECH_PROB", "0.6"))     # 세그먼트 버림: 무음 확률 초과
STT_LOGPROB_MIN = float(os.environ.get("STT_LOGPROB_MIN", "-1.0"))          # ... 그리고 avg_logprob 미만
STT_COMPRESSION_MAX = float(os.environ.get("STT_COMPRESSION_MAX", "2.4"))   # 세그먼트 버림: 반복(압축률) 초과
STT_MAX_CHARS_PER_SEC = float(os.environ.get("STT_MAX_CHARS_PER_SEC", "25"))  # 오디오 1초당 글자 수 상한
# 받아쓴 발화 저장(16 kHz wav + 같은 이름 .txt). 빈 문자열이면 끔. 최근 STT_SAVE_CLIPS_MAX개만 유지.
STT_SAVE_CLIPS_DIR = os.environ.get("STT_SAVE_CLIPS_DIR", str(ROOT / "tests" / "clips"))
STT_SAVE_CLIPS_MAX = int(os.environ.get("STT_SAVE_CLIPS_MAX", "200"))

# Supertonic 3 모델 폴더(onnx/, voice_styles/ 를 담은 곳). 기본 ~/.hermes/voice/models/supertonic3
_ST = str(Path(os.environ.get("SUPERTONIC_DIR") or str(Path.home() / ".hermes" / "voice" / "models" / "supertonic3")).expanduser())
SUPERTONIC_MODEL_DIR = os.environ.get("SUPERTONIC_MODEL_DIR") or f"{_ST}/onnx"
SUPERTONIC_VOICE = os.environ.get("SUPERTONIC_VOICE") or f"{_ST}/voice_styles/F1.json"
TTS_SPEED = float(os.environ.get("TTS_SPEED", "1.3"))  # 2026-11-02 사용자 요청으로 1.1 → 1.3
TTS_STEPS = int(os.environ.get("TTS_STEPS", "5"))
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "cpu").lower()  # cpu | coreml
# 합성 결과 라우드니스 정규화: 음성 구간(-40 dBFS 넘는 32 ms 프레임) RMS를 이 값으로, 피크는 TTS_PEAK_DB 이하.
TTS_TARGET_RMS_DB = float(os.environ.get("TTS_TARGET_RMS_DB", "-18"))
TTS_PEAK_DB = float(os.environ.get("TTS_PEAK_DB", "-1"))
# TTS 엔진: qwen(기본, Qwen3-TTS 스트리밍, 별도 venv 워커. 실패·지연 시 문장 단위로 Supertonic 폴백,
# Supertonic 모델이 없으면 폴백 없이 Qwen을 기다림) | supertonic(가벼운 대안: 메모리 적고 로드 빠름)
TTS_ENGINE = os.environ.get("TTS_ENGINE", "qwen").strip().lower()
QWEN_TTS_MODEL = os.environ.get("QWEN_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")
QWEN_TTS_VOICE = os.environ.get("QWEN_TTS_VOICE", "Sohee")
QWEN_TTS_PYTHON = os.environ.get("QWEN_TTS_PYTHON", str(ROOT / ".venv-tts" / "bin" / "python"))
QWEN_TTS_HF_HOME = os.environ.get("QWEN_TTS_HF_HOME") or _HF_HOME_DEFAULT  # 모델은 그 아래 hub/
QWEN_TTS_FIRST_CHUNK_TIMEOUT_S = float(os.environ.get("QWEN_TTS_FIRST_CHUNK_TIMEOUT_S", "2.5"))
# 앞 문장이 아직 재생 중이면 첫 청크를 (남은 재생 시간 - LEAD_MARGIN)까지 더 기다린다(상한 FIRST_CHUNK_MAX)
QWEN_TTS_LEAD_MARGIN_S = float(os.environ.get("QWEN_TTS_LEAD_MARGIN_S", "0.5"))
QWEN_TTS_FIRST_CHUNK_MAX_S = float(os.environ.get("QWEN_TTS_FIRST_CHUNK_MAX_S", "12"))
QWEN_TTS_STREAM_INTERVAL = float(os.environ.get("QWEN_TTS_STREAM_INTERVAL", "0.32"))  # 워커 청크 간격(초)
# 고정 이득: 벤치 샘플(tests/out/voices/qwen3tts-1.7B-8bit_Sohee_1~3.wav) 음성 구간 RMS -21.8/-20.5/-19.2 dBFS
# -> -18 dBFS 목표 대비 +3.8/+2.5/+1.2 dB, 중앙값 +2.5. 피크는 소프트 리미터로 -1 dBFS 이하.
QWEN_TTS_GAIN_DB = float(os.environ.get("QWEN_TTS_GAIN_DB", "2.5"))
# 말투 지시(1.7B CustomVoice만, 0.6B는 무시됨). mlx-audio의 speed 인자는 Qwen3-TTS에서 쓰이지 않음(TTS_SPEED는 Supertonic 전용).
# 말 속도: 음높이를 유지하는 시간 신축(WSOLA, audiotsm)을 스트리밍 청크에 적용. 1.0이면 우회.
QWEN_TTS_TEMPO = float(os.environ.get("QWEN_TTS_TEMPO", "1.2"))
QWEN_TTS_INSTRUCT = os.environ.get("QWEN_TTS_INSTRUCT", "").strip()
# 독음 정규화(server/tts_text.py): 사전에 없는 영어 단어를 그대로(keep) 둘지 글자별로(spell) 읽을지, 사용자 사전 경로
TTS_ENGLISH_MODE = os.environ.setdefault("TTS_ENGLISH_MODE", "keep").strip().lower()
TTS_LEXICON_USER = os.environ.setdefault("TTS_LEXICON_USER", str(VOICE_DATA_DIR / "tts-lexicon.tsv"))

# 스피커 에코 대책 (docs/프로토콜.md '에코 대책')
HOLD_CONFIRM_MS = int(os.environ.get("HOLD_CONFIRM_MS", "900"))   # 잠정 끼어들기 뒤 이만큼 말이 이어지면 확정
ECHO_TAIL_MS = int(os.environ.get("ECHO_TAIL_MS", "1500"))        # 재생 끝난 뒤 에코 꼬리 가드 시간
ECHO_MEMORY_S = float(os.environ.get("ECHO_MEMORY_S", "60"))      # 에코 비교용 문장 보관(그 문장 추정 재생 종료부터)
# pos_ms 보내는 클라이언트: 마지막 재생 위치 보고가 이 시간 안이어야 "재생 중"
PLAYBACK_REPORT_TIMEOUT_S = float(os.environ.get("PLAYBACK_REPORT_TIMEOUT_S", "1.5"))
# 잠정 끼어들기 방식: duck(재생을 낮춰 마이크가 따라 내려가는지 능동 검사, 기본) | pause(재생 일시정지, 예전 동작)
HOLD_MODE = os.environ.get("HOLD_MODE", "duck").strip().lower()
DUCK_GAIN = float(os.environ.get("DUCK_GAIN", "0.2"))  # 실기: 0.3에서 에코 하강 6.2 dB로 임계(6)에 아슬아슬 → 더 깊게(출력 이득 배수, 0.2 ≈ -14 dB)
DUCK_TEST_DELAY_MS = int(os.environ.get("DUCK_TEST_DELAY_MS", "180"))  # hold 뒤 이만큼 지나서부터
DUCK_TEST_MS = int(os.environ.get("DUCK_TEST_MS", "300"))              # 이 시간 동안 마이크 평균을 잰다
DUCK_DROP_DB = float(os.environ.get("DUCK_DROP_DB", "6"))              # hold 전보다 이만큼 내려가면 에코
# 덕킹 검사에서 이 레벨(dBFS) 이하 창은 디지털 무음(iOS가 재생 시작 직후 마이크를 0으로 보냄)으로 보고 pre/post 평균에서 뺀다
MIC_SILENCE_DB = float(os.environ.get("MIC_SILENCE_DB", "-65"))
# pre(또는 연장 뒤 post) 유효 창이 이보다 적으면 판정하지 않고 재생 복귀(말 끝 STT 에코 필터로)
DUCK_PRE_MIN_WINDOWS = int(os.environ.get("DUCK_PRE_MIN_WINDOWS", "4"))
# "사용자 말" 판정은 post가 hold 때 게이트 expected_db(에코 예상 레벨) + 이 값보다 커야 함(expected가 NaN이면 생략)
DUCK_USER_MARGIN_DB = float(os.environ.get("DUCK_USER_MARGIN_DB", "6"))
# 확실한 에코면 덕킹 검사 생략: 최근 DUCK_SKIP_WINDOWS창(판정된 것) corr 중앙값 ≥ DUCK_SKIP_CORR이고
# mic_db 중앙값 ≤ expected_db 중앙값 + DUCK_SKIP_LEVEL_DB면 hold 없이 재생 유지, 말 끝에서 대조로 판정. 0이면 끔
DUCK_SKIP_CORR = float(os.environ.get("DUCK_SKIP_CORR", "0.7"))
DUCK_SKIP_LEVEL_DB = float(os.environ.get("DUCK_SKIP_LEVEL_DB", "4.0"))
DUCK_SKIP_WINDOWS = int(os.environ.get("DUCK_SKIP_WINDOWS", "6"))
# 생략 뒤 말이 HOLD_CONFIRM_MS 넘게 이어지고 비에코 창이 이만큼 나오면 그때 hold(덕킹 검사)
DUCK_SKIP_REVOKE_WINDOWS = int(os.environ.get("DUCK_SKIP_REVOKE_WINDOWS", "4"))
# 덕킹 검사가 에코로 끝난(재생 복귀) 발화의 나머지: 복귀 뒤 마이크가 덕킹 pre 레벨보다 DUCK_ECHO_RISE_DB 이상
# 높은 구간이 DUCK_ECHO_RISE_MS 이상 이어지지 않았으면 말 끝에서 받아쓰기 없이 버림(이어졌으면 STT + 텍스트 에코 필터)
DUCK_ECHO_RISE_DB = float(os.environ.get("DUCK_ECHO_RISE_DB", "10"))
DUCK_ECHO_RISE_MS = int(os.environ.get("DUCK_ECHO_RISE_MS", "300"))
# 짧은 에코 조각 완화 필터: 재생 중(또는 꼬리) 시작된 발화가 음성 SHORT_ECHO_MS 이하면 STT 결과를 완화 기준으로 한 번 더
# 거름(숫자·단위만 있는 전사는 최근 문장 숫자와 맞으면 에코). 게이트 에코 비율이 SHORT_ECHO_FRAC 이상이면
# bigram 완화(SHORT_ECHO_BIGRAM, 최근 2문장)도 씀
SHORT_ECHO_MS = int(os.environ.get("SHORT_ECHO_MS", "1200"))
SHORT_ECHO_FRAC = float(os.environ.get("SHORT_ECHO_FRAC", "0.3"))
SHORT_ECHO_BIGRAM = float(os.environ.get("SHORT_ECHO_BIGRAM", "0.5"))
# 매 턴 사용자 글 앞에 붙이는 음성용 스타일 노트(빈 문자열이면 끔)
AGENT_NAME = os.environ.get("AGENT_NAME", "Herbot").strip() or "Herbot"  # 음성 대화 상대 이름(헤더·스타일 노트)
# 빠름/깊게 라우트에 실제로 붙은 모델의 표시 이름(설정 시트·스타일 노트). 비우면 표시·노트 생략
MODEL_LABEL_FAST = os.environ.get("MODEL_LABEL_FAST", "").strip()
MODEL_LABEL_DEEP = os.environ.get("MODEL_LABEL_DEEP", "").strip()
MODEL_LABELS = {k: v for k, v in (("voice-fast", MODEL_LABEL_FAST), ("voice-deep", MODEL_LABEL_DEEP)) if v}
VOICE_STYLE_NOTE = os.environ.get(
    "VOICE_STYLE_NOTE", "(음성 대화. 네 이름은 {agent}이야. 3문장 이내로 짧게 답하고, 내용이 많으면 핵심만 말한 뒤 더 들을지 물어봐.)"
).strip().replace("{agent}", AGENT_NAME)
# 독음 지시는 넣지 않는다: 헤르메스가 화면 글자까지 한글 독음으로 써 버림(2026-11-02 피드백). 독음은 합성 직전 tts_text가 함.
VOICE_INPUT_LOG = os.environ.get("VOICE_INPUT_LOG", "") not in ("", "0", "false", "no")  # 1초마다 수신 레벨 로그

PORT = int(os.environ.get("PORT", "8480"))
HOST = os.environ.get("HOST", "127.0.0.1")
WEB_DIR = ROOT / "web"
