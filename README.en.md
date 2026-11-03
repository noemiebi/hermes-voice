# Hermes Voice

English · [한국어](README.md)

A local voice server that lets you talk to [Hermes Agent](https://github.com/NousResearch/hermes-agent) running on your Mac, from an iPhone home-screen web app (PWA). Speech recognition (STT) and speech synthesis (TTS) both run on the Mac itself (mlx-whisper large-v3-turbo, Supertonic 3 ONNX (default) / Qwen3-TTS (optional)); no external speech APIs are used. Replies come from Hermes Agent's API server (the `/v1/chat/completions` SSE stream) and are synthesized and played back sentence by sentence as they arrive. Conversation is hands-free, and you can interrupt a reply just by talking over it. The assistant's name can be changed with `AGENT_NAME` (default `Herbot`).

The design lives in [`docs/설계안.md`](docs/설계안.md) (Korean), and the server/client message contract in [`docs/프로토콜.md`](docs/프로토콜.md) (Korean).

## Requirements

- An Apple Silicon Mac (mlx-whisper uses MLX)
- Python 3.12, [uv](https://docs.astral.sh/uv/)
- (Optional) If you use Qwen3-TTS: about 2 GB of disk (1.7B 8bit model) and about 4 GB of memory while running (separate worker process). See "Optional: Qwen3-TTS" below
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) installed, with its API server (`platforms.api_server`) enabled
- Safari on iPhone (used as a home-screen app)
- An HTTPS route to the server. Browsers only allow microphone access over HTTPS. [Tailscale](https://tailscale.com/) `serve` is recommended (see "Deployment" below).

## Quick start

### 1. Clone and create a virtualenv

```sh
git clone <this repository URL> hermes-voice
cd hermes-voice
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r pyproject.toml --extra dev
uv pip install --python .venv/bin/python --no-deps silero-vad   # only the VAD ONNX is used (no torch install)
```

### 2. Download models

The server reads Hugging Face in offline mode (`HF_HUB_OFFLINE=1`), so models must be downloaded ahead of time.

**Whisper (STT)**: download into the HF cache (default `~/.cache/huggingface/hub`).

```sh
.venv/bin/hf download mlx-community/whisper-large-v3-turbo
```

**Supertonic 3 (TTS, default)**: the default voice engine (voice F1, speaking rate `TTS_SPEED` 1.2x). When you use Qwen3-TTS, it is also the fallback for sentences Qwen is late on or fails. Download [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3) into `SUPERTONIC_DIR` (default `~/.hermes/voice/models/supertonic3`).

```sh
.venv/bin/hf download Supertone/supertonic-3 \
  --include "onnx/*" "voice_styles/*" LICENSE config.json \
  --local-dir ~/.hermes/voice/models/supertonic3
```

Afterwards the layout should look like this. The default voice is `voice_styles/F1.json`; pick another voice (F1–F5, M1–M5) with `SUPERTONIC_VOICE`.

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

The Supertonic 3 weights are licensed under **OpenRAIL-M**, not MIT (with use restrictions). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) (Korean).

**(Optional) Smart Turn v3**: an 8 MB model that helps decide when you've finished speaking. Only used when `TURN_DETECTOR=smart-turn`.

```sh
mkdir -p models && curl -L -o models/smart-turn-v3.2-cpu.onnx \
  https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx
```

### 3. Fill in `.env`

```sh
cp .env.example .env
```

At minimum, set `HERMES_API_KEY` (the Hermes API server key). For everything else, see the "Environment variables" tables below. `.env` is not committed to git.

### 4. Create two model routes in Hermes

The app sends requests to Hermes under two model names: `voice-fast` (the default, for quick replies) and `voice-deep` ("Deep" in settings). Define both names under the API server's `model_routes` in Hermes's `config.yaml`. Use whatever models and providers you already have configured in Hermes — for example, a small fast model for fast and a large model for deep.

```yaml
platforms:
  api_server:
    enabled: true
    extra:
      host: 127.0.0.1
      port: 8642
      key: "your-api-server-secret"     # = HERMES_API_KEY in hermes-voice
      model_routes:
        voice-fast:
          model: "<small, fast model>"
          provider: "<provider>"           # omit to use Hermes's global provider
        voice-deep:
          model: "<large model>"
          provider: "<provider>"
```

Spoken replies feel natural only if the first token arrives quickly, so a model with fast time-to-first-token is recommended for `voice-fast`. `voice-fast` requests carry a reasoning level set by `HERMES_REASONING` (default `off`).

### 5. Run and check

```sh
.venv/bin/python -m server                    # http://127.0.0.1:8480  (/ = web app, /ws, /health)
HERMES_MOCK=1 .venv/bin/python -m server      # try it without Hermes, using a fake stream
```

STT and TTS models are preloaded at startup (a few seconds for STT, around a second for TTS).

```sh
curl -s http://127.0.0.1:8480/health
```

`"ok": true` means it's ready. Without `VOICE_TOKEN`, the response also includes STT/TTS load status and `hermes.reachable` (whether the Hermes API server is reachable). With `VOICE_TOKEN` set, only `ok` is shown.

### 6. Open it on the iPhone

1. Expose the Mac's server over HTTPS. With Tailscale, `deploy/install.sh serve` maps `https://<Mac's Tailscale name>:8464` → `http://127.0.0.1:8480` and prints the URL to use. Any other reverse proxy works too, but it must be HTTPS for the microphone to open.
2. Open that URL in Safari on the iPhone and tap Share › **Add to Home Screen**. From then on, launch it from the home-screen icon.
3. If you set `VOICE_TOKEN`, open it once via the link `https://<address>/?token=<value>`. The token is stored on the device and removed from the address bar (`#token=<value>` works too). The home-screen app has storage separate from Safari, so you may need to enter the token once more in that app after adding it to the home screen.

## Environment variables (`.env`)

Path variables fall back to their defaults when left empty.

### Basics

| Name | Default | Description |
|---|---|---|
| `HERMES_API_URL` | `http://127.0.0.1:8642` | Hermes API server address |
| `HERMES_API_KEY` | (empty) | **Required.** Hermes API server key (`extra.key` / `API_SERVER_KEY`). Also needed to continue sessions |
| `HERMES_MOCK` | (empty) | `1` uses a fake stream instead of Hermes (includes one tool-start event) |
| `HERMES_REASONING` | `off` | Reasoning level for `voice-fast` requests: `off`, `minimal`, `low`, `medium`, `high`; `default` uses the Hermes setting. Not applied to `voice-deep`. Benchmark tool: `tests/bench_hermes.py` |
| `AGENT_NAME` | `Herbot` | Assistant name used on screen and in its speech |
| `MODEL_LABEL_FAST` / `MODEL_LABEL_DEEP` | (empty) | Actual model names shown under Fast/Deep in the settings sheet (e.g. `Claude Sonnet 5.5`). When set, the model is also told this on every turn. Omitted if empty |
| `VOICE_TOKEN` | (empty = no auth) | Access token. **Required** if exposed outside Tailscale (on the public internet) |
| `VOICE_ALLOWED_ORIGINS` | (empty) | Other origins allowed to connect to `/ws` (comma-separated, e.g. `https://a.example:8443`, or `*`). Same-origin clients and clients without an Origin header are always allowed |
| `VOICE_SESSION_IDLE_HOURS` | `0` (off) | If positive, starting a conversation this long after the last turn opens a new Hermes session. Never switches mid-conversation. When off, a new conversation only starts via the on-screen `+` button |
| `FILLER_ENABLED` | `1` | Short filler lines ("잠깐 확인해 볼게요." etc.) while waiting a long time for an answer. `0` turns them all off (including the one on tool start) |
| `FILLER_FIRST_S` | `4.0` | Seconds before the first filler while no token has arrived (immediately on a tool-start event) |
| `FILLER_EVERY_S` | `8.0` | Interval between further fillers until the first token. Skipped while the previous filler is still playing |
| `FILLER_MAX` | `4` | Maximum fillers per turn |
| `FILLER_STYLE` | `auto` | Filler speech level. `formal` (polite) · `casual` (banmal) · `auto` (each turn, follow the sentence endings of this session's recent answers; polite when there is nothing to go on) |
| `VOICE_DATA_DIR` | `~/.hermes/voice` | Server data folder (created if missing). Holds conversation history and the last session (`history.json`), the user pronunciation lexicon, and the STT correction table |
| `PORT` / `HOST` | `8480` / `127.0.0.1` | Bind address. Expose externally through a reverse proxy (tailscale serve, etc.) |

### STT

| Name | Default | Description |
|---|---|---|
| `STT_MODEL` | `mlx-community/whisper-large-v3-turbo` | Hugging Face model id |
| `HF_HUB_CACHE` | `$HF_HOME/hub` or `~/.cache/huggingface/hub` | Model cache. The server runs in offline mode and only reads the cache |
| `STT_CORRECTIONS` | `$VOICE_DATA_DIR/stt-corrections.tsv` | Correction table (`wrong<TAB>right`). No corrections if the file doesn't exist |
| `STT_PROMPT` | (empty = unused) | Vocabulary for whisper's `initial_prompt` (comma-separated). Off by default because it can actually hurt accuracy and cause prompt-repetition hallucinations on noise. When enabled, results that just echo the prompt are discarded |
| `STT_TEMPERATURES` | `0` | A single greedy pass. Give e.g. `0,0.2,0.4` to fall back to higher temperatures on compression-ratio/logprob failures |
| `STT_NO_SPEECH_PROB` / `STT_LOGPROB_MIN` | `0.6` / `-1.0` | Segments whose no-speech probability exceeds the first value and whose avg_logprob is below the second are discarded |
| `STT_COMPRESSION_MAX` | `2.4` | Segments with a compression ratio (repetition) above this are discarded. `0` disables |
| `STT_MAX_CHARS_PER_SEC` | `25` | Maximum characters per second of audio (anything under 1 s counts as 1 s). `0` disables |
| `STT_SAVE_CLIPS_DIR` | `tests/clips` | Saves transcribed utterances as 16 kHz wav plus a same-named `.txt` (transcript, raw text, ms, discard reason). Empty string disables. Excluded from git |
| `STT_SAVE_CLIPS_MAX` | `200` | Beyond this count, the oldest clips are deleted first |

### TTS

| Name | Default | Description |
|---|---|---|
| `TTS_ENGINE` | `supertonic` | `supertonic`: Supertonic 3 ONNX only. `qwen` (optional): Qwen3-TTS streaming with per-sentence Supertonic fallback (requires `.venv-tts`; see "Optional: Qwen3-TTS" below). Once a reply falls back, its remaining sentences stay on Supertonic so the voice never changes mid-reply. If the Supertonic model is missing, it waits for Qwen instead of falling back |
| `QWEN_TTS_LEAD_MARGIN_S` | `0.5` | (Only when `TTS_ENGINE=qwen`; the same goes for every `QWEN_TTS_*` row in this table) While earlier sentences are still playing, wait for Qwen's first chunk up to (remaining playback - this value), never less than `QWEN_TTS_FIRST_CHUNK_TIMEOUT_S` (2.5 s) |
| `QWEN_TTS_FIRST_CHUNK_MAX_S` | `12` | Upper bound (seconds) for that extended wait |
| `SUPERTONIC_DIR` | `~/.hermes/voice/models/supertonic3` | Folder containing `onnx/` and `voice_styles/` |
| `SUPERTONIC_MODEL_DIR` | `$SUPERTONIC_DIR/onnx` | |
| `SUPERTONIC_VOICE` | `$SUPERTONIC_DIR/voice_styles/F1.json` | Voice |
| `TTS_SPEED` | `1.2` | Supertonic speaking rate |
| `TTS_TAIL_MS` / `TTS_FADE_MS` | `300` / `20` | Both engines. Pause kept after each sentence (ms) and fade at the start/end of speech (ms). The pause is appended after time-stretching, so its length is exact |
| `TTS_FADE_IN_MS` | `60` | Fade-in at the start of each sentence (ms); softens abrupt sentence onsets |
| `DUCK_RAMP_DOWN_MS` / `DUCK_RAMP_UP_MS` | `60` / `250` | Gain ramps when the duck test lowers and restores playback (ms). The slow restore avoids a sudden jump in volume |
| `TTS_STEPS` | `5` | Supertonic denoising steps |
| `TTS_PROVIDER` | `cpu` | `coreml` also works, but CPU measured faster |
| `TTS_TARGET_RMS_DB` / `TTS_PEAK_DB` | `-18` / `-1` | Loudness normalization of synthesized audio (RMS target over voiced segments, peak ceiling) |
| `TTS_ENGLISH_MODE` | `keep` | In pronunciation normalization (`server/tts_text.py`, which rewrites text into Korean-readable form before synthesis), English words not in the lexicon are left as is. `spell` reads them letter by letter (acronyms and single capital letters are always spelled out) |
| `TTS_LEXICON_USER` | `$VOICE_DATA_DIR/tts-lexicon.tsv` | User pronunciation lexicon (`English<TAB>Hangul`, i.e. an English word and how to spell it out in Korean script for the TTS; `#` for comments). Takes precedence over the built-in lexicon `server/tts_lexicon.tsv`; ignored if missing |
| `QWEN_TTS_*` | | Qwen3-TTS settings, such as `QWEN_TTS_HF_HOME` (default `$HF_HOME` or `~/.cache/huggingface`). For the full list, see `server/config.py` and the "TTS 엔진" (TTS engine) section of `docs/프로토콜.md` (Korean) |

### Hands-free (VAD and barge-in)

| Name | Default | Meaning |
|---|---|---|
| `VAD_THRESHOLD` | `0.5` | Speech-start probability threshold |
| `VAD_START_MS` | `160` | How long the threshold must be exceeded continuously (speech start) |
| `VAD_END_THRESHOLD` | `0.35` | Below this counts as silence |
| `VAD_END_SILENCE_MS` | `700` | Trailing silence this long ends the utterance |
| `VAD_PREROLL_MS` | `500` | Audio before the speech start that is prepended for STT |
| `VAD_FIRST_PREROLL_MS` | `1000` | Preroll for the first utterance after the conversation starts (`conversation on`) |
| `VAD_MIN_SPEECH_MS` | `300` | Utterances shorter than this are discarded |
| `VAD_MAX_UTTERANCE_MS` | `60000` | Maximum length of one utterance (cut off and processed as a turn when exceeded) |
| `BARGE_THRESHOLD` / `BARGE_MS` | `0.7` / `400` | Speech-start threshold while thinking or speaking (or when the client reports `playback: speaking`) — i.e. the barge-in threshold |
| `TURN_DETECTOR` | `silence` | `smart-turn` uses Smart Turn v3 as an auxiliary end-of-turn check |
| `SMART_TURN_SILENCE_MS` | `300` | smart-turn: after this much silence, ask the model; if complete, end the turn immediately (otherwise wait until `VAD_END_SILENCE_MS`) |
| `SMART_TURN_THRESHOLD` | `0.5` | smart-turn completion probability threshold |
| `SMART_TURN_MODEL` | `models/smart-turn-v3.2-cpu.onnx` | Model path (if missing, logs a warning and uses the silence rule only) |
| `SILERO_VAD_MODEL` | the installed `silero_vad/data/silero_vad.onnx` | VAD model path |
| `PITCH_GATE` | `1` | When speech starts during playback, look at the pitch (F0) of its first 400 ms: in the TTS voice's range it is treated as echo and the duck test is skipped; in the user's range it barges in right away (voice profile in `VOICE_DATA_DIR/voice_profile.json`, used after 5 user utterances). `0` disables |
| `DUCK_SKIP_CORR` | `0.7` | When speech starts during playback and the median echo-gate corr of recent windows is at least this (and the level condition holds), treat it as certain echo and keep playing without the duck test. `0` disables |
| `DUCK_SKIP_LEVEL_DB` | `4` | Level condition for skipping the duck: median mic_db of recent windows must be at most median expected_db + this |

For speaker-echo countermeasure settings (`HOLD_*`, `ECHO_*`, `DUCK_*`, etc.), see `server/config.py` and the "스피커 에코 대책" (speaker echo countermeasures) section of `docs/프로토콜.md` (Korean).

## Usage

- **Start a conversation**: tap the orb in the middle to start a hands-free conversation. After that, just talk. When you stop speaking (about 0.7 s of silence), your speech is transcribed and sent to Hermes, and the reply is read out sentence by sentence.
- **Barge-in**: speak while a reply is playing and it stops; what you just said becomes the next turn. For replies you didn't hear to the end, the next request also tells the model "the part the user heard", so context carries over.
- **Bottom buttons**: mute mic, mute sound, end conversation. The `+` button at the top of the screen starts a new conversation (a new Hermes session).
- **Settings**:
  - Volume (100%, 150%, 200%, with a limiter)
  - Reasoning level: Fast (`voice-fast`) / Deep (`voice-deep`)
  - Push to talk: listens only while you hold the orb. Off means hands-free.
  - Echo cancellation: keeps the speaker output from being picked up as your speech. When off, audio plays through the earpiece and gets quieter.
  - Mute mic while replying: when on, you can't barge in.
  - Sound cues: a short sound when your speech is recognized and when a reply is taking a while.
  - Text input: shows a text box at the bottom of the screen for sending typed messages.

The hands-free flow and message formats are described in the "양방향(핸즈프리) 모드" (two-way / hands-free mode) section of `docs/프로토콜.md` (Korean).

## iOS limitations (home-screen web app, WebKit)

- The microphone stops when the app is in the background. This is intended WebKit behavior, so a web app can't work around it — only a native app could. The app keeps the conversation open and restores audio, microphone, and connection when you return to it.
- The home-screen app asks for microphone permission every time it's opened (WebKit bug 327237). That's why, while running, the mic track is never stopped, only disabled with `enabled=false`. Setting iOS Settings › Apps › Safari › Microphone › Allow has been reported as a workaround, but this hasn't been verified.
- While the microphone is in use, call volume applies and audio is quiet (WebKit bug 236219). Adjust with the volume buttons during a conversation, and boost further with "Volume" in settings.

## Optional: Qwen3-TTS

Supertonic is the default because Qwen3-TTS's tone tends to drift a little from sentence to sentence, but for a more natural voice you can use Qwen3-TTS (1.7B CustomVoice 8bit, streaming). It takes about 2 GB more disk and about 4 GB more memory while running.
Because mlx-audio's dependencies conflict with the main `.venv`, it runs as a worker process (`server/qwen_tts_worker.py`) in a separate venv, `.venv-tts`. The model is licensed under Apache-2.0.

```sh
deploy/install-tts.sh                  # .venv-tts (Python 3.12 + mlx-audio); --force recreates it
.venv-tts/bin/hf download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
#   to keep the cache elsewhere, prefix with HF_HOME=<path> and set QWEN_TTS_HF_HOME=<path> in .env
.venv/bin/python tests/smoke_tts_qwen.py   # (optional) first-chunk latency, RTF, memory; wavs go to tests/out/qwen-smoke/
```

To switch, add `TTS_ENGINE=qwen` to `.env` and restart the server.
Pick the voice with `QWEN_TTS_VOICE` (default `Sohee`) and the model cache with `QWEN_TTS_HF_HOME` (default `$HF_HOME` or `~/.cache/huggingface`). The other `QWEN_TTS_*` variables (speaking rate `QWEN_TTS_TEMPO`, sampling, and so on) are in `server/config.py` and the "TTS 엔진" (TTS engine) section of `docs/프로토콜.md` (Korean).

If Qwen fails or its first chunk is late, that sentence falls back to Supertonic, and once a reply falls back, its remaining sentences stay on Supertonic so the voice never changes mid-reply. So even with Qwen, keeping the Supertonic model (from "2. Download models") is recommended. If the Supertonic model is missing, the server waits for Qwen to finish instead of falling back, so silences can get long when Qwen is busy. The end-of-sentence pause and fades (`TTS_TAIL_MS`/`TTS_FADE_MS`) apply to both engines.

## Layout

| File | Role |
|---|---|
| `server/app.py` | FastAPI app: `/` (web app), `/ws` (voice WebSocket), `/health` |
| `server/session.py` | Per-connection state machine, turn pipeline, barge-in |
| `server/hermes.py` | Hermes SSE client, voice-specific instructions, fake stream |
| `server/stt.py` | Resident mlx-whisper (single thread), correction table and hallucination filters |
| `server/tts.py` | Resident TTS (single thread), engine selection, fallback, loudness normalization |
| `server/supertonic_helper.py` | Supertonic inference code (MIT, `server/SUPERTONIC_CODE_LICENSE.txt`) |
| `server/tts_qwen.py`, `server/qwen_tts_worker.py` | Qwen3-TTS (optional) worker connection, engine selection and fallback, and the worker itself |
| `server/tts_text.py`, `server/tts_lexicon.tsv` | Pronunciation normalization right before synthesis, and the built-in lexicon (numbers before Korean counters use native Korean numerals: "4번" -> "네 번", "11시" -> "열한 시") |
| `server/vad.py` | Silero VAD ONNX streaming wrapper (`SileroVAD`), speech start/end detection (`Endpointer`) |
| `server/smart_turn.py` | (Optional) Smart Turn v3 ONNX, numpy log-mel (taken from pipecat, BSD-2/Apache-2.0 notices) |
| `server/echo.py`, `server/aec_gate.py` | Speaker echo detection |
| `server/sentences.py` | Streaming sentence splitting, markdown stripping |
| `server/history.py` | On-screen conversation history storage (`history.json`) |
| `server/config.py` | Environment variable defaults |
| `web/` | PWA (`index.html`, `app.js`, `app.css`, audio worklets `worklet.js` and `playcore.js`, service worker `sw.js`) |
| `deploy/` | launchd install script, Qwen3-TTS venv script |
| `tests/` | Unit tests, smoke tests, e2e, benchmarks |

When you change anything in `web/`, bump `VERSION` in `web/sw.js` so the iPhone picks up the new files.

## Tests

```sh
.venv/bin/python -m pytest -q                 # unit tests (tests/test_*.py)
.venv/bin/python tests/smoke_tts.py           # synthesis time for 3 sentences, writes tests/out/smoke_ko.wav
.venv/bin/python tests/smoke_stt.py           # round-trips the wav above through STT
.venv/bin/python tests/bench_tts_provider.py  # TTS CPU vs. CoreML comparison

# e2e: after starting the server
HERMES_MOCK=1 PORT=8481 .venv/bin/python -m server &
.venv/bin/python tests/e2e_ws.py ws://127.0.0.1:8481/ws

# hands-free e2e (works with real Hermes too): streams speech and silence in real time, about 1 minute
PORT=8482 .venv/bin/python -m server &
.venv/bin/python tests/e2e_handsfree.py ws://127.0.0.1:8482/ws
```

- `e2e_ws.py`: hello → voice turn (transcript, assistant_delta, speak_start, audio) → barge-in (no audio from the old generation after interrupt) → next turn.
- `e2e_handsfree.py`: conversation on → silence → question audio → vad, transcript, speak_start, audio → speak again after the first audio to trigger `interrupt` with zero audio from the old generation, and a new turn for the interrupting utterance → conversation off. Then, on a new connection, checks that 10 seconds of silence plus noise produces no turn.

## Deployment (launchd service + tailscale serve)

```sh
deploy/install.sh install   # install and start the launchd service hermes-voice (default action)
deploy/install.sh restart   # restart (after changing .env)
deploy/install.sh stop      # stop (unload the service)
deploy/install.sh status    # status and /health
deploy/install.sh serve     # tailscale serve: https://<Tailscale name>:8464 -> http://127.0.0.1:8480, prints the URL
```

- `install` fills the placeholders in `deploy/hermes-voice.plist.template` with this folder's path, writes it to `~/Library/LaunchAgents/hermes-voice.plist`, and starts it. `.venv` must exist first. Logs go to `~/Library/Logs/hermes-voice.log`.
- `serve` looks for the `tailscale` binary on PATH or in `/Applications/Tailscale.app` (override with `TAILSCALE=<path>`). The hostname comes from `tailscale status`; if that fails, pass `TS_HOST=<name>`. If the serve config disappears after a reboot, run it again.
- Paths are derived from the script's location, so the repository can live anywhere. All personal settings go in `.env`.

## Security notes

- This app is a voice window onto **your entire Hermes agent**. Anyone who connects can use, by voice, whatever tools you've enabled in Hermes (terminal, files, mail, and so on).
- The default bind is `127.0.0.1`. Exposing it only on a network that only you can reach, like Tailscale, is recommended. When exposing it on the public internet (port forwarding, public tunnels, etc.), **always set `VOICE_TOKEN`** to a long random value. Treat the token link like a password.
- CORS does not apply to WebSockets, so if the server runs without a token, any web page opened in a browser on the same device could connect to `ws://127.0.0.1:8480/ws` and send commands to Hermes. To prevent this, the server checks `Origin` on `/ws` connections and accepts only same-origin connections (same host and port as the address being accessed) and non-browser clients that send no `Origin`; everything else is rejected (HTTP 403). If you need to use it from a page on another origin, add that origin to `VOICE_ALLOWED_ORIGINS`.
- It's a **single-user design**. There is only one Hermes session and one conversation history (`history.json`) for the whole app, so multiple people connecting to the same server share the same conversation.
- Keep `HERMES_API_KEY` and `VOICE_TOKEN` only in `.env` and never commit them to git (it's in `.gitignore`). `STT_SAVE_CLIPS_DIR` (default `tests/clips`) accumulates recordings of your voice, so set it to an empty value if you don't want that.

## Changelog

What changed in each version, and why, is in [`CHANGELOG.en.md`](CHANGELOG.en.md).

## License

MIT ([`LICENSE`](LICENSE)). Licenses for models and third-party code are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) (Korean). Notably, the Supertonic 3 model weights are OpenRAIL-M.
