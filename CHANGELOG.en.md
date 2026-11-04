# Changelog

English · [한국어](CHANGELOG.md)

What changed in each version, and why. Each entry follows symptom → cause → change. "App vNN" is the web app's service worker version; reopening the home-screen app picks up the new version. Entries that only touch the server are marked **(server)** and take effect after a server restart. Defaults and meanings of environment variables are in the [README](README.en.md#environment-variables-env).

## 0.8.0 (2026-11-04) — app v53

Bug-fix release.

**One sentence split into several turns (server)**
Pausing briefly mid-sentence split it into several turns that cut each other off, and the earlier request was lost ('오늘 일정 요약해주고' → '오늘' → …). The window for merging with the previous utterance only covered the time it was being transcribed (about 0.3 s). Now, if you keep talking within 2 s (`TURN_MERGE_WINDOW_S`) after the previous utterance ended and before the reply (first token or audio) has started, the new speech is merged with the previous utterance and transcribed together instead of counting as a barge-in. If the transcript looks unfinished (a connective ending such as '…하고'/'…는데', or a short fragment like '오늘'), the server waits another 0.7 s (`TURN_CONTINUE_WAIT_MS`) and merges what follows. The merged transcript replaces your last bubble and history item (`transcript.replace_last`, `log.replace_last`).

**Follow-up questions discarded as echo (server)**
Questions that repeat words from the answer ('…나오냐고', '…뭐야') were discarded as echo. The global similarity rule ("long, nearly identical sentence") applied regardless of playback and ignored question endings. Now speech with a question or request ending that is not in the answer is never treated as echo, and when nothing is playing, two or more words not in the answer mark it as the user's speech. The global rule only applies to speech that started during playback or within 1.5 s after it ended (`ECHO_GLOBAL_WINDOW_S`).

**Answer cut off by a wrong duck-test verdict (server)**
The duck test confirmed the assistant's own voice as "user speech" and cut the answer off (3 of 6 field cases). Now it does not cut right away: it switches to a pause `hold` to remove the echo, then barges in if the speech keeps going (`HOLD_CONFIRM_MS`) or the end-of-utterance STT is real speech, and resumes playback if it was echo. The 900 ms confirmation during ducking works the same way.

**Fillers always polite on the first turn (server)**
The filler register was based only on answer sentences, and casual endings like '…냐고/…고' were not recognized, so the first turn always used polite speech. With no answer to go on, fillers now follow the register of the user's speech.

**New turns after disconnect (server)**
After the connection dropped or the conversation was turned off, a pending decision could still start a new turn (a Hermes request). Pending decision tasks are now cancelled on disconnect and on turn-off.

**Stability (server)**
- If the speaker task died from an exception, the session could not speak again. Exceptions are now handled per sentence and the task logs when it exits.
- A new turn did not cancel the previous one, so fillers could get tangled. Fixed.
- Overall time limit for a Hermes response (`HERMES_TURN_MAX_S`, default 180 s); when exceeded, the turn ends with an error.
- History is saved off the event loop (batched over 0.5 s).
- The "heard so far" part of an interrupted-answer note is based on the actual playback position.

**No Tailscale path logs (server)**
Not a single Tailscale path log line was written. launchd provides no `TERM`, so the Tailscale app binary tried to start as the app instead of the CLI and failed. The server now passes `TERM`, cleans up the child process, and logs the reason for the first failure.

**Slow first sound after barge-in**
Right after a barge-in, the first sound of the next answer could come up to 2.5 s late. Rebuffer state was not reset when switching to a new generation, and the first answer sentence after a filler was also treated as a "stall" and rebuffered. Now the state is reset on switch, and if the queue is empty at a sentence start, playback starts immediately.

**Stuck after an iOS interruption**
If the mic track ended during an iOS interruption (phone call, Siri, etc.), it did not reconnect after the interruption ended; and if audio stayed suspended, only the "돌아오면 자동으로 이어집니다" (resumes automatically when you return) notice appeared and nothing happened. Now the mic reconnects when the interruption ends, a suspended context is resumed automatically (every 2 s, up to 5 times), the recovery ladder does not run during an interruption, and stalled playback is cleared after 3 s and stops reporting position.

**Other app fixes**
The mute and volume buttons were overridden by the duck ramp while it ran. Fixed. Unhandled Promise rejections cleaned up.

**TTS reading (server)**
Fixed number and symbol reading errors.
- '3~5도' was read as '35도' (sentence cleanup removed `~`). '3~5개' was read as '3에서 다섯 개'; now '세 개에서 다섯 개'.
- In time ranges like '10:00~12:00', '에서' was dropped; now '열 시에서 열두 시'. '24:00' is handled too.
- Hours 13–24 were read with native Korean numerals ('스무 시'); now '20:30' → '20시 30분'.
- '2번 출구', '1번 고객님' and '4대 보험' were read with native numerals as if they were counts. Now '2번 출구' stays as is, while a real count such as '4번 했어요' is still read '네 번 했어요'.
- '7시30분' written without a space was not handled; now '일곱 시 30분'.
- The year was dropped from a leading '2026. 11. 4.', and the underscore in 'my_file' was removed. Fixed.
- A single non-emoji special character (such as 𠀀) made the whole sentence drop out silently. Fixed.
- The Qwen sentence-start fade-in got shorter at chunk boundaries. Fixed.

## 0.7.0 (2026-11-03) — app v52

**Keeping and logging the Tailscale path (server)**
When the audio chopped, the Tailscale link between the Mac and the iPhone had no direct path and was going through the Tokyo relay; a single ping brought it back to the direct LAN path. During a conversation the server now runs `tailscale ping` every `NET_NUDGE_S` seconds (default 20) to prompt direct-path discovery, and logs the current path (direct/relay) whenever playback lag is detected.

**Voice silent for a while, then choppy**
The reply text appeared in full, but there was no sound at all for 10 seconds, and then it played in roughly one-second bursts (field log 23:38; same pattern at 22:25). The server had synthesized and sent three sentences within a second, yet the client had received 0 s of audio for 10 seconds, while small messages on the same connection (captions, diagnostics) and the microphone uplink were fine. Only the large binary frames (44.1 kHz int16, 88 KB/s) were late, so the cause appears to be stretches where downlink throughput drops to a few tens of KB/s (Tailscale path changes, Wi-Fi power saving, and so on). The network can't be fixed from here, so this release sends less data and stutters less when the link slows.
- **(server)** Audio is now sent at 24 kHz (`AUDIO_OUT_RATE`, default `24000`; `44100` also works), about 48 KB/s, nearly half. Supertonic output is resampled to 24 kHz after shaping; Qwen is already 24 kHz and is passed through. The echo-correlation reference, playback position and 0.5 s chunking follow this rate.
- **(server)** `AUDIO_OUT_CODEC=mulaw` sends 8-bit G.711 μ-law (24 KB/s). Default is `pcm16`. The server announces it in `ready.out_codec` and the app decodes accordingly.
- When the playback queue has run dry and audio starts arriving again, the app no longer plays each chunk immediately: it collects 1.2 s of audio or waits up to 2.5 s (or plays at once if the whole reply has arrived) and then continues. Instead of choppy half-second bursts there is one pause and then smooth playback. The status line shows "소리 받는 중…" (receiving audio) meanwhile. The first sound of a reply still plays right away.
- Diagnostics: the app's 2 s diagnostic `rx=` now includes the receive rate over the last 2 s in KB/s, `rb=` while rebuffering, and each rebuffer logs `[play] 재버퍼 N s (끊김 M회)`. **(server)** The `재생 지연` (playback lag) log now includes the bytes sent for that reply and the elapsed time, and a single audio chunk send taking over 50 ms logs `송신 지연 N ms`.

## 0.6.9 (2026-11-03) — app v51

**Removed background playback keep-alive**
The "Keep playing when the screen is off" attempt from 0.6.8 (a silent audio loop during conversations plus lock-screen media info) has been removed. It felt odd in real use, and the silent `<audio>` element risked side effects on the iOS audio session, so the feature is gone entirely, along with its settings toggle. The rest of 0.6.8 stays: no mic track re-acquisition during playback or within 3 s after it ends, and the `q=`/`rx=`/`out=`/`vis=`/`focus=` diagnostics and page-state logging.

**Telling my own voice apart by pitch (server)**
When the mic picked up speech during an answer, the server ran a duck test (dropping the volume to 20%) to tell echo from the user, and the volume dipped every time. The user's voice (median F0 about 109 Hz) and Supertonic F1 (about 218 Hz) are almost an octave apart, so the server now checks the pitch of the first 400 ms first. In the TTS range it keeps playing without ducking (the end-of-utterance text check stays, and it checks once more if the speech runs past 900 ms); in the user's range it barges in right away. Ambiguous cases fall back to the duck test as before. The TTS range is measured at server start by synthesizing four sentences in the current voice; the user range is built from transcribed utterances that did not overlap playback and kept in `voice_profile.json`. It is not used until 5 user utterances are collected. `PITCH_GATE=0` disables it.

**Gentler filler timing (server)**
Fillers felt forced, and the answer started right on the heels of a filler with no breathing room (on device: filler at 4.0 s, first token 0.5–0.8 s later). The first filler came early at 4 s, a tool-start event triggered one immediately, and nothing kept a gap between a filler and the answer. Now the first filler comes at 6 s (`FILLER_FIRST_S`), later ones every 14 s (`FILLER_EVERY_S`), at most 3 per turn (`FILLER_MAX`). A tool-start event waits another 2 s (`FILLER_TOOL_DELAY_S`) and drops the filler if the answer arrives meanwhile (never later than 6 s into the turn). When a filler was actually played, the first answer sentence starts 0.5 s (`FILLER_GAP_S`) after the filler finishes (cut short immediately on barge-in).

## 0.6.8 (2026-11-03) — app v50

**Audio kept cutting out during playback**
Speech kept breaking up while an answer was playing (iPhone, 2026-11-03 22:10). iOS normally feeds the mic as all-zero samples for seconds to tens of seconds during playback, but the dead-track recovery's "8 s during playback" rule re-acquired the mic track (`getUserMedia`) twice mid-answer, which makes iOS rebuild its audio session and briefly stalls output. The recovery now never starts while audio is playing (sources scheduled or time left in the queue) or within 3 s after playback ends; it only runs if zeros continue for another 3 s after that. To narrow down remaining causes, the 2-second diagnostic line now includes queued playback time (`q=`), received audio length (`rx=`), output latency (`out=`), visibility (`vis=`) and focus (`focus=`), and visibility changes, `pagehide`/`pageshow` and `blur`/`focus` are all logged.

## 0.6.7 (2026-11-03) — app v49

**Sentences felt like they started or ended abruptly**
When the duck test (which checks whether sound picked up by the mic during playback is just echo) fired right after a sentence began, the output gain dropped to 20% within 20 ms and came back within 20 ms, which sounded sudden; the first syllable of the next sentence also felt abrupt. The duck gain ramps are now 60 ms down and 250 ms back up (`DUCK_RAMP_DOWN_MS`/`DUCK_RAMP_UP_MS`, sent by the server in `hold`/`resume`), and the sentence fade-in is 60 ms (`TTS_FADE_IN_MS`).

**Fillers during long waits (server)**
When a search or tool call ran long, you waited 10 to almost 30 seconds in silence for the first answer (measured first tokens: 8.9, 18.9, 27.6 s). The only filler was a single line on tool start ("잠깐 확인해 볼게요."). Now, before the first token, the first filler plays immediately on a tool-start event or after 4 seconds (`FILLER_FIRST_S`) without one, then one more every 8 seconds (`FILLER_EVERY_S`) until the answer arrives, up to 4 per turn (`FILLER_MAX`). Lines follow the tool in use ("웹에서 찾아보는 중이에요, 조금만요." — searching the web; "메일을 보고 있어요, 조금만요." — reading mail, etc.), otherwise generic lines ("아직 찾는 중이에요, 조금만요." → "시간이 좀 걸리네요, 잠시만요." …) without repeats. When the answer arrives, fillers not yet spoken are skipped, and fillers are left out of the cut-answer note (what the user heard). `FILLER_ENABLED=0` turns them all off.

**Fillers follow the agent's speech level (server)**
When the Hermes persona answered in casual Korean (banmal), the fillers alone stayed polite ("잠깐 확인해 볼게요."), which sounded off. There was only one, polite, set of filler lines and the app has no polite/casual setting. Now a casual set ("잠깐 확인해 볼게.", "웹에서 찾아보는 중이야, 조금만.", "거의 다 됐어." etc.) mirrors the polite set one-to-one, chosen by `FILLER_STYLE` (default `auto`). `auto` counts, at the start of each turn, the sentence endings of the last 6 sentences the agent said in this session (`요`, `니다`… vs `어`, `야`, `지`…) and uses the majority; ties or no evidence fall back to polite. `formal` or `casual` pins it.

## 0.6.6 (2026-11-03) — app v48

**Default voice engine is now Supertonic 3 (server)**
Qwen3-TTS tended to shift tone from sentence to sentence, and tuning its sampling still did not make it satisfying in everyday use. The default is now `TTS_ENGINE=supertonic` (Supertonic 3, voice F1), and the `TTS_SPEED` default is 1.2. The end-of-sentence pause and fades (`TTS_TAIL_MS` 300 ms, `TTS_FADE_MS` 20 ms) apply to both engines. Qwen3-TTS remains an optional engine, enabled with `TTS_ENGINE=qwen` when you want a more natural voice (about 4 GB of memory, separate venv).

**Sentence endings cut off (server)**
Soft final consonants and last syllables sometimes sounded clipped, as confirmed in device logs and listening tests. The time-stretcher (WSOLA) drops its last leftover frame on flush, and this combined with a regression that left almost no silence at the end of each sentence. The audio is now padded with silence before stretching so that the dropped part is silence, the leading/trailing silence trimmer keeps soft tails (about 120 ms), and the end-of-sentence pause is appended after stretching and resampling at an exact length (`TTS_TAIL_MS`, default 300 ms). The fade (`TTS_FADE_MS`, default 20 ms) is applied at the actual end of the sound, so sentence transitions are smoother.

**Tone varies from sentence to sentence (server)**
Even within one reply, timbre and intonation could shift between sentences. Qwen3-TTS samples each sentence independently (model default temperature 0.9). The sampling parameters `QWEN_TTS_TEMPERATURE`, `QWEN_TTS_TOP_K`, `QWEN_TTS_TOP_P` are now exposed, along with `QWEN_TTS_TURN_SEED` to fix the seed within a reply. All defaults stay at the model's defaults (seed fixing is off), and a benchmark script for comparing tone was added.

**False "microphone stopped" detection**
On iOS, a system mute or an audio interruption (a phone call, another app's audio, etc.) could trigger the "microphone stopped" recovery and show a failure notice. The microphone sends zero samples during these states, and that was being judged as a dead track. Silence detection is now suspended while the track is muted or the audio context is interrupted/suspended, and recovery is re-enabled once the state returns to normal. The failure notice is also worded less definitively.

**Same sentence-end treatment for Supertonic (server)**
With Supertonic 3 there was no sentence-end pause or fade, so transitions between sentences felt abrupt. Both engines now share `TTS_TAIL_MS` and `TTS_FADE_MS` (the old `QWEN_TTS_*` names still work), so Supertonic sentences also get a fade at the start and end of speech and a pause of exact length.

## 0.6.5 (2026-11-03) — app v47

**No sound after reconnecting**
After the connection dropped and reconnected, reply text arrived but no audio played. On reconnect the server's reply numbering restarts at 0, but the client still held the numbers it had marked as "interrupted replies" on the previous connection, so it discarded the new reply's audio. The list of interrupted replies and the playback state are now reset when the server signals it is ready.

**Uneven speaking rate (server)**
At higher speeds (e.g. 1.3x) speech felt like it sped up and slowed down from one stretch to the next. The time-stretcher (WSOLA) used 20 ms frames, which cut syllables unevenly. After listening comparisons, the frame is now 40 ms (`QWEN_TTS_TSM_FRAME_MS`) and the search range ±15 ms (`QWEN_TTS_TSM_TOL_MS`).

## 0.6.4 (2026-11-03) — app v46

**Microphone-preparing indicator and ready chime**
In the first conversation, people sometimes started talking before the microphone was ready and lost the beginning (see 0.6.3). Turning on the conversation now shows that the microphone is being prepared, and a ready chime plays only if it took more than 300 ms after the tap. If the microphone is ready right away, no sound is played.

## 0.6.3 (2026-11-03) — app v45

**Beginning of the first sentence missing**
The start of the first sentence in the first conversation did not show up in the transcript. Acquiring the microphone (including the permission check) takes time in the first conversation, and whatever was said during that time was kept neither on the client nor on the server. The client now sends a pre-buffer collected right after the tap (up to 2 s), and the pre-roll prepended to the start of speech is longer (`VAD_PREROLL_MS` 500 ms; `VAD_FIRST_PREROLL_MS` 1000 ms for the first utterance after a conversation starts).

**Stop repeating silent-track recovery**
If silent-track recovery fails, it is not retried within the same conversation, so repeated recovery attempts no longer get in the way.

## 0.6.2 (2026-11-03) — app v44

**Earlier detection of silent tracks**
After a long time in the background, the microphone track could be live yet send only zeros, so nothing was heard. The old detection kicked in only after 8 s, which was too late. Now, if zero samples continue for 2 s after a conversation starts (1 s after returning from the background), the track is re-acquired, and if that does not help, a new audio context is created. The diagnostic line shows the zero-sample ratio (`zeros=`).

**Replies cut off by a wrong ducking baseline (server)**
In the first conversation, replies often cut themselves off right after they started. Right after the first playback the iOS microphone sends a few seconds of complete silence; the ducking check took that stretch as its baseline level and treated the microphone coming back (echo) as "the user speaking". Silent windows (at or below `MIC_SILENCE_DB`) are now excluded from the baseline, a decision is made only with enough baseline windows (`DUCK_PRE_MIN_WINDOWS`), and the microphone level must exceed the expected echo level by `DUCK_USER_MARGIN_DB` (6 dB) to count as user speech.

**Stronger echo text filter (server)**
Speaker output (echo) was still occasionally transcribed and created bogus turns. The reply text (e.g. "11월") differs from what was actually spoken (the reading, "십일월"), so it slipped past the text comparison, and in the first conversation the echo delay estimate had not been learned yet. Both the original text and the reading are now compared, and utterances under echo check use fuzzy similarity (`ECHO_FUZZY_RATIO`, 0.75). Echo delay and gain estimates are saved to `aec_state.json` (`AEC_STATE_PATH`) and used from the first conversation of the next run.

## 0.6.1 (2026-11-03) — app v43

**Three-step microphone recovery**
After returning from the background, the microphone sometimes stopped hearing anything. The audio context reported running while its clock was actually frozen, so audio processing (the worklet) never ran, and the existing recovery did not catch this. The client now checks whether the context clock has stopped and recovers in order: rebuild the audio graph → new microphone track → new audio context (at most twice per conversation). If a user gesture is required, it continues on the next tap. The diagnostic line shows context clock progress (`ct=`).

**Keep one voice within a reply (server)**
Qwen3-TTS and Supertonic voices were mixed within a single reply. When the server was busy and Qwen's first audio chunk took longer than 2.5 s, that sentence fell back to Supertonic, and this repeated sentence by sentence. Qwen's first chunk is now given extra time equal to the audio already queued for playback (`QWEN_TTS_LEAD_MARGIN_S`, capped at `QWEN_TTS_FIRST_CHUNK_MAX_S`), and once a reply falls back, it stays on Supertonic to the end. The fallback count is shown in `/health`.

**Fewer ducking checks (server)**
The volume often dipped briefly (about 0.3 s) while a reply was playing. The check that lowers the volume to see whether incoming sound is the user speaking (ducking) ran often, and about 70% of those runs were echo wrongly suspected as speech. When echo correlation is confident that it is echo (`DUCK_SKIP_CORR`, `DUCK_SKIP_LEVEL_DB`), the volume is no longer lowered and the decision is made by correlation after the utterance ends. Ducking still happens if the sound goes on long and non-echo windows accumulate.

**Don't transcribe confirmed echo (server)**
The tail of an utterance already confirmed as echo by the ducking check could be transcribed, creating bogus turns or cutting off the reply. Utterances confirmed as echo are now discarded without transcription unless the level rises clearly afterwards (`DUCK_ECHO_RISE_DB`, `DUCK_ECHO_RISE_MS`). Suspected echo during playback interrupts the reply only after the transcription verdict; if it turns out to be echo, playback simply continues. Short fragments that contain only numbers/years, or partially match recent reply sentences, are also filtered as echo.

## 0.6.0 (2026-11-03) — app v42

First public release. A local voice server for talking to Hermes Agent on your Mac from an iPhone home-screen web app, with mlx-whisper transcription, Silero VAD and Smart Turn end-of-turn detection, barge-in, server-side conversation history, and launchd/tailscale serve deployment scripts. An English README is included.

**Qwen3-TTS as the default TTS**
Qwen3-TTS is now the default engine (`TTS_ENGINE=qwen`), with Supertonic kept as the fallback and a lightweight alternative. The server starts even without the Supertonic model, and in that case waits for Qwen instead of falling back. A pre-flight check was added to the install script.

**Native Korean numerals before counting units**
"4번" was read with Sino-Korean numerals ("사 번"), which sounded unnatural for counts. Numbers 1–99 before counting units are now read with native Korean numerals (네 번, 한 번, 열한 시). Numbers before ordinal or Sino-Korean units are read as before.

**Access token and Origin check**
CORS does not apply to WebSockets, so any web page on the same device could connect to the server and send commands to Hermes. An optional access token (`VOICE_TOKEN`) and an `Origin` check on `/ws` connections were added, accepting only same-origin and non-browser clients. Add other origins to `VOICE_ALLOWED_ORIGINS` if needed.
