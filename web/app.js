// 헤르메스 음성 PWA (1단계: 핸즈프리 대화 기본, 누르고 말하기는 설정)
// 계약: docs/프로토콜.md
// 출력 유지(keep-alive): 대화 중 earBus에 -100 dBFS 상수 신호를 흘려 iOS가 재생 끝 무음 때 마이크를 mute하는 것을 줄인다.
//   끄려면 Safari 콘솔(또는 웹 검사기)에서 localStorage.setItem('hv.keepalive', '0') 후 새로 고침. 다시 켜려면 removeItem.
'use strict';

const APP_VERSION = '0.6.6';
const AGENT_NAME_DEFAULT = 'Herbot'; // 서버 ready의 agent_name이 오면 그걸 쓴다
const KEEPALIVE_LEVEL = 1e-5; // 약 -100 dBFS
const DIAG_BUF_MAX = 20; // WS가 닫힌 동안 모아 둘 진단 줄 수
const OUT_RATE_DEFAULT = 44100;
const TAP_MS = 300; // 이보다 짧게 누르면 토글 모드
const LS = { session: 'hv.session', model: 'hv.model', aecOff: 'hv.aecOff', ptt: 'hv.ptt', halfDuplex: 'hv.halfDuplex', earcon: 'hv.earcon', showText: 'hv.showText', boost: 'hv.boost', log: 'hv.log', keepalive: 'hv.keepalive', token: 'hv.token' };

const $ = (id) => document.getElementById(id);
const el = {
  conn: $('conn'), connText: $('connText'), log: $('log'), hint: $('hint'),
  status: $('status'), dock: $('dock'), bars: [...$('bars').children],
  orbCore: $('orbCore'), orbRing: $('orbRing'), orbPulse: $('orbPulse'),
  talk: $('talk'), micBtn: $('micBtn'), endBtn: $('endBtn'),
  soundBtn: $('soundBtn'), barsWrap: $('bars'),
  textForm: $('textForm'), textInput: $('textInput'), toast: $('toast'),
  menu: $('menu'), menuBtn: $('menuBtn'), menuClose: $('menuClose'),
  newSession: $('newSession'), aec: $('aec'), info: $('info'), infoSummary: $('infoSummary'), sheetVer: $('sheetVer'), appName: $('appName'),
  pttMode: $('pttMode'), halfDuplex: $('halfDuplex'), earcon: $('earcon'), showText: $('showText'),
  boostBtns: [...document.querySelectorAll('.seg-btn[data-boost]')],
  tokenForm: $('tokenForm'), tokenInput: $('tokenInput'),
};

const STATE_LABEL = {
  idle: '', waiting: '말씀하세요', listening: '듣는 중', transcribing: '생각 중', thinking: '생각 중', speaking: '말하는 중',
};

// 도구 이름 → 상태 문구용 한국어
const TOOL_KO = {
  weather: '날씨', apple_calendar: '캘린더', apple_reminders: '미리 알림', apple_notes: '메모', gmail_readonly: 'Gmail',
  naver_mail: '네이버 메일', korea_workday: '근무일', terminal: '터미널', process_manage: '터미널', web_search: '웹 검색',
  web_extract: '웹 검색', memory: '기억', session_search: '지난 대화', cronjob_manage: '예약',
};
// 헤르메스 내부 도구(모델이 도구 설명을 읽는 단계 등)는 사용자에게 보여 줄 가치가 없어 숨긴다.
const TOOL_HIDDEN = new Set(['tool_describe', 'tool_search', 'tool_call']);
// MCP 도구 이름은 `mcp__<서버>__<도구>` 꼴이라 서버 이름으로 매핑한다.
function toolKo(name) {
  if (!name) return '';
  const m = /^mcp__([^_]+(?:_[^_]+)*?)__(.+)$/.exec(name);
  if (m) return TOOL_KO[m[1]] || m[1];
  return TOOL_KO[name] || name;
}
const toolHidden = (name) => TOOL_HIDDEN.has(name);
const isBusy = (s) => s === 'transcribing' || s === 'thinking';
const reduceMotion = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };

// 접속 토큰(서버 VOICE_TOKEN). 주소의 ?token= 또는 #token= 으로 한 번 받아 저장하고 주소에서 지운다
// (아이폰: 링크를 한 번 열어 홈 화면에 추가하면 토큰이 주소·기록에 남지 않는다).
function takeUrlToken() {
  try {
    const q = new URLSearchParams(location.search);
    const h = new URLSearchParams(location.hash.replace(/^#/, ''));
    const tok = q.has('token') ? q.get('token') : (h.has('token') ? h.get('token') : null);
    if (tok === null) return;
    const v = tok.trim();
    if (v) localStorage.setItem(LS.token, v); else localStorage.removeItem(LS.token);
    q.delete('token');
    h.delete('token');
    const qs = q.toString(), hs = h.toString();
    history.replaceState(history.state, '', location.pathname + (qs ? '?' + qs : '') + (hs ? '#' + hs : ''));
  } catch { /* localStorage·history 불가 환경 무시 */ }
}
takeUrlToken();
function getToken() {
  try { return localStorage.getItem(LS.token) || ''; } catch { return ''; }
}

const st = {
  ws: null,
  authFail: false, // 서버가 토큰을 거부함(unauthorized / close 4401): 자동 재연결 멈춤, 토큰 저장 시 재연결
  wsOpen: false,
  retry: 0,
  retryTimer: 0,
  sessionId: localStorage.getItem(LS.session) || null,
  model: localStorage.getItem(LS.model) || 'voice-fast',
  aecOff: localStorage.getItem(LS.aecOff) === '1',
  ptt: localStorage.getItem(LS.ptt) === '1', // 누르고 말하기 모드(보조)
  halfDuplex: localStorage.getItem(LS.halfDuplex) === '1',
  earcon: localStorage.getItem(LS.earcon) !== '0', // 알림음(기본 켬)
  showText: localStorage.getItem(LS.showText) === '1', // 글 입력칸(기본 숨김: 조작부가 내려와 채팅 영역이 넓어짐. 설정 '글 입력 보이기'로 켬) 2026-11-03 사용자 결정
  boost: hvBoost(localStorage.getItem(LS.boost) ?? 1.5), // 소리 크기(TTS 출력 이득, 리미터 앞)
  // 핸즈프리
  conv: false, // conversation on 상태(끝낼 때까지 PCM 계속 전송)
  convStarting: false, // 마이크 준비 중
  readyWait: null, // 탭 뒤 첫 마이크 프레임 대기 { tapAt, frameAt, timer }
  convPaused: false, // 백그라운드로 가서 꺼짐 → "다시 시작" 안내
  vadSpeech: false, // 서버 vad 이벤트
  // 무음(저장하지 않음, 대화를 새로 시작하면 해제)
  micMuted: false, // 마이크 끔: 트랙 enabled=false + 전송 중지(트랙은 살려 둬 권한·세션 유지)
  soundOff: false, // 소리 끔: TTS·알림음 이득 0(재생 타이밍·playback 보고는 그대로)
  level: 0,
  pbState: 'idle', // 서버에 마지막으로 알린 playback 상태
  serverState: 'idle',
  outRate: OUT_RATE_DEFAULT,
  // 입력
  listening: false, // start_utterance ~ end_utterance 사이
  toggleMode: false, // 짧게 탭해서 켠 듣기
  pressing: false,
  pressAt: 0,
  ignoreUp: false,
  ending: false, // 워클릿 flush 대기 중
  // 오디오
  ctx: null,
  stream: null,
  source: null,
  capture: null, // AudioWorkletNode | ScriptProcessorNode
  captureSink: null,
  spDs: null, // ScriptProcessor 폴백용 다운샘플러
  spSending: false,
  micReady: null, // Promise
  diagTimer: 0, // 대화 중 [mic] 진단 로그
  diagSq: 0, // 입력 RMS² 누적(진단용)
  diagN: 0,
  recovering: null, // 복귀 복구 Promise
  // 재생
  gen: -1,
  deadGens: new Set(),
  nextTime: 0,
  sources: new Set(), // 예약·재생 중인 소스. 각 소스에 _hvAt(시작 시각)을 기록한다(hold용)
  held: null, // hold 중: { gen, bufs: AudioBuffer[] } 아직 안 들려준 오디오(순서대로)
  duck: null, // 서버 hold(mode duck): { gen, gain } 재생은 계속, 출력 이득만 낮춤
  zeroSince: 0, // 대화 중 입력이 정확히 0(디지털 무음)으로 이어진 시작 시각
  zeroTriedAt: 0, // 마지막 무음 트랙 재획득 시각(ZERO_RETRY_MS 뒤 다시 시도 가능)
  zeroRunAt: 0, // 진단: 0 샘플 레벨 보고가 이어진 시작 시각(대화 여부와 무관, zeros=Ns)
  lastNonZeroAt: 0, // 마지막으로 0이 아닌 샘플이 들어온 레벨 보고 시각
  lastPlayAt: 0, // 마지막으로 재생 중(playing())이던 것을 본 시각
  convStartAt: 0, // 이번 대화가 켜진 시각(시작 직후 무음 트랙 판정용)
  trackSuspect: false, // 백그라운드·interrupted가 60초 넘게 이어졌다 복귀함: 다음 무음 판정을 1초로
  awaySince: 0, // visibility hidden 또는 ctx interrupted가 시작된 시각
  keepAlive: null, // 출력 유지 상수 소스(st.sources와 별개: playing·재생 위치·분석기에 안 섞임)
  diagBuf: [], // WS가 닫힌 동안의 진단 줄 { text, at }
  wsOpens: 0, // ws.onopen 횟수(진단)
  reconnWhy: '', // 마지막 reconnectNow 이유(진단)
  // 전송 감시: 워클릿이 PCM 프레임을 실제로 내는지 본다(재시작 뒤 프레임 0개 버그, 2026-11-02 아이폰 실기)
  convSeq: 0, // 대화 회차(복구 도중 대화가 끝나거나 새로 시작됐는지 판별)
  lastPcmAt: 0, // 워클릿이 마지막으로 pcm 프레임을 낸 시각(전송 여부와 무관)
  capStartAt: 0, // 마지막 captureCmd('start') 시각(감시 기준점)
  prebuf: [], // 대화 전 사전 버퍼(pcm ArrayBuffer, 최근 PREBUF_MS). 서버로는 구체 탭 시작 때만 보냄
  prebufSamples: 0,
  tapAt: 0, // 마지막 구체 탭(onPressStart) 시각
  zeroFailed: false, // 이 대화에서 무음 복구가 3단계까지 실패함(다시 시도 안 함, 다음 대화 시작 때 초기화)
  // 시스템 음소거·중단(2026-11-03 21:05 실기: 재생 중 iOS가 track mute → ctx interrupted, 복구 사다리가 헛돌아 zeroFailed)
  muteSince: 0, // 트랙 muted=true가 이어진 시작 시각(진단 sysmute=Ns)
  holdSince: 0, // 대화 중 시스템 음소거·중단(muted 또는 ctx interrupted/suspended)으로 무음 판정을 보류한 시작 시각
  holdEndAt: 0, // 마지막 보류 해제 시각(복귀 뒤 0 샘플이 ZERO_IDLE_MS 이어지면 사다리 재허용). 실제 샘플이 들어오면 0
  deadFails: 0, // 페이지 수명 동안 마이크 복구 최종 실패 횟수(두 번째부터 앱 종료 권유)
  micWatch: 0, // 시작 직후 1.5초 감시 타이머
  micFix: null, // 진행 중인 프레임 복구 Promise
  // 멈춘 컨텍스트(running인데 렌더링 정지) 재생성(전송 감시 3단계)
  convRebuilds: 0, // 이번 대화에서 새로 만든 컨텍스트 수(최대 CTX_REBUILD_MAX)
  ctxRebuildSeq: 0, // 로그용 재생성 번호(#n)
  ctxStalled: false, // 복귀 점검에서 currentTime이 안 늘었음 → 다음 대화 시작 때 바로 재생성
  ctxPending: false, // 재생성한 컨텍스트가 제스처 없이 깨어나지 않음 → 다음 구체 탭에서 다시 재생성
  micNote: '', // 상태줄 안내(마이크 멈춤). 비어 있으면 평소 문구
  lastLevelAt: 0, // 워클릿 레벨 보고(빈 입력 포함)를 마지막으로 받은 시각: process()가 도는지
  ctWatch: 0, // 대화 시작 0.6초 뒤 currentTime 정지 검사 타이머
  diagCtx: null, // 2초 진단의 지난 컨텍스트·currentTime(ct 증가분 계산)
  diagCt: 0,
  stallTried: false, // 대화 중 10초 무프레임 복구를 이번 대화에서 이미 시도했나
  emptyN: 0, // 워클릿 입력 채널이 비어 있던 레벨 보고 수(진단용)
  chat: { session: null, items: [] }, // 화면 기록 보존(hv.log)
  agentName: AGENT_NAME_DEFAULT, // 대화 상대 이름(서버 AGENT_NAME). 정본은 서버(history 메시지)
  histSynced: false, // 서버 기록이 비었을 때 로컬 기록을 한 번 올렸는지
  pbBaseMs: 0, // 이 generation에서 다 나간(끝났거나 hold로 끊긴) 오디오 길이(ms). 재생 위치 보고용
  speakEnded: true,
  outBus: null, // TTS → analyser → outGain → destination
  outGain: null, // 소리 크기·소리 끔(분석기 뒤라 구체 반응은 유지)
  limiter: null, // outGain 뒤 리미터(부스트로 넘치는 피크를 누른다)
  analyser: null,
  earBus: null, // 알림음 전용(분석기 거치지 않음)
  // 준비 중 표시: 한 턴(발화 접수 ~ 답 재생 시작)의 상태
  turn: null, // { start, thinkAt, phase: 'think'|'tool'|'answer', toolName, gotDelta, firstAudio, reminders }
  shrinkAt: -1, // 끼어들기 수축 피드백 시작 시각
  // 자막
  botEl: null,
  toolEls: new Map(), // 진행 중 도구 호출: 키(toolCallId 또는 이름#순번) → { name, label }
  toolSeq: 0,
  chipBox: null, // 이번 턴 도구 칩 줄(가로 배치). 턴이 끝나면 null로 → 다음 턴은 새 줄
  chips: new Map(), // 이번 턴 칩: 한국어 라벨 → { el, count }
};

// ---------- UI ----------

function toast(msg, ms = 3500) {
  el.toast.textContent = msg;
  el.toast.hidden = false;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => { el.toast.hidden = true; }, ms);
}

function scrollLog() {
  el.log.scrollTop = el.log.scrollHeight;
}

function addLine(cls, text) {
  if (el.hint) { el.hint.remove(); el.hint = null; }
  const d = document.createElement('div');
  d.className = cls;
  d.textContent = text;
  el.log.appendChild(d);
  scrollLog();
  return d;
}

// ---------- 대화 기록 보존 (hv.log) ----------
// 앱을 껐다 켜도 화면 로그가 이어지게 한다. 저장: 확정 발화·글 입력·최종 답(끊기면 화면에 나온 데까지)·새 대화 구분선.
// 스트리밍 중간·도구 칩·상태 줄은 저장하지 않는다. '새 대화' 버튼에서만 지운다.
// iOS가 홈화면 앱의 localStorage를 지우는 일이 있어 항목마다 서버에도 보낸다(log). 정본은 서버(hello 뒤 history).
const LOG_MAX = 300;
const LOG_CLS = { user: 'msg me', bot: 'msg bot', sys: 'sys' };

function loadChat() {
  try {
    const o = JSON.parse(localStorage.getItem(LS.log) || 'null');
    if (o && Array.isArray(o.items)) return { session: o.session || null, items: o.items };
  } catch { /* 깨진 기록은 무시 */ }
  return { session: null, items: [] };
}

function saveChat() {
  try { localStorage.setItem(LS.log, JSON.stringify(st.chat)); } catch { /* 용량 초과 등 무시 */ }
}

function logItem(r, text, sync = true) {
  const t = (text || '').trim();
  if (!t) return;
  const item = { r, t, at: Date.now() };
  st.chat.items.push(item);
  if (st.chat.items.length > LOG_MAX) st.chat.items.splice(0, st.chat.items.length - LOG_MAX);
  saveChat();
  if (sync) send({ type: 'log', item }); // 닫혀 있으면 생략(나중에 맞춰 보내지 않음)
}

// 서버 history(hello 뒤 ready 다음에 옴): 서버 기록이 있으면 정본으로 덮어쓰고,
// 비어 있고 로컬 기록이 있으면 로컬을 유지한 채 서버로 한 번 올린다.
function onHistory(m) {
  const items = Array.isArray(m.items) ? m.items : [];
  if (items.length) {
    const next = { session: m.session_id || st.chat.session, items };
    if (JSON.stringify(next) === JSON.stringify(st.chat)) return;
    st.chat = next;
    el.log.textContent = '';
    st.botEl = null;
    restoreChat();
    saveChat();
    return;
  }
  if (st.histSynced || !st.chat.items.length) return;
  st.histSynced = true;
  for (const item of st.chat.items) send({ type: 'log', item });
}

// 페이지 로드 직후 저장소 상태 진단(iOS가 localStorage를 지우는지 확인용)
async function diagStore() {
  const sm = navigator.storage;
  let persisted = '지원 안 함';
  if (sm && sm.persisted) {
    try { persisted = String(await sm.persisted()); } catch (e) { persisted = `오류 ${e && e.name}`; }
  }
  diag(`[store] hv.session=${localStorage.getItem(LS.session) ? '있음' : '없음'} hv.log ${st.chat.items.length}개 persisted=${persisted}`);
  if (sm && sm.persist) {
    try { diag(`[store] persist() -> ${await sm.persist()}`); } catch (e) { diag(`[store] persist() 오류 ${e && e.name}`, 'warn'); }
  }
}

function dayKey(at) {
  const d = new Date(at);
  return `${d.getMonth() + 1}월 ${d.getDate()}일`;
}

// 저장된 기록을 같은 CSS 클래스로 다시 그린다. 날짜가 바뀐 첫 항목 앞에는 날짜 구분선(저장 안 함).
function restoreChat() {
  let day = null;
  for (const it of st.chat.items) {
    if (!it || typeof it.t !== 'string' || !LOG_CLS[it.r]) continue;
    const k = dayKey(it.at || 0);
    if (day && k !== day) addLine('sys', k);
    day = k;
    addLine(LOG_CLS[it.r], it.t);
  }
  scrollLog();
}

// 서버 ready의 session_id: 저장된 것과 다르면(유휴로 서버가 새 세션) 로그는 두고 구분선만 넣는다
function noteSession(sid) {
  if (!sid || st.chat.session === sid) return;
  if (st.chat.session && st.chat.items.length) {
    addLine('sys', '새 대화');
    // 서버로는 안 보낸다: 곧 올 history가 서버 기록이면 이 구분선째 덮어쓰고, 비었으면 로컬 동기화 때 함께 올라간다.
    logItem('sys', '새 대화', false);
  }
  st.chat.session = sid;
  saveChat();
}

function playing() {
  return st.sources.size > 0;
}

function displayState() {
  if (st.listening) return 'listening';
  if (st.convStarting || (st.conv && st.readyWait)) return 'starting';
  if (playing()) return 'speaking';
  const s = st.serverState || 'idle';
  if (st.conv) {
    if (s === 'idle' || s === 'waiting' || (st.micMuted && s === 'listening')) return st.vadSpeech && !st.micMuted ? 'listening' : 'waiting';
    return s;
  }
  return s === 'waiting' ? 'idle' : s;
}

function micSending() {
  // 지금 마이크 프레임을 서버로 보내는가
  if (!st.ws || !st.wsOpen) return false;
  if (st.listening) return true;
  if (!st.conv) return false;
  if (st.micMuted) return false;
  if (st.halfDuplex && (playing() || st.serverState === 'speaking')) return false;
  return true;
}

function trackTurn(s) {
  // 준비 중(받아쓰기·생각) 경과 시간을 재는 턴. 말하기 사이(채움 말 뒤 다시 생각)에도 이어진다.
  if (isBusy(s)) {
    if (!st.turn) st.turn = { start: performance.now(), thinkAt: 0, phase: 'think', toolName: '', gotDelta: false, firstAudio: false, reminders: 0 };
    if (s === 'thinking' && !st.turn.thinkAt) st.turn.thinkAt = performance.now();
  } else if (s !== 'speaking') {
    st.turn = null;
  }
}

function statusText(s) {
  if (st.authFail) return '접속 토큰이 필요합니다';
  if (st.micNote) return st.micNote;
  if (s === 'starting') return '마이크 준비 중…';
  if (s === 'idle' && st.convPaused && !st.ptt && !st.conv) return '탭해서 다시 시작';
  if (s === 'listening' && st.ptt && st.toggleMode) return '듣는 중 · 다시 누르면 끝';
  if (s === 'speaking') return '말하는 중';
  if (st.conv && st.micMuted && s === 'waiting') return '마이크 꺼짐';
  if (!isBusy(s) || !st.turn) return STATE_LABEL[s] ?? s;
  const t = st.turn;
  let txt = '생각 중';
  if (t.phase === 'tool') txt = `${toolKo(t.toolName)} 확인 중`;
  else if (t.phase === 'answer') txt = '답 준비 중';
  const sec = Math.floor((performance.now() - t.start) / 1000);
  return sec >= 2 ? `${txt} · ${sec}초` : txt; // 2초가 넘으면 경과 초
}

function setTip(b, txt) {
  if (b.getAttribute('aria-label') !== txt) { b.setAttribute('aria-label', txt); b.title = txt; }
}

// 단색 선 아이콘(획 1.5px)
const ICON = {
  wrench: '<path d="M14.5 5.5a4 4 0 0 0-5 5L4 16l4 4 5.5-5.5a4 4 0 0 0 5-5l-2.5 2.5-2.5-.5-.5-2.5z"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
};
function icon(name, size = 16) {
  return `<svg viewBox="0 0 24 24" width="${size}" height="${size}" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">${ICON[name]}</svg>`;
}
function setChip(chip, label, done, count = 1) {
  chip.classList.toggle('done', done);
  chip.innerHTML = icon('wrench', 12);
  const t = document.createElement('span');
  t.textContent = label;
  chip.appendChild(t);
  if (count > 1) {
    const n = document.createElement('span');
    n.className = 'n';
    n.textContent = '×' + count;
    chip.appendChild(n);
  }
  if (done) chip.insertAdjacentHTML('beforeend', icon('check', 12));
}

// 헤르메스 답 정리(표시·기록용): 첫 글자 앞 공백·줄바꿈 제거, 3개 이상 줄바꿈은 2개로, 끝 공백은 final에서만 제거
function cleanBot(raw, final) {
  let t = String(raw || '').replace(/\r\n?/g, '\n').replace(/^\s+/, '').replace(/\n[ \t]*(?:\n[ \t]*)+\n/g, '\n\n');
  if (final) t = t.trimEnd();
  return t;
}

// 턴이 끝나면 칩 줄을 닫는다(칩은 화면에 남음)
function endChipTurn() {
  st.chipBox = null;
  st.chips.clear();
  st.toolEls.clear();
}

function render() {
  const s = displayState();
  trackTurn(s);
  const txt = statusText(s);
  if (el.status.textContent !== txt) el.status.textContent = txt;
  el.status.className = 'status ' + s;
  if (el.dock.dataset.state !== s) el.dock.dataset.state = s;
  el.dock.classList.toggle('conv', st.conv);
  el.dock.classList.toggle('muted', st.conv && st.halfDuplex && s === 'speaking');
  el.dock.classList.toggle('mic-muted', st.conv && st.micMuted);
  // 소리 끔 + 말하는 중: 파형 막대 대신 "자막으로만"
  // 막대는 소리가 오가는 상태(듣기·생각·말하기)에서만. 대기·유휴에선 숨겨 화면을 비운다.
  const activeBars = s === 'listening' || s === 'transcribing' || s === 'thinking' || s === 'speaking';
  el.barsWrap.hidden = !activeBars;
  // 소리 끔 + 말하는 중: 문구 대신 막대를 흐리게만("진행은 되는데 소리는 없음")
  el.barsWrap.classList.toggle('silent', st.soundOff && s === 'speaking');
  // 버튼 줄: 마이크 끔 · 소리 끔 · 종료. 대화 전엔 비활성(0.35), 누르고 말하기 모드엔 마이크·종료 숨김
  el.micBtn.hidden = st.ptt;
  el.endBtn.hidden = st.ptt;
  el.micBtn.disabled = !st.conv;
  el.endBtn.disabled = !st.conv;
  el.soundBtn.disabled = !st.ptt && !st.conv;
  el.micBtn.classList.toggle('off', st.micMuted);
  setTip(el.micBtn, st.micMuted ? '마이크 켜기' : '마이크 끄기');
  el.soundBtn.classList.toggle('off', st.soundOff);
  setTip(el.soundBtn, st.soundOff ? '소리 켜기' : '소리 끄기');
  el.textForm.hidden = !st.showText;
  if (!isBusy(s) && s !== 'speaking') removeTyping();
  // 구체 크기: 대화 중(또는 누르고 말하기에서 듣기~말하기 동안)엔 크게, 그 밖엔 작게 + 마이크 아이콘
  const big = st.ptt ? s !== 'idle' : st.conv;
  el.dock.classList.toggle('small', !big);
  if (big && !anim.big && !reduceMotion.matches) {
    // 커지는 순간 얇은 링 하나가 한 번 퍼진다
    el.orbPulse.classList.remove('go');
    void el.orbPulse.offsetWidth;
    el.orbPulse.classList.add('go');
  }
  anim.big = big;
  const t = el.talk;
  t.disabled = st.convStarting;
  setTip(t, big ? '끼어들기' : (st.ptt ? '누르고 말하기' : '대화 시작'));
  el.conn.className = 'conn ' + (st.wsOpen ? 'on' : 'off');
  el.connText.textContent = st.wsOpen ? '연결됨' : (st.authFail ? '토큰 필요' : (st.retryTimer ? '다시 연결 중' : '연결 중'));
  for (const b of document.querySelectorAll('.seg-btn[data-model]')) {
    const on = b.dataset.model === st.model;
    b.classList.toggle('on', on);
    b.setAttribute('aria-checked', on ? 'true' : 'false');
  }
  for (const b of el.boostBtns) {
    const on = Number(b.dataset.boost) === st.boost;
    b.classList.toggle('on', on);
    b.setAttribute('aria-checked', on ? 'true' : 'false');
  }
  if (!st.listening && !st.conv) setLevel(0);
  kickAnim();
  if (el.hint) {
    el.hint.textContent = st.ptt
      ? '가운데 버튼을 누르는 동안 말하세요.'
      : '가운데 구체를 누르고 편하게 말하세요.';
  }
}

function setLevel(v) {
  st.level = v; // 마이크 레벨(0~1). 구체·막대는 애니메이션 루프가 읽는다.
}

// ---------- 구체·파형 애니메이션 (requestAnimationFrame 하나) ----------

const anim = { raf: 0, inLvl: 0, outLvl: 0, last: 0, td: null, fd: null, big: false };

function kickAnim() {
  if (anim.raf || document.hidden) return;
  anim.raf = requestAnimationFrame(animFrame);
}

function stopAnim() {
  if (anim.raf) cancelAnimationFrame(anim.raf);
  anim.raf = 0;
}

function outputRms() {
  const a = st.analyser;
  if (!a || !playing()) return 0;
  if (!anim.td || anim.td.length !== a.fftSize) anim.td = new Uint8Array(a.fftSize);
  a.getByteTimeDomainData(anim.td);
  let sum = 0;
  for (let i = 0; i < anim.td.length; i++) { const v = (anim.td[i] - 128) / 128; sum += v * v; }
  return Math.sqrt(sum / anim.td.length);
}

function outputBands(n) {
  // 출력 스펙트럼을 n개 대역(대략 로그 간격, ~4 kHz까지)으로
  const a = st.analyser;
  const out = new Array(n).fill(0);
  if (!a || !playing()) return out;
  if (!anim.fd || anim.fd.length !== a.frequencyBinCount) anim.fd = new Uint8Array(a.frequencyBinCount);
  a.getByteFrequencyData(anim.fd);
  const hz = (st.ctx ? st.ctx.sampleRate : 44100) / 2 / anim.fd.length;
  const edges = [80, 250, 500, 1000, 2000, 4000];
  for (let i = 0; i < n; i++) {
    const lo = Math.max(1, Math.floor(edges[i] / hz)), hi = Math.max(lo + 1, Math.floor(edges[i + 1] / hz));
    let m = 0;
    for (let k = lo; k < hi && k < anim.fd.length; k++) m = Math.max(m, anim.fd[k]);
    out[i] = m / 255;
  }
  return out;
}

function animFrame(now) {
  anim.raf = 0;
  if (document.hidden) return; // 숨겨지면 멈춘다(배터리). 다시 보이면 render()가 깨운다.
  const s = displayState();
  const rm = reduceMotion.matches;
  const t = now / 1000;
  const dt = anim.last ? Math.min(0.1, (now - anim.last) / 1000) : 0.016;
  anim.last = now;
  // 레벨 부드럽게(올라갈 땐 빠르게, 내려갈 땐 천천히)
  const smooth = (cur, target, up, down) => cur + (target - cur) * Math.min(1, dt * (target > cur ? up : down));
  const micTarget = s === 'listening' || s === 'waiting' ? st.level : 0;
  anim.inLvl = smooth(anim.inLvl, micTarget, 18, 6);
  const outTarget = s === 'speaking' ? Math.min(1, Math.sqrt(outputRms()) * 1.6) : 0;
  anim.outLvl = smooth(anim.outLvl, outTarget, 20, 8);
  const amp = rm ? 0.2 : 1; // 움직임 줄이기

  let scale = 1, ringScale = 1, ringOp = 0;
  if (s === 'waiting') {
    scale = 1 + (rm ? 0 : 0.02 * Math.sin((2 * Math.PI * t) / 3.5)) + anim.inLvl * 0.04 * amp;
  } else if (s === 'listening') {
    scale = 1 + anim.inLvl * 0.1 * amp;
    ringScale = 1 + 0.04 + anim.inLvl * 0.12 * amp; // 무대(140px) 안에서 최대 1.16배
    ringOp = 0.2 + anim.inLvl * 0.4;
  } else if (isBusy(s)) {
    // 미세한 호흡(0.97↔1.03, 3.2초, 사인 = ease-in-out). 링 스윕은 CSS(.orb-sweep)
    scale = 1 + (rm ? 0 : 0.03 * Math.sin((2 * Math.PI * t) / 3.2));
  } else if (s === 'speaking') {
    scale = 1 + anim.outLvl * 0.1 * amp;
    ringScale = 1 + 0.04 + anim.outLvl * 0.12 * amp;
    ringOp = anim.outLvl * 0.35;
  } else {
    scale = 1; // 대기 크기(약 90px)는 CSS scale이 맡는다
  }
  // 끼어들기 직후 300ms 수축
  let shrinking = false;
  if (st.shrinkAt >= 0) {
    const k = (now - st.shrinkAt) / 300;
    if (k >= 0 && k < 1) { scale *= 1 - 0.1 * Math.sin(Math.PI * k) * (rm ? 0.3 : 1); shrinking = true; }
    else if (k >= 1) st.shrinkAt = -1;
  }
  el.orbCore.style.transform = `scale(${scale.toFixed(4)})`;
  el.orbRing.style.transform = `scale(${ringScale.toFixed(4)})`;
  el.orbRing.style.opacity = ringOp.toFixed(3);

  // 파형 막대
  const nb = el.bars.length;
  const bands = s === 'speaking' ? outputBands(nb) : null;
  for (let i = 0; i < nb; i++) {
    let h;
    const wob = rm ? 1 : 0.75 + 0.25 * Math.sin(t * 7 + i * 1.7);
    if (s === 'speaking') h = 0.15 + (anim.outLvl * 0.6 * wob + bands[i] * 0.45) * (rm ? 0.6 : 1);
    else if (s === 'listening') h = 0.15 + anim.inLvl * 0.9 * wob * (1 - Math.abs(i - (nb - 1) / 2) * 0.12);
    else if (isBusy(s)) h = rm ? 0.25 : 0.2 + 0.14 * (0.5 + 0.5 * Math.sin(t * 5 - i * 0.9));
    else if (s === 'waiting') h = 0.14 + (rm ? 0 : 0.04 * Math.sin((2 * Math.PI * t) / 3.5 + i * 0.5));
    else h = 0.1;
    el.bars[i].style.transform = `scaleY(${Math.min(1, h).toFixed(3)})`;
  }

  // 상태 문구(경과 초)와 "아직 준비 중" 알림음
  if (isBusy(s)) {
    const txt = statusText(s);
    if (el.status.textContent !== txt) el.status.textContent = txt;
    checkReminder(now);
  }

  // 대기(대화 꺼짐)이고 움직일 게 없으면 루프를 쉰다
  const settling = anim.inLvl > 0.005 || anim.outLvl > 0.005;
  if (s !== 'idle' || shrinking || settling || playing()) anim.raf = requestAnimationFrame(animFrame);
}

// ---------- 알림음(earcon) ----------

function ensureOutput(ctx) {
  // TTS 재생 체인: source → outBus → analyser → outGain(소리 크기) → limiter → destination.
  // 알림음은 earBus → destination(분석기·부스트·리미터 안 거침).
  if (st.outBus && st.outBus.context === ctx) return;
  st.outBus = ctx.createGain();
  st.analyser = ctx.createAnalyser();
  st.analyser.fftSize = 512;
  st.analyser.smoothingTimeConstant = 0.6;
  st.outGain = ctx.createGain();
  st.outGain.gain.value = outGainValue();
  st.limiter = ctx.createDynamicsCompressor();
  st.limiter.threshold.value = -3;
  st.limiter.knee.value = 0;
  st.limiter.ratio.value = 20;
  st.limiter.attack.value = 0.002;
  st.limiter.release.value = 0.1;
  st.outBus.connect(st.analyser);
  st.analyser.connect(st.outGain);
  st.outGain.connect(st.limiter);
  st.limiter.connect(ctx.destination);
  st.earBus = ctx.createGain();
  st.earBus.gain.value = st.soundOff ? 0 : 1;
  st.earBus.connect(ctx.destination);
}

// ---------- 출력 유지(keep-alive) ----------
// 재생이 끝나 출력이 완전히 무음이 되면 iOS WebKit이 오디오 세션을 다시 짜며 마이크 트랙을 3~7초 mute한다
// (2026-11-03 서버 로그). 대화 중에는 earBus에 -100 dBFS 상수를 흘려 무음 판정을 막는다.
// st.sources에 넣지 않으므로 playing()·재생 위치(playedMs)·분석기(레벨 미터)에 섞이지 않는다.
function keepAliveEnabled() {
  try { return localStorage.getItem(LS.keepalive) !== '0'; } catch { return true; }
}

function startKeepAlive() {
  stopKeepAlive();
  const ctx = st.ctx;
  if (!ctx || !keepAliveEnabled()) return;
  ensureOutput(ctx);
  try {
    let src;
    if (typeof ctx.createConstantSource === 'function') {
      src = ctx.createConstantSource();
      src.offset.value = KEEPALIVE_LEVEL;
    } else {
      // 구형: 1초짜리 상수 버퍼를 반복 재생
      const buf = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
      buf.getChannelData(0).fill(KEEPALIVE_LEVEL);
      src = ctx.createBufferSource();
      src.buffer = buf;
      src.loop = true;
    }
    src.connect(ctx.destination); // 소리 끔(earBus gain 0)과 무관하게 유지
    src.start();
    st.keepAlive = src;
  } catch (e) {
    st.keepAlive = null;
    diag(`[audio] keep-alive 시작 실패 ${e && e.name} ${e && e.message}`, 'warn');
  }
}

function stopKeepAlive() {
  const src = st.keepAlive;
  st.keepAlive = null;
  if (!src) return;
  try { src.stop(); } catch { /* */ }
  try { src.disconnect(); } catch { /* */ }
}

function outGainValue() {
  return st.soundOff ? 0 : st.boost * (st.duck ? st.duck.gain : 1);
}

// 출력 이득을 20 ms 램프로 outGainValue()에 맞춘다(덕킹·복귀용)
function rampOutGain() {
  if (!st.outGain) return;
  const g = st.outGain.gain;
  const t = st.outGain.context.currentTime;
  g.cancelScheduledValues(t);
  g.setValueAtTime(g.value, t);
  g.linearRampToValueAtTime(outGainValue(), t + 0.02);
}

function tone(freq, at, dur, vol) {
  const ctx = st.ctx;
  const o = ctx.createOscillator();
  const g = ctx.createGain();
  o.type = 'sine';
  o.frequency.value = freq;
  // 부드럽게: 짧은 어택·릴리스
  g.gain.setValueAtTime(0, at);
  g.gain.linearRampToValueAtTime(vol, at + 0.008);
  g.gain.setValueAtTime(vol, at + dur - 0.02);
  g.gain.linearRampToValueAtTime(0, at + dur);
  o.connect(g);
  g.connect(st.earBus);
  o.start(at);
  o.stop(at + dur + 0.01);
  o.onended = () => { try { g.disconnect(); } catch { /* */ } };
}

function earcon(kind) {
  if (!st.earcon || st.soundOff || !st.ctx || st.ctx.state !== 'running') return;
  ensureOutput(st.ctx);
  const at = st.ctx.currentTime + 0.01;
  if (kind === 'ready') {
    // "준비됐어요": 523 → 784 Hz(도→솔), 각 45ms. 'heard'보다 낮게 시작해 더 크게 오른다.
    tone(523, at, 0.045, 0.07);
    tone(784, at + 0.05, 0.045, 0.07);
  } else if (kind === 'heard') {
    // "들었어요": 660 → 880 Hz, 각 60ms
    tone(660, at, 0.06, 0.08);
    tone(880, at + 0.065, 0.06, 0.08);
  } else {
    // "아직 준비 중": 아주 작은 한 음
    tone(740, at, 0.06, 0.04);
  }
}

function checkReminder(now) {
  const t = st.turn;
  if (!t || !t.thinkAt || t.firstAudio || t.reminders >= 3) return;
  if (now - t.thinkAt >= 4000 * (t.reminders + 1)) {
    t.reminders++;
    earcon('waiting');
  }
}

// ---------- 타이핑 표시(헤르메스 말풍선 자리) ----------

function showTyping() {
  if (st.botEl) return;
  const d = addLine('msg bot typing', '');
  for (let i = 0; i < 3; i++) d.appendChild(document.createElement('i'));
  d.setAttribute('aria-label', `${st.agentName}이 답을 준비하는 중`);
  st.botEl = d;
}

function removeTyping() {
  if (st.botEl && st.botEl.classList.contains('typing')) {
    st.botEl.remove();
    st.botEl = null;
  }
}

function fillBot() {
  // 타이핑 표시를 텍스트 말풍선으로 바꾼다
  if (!st.botEl) st.botEl = addLine('msg bot streaming', '');
  else if (st.botEl.classList.contains('typing')) {
    st.botEl.textContent = '';
    st.botEl._hvRaw = '';
    st.botEl.classList.remove('typing');
    st.botEl.removeAttribute('aria-label');
    st.botEl.classList.add('streaming');
  }
  return st.botEl;
}

function renderInfo() {
  const standalone = (navigator.standalone === true) || (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches);
  if (el.sheetVer) {
    el.sheetVer.textContent = `Hermes Voice v${APP_VERSION}${st.swVer ? ` (캐시 ${st.swVer})` : ''} · ${standalone ? '홈 화면 앱' : '브라우저'} · 세션 ${st.sessionId ? st.sessionId.slice(0, 8) : '없음'}`;
  }
  el.info.textContent = [
    `버전 ${APP_VERSION}`,
    `세션 ${st.sessionId || '(없음)'}`,
    `모델 ${st.model}`,
    `무음 ${st.micMuted ? '마이크 끔' : '마이크 켬'}, ${st.soundOff ? '소리 끔' : '소리 켬'}`,
    `모드 ${st.ptt ? '누르고 말하기' : '핸즈프리'}${st.halfDuplex ? ', 답하는 동안 마이크 끔' : ''}${st.aecOff ? ', 에코 제거 끔' : ''}${st.earcon ? '' : ', 알림음 끔'}`,
    `캡처 ${st.capture ? (st.capture.port ? 'AudioWorklet' : 'ScriptProcessor') : '(아직 없음)'}`,
    st.ctx ? `AudioContext ${st.ctx.sampleRate}Hz, ${st.ctx.state}` : 'AudioContext (아직 없음)',
  ].join('\n');
}

// ---------- WebSocket ----------

function wsUrl() {
  return (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
}

function send(obj) {
  if (st.ws && st.wsOpen) {
    st.ws.send(JSON.stringify(obj));
    return true;
  }
  return false;
}

// 진단 한 줄: 콘솔에 찍고 서버 로그(client:)로도 보낸다. 연결이 닫혀 있으면 최근 20줄을 모았다가 onopen에서 보낸다.
function diag(text, level = 'info') {
  (console[level] || console.info).call(console, text);
  let ok = false;
  try { ok = send({ type: 'diag', text }); } catch { /* */ }
  if (!ok) {
    st.diagBuf.push({ text, at: performance.now() });
    if (st.diagBuf.length > DIAG_BUF_MAX) st.diagBuf.shift();
  }
}

function flushDiag(ws) {
  const buf = st.diagBuf;
  st.diagBuf = [];
  const now = performance.now();
  for (const d of buf) {
    try {
      ws.send(JSON.stringify({ type: 'diag', text: `${d.text} (${((now - d.at) / 1000).toFixed(1)}s 전)` }));
    } catch { /* */ }
  }
}

function connect() {
  clearTimeout(st.retryTimer);
  st.retryTimer = 0;
  if (st.authFail) return; // 토큰을 새로 저장할 때까지 접속하지 않는다
  if (st.ws && (st.ws.readyState === WebSocket.OPEN || st.ws.readyState === WebSocket.CONNECTING)) return;
  let ws;
  try {
    ws = new WebSocket(wsUrl());
  } catch (e) {
    scheduleReconnect();
    return;
  }
  ws.binaryType = 'arraybuffer';
  st.ws = ws;
  ws.onopen = () => {
    if (st.ws !== ws) return;
    st.wsOpen = true;
    st.retry = 0;
    const hello = { type: 'hello', resume: st.sessionId || null, client: 'pwa', vp: viewportInfo() };
    const token = getToken();
    if (token) hello.token = token;
    ws.send(JSON.stringify(hello));
    flushDiag(ws);
    st.wsOpens++;
    diag(`[ws] open #${st.wsOpens}${st.wsOpens > 1 ? ` (재연결 ${st.wsOpens - 1}번째)` : ''}${st.reconnWhy ? ` 즉시 재연결 이유: ${st.reconnWhy}` : ''}, conv ${st.conv ? 'on' : 'off'}`);
    st.reconnWhy = '';
    sendSettings();
    st.pbState = 'idle';
    if (playing()) reportPlayback('speaking');
    // 끊기기 전 대화 중이었으면 이어서
    if (st.conv) { dropPrebuf(); send({ type: 'conversation', on: true }); }
    render();
  };
  ws.onmessage = (ev) => {
    if (st.ws !== ws) return;
    if (typeof ev.data === 'string') {
      let m;
      try { m = JSON.parse(ev.data); } catch { return; }
      onEvent(m);
    } else {
      onAudio(ev.data);
    }
  };
  ws.onclose = (ev) => {
    if (st.ws !== ws) return;
    st.ws = null;
    st.wsOpen = false;
    diag(`[ws] close code=${ev && ev.code} reason=${(ev && ev.reason) || '-'} wasClean=${ev && ev.wasClean}, conv ${st.conv ? 'on' : 'off'}`, 'warn');
    // 말하던 중이면 그 발화는 버린다
    if (st.listening) abortListening();
    st.serverState = 'idle';
    st.vadSpeech = false;
    if (ev && ev.code === 4401) onUnauthorized();
    if (!st.authFail) scheduleReconnect();
    render();
  };
  ws.onerror = () => { /* onclose가 뒤따른다 */ };
}

function sendSettings() {
  send({ type: 'settings', aec: !st.aecOff, half_duplex: st.halfDuplex });
}

function scheduleReconnect() {
  if (st.retryTimer || st.authFail) return;
  const delay = Math.min(8000, 500 * 2 ** st.retry);
  st.retry++;
  st.retryTimer = setTimeout(() => { st.retryTimer = 0; connect(); }, delay);
}

function reconnectNow(why = '') {
  if (st.wsOpen || st.authFail) return;
  st.reconnWhy = why || '-';
  diag(`[ws] 즉시 재연결(${st.reconnWhy})`);
  st.retry = 0;
  clearTimeout(st.retryTimer);
  st.retryTimer = 0;
  connect();
}

// ---------- 서버 이벤트 ----------

function onEvent(m) {
  switch (m.type) {
    case 'ready':
      // 새 서버 세션은 generation을 0부터 다시 센다. 이전 연결에서 죽인 번호 목록을 들고 있으면
      // 새 답의 오디오를 버려 소리가 안 나므로(2026-11-03 20:33 실기) 여기서 비운다.
      stopPlayback();
      st.deadGens = new Set();
      st.gen = -1;
      st.pbBaseMs = 0;
      if (m.session_id) {
        st.sessionId = m.session_id;
        noteSession(m.session_id);
        localStorage.setItem(LS.session, m.session_id);
      }
      if (m.out_sample_rate) st.outRate = m.out_sample_rate;
      if (typeof m.agent_name === 'string' && m.agent_name.trim()) { st.agentName = m.agent_name.trim(); if (el.appName) el.appName.textContent = st.agentName; }
      if (m.model_labels && typeof m.model_labels === 'object') {
        const hint = document.getElementById('modelHint');
        const f = m.model_labels['voice-fast'], d = m.model_labels['voice-deep'];
        if (hint && (f || d)) hint.textContent = [f ? `빠름: ${f}` : '', d ? `깊게: ${d}` : ''].filter(Boolean).join(' · ');
      }
      if (!el.menu.hidden) renderInfo(); // 시트가 열려 있으면 세션 표시 갱신
      if (m.model && m.model !== st.model) send({ type: 'set_model', model: st.model });
      break;
    case 'history':
      onHistory(m);
      break;
    case 'status':
      st.serverState = m.state;
      if (m.state === 'idle' || m.state === 'waiting') clearToolBadge();
      if (m.state !== 'listening' && m.state !== 'waiting' && m.state !== 'idle') st.vadSpeech = false;
      if (m.state === 'thinking') showTyping();
      break;
    case 'transcript': {
      removeTyping();
      const d = addLine('msg me', m.text || '');
      logItem('user', m.text);
      if (m.ms != null) console.info('[stt]', m.ms, 'ms'); // 진단 수치는 화면 대신 콘솔로
      st.botEl = null;
      endChipTurn();
      // 발화 접수 → 생각 시작: "들었어요" 알림음과 타이핑 표시
      trackTurn('thinking');
      earcon('heard');
      showTyping();
      break;
    }
    case 'assistant_delta':
    {
      // 원문은 _hvRaw에 모으고 화면에는 정리한 텍스트(첫 글자 앞 줄바꿈 제거 등)
      const b = fillBot();
      b._hvRaw = (b._hvRaw || '') + (m.text || '');
      b.textContent = cleanBot(b._hvRaw, false);
    }
      if (st.turn && !st.turn.gotDelta) {
        st.turn.gotDelta = true;
        if (st.toolEls.size === 0) st.turn.phase = 'answer';
      }
      scrollLog();
      break;
    case 'assistant_done':
      fillBot();
      st.botEl.textContent = cleanBot(typeof m.text === 'string' && m.text ? m.text : (st.botEl._hvRaw ?? st.botEl.textContent), true);
      st.botEl.classList.remove('streaming');
      st.botEl._hvSaved = true;
      logItem('bot', st.botEl.textContent);
      st.botEl = null;
      endChipTurn();
      scrollLog();
      break;
    case 'tool':
      onTool(m);
      break;
    case 'speak_start':
      onSpeakStart(m.generation);
      break;
    case 'speak_end':
      if (m.generation === st.gen) st.speakEnded = true;
      break;
    case 'vad':
      st.vadSpeech = !!m.speech;
      break;
    case 'hold':
      // 서버 잠정 끼어들기. mode duck: 재생은 계속하고 소리만 낮춤(서버가 마이크가 따라 내려가는지 검사).
      // mode 없음·pause: 소리만 즉시 멈추고 남은 오디오는 보관
      if (m.mode === 'duck') onDuck(m.generation, typeof m.gain === 'number' ? m.gain : 0.3);
      else onHold(m.generation);
      break;
    case 'resume':
      // 잠정 끼어들기 취소: 보관분을 멈춘 지점부터 이어서
      onResume(m.generation);
      break;
    case 'interrupt':
      // 서버가 끼어들기를 판정해 재생을 끊음
      if (st.shrinkAt < 0 && (playing() || isBusy(displayState()))) st.shrinkAt = performance.now();
      stopLocal(typeof m.generation === 'number' ? m.generation : null);
      break;
    case 'error':
      if (m.code === 'unauthorized') { onUnauthorized(); break; }
      addLine('sys err', '오류: ' + (m.message || '알 수 없음'));
      break;
    default:
      break;
  }
  render();
}

function onTool(m) {
  const name = m.name || '도구';
  if (toolHidden(name)) return; // 내부 도구는 칩·상태 텍스트에 안 보임
  const label = toolKo(name);
  const running = m.state === 'started' || m.status === 'running';
  if (running) {
    // 같은 라벨 칩은 턴당 하나만 두고 재사용, 두 번째부터 횟수 표시. 칩은 한 줄 컨테이너에 가로로
    if (!st.chipBox) st.chipBox = addLine('chips', '');
    let c = st.chips.get(label);
    if (!c) {
      const chip = document.createElement('span');
      chip.className = 'chip';
      st.chipBox.appendChild(chip);
      c = { el: chip, count: 0 };
      st.chips.set(label, c);
    }
    c.count++;
    setChip(c.el, label, false, c.count);
    const key = m.toolCallId || `${name}#${++st.toolSeq}`;
    st.toolEls.set(key, { name, label });
    // 타이핑 표시는 늘 맨 아래에
    if (st.botEl && st.botEl.classList.contains('typing')) { el.log.appendChild(st.botEl); scrollLog(); }
    if (st.turn) { st.turn.phase = 'tool'; st.turn.toolName = name; }
  } else {
    // completed: toolCallId가 있으면 그것으로, 없으면 같은 이름의 가장 오래된 진행 중 호출을 끝낸다
    let key = m.toolCallId && st.toolEls.has(m.toolCallId) ? m.toolCallId : null;
    if (!key) for (const [k, v] of st.toolEls) if (v.name === name) { key = k; break; }
    if (key) st.toolEls.delete(key);
    const c = st.chips.get(label);
    // 같은 라벨 호출이 하나도 안 남았을 때만 체크
    if (c && ![...st.toolEls.values()].some((v) => v.label === label)) setChip(c.el, label, true, c.count);
    if (st.toolEls.size === 0) clearToolBadge();
    else if (st.turn) st.turn.toolName = [...st.toolEls.values()].pop().name;
  }
}

function clearToolBadge() {
  if (st.turn && st.turn.phase === 'tool') st.turn.phase = st.turn.gotDelta ? 'answer' : 'think';
}

// ---------- 재생 (AudioBufferSourceNode 이어 붙이기) ----------
// AudioWorklet 재생은 iOS AEC 잡음 버그(WebKit 326286)가 있어 쓰지 않는다.

function onSpeakStart(gen) {
  if (typeof gen !== 'number' || st.deadGens.has(gen)) return;
  if (gen !== st.gen) {
    // 새 generation: 이전 재생 큐를 비운다. 같은 generation의 다음 문장은 이어서 붙인다.
    stopPlayback();
    st.gen = gen;
    st.pbBaseMs = 0; // 재생 위치는 generation마다 0부터
  }
  st.speakEnded = false;
}

function onAudio(buf) {
  if (!(buf instanceof ArrayBuffer) || buf.byteLength < 6) return;
  const gen = new DataView(buf).getUint32(0, true);
  if (gen !== st.gen || st.deadGens.has(gen)) return;
  const ctx = st.ctx;
  if (!ctx) return; // 아직 사용자 제스처 전이면 재생 불가
  const n = (buf.byteLength - 4) >> 1;
  const pcm = new Int16Array(buf, 4, n);
  const ab = ctx.createBuffer(1, n, st.outRate);
  const ch = ab.getChannelData(0);
  for (let i = 0; i < n; i++) ch[i] = pcm[i] / 32768;
  if (st.turn) st.turn.firstAudio = true; // 첫 오디오가 나오면 "아직 준비 중" 알림음은 그만
  if (st.held) {
    // hold 중: 재생하지 않고 보관 목록 뒤에 붙인다
    if (st.held.gen === gen) st.held.bufs.push(ab);
    return;
  }
  scheduleBuffer(ab);
}

// AudioBuffer 하나를 재생 큐 끝(st.nextTime)에 이어 붙인다.
function scheduleBuffer(ab) {
  const ctx = st.ctx;
  const src = ctx.createBufferSource();
  src.buffer = ab;
  ensureOutput(ctx);
  src.connect(st.outBus); // 분석기를 거쳐 출력(구체·파형용)
  const now = ctx.currentTime;
  // 드리프트·끊김 보정: 예정 시각이 이미 지났거나 너무 가까우면 약간 앞으로 다시 잡는다.
  if (st.nextTime < now + 0.02) st.nextTime = now + 0.06;
  src.start(st.nextTime);
  src._hvAt = st.nextTime; // hold 때 남은 부분 계산용
  st.nextTime += ab.duration;
  st.sources.add(src);
  src.onended = () => {
    if (st.sources.has(src)) st.pbBaseMs += ab.duration * 1000;
    st.sources.delete(src);
    if (st.sources.size === 0 && !st.held) {
      reportPlayback('idle');
      render();
    }
  };
  if (st.sources.size === 1) {
    reportPlayback('speaking');
    render();
  }
}

// AudioBuffer의 from 샘플 이후만 새 버퍼로
function sliceBuffer(ab, from) {
  if (from <= 0) return ab;
  const n = ab.length - from;
  const out = st.ctx.createBuffer(ab.numberOfChannels, n, ab.sampleRate);
  for (let c = 0; c < ab.numberOfChannels; c++) out.getChannelData(c).set(ab.getChannelData(c).subarray(from));
  return out;
}

// 서버 hold: 재생을 즉시 멈추되 안 들려준 부분은 보관. playback 보고는 하지 않는다(pbState 유지).
function onHold(gen) {
  if (gen !== st.gen || st.deadGens.has(gen)) return;
  if (st.held) return; // 이미 hold 중
  const ctx = st.ctx;
  const list = [...st.sources];
  const bufs = [];
  // 끊기 전까지 나간 부분을 위치에 더한다(hold 동안 위치 고정, 보고도 멈춤)
  if (ctx) st.pbBaseMs = Math.max(st.pbBaseMs, playedMs(true));
  if (ctx && list.length) {
    const plan = hvHoldPlan(list.map((x) => ({ at: x._hvAt, length: x.buffer.length, rate: x.buffer.sampleRate })), ctx.currentTime);
    for (const p of plan) bufs.push(sliceBuffer(list[p.i].buffer, p.from));
  }
  for (const x of list) {
    x.onended = null; // stop 전에 해제: idle 보고가 나가지 않게
    try { x.stop(); } catch { /* 이미 끝남 */ }
    try { x.disconnect(); } catch { /* */ }
  }
  st.sources.clear();
  st.nextTime = 0;
  st.held = { gen, bufs };
  console.info('[play] hold', gen, `${bufs.reduce((a, b) => a + b.duration, 0).toFixed(2)}s 보관`);
  render();
}

// 서버 hold(duck): 재생·위치 보고는 그대로, 출력 이득만 boost×gain으로 20 ms 램프
function onDuck(gen, gain) {
  if (gen !== st.gen || st.deadGens.has(gen) || st.held) return;
  st.duck = { gen, gain: Math.max(0, Math.min(1, gain)) };
  rampOutGain();
  console.info('[play] duck', gen, st.duck.gain);
}

function clearDuck() {
  if (!st.duck) return;
  st.duck = null;
  rampOutGain();
}

// 서버 resume: 덕킹이면 이득 복귀, 일시정지면 보관분을 now+0.06부터 순서대로 이어 붙인다.
function onResume(gen) {
  if (st.duck && st.duck.gen === gen) {
    clearDuck();
    console.info('[play] unduck', gen);
    return;
  }
  if (!st.held || gen !== st.gen || st.held.gen !== gen) return;
  const bufs = st.held.bufs;
  st.held = null;
  console.info('[play] resume', gen, `${bufs.reduce((a, b) => a + b.duration, 0).toFixed(2)}s`);
  if (st.ctx && bufs.length) {
    st.nextTime = 0; // scheduleBuffer가 now+0.06으로 잡는다
    for (const ab of bufs) scheduleBuffer(ab);
  } else if (!playing()) {
    reportPlayback('idle');
  }
  render();
}

function reportPlayback(state) {
  if (st.pbState === state) return;
  st.pbState = state;
  sendPlayback(state);
}

// 이 generation에서 지금까지 스피커로 나간 길이(ms). clamp면 시작 전 소스를 음수로 치지 않는다.
function playedMs(clamp = false) {
  const ctx = st.ctx;
  if (!ctx) return st.pbBaseMs;
  const items = [...st.sources].map((x) => ({ at: x._hvAt, dur: x.buffer.duration }));
  const v = hvPlayedMs(st.pbBaseMs, items, ctx.currentTime);
  return clamp ? Math.max(st.pbBaseMs, v) : v;
}

// playback 보고(서버 에코 대조용 재생 위치 pos_ms, 지원하면 출력 지연 out_latency_ms 포함)
function sendPlayback(state) {
  const m = { type: 'playback', state, generation: st.gen, pos_ms: Math.round(playedMs()) };
  const lat = st.ctx && st.ctx.outputLatency;
  if (typeof lat === 'number' && isFinite(lat) && lat >= 0) m.out_latency_ms = Math.round(lat * 1000);
  send(m);
}

// 재생 중 250 ms마다 위치 보고. hold 중(st.held)에는 멈춘다(위치 고정), resume 뒤 다시.
setInterval(() => {
  if (st.wsOpen && st.pbState === 'speaking' && !st.held && playing()) sendPlayback('speaking');
}, 250);

function stopPlayback() {
  for (const s of st.sources) {
    s.onended = null;
    try { s.stop(); } catch { /* 이미 끝남 */ }
    try { s.disconnect(); } catch { /* */ }
  }
  st.sources.clear();
  st.held = null; // 보관분도 버린다
  clearDuck();
  st.nextTime = 0;
  reportPlayback('idle');
  st.pbBaseMs = 0;
}

// 재생 큐를 비우고 지금 generation을 죽인다. 클라이언트 interrupt와 서버 interrupt가 같이 쓴다.
// newGen: 서버가 알려 준 generation(그 뒤 오는 새 오디오용). 지금 것과 같으면 그것도 죽은 것으로 본다.
function stopLocal(newGen = null) {
  if (st.gen >= 0) st.deadGens.add(st.gen);
  if (st.deadGens.size > 50) st.deadGens = new Set([...st.deadGens].slice(-20));
  stopPlayback();
  if (typeof newGen === 'number' && newGen !== st.gen && !st.deadGens.has(newGen)) st.gen = newGen;
  st.speakEnded = true;
  if (st.botEl) {
    if (st.botEl.classList.contains('typing')) st.botEl.remove();
    else {
      st.botEl.classList.remove('streaming');
      // 끊긴 답: 화면에 나온 데까지 저장
      if (!st.botEl._hvSaved) { st.botEl._hvSaved = true; logItem('bot', st.botEl.textContent); }
    }
    st.botEl = null;
  }
  render();
}

function interrupt() {
  if (playing() || isBusy(displayState())) st.shrinkAt = performance.now();
  send({ type: 'interrupt' });
  stopLocal();
}

// ---------- 마이크 ----------

function newContext() {
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  ctx.onstatechange = () => {
    diag(`[audio] ctx ${ctx.state}`);
    if (ctx.state === 'interrupted') noteAway();
    else if (ctx.state === 'running') noteBack(`ctx ${ctx.state}`);
    checkSysHold();
  };
  st.ctx = ctx;
  ensureOutput(ctx);
  return ctx;
}

function ensureContext() {
  // 사용자 제스처 안에서 동기로 만들고 resume (iOS 자동재생 정책)
  if (!st.ctx) newContext();
  if (st.ctx.state !== 'running') st.ctx.resume().catch(() => {});
  return st.ctx;
}

function withTimeout(p, ms) {
  return Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), ms))]);
}

// 백그라운드 복귀: suspended·interrupted면 suspend() → resume(), 그래도 안 되면 컨텍스트와 재생 체인을 새로 만든다.
async function recoverContext() {
  const ctx = st.ctx;
  if (!ctx || ctx.state === 'running') return;
  diag(`[audio] 복구 시작 ${ctx.state}`);
  try { await withTimeout(ctx.suspend(), 1000); } catch { /* */ }
  try { await withTimeout(ctx.resume(), 1500); } catch { /* */ }
  if (st.ctx !== ctx || ctx.state === 'running') return;
  diag(`[audio] resume 실패, 컨텍스트를 새로 만듦 ${ctx.state}`, 'warn');
  // 옛 컨텍스트의 예약 소스는 버린다. hold 보관분(AudioBuffer)은 새 컨텍스트에서도 쓸 수 있어 남긴다.
  const held = st.held;
  stopPlayback();
  st.held = held;
  disconnectCapture();
  stopKeepAlive();
  st.outBus = st.analyser = st.outGain = st.limiter = st.earBus = null;
  try { ctx.onstatechange = null; ctx.close(); } catch { /* */ }
  const fresh = newContext(); // ensureOutput 포함
  diag(`[audio] 새 컨텍스트 ${fresh.state}`);
  if (st.conv) startKeepAlive();
  try { await withTimeout(fresh.resume(), 1500); } catch { /* 제스처가 필요하면 다음 탭에서(아래 pointerdown) */ }
  // 마이크 그래프(워크렛)는 recoverMic이 새 컨텍스트에 다시 잇는다
}

// ---------- 멈춘 컨텍스트 재생성(전송 감시 3단계) ----------
// iOS WebKit은 오디오 세션 중단(끝내기 → 백그라운드 → 복귀) 뒤 AudioContext를 running으로 두면서
// 그래프를 돌리지 않는 일이 있다(2026-11-03 17:46 실기: ctx running·track live인데 워클릿 process()가
// 한 번도 안 불림, 빈 입력 0, 새 트랙도 같음). 이때는 컨텍스트를 버리고 새로 만들어야 한다.
const CTX_REBUILD_MAX = 2; // 대화당
const NOTE_TAP = '구체를 한 번 더 눌러 주세요';
const NOTE_DEAD = '마이크가 멈췄습니다. 앱을 닫았다 다시 여세요';
const NOTE_RETRY = '마이크가 응답하지 않습니다. 구체를 눌러 대화를 끝냈다가 다시 시작해 보세요';
const NOTE_SYSMUTE = 'iOS가 마이크를 멈췄습니다(다른 앱·통화·알림). 돌아오면 자동으로 이어집니다';

// 복구 사다리 최종 실패 안내. 첫 실패는 끝내기→다시 시작 권유(실기에서 실제로 복구됨), 두 번째부터 앱 종료 권유.
function micDeadNote(withToast) {
  if (st.micNote !== NOTE_RETRY && st.micNote !== NOTE_DEAD) st.deadFails++;
  const txt = st.deadFails >= 2 ? NOTE_DEAD : NOTE_RETRY;
  setMicNote(txt);
  if (withToast) toast(txt, 7000);
}

function setMicNote(txt) {
  if (st.micNote === txt) return;
  st.micNote = txt;
  render();
}

// ms 동안 ctx.currentTime 증가분(초). 그 사이 컨텍스트가 바뀌거나 running이 아니면 null.
async function ctAdvance(ms) {
  const ctx = st.ctx;
  if (!ctx || ctx.state !== 'running') return null;
  const a = ctx.currentTime;
  await new Promise((r) => setTimeout(r, ms));
  if (st.ctx !== ctx || ctx.state !== 'running') return null;
  return ctx.currentTime - a;
}

// 옛 컨텍스트를 닫고 새 컨텍스트에 재생 체인(ensureOutput: outBus→analyser→outGain→limiter, earBus)과
// keep-alive를 다시 만든다. 동기 함수라 탭 핸들러 안에서 부르면 제스처 안에서 만들어지고 resume된다.
// 재생 큐(hold 보관분 포함)는 비운다. 캡처 그래프(워크렛 모듈 포함)는 다음 ensureMic이 새 컨텍스트에 만든다.
function swapContext(why) {
  const old = st.ctx;
  const n = ++st.ctxRebuildSeq;
  diag(`[mic] 컨텍스트 재생성 #${n} (${why}) 옛 ctx ${old ? `${old.state} ct ${old.currentTime.toFixed(3)}` : '-'}`, 'warn');
  stopPlayback();
  disconnectCapture();
  stopKeepAlive();
  st.outBus = st.analyser = st.outGain = st.limiter = st.earBus = null;
  st.ctx = null;
  if (old) {
    old.onstatechange = null;
    try { old.close().catch(() => {}); } catch { /* 이미 닫힘 */ }
  }
  const fresh = newContext(); // ensureOutput 포함. 옵션은 기존 newContext와 같음(기기 기본 샘플레이트)
  try { fresh.resume().catch(() => {}); } catch { /* */ }
  if (st.conv) startKeepAlive();
  st.ctxStalled = false;
  st.ctxPending = false;
  st.diagCtx = null;
  return fresh;
}

// 새 컨텍스트에 캡처 그래프를 붙이고 전송을 다시 켠 뒤 pcm 프레임을 기다린다.
async function attachAndWait(alive) {
  const t = performance.now();
  try {
    await ensureMic();
  } catch (e) {
    diag(`[mic] 새 컨텍스트 캡처 그래프 실패 ${e && e.name} ${e && e.message}`, 'warn');
  }
  if (!alive()) return false;
  setTrackEnabled(true);
  captureCmd('start');
  if (await waitPcm(t, MIC_WATCH_MS, alive)) {
    diag(`[mic] 프레임 복구됨(새 컨텍스트), ${micDesc()}`);
    setMicNote('');
    return true;
  }
  if (alive()) {
    const ctx = st.ctx;
    diag(`[mic] 새 컨텍스트도 프레임 없음, 레벨 보고 ${st.lastLevelAt >= t ? '있음' : '없음'}, ct ${ctx ? ctx.currentTime.toFixed(3) : '-'}, ${micDesc()}`, 'warn');
  }
  return false;
}

// 3단계 한 번: 새 컨텍스트 → resume(1.5초) → 캡처 그래프 → 프레임 확인.
// resume이 안 되면(제스처 밖) 상태줄에 탭 안내를 띄우고 다음 구체 탭에서 다시 만든다(runPendingRebuild).
async function rebuildContext(why, alive) {
  if (st.convRebuilds >= CTX_REBUILD_MAX) {
    diag(`[mic] 컨텍스트 재생성 한도(${CTX_REBUILD_MAX}회) 도달, ${micDesc()}`, 'warn');
    micDeadNote(false);
    return false;
  }
  st.convRebuilds++;
  const ctx = swapContext(why);
  try { await withTimeout(ctx.resume(), 1500); } catch { /* */ }
  if (!alive() || st.ctx !== ctx) return false;
  if (ctx.state !== 'running') {
    st.ctxPending = true;
    diag(`[mic] 새 컨텍스트 ${ctx.state}: 제스처가 필요해 다음 탭에서 재생성`, 'warn');
    setMicNote(NOTE_TAP);
    return false;
  }
  return attachAndWait(alive);
}

// 3단계: 대화당 CTX_REBUILD_MAX번까지. 다 실패하면 앱을 다시 열라고 안내.
async function contextStage(why, alive) {
  while (alive()) {
    if (await rebuildContext(why, alive)) return true;
    if (!alive() || st.ctxPending) return false;
    if (st.convRebuilds >= CTX_REBUILD_MAX) {
      micDeadNote(true);
      return false;
    }
  }
  return false;
}

// 대기 중인 재생성(제스처 필요)을 구체 탭 안에서 실행한다. 컨텍스트는 탭 핸들러 안에서 동기로 만든다.
function runPendingRebuild() {
  const seq = st.convSeq;
  const alive = () => st.conv && st.convSeq === seq && !st.micMuted;
  const ctx = swapContext('탭: 대기 중 재생성');
  setMicNote('');
  st.micFix = (async () => {
    try { await withTimeout(ctx.resume(), 1500); } catch { /* */ }
    if (!alive() || st.ctx !== ctx) return;
    if (ctx.state !== 'running') {
      st.ctxPending = true;
      diag(`[mic] 탭으로 만든 컨텍스트도 ${ctx.state}`, 'warn');
      setMicNote(NOTE_TAP);
      return;
    }
    if (await attachAndWait(alive)) return;
    if (alive()) await contextStage('탭 재생성 뒤에도 프레임 없음', alive);
  })().catch((e) => diag(`[mic] 탭 재생성 오류 ${e && e.name} ${e && e.message}`, 'warn')).finally(() => {
    st.micFix = null;
    render();
  });
  return st.micFix;
}

function micTrack() {
  return st.stream ? st.stream.getAudioTracks()[0] || null : null;
}

function micAlive() {
  const t = micTrack();
  return !!t && t.readyState === 'live';
}

function setTrackEnabled(on) {
  // 트랙은 실행 중 늘 켜 둔다. stop하면 권한을 다시 묻고(WebKit 327237), enabled=false로 두면 iOS가
  // 음성 처리 유닛을 내려 출력 경로가 수화부로 바뀌어 소리가 작아지는 것으로 보인다(v16 실기 보고).
  // 보내지 않을 때는 micSending()과 워클릿 abort가 전송을 막는다.
  const t = micTrack();
  if (t && !t.enabled) t.enabled = true;
  void on;
}

function captureReady() {
  return !!st.capture && !!st.ctx && st.capture.context === st.ctx;
}

function ensureMic() {
  if (micAlive() && captureReady()) return Promise.resolve();
  if (st.micReady) return st.micReady;
  st.micReady = setupMic().finally(() => { st.micReady = null; });
  return st.micReady;
}

function micConstraints() {
  return { echoCancellation: !st.aecOff, noiseSuppression: true, autoGainControl: true, channelCount: 1 };
}

async function acquireStream() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw Object.assign(new Error('이 브라우저는 마이크를 쓸 수 없어요(HTTPS 필요).'), { name: 'NoMedia' });
  }
  // audioSession.type은 지정하지 않는다(v16에서 지정했더니 두 번째 대화부터 소리가 작아짐 — 출력 경로가 바뀌는 것으로 추정).
  const stream = await navigator.mediaDevices.getUserMedia({ audio: micConstraints() });
  diag('[mic] getUserMedia 획득');
  return stream;
}

function watchTrack(track) {
  track.onmute = () => {
    diag(`[mic] track mute (재생 ${st.sources.size}, ctx ${st.ctx ? st.ctx.state : '-'})`);
    checkSysHold();
  };
  track.onunmute = () => {
    diag(`[mic] track unmute (재생 ${st.sources.size}, ctx ${st.ctx ? st.ctx.state : '-'})`);
    checkSysHold();
  };
  track.onended = () => {
    diag('[mic] track ended', 'warn');
    if (st.listening) abortListening();
    // 대화 중이고 화면에 보이면 제스처 없이 다시 받아 본다. 숨겨져 있으면 복귀 때 처리.
    if (st.conv && !document.hidden) recoverMic();
    render();
  };
}

async function setupMic() {
  const ctx = ensureContext();
  disconnectCapture();
  if (!micAlive()) {
    // 끝난(ended) 트랙만 버리고 새로 받는다. live 트랙은 재사용(권한 재요청 방지)
    releaseMic();
    st.stream = await acquireStream();
  }
  const stream = st.stream;
  const track = micTrack();
  if (track) {
    track.enabled = true;
    watchTrack(track);
  }
  if (st.ctx !== ctx) return setupMic(); // 기다리는 사이 컨텍스트가 바뀜
  st.source = ctx.createMediaStreamSource(stream);
  // 소리는 내지 않지만 그래프가 돌도록 0 이득 노드로 출력에 잇는다
  st.captureSink = ctx.createGain();
  st.captureSink.gain.value = 0;
  st.captureSink.connect(ctx.destination);

  let worklet = false;
  if (ctx.audioWorklet && window.AudioWorkletNode) {
    try {
      if (!ctx._hvWorklet) {
        await ctx.audioWorklet.addModule('worklet.js?v=' + APP_VERSION);
        ctx._hvWorklet = true; // 모듈은 컨텍스트마다 올린다
      }
      const node = new AudioWorkletNode(ctx, 'pcm-capture', {
        numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCount: 1, channelCountMode: 'explicit',
        processorOptions: { prebuf: true },
      });
      node.port.onmessage = (e) => onCaptureMsg(e.data);
      st.source.connect(node);
      node.connect(st.captureSink);
      st.capture = node;
      worklet = true;
    } catch (e) {
      console.warn('AudioWorklet 실패, ScriptProcessor로 대체', e);
    }
  }
  if (!worklet) {
    const sp = ctx.createScriptProcessor(4096, 1, 1);
    st.spDs = new HvDownsampler(ctx.sampleRate);
    st.spSending = true; // 워클릿과 같이 사전 버퍼용으로 바로 낸다
    let acc = 0, cnt = 0;
    sp.onaudioprocess = (e) => {
      const ch = e.inputBuffer.getChannelData(0);
      acc += hvRms(ch); cnt++;
      if (cnt * ch.length >= ctx.sampleRate / 20) { onCaptureMsg({ type: 'level', rms: acc / cnt, zero: acc === 0 }); acc = 0; cnt = 0; }
      if (st.spSending) st.spDs.push(ch, (buf) => onCaptureMsg({ type: 'pcm', buf }));
    };
    st.source.connect(sp);
    sp.connect(st.captureSink);
    st.capture = sp;
  }
}

// 캡처 그래프만 뗀다(스트림·트랙은 유지)
function disconnectCapture() {
  try { st.source && st.source.disconnect(); } catch { /* */ }
  try { st.capture && st.capture.disconnect(); } catch { /* */ }
  try { st.capture && st.capture.port && (st.capture.port.onmessage = null); } catch { /* */ }
  try { st.captureSink && st.captureSink.disconnect(); } catch { /* */ }
  st.source = st.capture = st.captureSink = st.spDs = null;
  st.spSending = false;
}

// 트랙까지 정리. 페이지를 완전히 닫을 때와 끝난 트랙을 버릴 때만.
function releaseMic() {
  disconnectCapture();
  if (st.stream) {
    for (const t of st.stream.getTracks()) {
      t.onended = t.onmute = t.onunmute = null;
      try { t.stop(); } catch { /* */ }
    }
  }
  st.stream = null;
}

// 대화 중 마이크 확인: 트랙이 끝났거나 캡처 그래프가 옛 컨텍스트면 다시 잇는다(제스처 없이).
// 실패하면 조용히 대화를 멈추고 구체 탭으로 다시 시작하게 둔다.
async function recoverMic() {
  if (!st.conv) return;
  if (micAlive() && captureReady()) return;
  try {
    await ensureMic();
    if (!st.conv) return;
    setTrackEnabled(!st.micMuted);
    if (!st.micMuted) captureCmd('start');
    diag('[mic] 복구됨');
  } catch (e) {
    diag(`[mic] 복구 실패 ${e && e.name} ${e && e.message}`, 'warn');
    if (st.conv) {
      stopConversation('마이크 복구 실패(recoverMic)');
      st.convPaused = true;
    }
  }
  render();
}

function captureCmd(type) {
  const c = st.capture;
  if (!c) return;
  if (type === 'start') st.capStartAt = performance.now();
  if (c.port) { c.port.postMessage({ type }); return; }
  // ScriptProcessor 폴백
  if (type === 'start') { if (!st.spSending) st.spDs.reset(); st.spSending = true; }
  else if (type === 'stop') {
    if (st.spSending) st.spDs.flush((buf) => onCaptureMsg({ type: 'pcm', buf }));
    st.spSending = false;
    onCaptureMsg({ type: 'flushed' });
  } else if (type === 'abort') { st.spSending = false; st.spDs.reset(); }
}

// ---------- 사전 버퍼 ----------
// 캡처 그래프가 올라온 뒤 대화가 켜지기 전까지의 프레임을 최근 PREBUF_MS만큼 보관했다가
// 구체 탭으로 새로 시작할 때 conversation on 직후 먼저 보낸다(탭하고 바로 말한 첫 음절 보존).
const PREBUF_MS = 2000;
const PREBUF_MAX = 16 * PREBUF_MS; // 16 kHz 샘플 수
function keepPrebuf(buf) {
  st.prebuf.push(buf);
  st.prebufSamples += buf.byteLength >> 1;
  while (st.prebuf.length > 1 && st.prebufSamples - (st.prebuf[0].byteLength >> 1) >= PREBUF_MAX) {
    st.prebufSamples -= st.prebuf.shift().byteLength >> 1;
  }
}
function dropPrebuf() {
  st.prebuf = [];
  st.prebufSamples = 0;
}
// 앞쪽의 전부 0인 프레임(트랙이 막 올라온 구간)은 버리고 나머지를 순서대로 보낸다. 보낸 ms를 돌려준다.
function sendPrebuf() {
  const bufs = st.prebuf;
  dropPrebuf();
  let i = 0;
  for (; i < bufs.length; i++) {
    const a = new Int16Array(bufs[i]);
    let nz = false;
    for (let j = 0; j < a.length; j++) if (a[j] !== 0) { nz = true; break; }
    if (nz) break;
  }
  let n = 0;
  for (; i < bufs.length; i++) {
    if (!st.ws || !st.wsOpen) break;
    st.ws.send(bufs[i]);
    n += bufs[i].byteLength >> 1;
  }
  return Math.round(n / 16);
}

function onCaptureMsg(m) {
  if (m.type === 'pcm') {
    st.lastPcmAt = performance.now();
    if (micSending()) st.ws.send(m.buf);
    else if (!st.conv && !st.listening) keepPrebuf(m.buf); // 대화 전: 보관만(전송 안 함)
  } else if (m.type === 'flushed') {
    if (st.ending) {
      st.ending = false;
      st.listening = false;
      if (!st.conv) setTrackEnabled(false);
      send({ type: 'end_utterance' });
      render();
    }
  } else if (m.type === 'level') {
    const now = performance.now();
    st.lastLevelAt = now;
    if (st.readyWait && !st.readyWait.frameAt) { st.readyWait.frameAt = now; checkReady(); }
    st.diagSq += m.rms * m.rms;
    st.diagN++;
    if (playing()) st.lastPlayAt = now;
    const zero = m.empty ? false : (m.zero !== undefined ? m.zero : m.rms === 0);
    if (m.empty || !zero) st.zeroRunAt = 0;
    else if (!st.zeroRunAt) st.zeroRunAt = now;
    if (!m.empty && !zero) {
      st.lastNonZeroAt = now;
      st.holdEndAt = 0;
      if (st.conv) st.trackSuspect = false; // 대화 중 실제 샘플이 들어옴: 트랙은 살아 있다
    }
    checkSysHold(now);
    if (m.empty) st.emptyN++; // 빈 입력은 무음 트랙 검사 대신 전송 감시(pcm 무프레임)가 처리
    else checkDeadTrack(zero, now);
    if (st.micMuted) { setLevel(0); return; }
    if (st.listening || st.conv) {
      const v = Math.min(1, Math.sqrt(m.rms) * 2.2);
      setLevel(st.conv && !micSending() ? v * 0.3 : v);
    }
  }
}

// 입력이 디지털 무음(샘플이 정확히 0)으로 이어지면 iOS가 죽은 트랙을 준 것으로 보고 트랙을 다시 받는다
// (2026-11-03 18:55 실기: 20분 백그라운드 뒤 ctx running·track live·워클릿 동작인데 샘플 전부 0, getUserMedia로 복구).
// - 시작 직후: 대화 시작 뒤 0이 아닌 샘플도 재생도 없이 0만 ZERO_START_MS(백그라운드 복귀 의심이면 1000) → 바로
// - 대화 중 한가함: 재생이 아니고 마지막 재생 뒤 ZERO_PLAY_GAP_MS 지난 시점부터 0이 ZERO_IDLE_MS
// - 재생 중·직후: iOS가 재생 시작·끝 뒤 몇 초 0을 보내다 저절로 돌아오므로(2026-11-03 서버 로그) ZERO_TRACK_MS
const ZERO_START_MS = 2000;
const ZERO_START_SUSPECT_MS = 1000;
const ZERO_IDLE_MS = 3000;
const ZERO_PLAY_GAP_MS = 3000;
const ZERO_TRACK_MS = 8000;
const ZERO_RETRY_MS = 20000;
const ZERO_VERIFY_MS = 1500; // 새 트랙·새 컨텍스트에서 0이 아닌 샘플을 기다리는 시간
const AWAY_SUSPECT_MS = 60000;
const MUTED_WAIT_MS = 15000; // 시스템 음소거·중단이 이만큼 이어지면 상태줄 안내

// 시스템 음소거·중단: iOS가 트랙을 muted로 두거나(통화·알림·다른 앱) 오디오 세션을 끊은 동안은 0 샘플이 정상이다.
// 이때 무음 복구 사다리를 돌리면 헛돌아 zeroFailed가 되므로(2026-11-03 21:05 실기) 판정을 보류한다.
// ctx suspended는 재생성이 탭을 기다리는 중(ctxPending)이면 보류로 치지 않는다(그 안내가 따로 있음).
function sysHoldWhy() {
  const t = micTrack();
  if (t && t.readyState === 'live' && t.muted) return 'track muted';
  const ctx = st.ctx;
  if (ctx && ctx.state === 'interrupted') return 'ctx interrupted';
  if (ctx && ctx.state === 'suspended' && !st.ctxPending) return 'ctx suspended';
  return '';
}
function checkSysHold(now = performance.now()) {
  const t = micTrack();
  if (t && t.readyState === 'live' && t.muted) { if (!st.muteSince) st.muteSince = now; }
  else st.muteSince = 0;
  const why = st.conv && !st.convStarting ? sysHoldWhy() : '';
  if (why) {
    if (!st.holdSince) {
      st.holdSince = now;
      st.zeroSince = 0;
      diag(`[mic] 시스템 음소거 중: 무음 판정 보류(${why}), ${micDesc()}`);
    } else if (now - st.holdSince >= MUTED_WAIT_MS && st.micNote !== NOTE_SYSMUTE) {
      diag(`[mic] 시스템 음소거·중단 ${Math.round((now - st.holdSince) / 1000)}초 지속(${why}): 상태줄 안내`);
      setMicNote(NOTE_SYSMUTE);
    }
    return true;
  }
  if (st.holdSince) {
    const ms = now - st.holdSince;
    st.holdSince = 0;
    if (!st.conv) { if (st.micNote === NOTE_SYSMUTE) setMicNote(''); return false; }
    // 외부 중단 뒤 복귀: 이번 대화에서 이미 실패했더라도 새 기회로 친다
    st.holdEndAt = now;
    st.zeroSince = 0;
    st.zeroTriedAt = 0;
    const wasFailed = st.zeroFailed;
    st.zeroFailed = false;
    if (st.micNote === NOTE_SYSMUTE || st.micNote === NOTE_RETRY || st.micNote === NOTE_DEAD) setMicNote('');
    diag(`[mic] 음소거 해제·복귀: 무음 복구 다시 허용(보류 ${(ms / 1000).toFixed(1)}초${wasFailed ? ', 실패 표시 해제' : ''}), ${micDesc()}`);
  }
  return false;
}

function checkDeadTrack(zero, now = performance.now()) {
  if (st.holdSince) { st.zeroSince = 0; return; } // 시스템 음소거·중단 중: 0 샘플을 세지 않음
  if (!st.conv || st.micMuted || !zero || st.micFix || st.convStarting || document.hidden || st.zeroFailed) {
    st.zeroSince = 0;
    return;
  }
  if (!st.zeroSince) st.zeroSince = now;
  if (st.zeroTriedAt && now - st.zeroTriedAt < ZERO_RETRY_MS) return;
  const dur = now - st.zeroSince;
  const startPhase = st.lastNonZeroAt < st.convStartAt && st.lastPlayAt < st.convStartAt && !playing();
  const startMs = st.trackSuspect ? ZERO_START_SUSPECT_MS : ZERO_START_MS;
  let why = '';
  const resumed = st.holdEndAt && st.zeroSince >= st.holdEndAt; // 외부 중단 복귀 뒤 실제 샘플이 아직 없음
  if (startPhase && dur >= startMs) {
    why = `시작 뒤 ${startMs / 1000}초${st.trackSuspect ? ', 백그라운드 복귀' : ''}`;
  } else if (resumed && !playing() && dur >= ZERO_IDLE_MS) {
    why = `외부 중단 복귀 뒤 ${Math.round(dur / 1000)}초`;
  } else if (!playing() && now - st.lastPlayAt >= ZERO_PLAY_GAP_MS
    && now - Math.max(st.zeroSince, st.lastPlayAt + ZERO_PLAY_GAP_MS) >= ZERO_IDLE_MS) {
    why = `대화 중 ${Math.round(dur / 1000)}초`;
  } else if (dur > ZERO_TRACK_MS) {
    why = `재생 중·직후 ${Math.round(dur / 1000)}초`;
  }
  if (!why) return;
  st.zeroTriedAt = now;
  st.zeroSince = 0;
  fixZeroTrack(why);
}

function fixZeroTrack(why) {
  if (st.micFix) return st.micFix;
  st.trackSuspect = false;
  st.micFix = runZeroFix(why).catch((e) => diag(`[mic] 무음 복구 오류 ${e && e.name} ${e && e.message}`, 'warn')).finally(() => {
    st.micFix = null;
    st.zeroSince = 0;
    render();
  });
  return st.micFix;
}

// t 이후 0이 아닌 샘플이 들어오면 true
async function waitNonZero(t, ms, alive) {
  const end = performance.now() + ms;
  while (performance.now() < end) {
    if (!alive()) return false;
    if (st.lastNonZeroAt >= t) return true;
    await new Promise((r) => setTimeout(r, 100));
  }
  return st.lastNonZeroAt >= t;
}

// 무음 트랙 복구: fixMic의 2단계(트랙 재획득)부터. 새 트랙도 0이면 3단계(새 컨텍스트), 그래도 0이면 안내.
async function runZeroFix(why) {
  const seq = st.convSeq;
  // 도중에 시스템 음소거·ctx interrupted가 오면 조용히 그만둔다(zeroFailed 안 켬, 복귀 때 다시 판정)
  const extHold = () => { const t = micTrack(); return (t && t.readyState === 'live' && t.muted) || (st.ctx && st.ctx.state === 'interrupted'); };
  const alive = () => st.conv && st.convSeq === seq && !st.micMuted && !extHold();
  if (!alive() || st.zeroFailed) return;
  const t0 = micTrack();
  diag(`[mic] 무음 트랙(${why}, 0 샘플만): 트랙 다시 받음, muted=${t0 ? t0.muted : '-'} ${micDesc()}`, 'warn');
  toast('마이크를 다시 연결해요');
  releaseMic();
  const t2 = performance.now();
  try {
    await ensureMic();
  } catch (e) {
    diag(`[mic] 트랙 재획득 실패 ${e && e.name} ${e && e.message}`, 'warn');
    if (alive()) {
      stopConversation('무음 트랙 재획득 실패');
      st.convPaused = true;
      toast(micErrorText(e), 7000);
    }
    return;
  }
  if (!alive()) return;
  setTrackEnabled(true);
  captureCmd('start');
  if (await waitNonZero(t2, ZERO_VERIFY_MS, alive)) { diag(`[mic] 무음 복구됨(트랙 재획득), ${micDesc()}`); return; }
  if (!alive()) return;
  diag(`[mic] 새 트랙도 ${ZERO_VERIFY_MS / 1000}초 동안 0 샘플: 컨텍스트를 새로 만듦, ${micDesc()}`, 'warn');
  const t3 = performance.now();
  if (!await contextStage(`${why}, 새 트랙도 0 샘플`, alive)) { // 실패·탭 대기는 contextStage가 안내
    if (alive() && !st.ctxPending) st.zeroFailed = true; // 3단계 실패: 이 대화에서는 더 시도 안 함
    return;
  }
  if (await waitNonZero(t3, ZERO_VERIFY_MS, alive)) { diag(`[mic] 무음 복구됨(새 컨텍스트), ${micDesc()}`); return; }
  if (!alive()) return;
  st.zeroFailed = true; // 이 대화에서는 다시 시도하지 않음(권한 프롬프트 반복 방지). 다음 대화 시작 때 초기화
  diag(`[mic] 무음 복구 실패: 새 컨텍스트에서도 0 샘플, ${micDesc()} — 이 대화에서는 더 시도하지 않음(외부 중단 뒤 복귀하면 다시 허용)`, 'warn');
  micDeadNote(true);
}

// 백그라운드(visibility hidden)나 ctx interrupted가 AWAY_SUSPECT_MS 넘게 이어졌다 돌아오면 트랙을 의심한다.
// 홈 화면 앱에서 getUserMedia는 권한 프롬프트를 다시 띄우므로 미리 다시 받지 않고, 다음 대화의 무음 판정만 당긴다.
function noteAway() {
  if (!st.awaySince) st.awaySince = performance.now();
}
function noteBack(why) {
  if (!st.awaySince || document.hidden) return;
  const ms = performance.now() - st.awaySince;
  st.awaySince = 0;
  if (ms > AWAY_SUSPECT_MS) {
    st.trackSuspect = true;
    diag(`[mic] ${Math.round(ms / 1000)}초 만에 복귀(${why}): 트랙 의심, 다음 무음 판정 ${ZERO_START_SUSPECT_MS} ms`);
  }
}

function micErrorText(e) {
  const n = e && e.name;
  if (n === 'NotAllowedError' || n === 'SecurityError') {
    return '마이크 권한이 거부됐어요.\n설정 앱 > Safari > 마이크(또는 주소창 왼쪽 "가가" > 웹사이트 설정)에서 허용한 뒤 다시 눌러 주세요.';
  }
  if (n === 'NotFoundError') return '마이크를 찾지 못했어요.';
  if (n === 'NotReadableError') return '마이크를 다른 앱이 쓰고 있어요. 통화나 녹음을 끝내고 다시 눌러 주세요.';
  return (e && e.message) || '마이크를 켜지 못했어요.';
}

function startListening() {
  if (!st.wsOpen) {
    toast('서버에 연결되지 않았어요. 다시 연결하는 중이에요.');
    reconnectNow('듣기 시작');
    return false;
  }
  if (!st.capture) return false;
  st.listening = true;
  st.ending = false;
  st.botEl = null;
  setTrackEnabled(true);
  dropPrebuf(); // 누르고 말하기는 사전 버퍼를 쓰지 않음
  send({ type: 'start_utterance' });
  captureCmd('start');
  render();
  return true;
}

function stopListening() {
  st.toggleMode = false;
  if (!st.listening || st.ending) return;
  st.ending = true;
  captureCmd('stop'); // flushed 받으면 end_utterance
  // 워클릿이 응답하지 않을 때 대비
  setTimeout(() => { if (st.ending) onCaptureMsg({ type: 'flushed' }); }, 500);
  render();
}

function abortListening() {
  // 연결이 끊기거나 마이크가 죽었을 때: 서버에 보낼 수 있으면 end_utterance로 마무리
  st.toggleMode = false;
  if (!st.listening) return;
  captureCmd('abort');
  st.ending = false;
  st.listening = false;
  if (!st.conv) setTrackEnabled(false);
  send({ type: 'end_utterance' });
  render();
}

// ---------- 핸즈프리 대화 ----------

async function startConversation(why = '') {
  if (st.conv || st.convStarting) return;
  diag(`[conv] 시작 요청(${why || '-'}), ws ${st.wsOpen ? 'open' : 'closed'}, ctx ${st.ctx ? st.ctx.state : '-'}`);
  st.convRebuilds = 0;
  st.micNote = '';
  // 복귀 점검에서 멈춘 컨텍스트로 표시됐거나 재생성이 탭을 기다리면 제스처 안에서 바로 새로 만든다
  if (st.ctx && (st.ctxStalled || st.ctxPending)) {
    st.convRebuilds = 1;
    swapContext(st.ctxPending ? '대화 시작: 대기 중 재생성' : '대화 시작: 복귀 때 ct 정지');
  }
  const ctx = ensureContext(); // 제스처 안에서 동기로(resume 포함)
  if (!st.wsOpen) {
    toast('서버에 연결되지 않았어요. 다시 연결하는 중이에요.');
    reconnectNow('대화 시작');
    return;
  }
  resetConvState();
  st.zeroFailed = false; // 무음 복구 실패 표시는 새 대화마다 초기화
  const tap = why === '구체 탭';
  const tapAt = tap ? st.tapAt : 0;
  const seq = ++st.convSeq;
  st.convStarting = true;
  st.convPaused = false;
  setMicMuted(false); // 새 대화는 무음 해제
  setSoundOff(false);
  // 캡처 그래프는 대화마다 새로 만든다(트랙은 재사용). 옛 워클릿 노드가 끊기거나 멈춘 채
  // captureReady()만 true로 남아 프레임이 0개가 되는 경로를 없앤다.
  if (!st.micReady) disconnectCapture();
  dropPrebuf(); // 탭 이전 프레임은 보내지 않는다(새 캡처 그래프부터 보관)
  clearReadyWait();
  if (tap) st.readyWait = { tapAt, frameAt: 0, timer: 0 };
  render();
  try {
    await ensureMic();
    // 재시작 때 컨텍스트가 suspended·interrupted로 남아 있으면 워클릿 process가 돌지 않는다(프레임 0개).
    // ensureContext의 resume은 기다리지 않으므로 여기서 확인하고, 안 깨어나면 새 컨텍스트로 바꾼다.
    if (st.ctx.state !== 'running') {
      diag(`[mic] 시작 시 컨텍스트 ${st.ctx.state}`, 'warn');
      await wakeContext(ctx === st.ctx ? 600 : 1500);
      if (!captureReady()) await ensureMic();
    }
  } catch (err) {
    st.convStarting = false;
    clearReadyWait();
    diag(`[conv] 시작 실패 ${err && err.name} ${err && err.message}`, 'warn');
    toast(micErrorText(err), 7000);
    addLine('sys err', micErrorText(err));
    render();
    return;
  }
  st.convStarting = false;
  if (seq !== st.convSeq) return;
  if (!st.capture) { clearReadyWait(); render(); return; }
  st.conv = true;
  st.convStartAt = performance.now();
  st.vadSpeech = false;
  st.botEl = null;
  setTrackEnabled(true);
  send({ type: 'conversation', on: true });
  if (tap) {
    const readyMs = Math.round(performance.now() - tapAt);
    const preMs = sendPrebuf(); // conversation on 직후, 실시간 프레임보다 먼저
    diag(`[conv] 사전 버퍼 ${preMs} ms 전송(탭→준비 완료 ${readyMs} ms)`);
  } else dropPrebuf();
  captureCmd('start'); // 사전 버퍼로 이미 내보내는 중이면 다운샘플러를 reset하지 않고 이어감
  startKeepAlive();
  armMicWatch();
  startDiag();
  diag(`[conv] 켜짐, ${micDesc()}`);
  addLine('sys', '대화를 시작했어요. 그냥 말씀하세요.');
  if (st.readyWait) {
    // 프레임이 끝내 안 오면 준비 표시를 3초 뒤 거둔다(무음·무프레임 복구는 전송 감시가 맡음)
    st.readyWait.timer = setTimeout(() => {
      if (!st.readyWait) return;
      diag('[conv] 준비 완료 미확인(3초 안에 마이크 프레임 없음)', 'warn');
      clearReadyWait();
      render();
    }, 3000);
    checkReady();
  }
  render();
}

// 첫 대화는 탭 뒤 getUserMedia(권한 창 포함)까지 0.8초 넘게 걸려, 그 사이 말하면 남지 않는다.
// 탭→첫 마이크 프레임이 READY_EARCON_MS를 넘으면 준비 완료 때 짧은 상승 2음으로 알린다(트랙 재사용이면 소리 없음).
const READY_EARCON_MS = 300;
function checkReady() {
  const r = st.readyWait;
  if (!r || !r.frameAt || !st.conv) return;
  const ms = Math.round(r.frameAt - r.tapAt);
  const beep = ms > READY_EARCON_MS && st.earcon && !st.soundOff;
  clearReadyWait();
  if (beep) earcon('ready');
  diag(`[conv] 준비 완료 ${ms} ms(알림음 ${beep ? 'Y' : 'N'})`);
  render();
}

function clearReadyWait() {
  if (st.readyWait) clearTimeout(st.readyWait.timer);
  st.readyWait = null;
}

// 대화를 시작·끝낼 때 남은 상태를 비운다(재생 큐, 듣기, 무음 감시, 전송 감시)
function resetConvState() {
  clearTimeout(st.micWatch);
  st.micWatch = 0;
  clearTimeout(st.ctWatch);
  st.ctWatch = 0;
  st.zeroTriedAt = 0;
  st.zeroSince = 0;
  st.holdSince = 0;
  st.holdEndAt = 0;
  st.stallTried = false;
  st.vadSpeech = false;
  st.emptyN = 0;
  st.toggleMode = false;
  st.ending = false;
  if (st.listening) abortListening();
  st.listening = false;
  // 멈춘 컨텍스트에 예약된 소스는 onended가 오지 않아 playing()이 true로 남는다(반이중 게이트를 막음)
  if (st.sources.size && (!st.ctx || st.ctx.state !== 'running')) stopPlayback();
  if (!st.sources.size && !st.held) {
    st.nextTime = 0;
    reportPlayback('idle');
  }
}

function stopConversation(why = '') {
  if (!st.conv) return;
  diag(`[conv] 끝(${why || '-'})`);
  st.conv = false;
  clearReadyWait();
  stopKeepAlive();
  st.convSeq++; // 진행 중인 전송 복구를 무효로
  st.vadSpeech = false;
  captureCmd('abort'); // 전송 중지
  setTrackEnabled(false); // stop하지 않음(다음 대화에서 재사용)
  stopDiag();
  send({ type: 'conversation', on: false });
  stopLocal(); // 서버가 턴을 정리하므로 interrupt는 따로 보내지 않는다
  resetConvState();
  st.serverState = 'idle';
  setLevel(0);
  render();
}

// ---------- 전송 감시 ----------
// 시작 뒤 1.5초 안에 워클릿 pcm 프레임이 하나도 없으면
//  1단계: 컨텍스트를 깨우고(안 되면 새 컨텍스트) 캡처 그래프를 다시 만든다
//  2단계: 그래도 1.5초 안에 없으면 트랙을 버리고 다시 받는다(권한 창이 뜰 수 있음)
//  3단계: 새 트랙에도 1.5초 안에 레벨 보고(빈 입력 포함)가 없으면 컨텍스트 렌더링이 멈춘 것으로 보고
//         컨텍스트를 새로 만든다(대화당 최대 2회). running인데 currentTime이 0.5초 넘게 안 늘면 바로 3단계.
// 대화 중에도 보내야 할 때(micSending) 10초 넘게 프레임이 없으면 같은 복구를 한 번 한다.
const MIC_WATCH_MS = 1500;
const MIC_STALL_MS = 10000;

// 컨텍스트를 깨운다. ms 안에 running이 안 되면 recoverContext(suspend→resume→새 컨텍스트)
async function wakeContext(ms = 600) {
  const ctx = st.ctx;
  if (!ctx || ctx.state === 'running') return;
  try { await withTimeout(ctx.resume(), ms); } catch { /* */ }
  if (st.ctx === ctx && ctx.state !== 'running') await recoverContext();
}

function micDesc() {
  const t = micTrack();
  return `ctx ${st.ctx ? st.ctx.state : '-'}, track ${t ? t.readyState + (t.muted ? ' muted' : '') + (t.enabled ? '' : ' disabled') : '-'}, capture ${st.capture ? 'ok' : '-'}, 빈 입력 ${st.emptyN}`;
}

function armMicWatch() {
  clearTimeout(st.micWatch);
  clearTimeout(st.ctWatch);
  const t0 = performance.now();
  st.micWatch = setTimeout(() => {
    st.micWatch = 0;
    if (st.conv && !st.micMuted && st.lastPcmAt < t0) fixMic('시작 뒤 1.5초');
  }, MIC_WATCH_MS);
  // running인데 0.6초 동안 currentTime이 그대로면 1.5초를 기다리지 않고 바로 새 컨텍스트로
  const ctx0 = st.ctx;
  const ct0 = ctx0 ? ctx0.currentTime : 0;
  st.ctWatch = setTimeout(() => {
    st.ctWatch = 0;
    if (!st.conv || st.micMuted || st.micFix || st.ctx !== ctx0 || !ctx0 || ctx0.state !== 'running') return;
    if (ctx0.currentTime > ct0 || playing() || st.lastLevelAt >= t0) return;
    clearTimeout(st.micWatch);
    st.micWatch = 0;
    fixMic('시작 뒤 ct 정지 0.6초', true);
  }, 600);
}

function fixMic(why, frozen = false) {
  if (st.micFix) return st.micFix;
  diag(`[mic] fixMic(${why})`, 'warn');
  st.micFix = runMicFix(why, frozen).catch((e) => diag(`[mic] 복구 오류 ${e && e.name} ${e && e.message}`, 'warn')).finally(() => {
    st.micFix = null;
    render();
  });
  return st.micFix;
}

// t 이후 pcm 프레임이 오면 true. alive()가 false가 되면 바로 false.
async function waitPcm(t, ms, alive) {
  const end = performance.now() + ms;
  while (performance.now() < end) {
    if (!alive()) return false;
    if (st.lastPcmAt >= t) return true;
    await new Promise((r) => setTimeout(r, 100));
  }
  return st.lastPcmAt >= t;
}

async function runMicFix(why, frozen = false) {
  const seq = st.convSeq;
  const alive = () => st.conv && st.convSeq === seq && !st.micMuted;
  if (!alive()) return;
  // 컨텍스트 시간이 멈췄으면(running인데 0.5초 동안 currentTime 그대로) 1·2단계는 소용없다 → 바로 3단계
  if (!frozen && !playing()) {
    const d = await ctAdvance(500);
    if (!alive()) return;
    if (d !== null && d <= 0) frozen = true;
  }
  if (frozen) {
    diag(`[mic] 컨텍스트 시간 정지(running, ct 증가 0): 바로 새 컨텍스트로, ${micDesc()}`, 'warn');
    await contextStage(`${why}, ct 정지`, alive);
    return;
  }
  // 1단계
  diag(`[mic] 프레임 없음(${why}): 컨텍스트 재개·캡처 그래프 재구성, ${micDesc()}`, 'warn');
  await wakeContext(600);
  if (!alive()) return;
  disconnectCapture();
  const t1 = performance.now();
  try {
    await ensureMic();
  } catch (e) {
    diag(`[mic] 캡처 그래프 재구성 실패 ${e && e.name} ${e && e.message}`, 'warn');
  }
  if (!alive()) return;
  setTrackEnabled(true);
  captureCmd('start');
  if (await waitPcm(t1, MIC_WATCH_MS, alive)) { diag('[mic] 프레임 복구됨(그래프 재구성)'); return; }
  if (!alive()) return;
  // 2단계
  diag(`[mic] 그래도 프레임 없음: 트랙을 버리고 다시 받음, ${micDesc()}`, 'warn');
  toast('마이크를 다시 연결해요');
  releaseMic();
  const t2 = performance.now();
  try {
    await ensureMic();
  } catch (e) {
    diag(`[mic] 트랙 재획득 실패 ${e && e.name} ${e && e.message}`, 'warn');
    if (alive()) {
      stopConversation('micfix 트랙 재획득 실패');
      st.convPaused = true;
      toast(micErrorText(e), 7000);
    }
    return;
  }
  if (!alive()) return;
  setTrackEnabled(true);
  captureCmd('start');
  if (await waitPcm(t2, MIC_WATCH_MS, alive)) { diag('[mic] 프레임 복구됨(트랙 재획득)'); return; }
  if (!alive()) return;
  // 3단계: 새 트랙에도 워클릿 process()가 안 돈다(레벨 보고·빈 입력조차 없음) = 컨텍스트 렌더링 정지
  if (st.lastLevelAt < t2) {
    diag(`[mic] 새 트랙에도 레벨 보고 없음: 컨텍스트를 새로 만듦, ${micDesc()}`, 'warn');
    await contextStage(`${why}, 새 트랙 뒤 레벨 없음`, alive);
    return;
  }
  diag(`[mic] 복구 실패: 프레임이 여전히 없음(레벨 보고는 있음), ${micDesc()}`, 'warn');
  toast('마이크 입력이 들어오지 않아요. 끝내기를 누르고 앱을 다시 열어 주세요.', 6000);
}

// 대화 중 무프레임 감시(1초마다). 화면이 숨겨진 동안은 iOS가 캡처를 멈추므로 보지 않는다.
setInterval(() => {
  if (!st.conv || st.micFix || st.stallTried || st.convStarting || document.hidden) return;
  if (!micSending()) return;
  const now = performance.now();
  if (now - Math.max(st.lastPcmAt, st.capStartAt) > MIC_STALL_MS) {
    st.stallTried = true;
    fixMic('대화 중 10초');
  }
}, 1000);

// 대화 중 2초마다 입력 레벨과 오디오 상태를 콘솔에(화면에는 표시하지 않음)
function startDiag() {
  stopDiag();
  st.diagSq = 0;
  st.diagN = 0;
  st.diagCtx = null;
  st.diagTimer = setInterval(() => {
    const t = micTrack();
    // ctx.currentTime의 2초간 증가분: running인데 +0.000이면 렌더링이 멈춘 컨텍스트
    const ctx = st.ctx;
    const dCt = ctx && ctx === st.diagCtx ? ctx.currentTime - st.diagCt : null;
    st.diagCtx = ctx;
    st.diagCt = ctx ? ctx.currentTime : 0;
    const db = st.diagN ? hvDbfs(Math.sqrt(st.diagSq / st.diagN)).toFixed(1) : '-';
    st.diagSq = 0;
    st.diagN = 0;
    const since = st.lastPcmAt ? `${Math.round(performance.now() - st.lastPcmAt)}ms` : '-';
    diag(`[mic] ${db} dBFS ctx ${st.ctx ? st.ctx.state : '-'} ct=${dCt === null ? '-' : '+' + dCt.toFixed(3)} `
      + (t ? `track ${t.readyState} muted=${t.muted} enabled=${t.enabled}` : 'track -')
      + ` pcm ${since} 전 빈입력 ${st.emptyN} 송신 ${micSending() ? 'Y' : 'N'} 재생 ${st.sources.size} ka ${st.keepAlive ? 'on' : 'off'}`
      + ` mute=${st.micMuted} zeros=${st.zeroRunAt ? ((performance.now() - st.zeroRunAt) / 1000).toFixed(1) : '0'}s`
      + ` sysmute=${st.muteSince ? ((performance.now() - st.muteSince) / 1000).toFixed(1) : '0'}s${st.holdSince ? ' 보류' : ''}`);
    checkSysHold(); // ctx interrupted 동안은 레벨 보고가 없으므로 여기서도 본다
    if (dCt !== null && dCt <= 0 && ctx.state === 'running' && st.conv && !st.micMuted && !st.micFix && !st.convStarting
      && !document.hidden && !playing() && !st.ctxPending && st.convRebuilds < CTX_REBUILD_MAX) {
      fixMic('진단: ct 정지 2초', true);
    }
  }, 2000);
}

function stopDiag() {
  clearInterval(st.diagTimer);
  st.diagTimer = 0;
}

// ---------- 무음 ----------

function setMicMuted(on) {
  st.micMuted = !!on;
  if (st.conv) setTrackEnabled(!st.micMuted); // stop하지 않음
  if (st.micMuted) {
    st.vadSpeech = false;
    setLevel(0);
    if (st.conv) captureCmd('abort'); // 워클릿 전송 중지
  } else if (st.conv && st.capture) {
    captureCmd('start');
  }
}

function setSoundOff(on) {
  st.soundOff = !!on;
  const v = st.soundOff ? 0 : 1;
  if (st.outGain) st.outGain.gain.setValueAtTime(outGainValue(), st.outGain.context.currentTime);
  if (st.earBus) st.earBus.gain.setValueAtTime(v, st.earBus.context.currentTime);
}

el.micBtn.addEventListener('click', () => {
  if (!st.conv) return;
  setMicMuted(!st.micMuted);
  render();
});
el.soundBtn.addEventListener('click', () => {
  ensureContext();
  setSoundOff(!st.soundOff);
  render();
});
el.endBtn.addEventListener('click', () => {
  if (!st.conv) return;
  st.convPaused = false;
  stopConversation('끝내기 버튼');
});

// ---------- 버튼 ----------

async function onPressStart(e) {
  e.preventDefault();
  if (!st.ptt) {
    // 핸즈프리: 대화 전 탭은 시작. 대화 중엔 생각·말하는 중일 때만 끼어들기(끝내기는 종료 버튼)
    if (st.conv && st.ctxPending && !st.micFix && !st.micMuted) { runPendingRebuild(); return; }
    if (st.conv) {
      const s = displayState();
      if (s === 'speaking' || isBusy(s) || playing()) interrupt();
    } else {
      st.tapAt = performance.now();
      startConversation('구체 탭');
    }
    return;
  }
  if (st.pressing) return;
  try { el.talk.setPointerCapture(e.pointerId); } catch { /* */ }
  st.pressing = true;
  st.pressAt = performance.now();
  st.ignoreUp = false;
  ensureContext();

  // 토글 모드로 듣는 중이면 이번 탭은 "끝"
  if (st.listening && st.toggleMode) {
    st.ignoreUp = true;
    stopListening();
    return;
  }
  if (st.listening || st.ending) { st.ignoreUp = true; return; }

  // 재생·생각 중이면 끼어들기
  const s = displayState();
  if (playing() || s === 'thinking' || s === 'speaking' || s === 'transcribing') interrupt();

  if (!micAlive() || !captureReady()) {
    try {
      await ensureMic();
    } catch (err) {
      st.pressing = false;
      toast(micErrorText(err), 7000);
      addLine('sys err', micErrorText(err));
      return;
    }
    // 권한 창이 떠 있던 사이 손을 뗀 경우
    if (!st.pressing) {
      const held = st.releasedAt - st.pressAt;
      if (held < TAP_MS) {
        if (startListening()) st.toggleMode = true;
      } else {
        toast('마이크가 준비됐어요. 다시 눌러 말하세요.');
      }
      render();
      return;
    }
  }
  startListening();
}

function onPressEnd(e) {
  if (!st.pressing) return;
  st.pressing = false;
  st.releasedAt = performance.now();
  try { el.talk.releasePointerCapture(e.pointerId); } catch { /* */ }
  if (st.ignoreUp) { st.ignoreUp = false; return; }
  if (!st.listening) return; // 마이크 준비 중이었으면 onPressStart가 처리
  const held = st.releasedAt - st.pressAt;
  if (held < TAP_MS && e.type !== 'pointercancel') {
    st.toggleMode = true; // 짧은 탭: 계속 듣다가 다음 탭에서 끝
    render();
  } else {
    stopListening();
  }
}

el.talk.addEventListener('pointerdown', onPressStart);
el.talk.addEventListener('pointerup', onPressEnd);
el.talk.addEventListener('pointercancel', onPressEnd);
el.talk.addEventListener('contextmenu', (e) => e.preventDefault());
el.talk.addEventListener('touchstart', (e) => e.preventDefault(), { passive: false });

for (const b of el.boostBtns) {
  b.addEventListener('click', () => {
    st.boost = hvBoost(b.dataset.boost);
    localStorage.setItem(LS.boost, String(st.boost));
    if (st.outGain) st.outGain.gain.setValueAtTime(outGainValue(), st.outGain.context.currentTime);
    render();
  });
}

for (const b of document.querySelectorAll('.seg-btn[data-model]')) {
  b.addEventListener('click', () => {
    st.model = b.dataset.model;
    localStorage.setItem(LS.model, st.model);
    send({ type: 'set_model', model: st.model });
    render();
  });
}

el.textForm.addEventListener('submit', (e) => {
  e.preventDefault();
  ensureContext(); // 답 음성을 재생할 수 있게
  const text = el.textInput.value.trim();
  if (!text) return;
  if (!st.wsOpen) { toast('서버에 연결되지 않았어요.'); reconnectNow('텍스트 전송'); return; }
  if (playing() || st.serverState === 'thinking' || st.serverState === 'speaking') interrupt();
  send({ type: 'text', text });
  removeTyping();
  addLine('msg me', text);
  logItem('user', text);
  st.botEl = null;
  showTyping();
  el.textInput.value = '';
  el.textInput.blur();
});

// 접속 토큰 거부: 재연결을 멈추고 설정 시트의 토큰 칸을 연다(한 번만)
function onUnauthorized() {
  if (st.authFail) return;
  st.authFail = true;
  clearTimeout(st.retryTimer);
  st.retryTimer = 0;
  diag('[ws] 접속 토큰 거부(unauthorized)', 'warn');
  stopConversation('토큰 거부');
  renderToken();
  renderInfo();
  el.menu.hidden = false;
  setTimeout(() => { try { el.tokenInput.focus(); } catch { /* */ } }, 50);
  render();
}

function renderToken() {
  el.tokenInput.value = '';
  el.tokenInput.placeholder = getToken() ? '저장됨' : '';
  el.tokenForm.classList.toggle('need', st.authFail);
}

el.tokenForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const v = el.tokenInput.value.trim();
  const had = getToken();
  if (!v && !had && !st.authFail) return;
  try {
    if (v) localStorage.setItem(LS.token, v); else localStorage.removeItem(LS.token);
  } catch { /* */ }
  el.tokenInput.blur();
  toast(v ? '접속 토큰을 저장했어요.' : '접속 토큰을 지웠어요.');
  st.authFail = false;
  renderToken();
  // 새 토큰으로 다시 접속(열린 연결은 닫고 onclose가 재연결)
  if (st.ws && st.wsOpen) { try { st.ws.close(); } catch { /* */ } } else reconnectNow('토큰 저장');
  render();
});

// 메뉴
el.menuBtn.addEventListener('click', () => {
  renderToken();
  renderInfo();
  el.aec.checked = !st.aecOff;
  el.showText.checked = st.showText;
  el.pttMode.checked = st.ptt;
  el.halfDuplex.checked = st.halfDuplex;
  el.earcon.checked = st.earcon;
  el.menu.hidden = false;
});
el.menuClose.addEventListener('click', () => { el.menu.hidden = true; });
el.menu.addEventListener('click', (e) => { if (e.target === el.menu) el.menu.hidden = true; });
el.newSession.addEventListener('click', () => {
  if (st.listening) abortListening();
  interrupt();
  send({ type: 'new_session' });
  if (st.conv) { dropPrebuf(); send({ type: 'conversation', on: true }); } // 새 세션에서도 대화 유지
  st.sessionId = null;
  localStorage.removeItem(LS.session);
  localStorage.removeItem(LS.log); // 화면 기록은 여기서만 지운다
  st.chat = { session: null, items: [] };
  el.log.textContent = '';
  st.botEl = null;
  endChipTurn();
  clearToolBadge();
  addLine('sys', '새 대화를 시작했어요.');
  el.menu.hidden = true; // 헤더 + 버튼(예전엔 설정 시트 항목). 시트가 열려 있었다면 닫는다
});
el.aec.addEventListener('change', () => {
  // iOS에서 트랙을 다시 받는 경로가 죽은 트랙(-180 dBFS)을 주는 일이 있어(2026-11-02 실기),
  // 설정만 저장하고 마이크를 놓은 뒤 페이지를 새로 연다(권한 창이 다시 뜰 수 있음).
  st.aecOff = !el.aec.checked;
  localStorage.setItem(LS.aecOff, st.aecOff ? '1' : '0');
  toast(st.aecOff ? '에코 제거를 껐어요. 다시 시작할게요.' : '에코 제거를 켰어요. 다시 시작할게요.');
  if (st.listening) abortListening();
  if (st.conv) stopConversation('에코 캔슬 설정 변경');
  releaseMic();
  setTimeout(() => location.reload(), 300);
});

el.halfDuplex.addEventListener('change', () => {
  st.halfDuplex = el.halfDuplex.checked;
  localStorage.setItem(LS.halfDuplex, st.halfDuplex ? '1' : '0');
  sendSettings();
  toast(st.halfDuplex ? `${st.agentName}이 답하는 동안 마이크를 꺼요(끼어들기는 멈춤 버튼으로).` : '답하는 동안에도 마이크를 켜 둬요.');
  renderInfo();
  render();
});
el.earcon.addEventListener('change', () => {
  st.earcon = el.earcon.checked;
  localStorage.setItem(LS.earcon, st.earcon ? '1' : '0');
  if (st.earcon) { ensureContext(); earcon('heard'); } // 미리 들려 주기
});
el.showText.addEventListener('change', () => {
  st.showText = el.showText.checked;
  localStorage.setItem(LS.showText, st.showText ? '1' : '0');
  render();
});
el.pttMode.addEventListener('change', () => {
  st.ptt = el.pttMode.checked;
  localStorage.setItem(LS.ptt, st.ptt ? '1' : '0');
  if (st.conv) stopConversation('누르고 말하기 모드 변경');
  if (st.listening) abortListening();
  st.convPaused = false;
  st.pressing = false;
  toast(st.ptt ? '누르고 말하기 모드로 바꿨어요.' : '핸즈프리 대화 모드로 바꿨어요.');
  renderInfo();
  render();
});

// ---------- 백그라운드·복귀 (iOS) ----------

// 백그라운드에서는 WebKit이 마이크 캡처를 멈춘다(홈 화면 앱 포함). 숨겨질 때는 대화를 끝내지 않고
// 아무것도 파괴하지 않는다. 다시 보이면 오디오 컨텍스트 → 마이크 → 연결 순으로 복구한다.
function onForeground(why) {
  if (document.hidden) return;
  render();
  if (st.recovering) return;
  st.recovering = recoverAll(why).finally(() => { st.recovering = null; });
}

async function recoverAll(why) {
  diag(`[audio] 복귀 ${why} ctx ${st.ctx ? st.ctx.state : '-'}${st.conv ? ' conv' : ''} ws ${st.wsOpen ? 'open' : 'closed'}`);
  if (!st.wsOpen) reconnectNow(`복귀 ${why}`); // onopen이 hello(resume 세션 id)와 conversation on을 다시 보낸다
  if (!st.conv) {
    if (st.ctx && st.ctx.state !== 'running') st.ctx.resume().catch(() => {});
    if (st.convPaused && !st.ptt) toast('대화가 멈췄어요. 가운데 구체를 탭하면 다시 시작해요.', 5000);
    await checkCtxStall(why);
    return;
  }
  await recoverContext();
  await recoverMic();
  if (await checkCtxStall(why) && st.conv && !playing() && !st.micMuted) fixMic(`복귀 ${why}: ct 정지`, true);
  if (!st.wsOpen) reconnectNow(`복귀 ${why}(복구 뒤)`);
  render();
}

// 복귀 점검: running인데 currentTime이 300 ms 동안 안 늘면 렌더링이 멈춘 컨텍스트로 표시한다
// (다음 대화 시작 때 startConversation이 제스처 안에서 새로 만든다).
async function checkCtxStall(why) {
  const ctx = st.ctx;
  if (!ctx || ctx.state !== 'running' || document.hidden) return false;
  const a = ctx.currentTime;
  await new Promise((r) => setTimeout(r, 300));
  if (st.ctx !== ctx || ctx.state !== 'running' || document.hidden) return false;
  const d = ctx.currentTime - a;
  if (d > 0) return false;
  st.ctxStalled = true;
  diag(`[audio] 복귀 점검(${why}): ctx running인데 currentTime 정지(300ms 증가 ${d.toFixed(3)}) → 다음 대화 시작 때 새 컨텍스트`, 'warn');
  return true;
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    // 누르고 말하기 중이던 발화만 마무리. 대화·재생·마이크·연결은 그대로 둔다.
    if (st.listening) stopListening();
    st.pressing = false;
    noteAway();
    stopAnim();
    render();
  } else {
    st.capStartAt = performance.now(); // 숨겨진 동안의 무프레임은 감시에서 빼기
    noteBack('visible');
    onForeground('visible');
  }
});
window.addEventListener('pageshow', () => onForeground('pageshow'));
window.addEventListener('focus', () => onForeground('focus'));

// 진단: 아이폰이 보는 화면·뷰포트·안전 영역 크기(hello에 실어 서버 로그로 확인)
function viewportInfo() {
  try {
    const vv = window.visualViewport;
    const cs = getComputedStyle(document.documentElement);
    const probe = document.createElement('div');
    probe.style.cssText = 'position:fixed;left:0;top:0;width:0;height:0;padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left);visibility:hidden';
    document.body.appendChild(probe);
    const pcs = getComputedStyle(probe);
    const sa = [pcs.paddingTop, pcs.paddingRight, pcs.paddingBottom, pcs.paddingLeft].join(' ');
    probe.remove();
    const br = document.body.getBoundingClientRect();
    return {
      inner: [window.innerWidth, window.innerHeight], outer: [window.outerWidth, window.outerHeight],
      screen: [screen.width, screen.height, screen.availHeight], dpr: window.devicePixelRatio,
      client: [document.documentElement.clientWidth, document.documentElement.clientHeight],
      vv: vv ? [Math.round(vv.width), Math.round(vv.height), Math.round(vv.offsetTop), vv.scale] : null,
      body: [Math.round(br.top), Math.round(br.height)], scrollY: window.scrollY, safe: sa,
      vvh: cs.getPropertyValue('--vvh').trim(), standalone: navigator.standalone === true,
      ua: navigator.userAgent.slice(0, 80),
    };
  } catch (e) { return { err: String(e) }; }
}

// (v26~v30의 visualViewport 높이·위치 맞춤 코드는 키보드가 입력칸을 가려 제거. 2026-11-03)
// 제스처 없이 새로 만든 컨텍스트가 suspended로 남은 경우 다음 터치에서 깨운다
document.addEventListener('pointerdown', () => {
  if (st.ctx && st.ctx.state !== 'running') st.ctx.resume().catch(() => {});
}, true);
window.addEventListener('pagehide', (e) => {
  if (e.persisted) return; // bfcache로 들어가면 그대로 둔다
  // 페이지를 완전히 닫을 때만 정리
  if (st.listening) stopListening();
  if (st.conv) stopConversation('pagehide');
  releaseMic();
});
window.addEventListener('online', () => reconnectNow('online'));

// ---------- 시작 ----------

if ('serviceWorker' in navigator && location.protocol !== 'file:') {
  navigator.serviceWorker.register('sw.js').catch((e) => console.warn('SW 등록 실패', e));
}

// 설정 시트에 보일 서비스 워커 캐시 버전(sw.js의 VERSION): 앱이 새 버전으로 갱신됐는지 확인용.
// 지금 페이지를 맡고 있는 워커에게 물어본다(caches.keys()는 아이폰 홈 화면 앱에서 빈 결과였음, 2026-11-03).
function askSwVersion() {
  const sw = navigator.serviceWorker;
  if (!sw) return;
  const ctl = sw.controller;
  if (!ctl) return;
  try { ctl.postMessage({ type: 'version' }); } catch { /* */ }
}
if (navigator.serviceWorker) {
  navigator.serviceWorker.addEventListener('message', (e) => {
    const d = e.data || {};
    if (d.type === 'version' && typeof d.version === 'string') {
      st.swVer = d.version.replace('hermes-voice-', '');
      renderInfo();
    }
  });
  navigator.serviceWorker.addEventListener('controllerchange', askSwVersion);
  navigator.serviceWorker.ready.then(askSwVersion).catch(() => {});
}
st.chat = loadChat();
restoreChat();
diagStore();
render();
connect();
// 디버그 훅(콘솔·CDP 점검용): window.hv.recreateContext()는 3단계와 같은 경로로 컨텍스트를 새로 만든다
window.hv = {
  st,
  async recreateContext(why = '디버그') {
    const ctx = swapContext(why);
    try { await withTimeout(ctx.resume(), 1500); } catch { /* */ }
    let micErr = null;
    try { await ensureMic(); } catch (e) { micErr = `${e && e.name} ${e && e.message}`; }
    if (st.conv && st.capture) captureCmd('start');
    return {
      state: ctx.state, same: st.ctx === ctx, out: !!st.outBus && st.outBus.context === ctx && !!st.earBus && st.earBus.context === ctx,
      keepAlive: st.keepAlive ? st.keepAlive.context === ctx : null, capture: st.capture ? st.capture.context === ctx : null,
      worklet: !!ctx._hvWorklet, micErr,
    };
  },
  ctAdvance,
};
if (location.hash === '#settings') { renderInfo(); el.menu.hidden = false; } // 시안·점검용
