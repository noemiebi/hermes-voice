#!/usr/bin/env node
// 개발용 가짜 서버: web/ 정적 파일 + /ws (docs/프로토콜.md 흉내). 의존성 없음.
// 실행: node tools/dev-mock-server.mjs   (PORT 기본 8481, 진짜 서버 8480과 겹치지 않게)
// 가짜 답은 생각 지연 MOCK_THINK_MS(기본 5000)와 tool 이벤트 1회를 넣어 준비 중 UI를 확인할 수 있다.
// 잠정 끼어들기 흉내: 글 입력에 "보류"가 들어가면 3초 답 재생 0.8초 뒤 hold → 1초 뒤 resume,
// "끊기"가 들어가면 hold → 1초 뒤 interrupt. hold 중에도 남은 오디오 조각을 보내 클라이언트 보관을 시험한다.
// "덕킹"이 들어가면 hold를 mode duck(gain 0.3)으로 보내고 0.5초 뒤 resume("덕킹 끊기"면 interrupt).
// 덕킹은 재생이 멈추지 않으므로 소리가 작아졌다 돌아오는지, playback pos_ms 보고가 이어지는지 본다.
import http from 'node:http';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const PORT = Number(process.env.PORT || 8481);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'web');
const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png',
};

const server = http.createServer((req, res) => {
  let p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (p === '/') p = '/index.html';
  const f = path.join(ROOT, p);
  if (!f.startsWith(ROOT + path.sep)) { res.writeHead(403).end(); return; }
  fs.readFile(f, (err, data) => {
    if (err) { res.writeHead(404).end('not found'); return; }
    res.writeHead(200, { 'content-type': MIME[path.extname(f)] || 'application/octet-stream', 'cache-control': 'no-cache' });
    res.end(data);
  });
});

// ---------- 최소 WebSocket ----------

function encodeFrame(opcode, payload) {
  const len = payload.length;
  let head;
  if (len < 126) { head = Buffer.alloc(2); head[1] = len; }
  else if (len < 65536) { head = Buffer.alloc(4); head[1] = 126; head.writeUInt16BE(len, 2); }
  else { head = Buffer.alloc(10); head[1] = 127; head.writeBigUInt64BE(BigInt(len), 2); }
  head[0] = 0x80 | opcode;
  return Buffer.concat([head, payload]);
}

class Conn {
  constructor(sock) {
    this.sock = sock;
    this.buf = Buffer.alloc(0);
    this.frag = null; // { opcode, parts }
    this.closed = false;
    sock.on('data', (d) => { this.buf = Buffer.concat([this.buf, d]); this.parse(); });
    sock.on('close', () => { this.closed = true; this.onclose?.(); });
    sock.on('error', () => {});
  }

  parse() {
    for (;;) {
      if (this.buf.length < 2) return;
      const b0 = this.buf[0], b1 = this.buf[1];
      const fin = (b0 & 0x80) !== 0, opcode = b0 & 0x0f, masked = (b1 & 0x80) !== 0;
      let len = b1 & 0x7f, off = 2;
      if (len === 126) { if (this.buf.length < 4) return; len = this.buf.readUInt16BE(2); off = 4; }
      else if (len === 127) { if (this.buf.length < 10) return; len = Number(this.buf.readBigUInt64BE(2)); off = 10; }
      const need = off + (masked ? 4 : 0) + len;
      if (this.buf.length < need) return;
      let payload = this.buf.subarray(off + (masked ? 4 : 0), need);
      if (masked) {
        const mask = this.buf.subarray(off, off + 4);
        payload = Buffer.from(payload);
        for (let i = 0; i < payload.length; i++) payload[i] ^= mask[i & 3];
      }
      this.buf = this.buf.subarray(need);
      if (opcode === 0x8) { this.close(); return; }
      if (opcode === 0x9) { this.raw(0xa, payload); continue; }
      if (opcode === 0xa) continue;
      if (opcode === 0x0) {
        if (!this.frag) continue;
        this.frag.parts.push(payload);
        if (fin) { const { opcode: op, parts } = this.frag; this.frag = null; this.deliver(op, Buffer.concat(parts)); }
        continue;
      }
      if (!fin) { this.frag = { opcode, parts: [payload] }; continue; }
      this.deliver(opcode, payload);
    }
  }

  deliver(opcode, payload) {
    if (opcode === 0x1) this.ontext?.(payload.toString('utf8'));
    else if (opcode === 0x2) this.onbinary?.(payload);
  }

  raw(opcode, payload) { if (!this.closed) this.sock.write(encodeFrame(opcode, payload)); }
  json(obj) { this.raw(0x1, Buffer.from(JSON.stringify(obj))); }
  binary(buf) { this.raw(0x2, buf); }
  close() { if (!this.closed) { try { this.sock.end(encodeFrame(0x8, Buffer.alloc(0))); } catch {} this.closed = true; } }
}

server.on('upgrade', (req, sock) => {
  if (new URL(req.url, 'http://x').pathname !== '/ws' || !req.headers['sec-websocket-key']) { sock.destroy(); return; }
  const accept = crypto.createHash('sha1').update(req.headers['sec-websocket-key'] + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest('base64');
  sock.write('HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n' + `Sec-WebSocket-Accept: ${accept}\r\n\r\n`);
  session(new Conn(sock));
});

// ---------- 가짜 음성 세션 ----------

const OUT_RATE = 24000; // 서버 기본(AUDIO_OUT_RATE)과 같게, ready.out_sample_rate로 알림
// 핸즈프리 흉내 VAD(RMS 기준, 0~1). 진짜 서버는 Silero VAD.
const VAD_RMS = Number(process.env.MOCK_VAD_RMS || 0.02);
const BARGE_RMS = Number(process.env.MOCK_BARGE_RMS || 0.05);
// 가짜 답의 생각 지연(첫 토큰까지). 준비 중 UI 확인용.
const THINK_MS = Number(process.env.MOCK_THINK_MS ?? 5000);
const START_S = 0.1, END_SILENCE_S = 0.7, MIN_SPEECH_S = 0.3, BARGE_S = 0.3;

function frameRms(b) {
  const n = b.length >> 1;
  if (!n) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) { const v = b.readInt16LE(i * 2) / 32768; s += v * v; }
  return Math.sqrt(s / n);
}

// 위로 올라가는 음(이어짐을 귀로 확인하기 쉬움). 앞 4바이트 generation
function sweepPcm(seconds, f0, f1, gen) {
  const n = Math.round(OUT_RATE * seconds);
  const b = Buffer.alloc(4 + n * 2);
  b.writeUInt32LE(gen, 0);
  let ph = 0;
  for (let i = 0; i < n; i++) {
    const f = f0 + (f1 - f0) * (i / n);
    ph += (2 * Math.PI * f) / OUT_RATE;
    const fade = Math.min(1, i / 400, (n - i) / 400);
    b.writeInt16LE(Math.round(Math.sin(ph) * 0.25 * fade * 32767), 4 + i * 2);
  }
  return b;
}

function chunks(whole, gen, seconds = 0.5) {
  const size = Math.round(OUT_RATE * seconds) * 2;
  const pcm = whole.subarray(4);
  const out = [];
  for (let off = 0; off < pcm.length; off += size) {
    const part = Buffer.alloc(4 + Math.min(size, pcm.length - off));
    part.writeUInt32LE(gen, 0);
    pcm.copy(part, 4, off, off + size);
    out.push(part);
  }
  return out;
}

function sinePcm(seconds, freq, gen) {
  const n = Math.round(OUT_RATE * seconds);
  const b = Buffer.alloc(4 + n * 2);
  b.writeUInt32LE(gen, 0);
  for (let i = 0; i < n; i++) {
    const fade = Math.min(1, i / 400, (n - i) / 400);
    b.writeInt16LE(Math.round(Math.sin((2 * Math.PI * freq * i) / OUT_RATE) * 0.25 * fade * 32767), 4 + i * 2);
  }
  return b;
}

function session(c) {
  let sessionId = null;
  let model = 'voice-fast';
  let gen = 0;
  let pcmBytes = 0;
  let timers = [];
  const log = (...a) => console.log(new Date().toISOString().slice(11, 23), ...a);
  const later = (ms, fn) => { timers.push(setTimeout(fn, ms)); };
  const cancel = () => { timers.forEach(clearTimeout); timers = []; };
  let state = 'idle';
  const status = (s) => { state = s; c.json({ type: 'status', state: s }); };
  // 핸즈프리
  let conv = false;
  let playback = 'idle';
  let inSpeech = false, speechS = 0, silenceS = 0, loudS = 0, bargeS = 0;
  const restState = () => (conv ? 'waiting' : 'idle');
  const resetVad = () => { inSpeech = false; speechS = silenceS = loudS = bargeS = 0; };

  function reply(userText) {
    const g = gen;
    let t = 0;
    later((t += 200), () => status('thinking'));
    if (/도구반복/.test(userText)) {
      // 같은 도구 4회(running→completed) 뒤 "\n\n"으로 시작하는 답(헤르메스 실제 출력 꼴): 칩 합치기·말풍선 앞 공백 확인용
      for (let i = 1; i <= 4; i++) {
        const id = `call_${g}_${i}`;
        later((t += 300), () => c.json({ type: 'tool', name: 'mcp__apple_calendar__calendar_events', state: 'started', toolCallId: id }));
        later((t += 300), () => c.json({ type: 'tool', name: 'mcp__apple_calendar__calendar_events', state: 'completed', toolCallId: id }));
      }
      later((t += 200), () => c.json({ type: 'tool', name: 'weather', state: 'started' }));
      later((t += 200), () => c.json({ type: 'tool', name: 'weather', state: 'completed' }));
      const ans = '\n\n이번 주 일정은 세 건이에요.\n\n\n\n수요일 오후 2시 회의, 목요일 저녁 약속, 금요일 오전 진료가 있어요.\n\n';
      for (const tok of ans.match(/[\s\S]{1,4}/gu)) later((t += 40), () => c.json({ type: 'assistant_delta', text: tok }));
      later((t += 50), () => {
        status('speaking');
        c.json({ type: 'speak_start', generation: g, text: ans.trim() });
        c.binary(sinePcm(0.5, 440, g));
        c.json({ type: 'speak_end', generation: g });
        c.json({ type: 'assistant_done', text: ans });
      });
      later((t += 700), () => status(restState()));
      return;
    }
    // 도구 이벤트 1회. "날씨/검색/도구"가 들어가면 web_search + 채움 말("잠깐 확인해 볼게요."), 아니면 weather(채움 말 없음)
    const filler = /날씨|검색|도구/.test(userText);
    const toolName = filler ? 'web_search' : 'weather';
    const toolAt = Math.min(1500, Math.round(THINK_MS * 0.3));
    later(t + toolAt, () => c.json({ type: 'tool', name: toolName, state: 'started' }));
    if (filler) {
      later(t + toolAt + 50, () => {
        c.json({ type: 'speak_start', generation: g, text: '잠깐 확인해 볼게요.' });
        c.binary(sinePcm(0.4, 660, g));
        c.json({ type: 'speak_end', generation: g });
      });
    }
    later(t + Math.round(THINK_MS * 0.8), () => c.json({ type: 'tool', name: toolName, state: 'completed' }));
    t += THINK_MS; // 생각 지연(MOCK_THINK_MS)
    const answer = `(가짜 ${model}) "${userText}"라고 하셨네요. 사인파로 대답할게요.`;
    const duck = /덕킹/.test(userText);
    const holdMode = /끊기/.test(userText) ? 'interrupt' : (/보류/.test(userText) || duck) ? 'resume' : null;
    if (holdMode) {
      // 3초 상승음: 앞 4조각(2초)을 먼저 보내고, 0.8초 뒤 hold, hold 중에 나머지 2조각, 1초 뒤 resume 또는 interrupt
      for (const tok of answer.match(/.{1,4}/gu)) later((t += 20), () => c.json({ type: 'assistant_delta', text: tok }));
      const parts = chunks(sweepPcm(3.0, 300, 900, g), g);
      later((t += 50), () => {
        status('speaking');
        c.json({ type: 'speak_start', generation: g, text: answer });
        parts.slice(0, 4).forEach((p) => c.binary(p));
      });
      later((t += 800), () => {
        log(`hold${duck ? '(duck)' : ''} gen ${g}`);
        c.json(duck ? { type: 'hold', generation: g, mode: 'duck', gain: 0.3 } : { type: 'hold', generation: g });
        status('listening');
      });
      later((t += 200), () => { parts.slice(4).forEach((p) => c.binary(p)); c.json({ type: 'speak_end', generation: g }); });
      later((t += duck ? 300 : 800), () => {
        if (holdMode === 'resume') {
          log(`resume gen ${g}`);
          c.json({ type: 'resume', generation: g });
          status('speaking');
          c.json({ type: 'assistant_done', text: answer });
        } else {
          gen++;
          log(`interrupt → gen ${gen}`);
          c.json({ type: 'interrupt', generation: gen });
          status(restState());
        }
      });
      if (holdMode === 'resume') later((t += 2500), () => status(restState()));
      return;
    }
    for (const tok of answer.match(/.{1,4}/gu)) later((t += 60), () => c.json({ type: 'assistant_delta', text: tok }));
    later((t += 50), () => {
      status('speaking');
      c.json({ type: 'speak_start', generation: g, text: answer });
      // 1초 사인파를 0.5초씩 두 조각
      const whole = sinePcm(1.0, 440, g);
      const half = Math.round(OUT_RATE * 0.5) * 2;
      const pcm = whole.subarray(4);
      for (let off = 0; off < pcm.length; off += half) {
        const part = Buffer.alloc(4 + Math.min(half, pcm.length - off));
        part.writeUInt32LE(g, 0);
        pcm.copy(part, 4, off, off + half);
        c.binary(part);
      }
      c.json({ type: 'speak_end', generation: g });
      c.json({ type: 'assistant_done', text: answer });
    });
    later((t += 1100), () => status(restState()));
  }

  function onConvFrame(b) {
    const dur = (b.length >> 1) / 16000;
    const rms = frameRms(b);
    const busy = playback === 'speaking' || state === 'thinking' || state === 'speaking' || state === 'transcribing';
    if (busy && !inSpeech) {
      // 끼어들기: 기준을 올려서 판정
      bargeS = rms > BARGE_RMS ? bargeS + dur : 0;
      if (bargeS >= BARGE_S) {
        cancel();
        gen++;
        log(`끼어들기 (rms ${rms.toFixed(3)}) → gen ${gen}`);
        c.json({ type: 'interrupt', generation: gen });
        inSpeech = true; speechS = bargeS; silenceS = 0; bargeS = 0;
        c.json({ type: 'vad', speech: true });
        status('listening');
      }
      return;
    }
    if (!inSpeech) {
      loudS = rms > VAD_RMS ? loudS + dur : 0;
      if (loudS >= START_S) {
        inSpeech = true; speechS = loudS; silenceS = 0;
        log(`말 시작 (rms ${rms.toFixed(3)})`);
        c.json({ type: 'vad', speech: true });
        status('listening');
      }
      return;
    }
    speechS += dur;
    silenceS = rms > VAD_RMS ? 0 : silenceS + dur;
    if (silenceS < END_SILENCE_S) return;
    const secs = speechS - silenceS;
    resetVad();
    c.json({ type: 'vad', speech: false });
    log(`말 끝: 음성 ${secs.toFixed(2)}초`);
    if (secs < MIN_SPEECH_S) { status('waiting'); return; }
    status('transcribing');
    later(300, () => {
      const text = `가짜 받아쓰기(핸즈프리) ${secs.toFixed(1)}초`;
      c.json({ type: 'transcript', text, ms: 300 });
      reply(text);
    });
  }

  c.ontext = (s) => {
    let m;
    try { m = JSON.parse(s); } catch { c.json({ type: 'error', message: 'JSON 아님' }); return; }
    log('<-', s.length > 200 ? s.slice(0, 200) + '…' : s);
    switch (m.type) {
      case 'hello':
        sessionId = m.resume || 'mock-' + crypto.randomUUID().slice(0, 8);
        c.json({ type: 'ready', session_id: sessionId, model, out_sample_rate: OUT_RATE });
        status(restState());
        break;
      case 'conversation':
        cancel();
        resetVad();
        conv = !!m.on;
        if (!conv) gen++; // 진행 중인 턴 정리
        status(restState());
        break;
      case 'playback':
        playback = m.state === 'speaking' ? 'speaking' : 'idle';
        break;
      case 'settings':
        break;
      case 'start_utterance':
        cancel();
        pcmBytes = 0;
        status('listening');
        break;
      case 'end_utterance': {
        const secs = pcmBytes / 2 / 16000;
        log(`발화 ${secs.toFixed(2)}초 (${pcmBytes} bytes)`);
        status('transcribing');
        if (secs < 0.2) { later(200, () => status(restState())); break; }
        later(300, () => {
          const text = `가짜 받아쓰기 ${secs.toFixed(1)}초`;
          c.json({ type: 'transcript', text, ms: 300 });
          reply(text);
        });
        break;
      }
      case 'text':
        cancel();
        reply(String(m.text || ''));
        break;
      case 'interrupt':
        cancel();
        gen++;
        resetVad();
        c.json({ type: 'interrupt', generation: gen }); // 프로토콜: 클라이언트 interrupt에도 같은 메시지로 답한다
        status(restState());
        break;
      case 'set_model':
        model = m.model === 'voice-deep' ? 'voice-deep' : 'voice-fast';
        break;
      case 'new_session':
        cancel();
        gen++;
        sessionId = 'mock-' + crypto.randomUUID().slice(0, 8);
        c.json({ type: 'ready', session_id: sessionId, model, out_sample_rate: OUT_RATE });
        status(restState());
        break;
      default:
        c.json({ type: 'error', message: '모르는 type: ' + m.type });
    }
  };
  // 진짜 서버의 1초 입력 로그 흉내: 마이크 프레임이 실제로 오는지 재시작 검증에 쓴다
  let binN = 0;
  const binLog = setInterval(() => { if (binN) log(`입력 ${binN}프레임/1초${conv ? '' : ' (대화 꺼짐)'}`); binN = 0; }, 1000);
  c.onbinary = (b) => { binN++; pcmBytes += b.length; if (conv) onConvFrame(b); };
  c.onclose = () => { clearInterval(binLog); cancel(); log('연결 끊김'); };
  log('연결됨');
}

server.listen(PORT, '127.0.0.1', () => console.log(`가짜 서버: http://127.0.0.1:${PORT}/  (web: ${ROOT})`));
