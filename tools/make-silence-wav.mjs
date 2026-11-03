// web/silence.wav 생성: 44.1 kHz 16-bit 모노 1초 무음(백그라운드 재생 유지용 <audio id="bgKeep"> 루프 소스)
// 사용: node tools/make-silence-wav.mjs
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const rate = 44100;
const n = rate; // 1초
const data = n * 2;
const b = Buffer.alloc(44 + data);
b.write('RIFF', 0); b.writeUInt32LE(36 + data, 4); b.write('WAVE', 8);
b.write('fmt ', 12); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22);
b.writeUInt32LE(rate, 24); b.writeUInt32LE(rate * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34);
b.write('data', 36); b.writeUInt32LE(data, 40);
const out = fileURLToPath(new URL('../web/silence.wav', import.meta.url));
writeFileSync(out, b);
console.log(out, b.length, 'bytes');
