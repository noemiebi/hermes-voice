// 마이크 캡처: 입력 샘플레이트(보통 48k) → 16k int16 mono, 약 100ms(1600샘플)씩 묶어 보낸다.
// 이 파일은 두 곳에서 쓴다.
//  1) AudioWorklet 모듈(audioWorklet.addModule) — PcmCaptureProcessor 등록
//  2) 일반 <script> — ScriptProcessorNode 폴백이 HvDownsampler를 전역으로 쓴다

const HV_OUT_RATE = 16000;
const HV_FRAME = 1600; // 100ms @16k

class HvDownsampler {
  constructor(inRate) {
    this.inRate = inRate;
    this.ratio = inRate / HV_OUT_RATE;
    // 간단한 앨리어싱 방지: 비율 길이만큼 이동 평균한 뒤 선형 보간
    this.k = Math.max(1, Math.round(this.ratio));
    this.reset();
  }

  reset() {
    this.ring = new Float32Array(this.k);
    this.ringPos = 0;
    this.ringSum = 0;
    this.prev = 0; // 직전 블록의 마지막 필터 값
    this.t = 0; // 다음 출력 샘플의 위치(현재 블록 기준, prev = -1)
    this.out = new Int16Array(HV_FRAME);
    this.outLen = 0;
  }

  // input: Float32Array. 꽉 찬 프레임마다 onFrame(ArrayBuffer) 호출.
  push(input, onFrame) {
    const n = input.length;
    const f = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      this.ringSum += input[i] - this.ring[this.ringPos];
      this.ring[this.ringPos] = input[i];
      this.ringPos = (this.ringPos + 1) % this.k;
      f[i] = this.ringSum / this.k;
    }
    // 위치 t: -1은 prev, 0..n-1은 f
    let t = this.t;
    while (t <= n - 1) {
      const i0 = Math.floor(t);
      const frac = t - i0;
      const a = i0 < 0 ? this.prev : f[i0];
      const b = f[i0 + 1 < n ? i0 + 1 : n - 1];
      let s = a + (b - a) * frac;
      if (s > 1) s = 1; else if (s < -1) s = -1;
      this.out[this.outLen++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.outLen === HV_FRAME) {
        onFrame(this.out.slice(0, HV_FRAME).buffer);
        this.outLen = 0;
      }
      t += this.ratio;
    }
    this.t = t - n;
    if (n > 0) this.prev = f[n - 1];
  }

  // 남은 샘플을 내보낸다(말 끝).
  flush(onFrame) {
    if (this.outLen > 0) onFrame(this.out.slice(0, this.outLen).buffer);
    this.outLen = 0;
  }
}

function hvRms(input) {
  let s = 0;
  for (let i = 0; i < input.length; i++) s += input[i] * input[i];
  return Math.sqrt(s / (input.length || 1));
}

if (typeof registerProcessor === 'function') {
  class PcmCaptureProcessor extends AudioWorkletProcessor {
    constructor(options) {
      super();
      this.ds = new HvDownsampler(sampleRate);
      // prebuf: 노드가 만들어지자마자 프레임을 낸다(메인이 대화 전엔 링 버퍼에만 보관, 탭 직후 말 보존)
      this.sending = !!(options && options.processorOptions && options.processorOptions.prebuf);
      this.levelAcc = 0;
      this.levelN = 0;
      this.port.onmessage = (e) => {
        const m = e.data || {};
        if (m.type === 'start') {
          // 이미 내보내는 중(사전 버퍼)이면 reset하지 않는다: 사전 버퍼와 실시간 프레임 경계가 끊기지 않게
          if (!this.sending) this.ds.reset();
          this.sending = true;
        } else if (m.type === 'stop') {
          // 남은 샘플을 보내고 끝났다고 알린다(메인은 그 뒤 end_utterance 전송)
          if (this.sending) this.ds.flush((buf) => this.port.postMessage({ type: 'pcm', buf }, [buf]));
          this.sending = false;
          this.port.postMessage({ type: 'flushed' });
        } else if (m.type === 'abort') {
          this.sending = false;
          this.ds.reset();
        }
      };
    }

    process(inputs) {
      const ch = inputs[0] && inputs[0][0];
      if (!ch) {
        // 입력이 비어 있음(소스가 끊겼거나 트랙이 데이터를 안 줌): 프레임은 못 내지만 process가 도는지는
        // 메인이 알 수 있게 약 50ms마다 빈 레벨을 알린다(전송 감시·진단용).
        this.emptyN = (this.emptyN || 0) + 128;
        if (this.emptyN >= sampleRate / 20) {
          this.emptyN = 0;
          this.port.postMessage({ type: 'level', rms: 0, empty: true });
        }
        return true;
      }
      // 레벨 미터: 약 50ms마다
      this.levelAcc += hvRms(ch);
      this.levelN++;
      if (this.levelN * ch.length >= sampleRate / 20) {
        // zero: 이 구간 샘플이 전부 정확히 0(디지털 무음, iOS가 죽은 트랙을 준 경우). 아주 작은 값과 구분.
        this.port.postMessage({ type: 'level', rms: this.levelAcc / this.levelN, zero: this.levelAcc === 0 });
        this.levelAcc = 0;
        this.levelN = 0;
      }
      if (this.sending) {
        this.ds.push(ch, (buf) => this.port.postMessage({ type: 'pcm', buf }, [buf]));
      }
      return true;
    }
  }
  registerProcessor('pcm-capture', PcmCaptureProcessor);
}
