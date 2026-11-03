// 재생 보관(hold) 계산: DOM·Web Audio 없이 쓰는 순수 함수. 브라우저(<script>)와 node 시험이 같이 쓴다.
'use strict';

// items: [{ at, length, rate }] 예약된 소스(at = ctx 시작 시각 초, length = 샘플 수, rate = Hz)
// now: ctx.currentTime. 반환: [{ i, from }] 시작 시각 순서로, 보관할 소스 번호와 남은 부분의 시작 샘플.
// 이미 다 재생된 소스(남은 샘플 < minLeft)는 빠진다. 아직 시작 안 한 소스는 from = 0(통째로).
function hvHoldPlan(items, now, minLeft = 32) {
  const order = items.map((it, i) => ({ ...it, i })).sort((a, b) => a.at - b.at || a.i - b.i);
  const out = [];
  for (const it of order) {
    const elapsed = now - it.at;
    const from = elapsed <= 0 ? 0 : Math.min(it.length, Math.round(elapsed * it.rate));
    if (it.length - from >= minLeft) out.push({ i: it.i, from });
  }
  return out;
}

// 이 generation에서 지금까지 스피커로 나간 오디오 길이(ms).
// base: 끝났거나 hold로 끊긴 부분의 합(ms). items: [{ at, dur }] 예약·재생 중인 소스(초). now: ctx.currentTime.
// 시작한 소스는 min(now - at, dur)만큼 더한다. 아무것도 시작 전이면 base + (now - 가장 이른 at)(음수 = 곧 시작).
function hvPlayedMs(base, items, now) {
  let sum = base;
  let started = false;
  let first = Infinity;
  for (const it of items) {
    first = Math.min(first, it.at);
    if (now >= it.at) {
      sum += Math.min(now - it.at, it.dur) * 1000;
      started = true;
    }
  }
  if (!started && items.length) sum += (now - first) * 1000;
  return sum;
}

// 선형 이득 → dBFS(무음은 -120)
function hvDbfs(rms) {
  return rms > 1e-6 ? 20 * Math.log10(rms) : -120;
}

// 소리 크기 설정값(1 · 1.5 · 2). 그 밖의 값은 기본 1.5
const HV_BOOSTS = [1, 1.5, 2];
function hvBoost(v) {
  const n = Number(v);
  return HV_BOOSTS.includes(n) ? n : 1.5;
}

if (typeof module !== 'undefined' && module.exports) module.exports = { hvHoldPlan, hvPlayedMs, hvDbfs, hvBoost, HV_BOOSTS };
