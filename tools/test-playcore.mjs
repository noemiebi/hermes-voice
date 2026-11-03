// web/playcore.js 순수 함수 시험: node tools/test-playcore.mjs
import { createRequire } from 'node:module';
import assert from 'node:assert/strict';
const require = createRequire(import.meta.url);
const { hvHoldPlan, hvDbfs, hvBoost } = require('../web/playcore.js');

const R = 44100;
// 0.5초 조각 4개를 1.0초부터 이어 붙임
const items = [0, 1, 2, 3].map((k) => ({ at: 1.0 + 0.5 * k, length: R / 2, rate: R }));

// 재생 전: 전부 통째로
assert.deepEqual(hvHoldPlan(items, 0.9), [0, 1, 2, 3].map((i) => ({ i, from: 0 })));
// 1.2초: 첫 조각은 0.2초(8820샘플) 지난 곳부터, 나머지는 통째로
assert.deepEqual(hvHoldPlan(items, 1.2), [{ i: 0, from: 8820 }, { i: 1, from: 0 }, { i: 2, from: 0 }, { i: 3, from: 0 }]);
// 1.75초: 첫 조각은 끝나서 빠지고 둘째는 0.25초부터
assert.deepEqual(hvHoldPlan(items, 1.75), [{ i: 1, from: 11025 }, { i: 2, from: 0 }, { i: 3, from: 0 }]);
// 끝 직전(남은 샘플 < 32)은 버림
assert.deepEqual(hvHoldPlan([{ at: 0, length: 1000, rate: 1000 }], 0.99), []);
// 모두 끝남
assert.deepEqual(hvHoldPlan(items, 5), []);
// 입력 순서가 섞여도 시작 시각 순으로
const mixed = [items[2], items[0], items[1]];
assert.deepEqual(hvHoldPlan(mixed, 1.1).map((p) => p.i), [1, 2, 0]);
// 보관 + 이미 들은 양 = 전체
const now = 1.37;
const kept = hvHoldPlan(items, now).reduce((a, p) => a + (items[p.i].length - p.from), 0);
assert.equal(kept + Math.round((now - 1.0) * R), 4 * R / 2);

assert.equal(Math.round(hvDbfs(1)), 0);
assert.equal(Math.round(hvDbfs(0.1)), -20);
assert.equal(hvDbfs(0), -120);
assert.equal(hvBoost('2'), 2);
assert.equal(hvBoost(null), 1.5);
assert.equal(hvBoost('3'), 1.5);
console.log('playcore ok');
