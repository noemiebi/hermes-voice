// 서비스 워커: 앱 껍데기는 캐시 우선. 파일을 고치면 VERSION을 올려야 새 껍데기가 내려간다.
// /ws(WebSocket)와 그 밖의 서버 경로는 건드리지 않는다.

const VERSION = 'hermes-voice-v45';
const SHELL = [
  './',
  'index.html',
  'app.css',
  'app.js',
  'worklet.js',
  'playcore.js',
  'manifest.json',
  'icon.svg',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'icons/apple-touch-icon.png',
];

// 설치 때는 HTTP 캐시를 우회해(cache: 'reload') 받는다. addAll은 브라우저 캐시를 거쳐서
// 새 VERSION에 옛 app.js·index.html이 담기는 일이 있었다(2026-11-02 아이폰 실기).
self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(VERSION)
      .then((c) => Promise.all(SHELL.map((u) => fetch(u, { cache: 'reload' }).then((res) => {
        if (!res.ok) throw new Error(`${u} ${res.status}`);
        return c.put(u, res);
      }))))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

// 페이지가 물으면 지금 워커의 VERSION을 알려 준다(설정 시트의 캐시 버전 표시).
self.addEventListener('message', (e) => {
  const d = e.data || {};
  if (d.type === 'version' && e.source) e.source.postMessage({ type: 'version', version: VERSION });
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname === '/ws' || url.pathname.startsWith('/ws/') || url.pathname.startsWith('/api/')) return;
  if (req.headers.get('upgrade')) return;
  e.respondWith(
    caches.match(req, { ignoreSearch: true }).then((hit) => {
      if (hit) return hit;
      return fetch(req).then((res) => {
        // 껍데기 목록에 있는 것만 새로 저장
        const rel = url.pathname.replace(/^\//, '') || './';
        if (res.ok && (SHELL.includes(rel) || req.mode === 'navigate')) {
          const copy = res.clone();
          caches.open(VERSION).then((c) => c.put(req, copy));
        }
        return res;
      }).catch(() => (req.mode === 'navigate' ? caches.match('./') : Response.error()));
    }),
  );
});
