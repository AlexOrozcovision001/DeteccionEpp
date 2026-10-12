const CACHE_NAME = 'epp-guard-shell-v15';
const APP_SHELL = [
  './',
  './index.html',
  './css/styles.css?v=15',
  './js/app.js?v=15',
  './manifest.webmanifest',
  './assets/icon-192.png',
  './assets/icon-512.png'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys.filter(key => key.startsWith('epp-guard-shell-') && key !== CACHE_NAME).map(key => caches.delete(key))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);

  // Los modelos y el runtime se obtienen de red para evitar duplicar cientos
  // de MB en la caché administrada por el service worker.
  if (url.pathname.endsWith('.onnx') || url.hostname.includes('jsdelivr.net')) return;

  event.respondWith(
    caches.match(event.request).then(cached => cached || fetch(event.request))
  );
});

