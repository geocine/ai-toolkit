// App-shell service worker: lets the studio install to a phone home screen
// and open without a network. Only the studio's own static files are cached;
// the API (/v1, /api, /mcp, event streams) and anything cross-origin always
// go to the network, untouched.
const CACHE = 'aitk-studio-v1';
const SCOPE = new URL('./', self.location).pathname; // "/studio/" when served by the server
const SHELL = ['./', './manifest.json', './icon-192.png', './icon-512.png', './apple-touch-icon.png'];

self.addEventListener('install', event => {
  event.waitUntil(
    caches
      .open(CACHE)
      .then(c => c.addAll(SHELL))
      .catch(() => undefined)
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches
      .keys()
      .then(keys => Promise.all(keys.filter(k => k.startsWith('aitk-studio-') && k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

function isShell(url) {
  if (url.origin !== self.location.origin || !url.pathname.startsWith(SCOPE)) return false;
  const rest = url.pathname.slice(SCOPE.length);
  // never the API, whatever prefix the studio is served under
  return !/^(v1|api|mcp)(\/|$)/.test(rest);
}

self.addEventListener('fetch', event => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (!isShell(url)) return;

  if (req.mode === 'navigate') {
    // network first so a new build shows up; the cached shell when offline
    event.respondWith(
      fetch(req)
        .then(res => {
          if (res.ok) {
            const copy = res.clone();
            caches.open(CACHE).then(c => c.put('./', copy));
          }
          return res;
        })
        .catch(() => caches.match('./').then(r => r || Response.error())),
    );
    return;
  }
  if (url.pathname.startsWith(SCOPE + 'assets/')) {
    // content-hashed: cache first
    event.respondWith(
      caches.match(req).then(
        hit =>
          hit ||
          fetch(req).then(res => {
            if (res.ok) {
              const copy = res.clone();
              caches.open(CACHE).then(c => c.put(req, copy));
            }
            return res;
          }),
      ),
    );
    return;
  }
  // other shell files (icons, manifest): network, falling back to the cache
  event.respondWith(fetch(req).catch(() => caches.match(req).then(r => r || Response.error())));
});
