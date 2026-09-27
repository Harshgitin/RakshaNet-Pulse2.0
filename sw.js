/* RakshaNet offline-first service worker. */
const CACHE_NAME = 'rakshanet-static-v3';
const CORE_ASSETS = [
  '/',
  '/index.html',
  '/css/style.css',
  '/js/script.js',
  '/js/location.js',
  '/pages/intelligence.html',
  '/pages/map.html',
  '/pages/emergency.html',
  '/pages/help.html',
  '/pages/profile.html',
  '/pages/login.html',
  '/pages/register.html',
  '/pages/dashboard.html',
  '/pages/admin-login.html',
  '/js/notifications.js',
  '/favicon.svg'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(CORE_ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys
        .filter(key => key !== CACHE_NAME)
        .map(key => caches.delete(key))
    )).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.method !== 'GET') return;
  if (!request.url.startsWith(self.location.origin)) return;
  const pathname = new URL(request.url).pathname;
  // API responses and uploaded evidence are dynamic/user-specific and must
  // never be served from the static offline cache.
  if (pathname.startsWith('/api/') || pathname.startsWith('/uploads/')) return;

  event.respondWith(
    fetch(request)
      .then(response => {
        if (response.ok && request.destination !== 'image') {
          const copy = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(request, copy)).catch(() => {});
        }
        return response;
      })
      .catch(() => caches.match(request).then(cached => cached || caches.match('/index.html')))
  );
});


self.addEventListener('push', event => {
  event.waitUntil((async () => {
    let data = {};
    try {
      data = event.data ? event.data.json() : {};
    } catch (_) {
      data = { body: event.data ? event.data.text() : 'New RakshaNet warning.' };
    }

    const severity = String(data.severity || 'INFO').toUpperCase();
    const title = data.title || 'RakshaNet Early Warning';
    const options = {
      body: data.body || 'New disaster intelligence update available.',
      tag: `rakshanet-${data.type || 'alert'}-${severity}`,
      renotify: severity === 'EXTREME',
      requireInteraction: severity === 'EXTREME' || severity === 'HIGH',
      data: {
        url: data.url || '/pages/intelligence.html#alerts',
        score: data.score ?? null,
        location: data.location || null
      },
      icon: '/favicon.svg',
      badge: '/favicon.svg'
    };

    await self.registration.showNotification(title, options);
  })());
});

self.addEventListener('notificationclick', event => {
  event.notification.close();
  const target = new URL(event.notification.data?.url || '/pages/intelligence.html#alerts', self.location.origin).href;
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then(clientList => {
      const existing = clientList.find(client => 'focus' in client && client.url.startsWith(self.location.origin));
      if (existing) return existing.focus().then(() => existing.navigate?.(target));
      return clients.openWindow(target);
    })
  );
});
