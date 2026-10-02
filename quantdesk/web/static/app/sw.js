/* QuantDesk service worker (published site only): the app opens instantly and works offline.
 * Everything is network-first, so a live desk is never shown stale while there's a connection;
 * the last good copy of each file (data.json included) is what you see offline. Fonts, which
 * never change, are cache-first. The cache is named after the app version, so an update
 * replaces it. */
"use strict";
const CACHE = "qd-__QD_VERSION__";
const SHELL = ["./", "index.html", "app.js", "lightweight-charts.js", "manifest.webmanifest", "icon-192.png",
	"fonts/plex-sans-latin.woff2", "fonts/plex-sans-latin-ext.woff2", "fonts/plex-mono-400.woff2", "fonts/plex-mono-600.woff2"];

self.addEventListener("install", (e) => {
	e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
	e.waitUntil(caches.keys()
		.then((keys) => Promise.all(keys.filter((k) => k.startsWith("qd-") && k !== CACHE).map((k) => caches.delete(k))))
		.then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
	const req = e.request;
	if (req.method !== "GET" || new URL(req.url).origin !== self.location.origin) return;
	if (/\/fonts\/[^/]+\.woff2$/.test(new URL(req.url).pathname)) {
		e.respondWith(caches.match(req).then((hit) => hit || fetch(req).then((res) => keep(req, res))));
		return;
	}
	e.respondWith(fetch(req).then((res) => keep(req, res)).catch(() =>
		caches.match(req, { ignoreSearch: true }).then((hit) => hit || (req.mode === "navigate" ? caches.match("index.html") : Response.error()))));
});

function keep(req, res) {
	if (res && res.ok && res.type === "basic") {
		const copy = res.clone();
		caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => { /* quota: fine */ });
	}
	return res;
}
