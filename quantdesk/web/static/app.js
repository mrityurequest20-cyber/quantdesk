/* QuantDesk desk UI.
 * Chart: GoCharting SDK (hosted UMD build + license key from /api/config) with the QuantDesk
 * datafeed and a broker bridge (setBrokerAccounts ← /api/broker; appCallback → /api/order…).
 * If the SDK can't load (no network, no license, blocked domain) a built-in candlestick chart
 * takes over, drawing the same data plus journal markers and open-trade stop/target levels.
 * Strings from the server are inserted with textContent only. */
"use strict";
const $ = (s) => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
let CFG = null, current = null, mode = "fallback", RANGE = 250;
let sdkChart = null, chartInstance = null, brokerTimer = null;

async function api(path, opt) {
	const r = await fetch(path, opt);
	const j = await r.json().catch(() => ({}));
	if (!r.ok) throw new Error(j.error || "HTTP " + r.status);
	return j;
}
const post = (p, body) => api(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const bare = (k) => String(k || "").split(":").pop();
const inr = (v) => {
	if (v == null || !isFinite(v)) return "—";
	const a = Math.abs(v), s = v < 0 ? "-" : "";
	return s + (a >= 1e7 ? "₹" + (a / 1e7).toFixed(2) + " Cr" : a >= 1e5 ? "₹" + (a / 1e5).toFixed(2) + " L" : "₹" + Math.round(a).toLocaleString("en-IN"));
};
const px = (v) => (v == null || !isFinite(v) ? "—" : Number(v).toLocaleString("en-IN", { maximumFractionDigits: 2 }));
function h(tag, attrs, ...kids) {
	const e = document.createElement(tag);
	for (const [k, v] of Object.entries(attrs || {})) {
		if (k === "class") e.className = v;
		else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
		else e.setAttribute(k, v);
	}
	for (const k of kids.flat()) if (k != null) e.appendChild(typeof k === "string" || typeof k === "number" ? document.createTextNode(String(k)) : k);
	return e;
}
function svg(tag, attrs, parent) {
	const e = document.createElementNS(NS, tag);
	for (const k in attrs) e.setAttribute(k, attrs[k]);
	if (parent) parent.appendChild(e);
	return e;
}
function toast(msg, bad) {
	const t = $("#toast");
	t.textContent = msg;
	t.style.color = bad ? "var(--bad)" : "var(--ink)";
	t.style.display = "block";
	clearTimeout(toast._t);
	toast._t = setTimeout(() => (t.style.display = "none"), 3500);
}
const tt = () => $("#tt");
function showTT(e, lines) {
	const el = tt();
	el.textContent = "";
	lines.forEach((l, i) => el.appendChild(h(i === 0 ? "div" : "div", i === 0 ? { style: "color:var(--ink2)" } : {}, l)));
	el.style.display = "block";
	el.style.left = Math.min(e.clientX + 14, innerWidth - el.offsetWidth - 8) + "px";
	el.style.top = e.clientY + 14 + "px";
}
const hideTT = () => (tt().style.display = "none");

// ---- boot -------------------------------------------------------------------------------------
async function boot() {
	CFG = await api("/api/config");
	current = CFG.default;
	$("#theme").addEventListener("click", toggleTheme);
	["t-buy", "t-sell"].forEach((id) => $("#" + id).addEventListener("click", () => ticket(id === "t-buy" ? "buy" : "sell")));
	const rs = $("#ranges");
	[["3M", 63], ["6M", 126], ["1Y", 250], ["3Y", 750]].forEach(([l, n]) =>
		rs.appendChild(h("button", { class: n === RANGE ? "on" : "", onclick: (e) => { RANGE = n; rs.querySelectorAll("button").forEach((b) => b.classList.remove("on")); e.target.classList.add("on"); if (mode === "fallback") drawFallback(); } }, l)));
	await Promise.all([renderWatchlist(), refreshStatus()]);
	setInterval(refreshStatus, 30000);
	const ok = CFG.gocharting.enabled && (await loadSDK(CFG.gocharting.sdkUrl, 12000));
	if (ok) mountGoCharting();
	else useFallback("GoCharting SDK unavailable — built-in chart");
	selectSymbol(current, true);
}

function loadSDK(url, timeout) {
	return new Promise((resolve) => {
		if (!url) return resolve(false);
		let done = false;
		const fin = (v) => { if (!done) { done = true; resolve(v); } };
		const s = document.createElement("script");
		s.src = url;
		s.async = true;
		s.onerror = () => fin(false);
		s.onload = () => {
			const t0 = Date.now();
			(function wait() {
				if (window.GoChartingSDK && typeof window.GoChartingSDK.createChart === "function") fin(true);
				else if (Date.now() - t0 > 5000) fin(false);
				else setTimeout(wait, 100);
			})();
		};
		setTimeout(() => fin(false), timeout);
		document.head.appendChild(s);
	});
}

function theme() {
	const t = document.documentElement.getAttribute("data-theme");
	if (t) return t;
	return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function toggleTheme() {
	document.documentElement.setAttribute("data-theme", theme() === "dark" ? "light" : "dark");
	if (mode === "gocharting") mountGoCharting();
	else drawFallback();
}

// ---- GoCharting mode ----------------------------------------------------------------------------
function mountGoCharting() {
	mode = "gocharting";
	$("#mode").textContent = "GoCharting SDK";
	const el = $("#chart");
	el.textContent = "";
	if (sdkChart && typeof sdkChart.destroy === "function") try { sdkChart.destroy(); } catch (e) { /* ignore */ }
	chartInstance = null;
	sdkChart = window.GoChartingSDK.createChart("#chart", {
		symbol: current,
		interval: "1D",
		datafeed: window.createQuantDeskDatafeed(""),
		licenseKey: CFG.gocharting.licenseKey,
		theme: theme(),
		enableTrading: CFG.paperOnly,
		trading: { showReverseButton: false, supportStopOrders: false, supportStopLimitOrders: false },
		appCallback: ({ eventType, message }) => onTradingEvent(eventType, message),
		onReady: (inst) => {
			chartInstance = inst;
			pushBroker();
			clearInterval(brokerTimer);
			brokerTimer = setInterval(pushBroker, 20000);
		},
		onError: (err) => {
			console.error("GoCharting error", err);
			useFallback("GoCharting error — built-in chart");
		},
	});
	$("#legend").textContent = "";
}

async function pushBroker() {
	if (!chartInstance || typeof chartInstance.setBrokerAccounts !== "function") return;
	try {
		chartInstance.setBrokerAccounts(await api("/api/broker"));
	} catch (e) {
		console.warn("broker sync failed", e);
	}
}

async function onTradingEvent(type, msg) {
	msg = msg || {};
	try {
		if (type === "PLACE_ORDER") {
			const o = msg.order || msg;
			if (o.exitPosition) await post("/api/close", { id: o.id || o.positionId || (o.position && o.position.id) });
			else await post("/api/order", { symbol: o.symbol || o.productId || current, side: o.side, quantity: o.quantity || o.size, stopLoss: o.stopLoss, takeProfit: o.takeProfit });
		} else if (type === "CLOSE_POSITION") {
			await post("/api/close", { id: (msg.position && msg.position.id) || msg.id || msg.positionId });
		} else if (type === "MODIFY_POSITION") {
			const p = msg.position || msg;
			await post("/api/modify", { id: p.id, stopLoss: p.stopLoss != null ? p.stopLoss : msg.stopLoss, takeProfit: p.takeProfit != null ? p.takeProfit : msg.takeProfit });
		} else if (type === "CANCEL_ORDER") {
			await post("/api/cancel", { orderId: msg.orderId || (msg.order && msg.order.orderId) || msg.id });
		} else return;
		toast(type.toLowerCase().replace(/_/g, " ") + " ✓");
		await Promise.all([pushBroker(), refreshStatus()]);
	} catch (e) {
		toast(e.message, true);
		pushBroker();
	}
}

// ---- symbol selection ---------------------------------------------------------------------------
async function selectSymbol(key, first) {
	current = key;
	document.querySelectorAll("#watchlist li").forEach((li) => li.classList.toggle("sel", li.dataset.key === key));
	const w = (window._wl || []).find((s) => s.key === key);
	$("#sym-title").textContent = bare(key);
	$("#sym-meta").textContent = w ? `${w.segment.toLowerCase()} · ${px(w.last)} · ${w.date}${w.regime ? " · " + w.regime : ""}` : "";
	if (mode === "gocharting" && !first) {
		const target = typeof (sdkChart && sdkChart.setSymbol) === "function" ? sdkChart : chartInstance;
		if (target && typeof target.setSymbol === "function") target.setSymbol(key);
		else mountGoCharting();
		pushBroker();
	} else if (mode === "fallback") drawFallback();
	$("#analysis").textContent = "…";
	api("/api/analysis?symbol=" + bare(key)).then((a) => ($("#analysis").textContent = a.text)).catch((e) => ($("#analysis").textContent = e.message));
}

async function renderWatchlist() {
	const list = await api("/api/symbols");
	window._wl = list;
	const ul = $("#watchlist");
	ul.textContent = "";
	for (const s of list) {
		const cls = s.change >= 0 ? "num pos" : "num neg";
		ul.appendChild(h("li", { "data-key": s.key, onclick: () => selectSymbol(s.key) },
			h("span", {}, s.symbol, h("small", {}, s.regime ? s.regime : s.segment.toLowerCase())),
			h("span", { class: cls }, px(s.last), h("small", {}, (s.change >= 0 ? "+" : "") + (s.change * 100).toFixed(2) + "%"))));
	}
}

// ---- account panels -----------------------------------------------------------------------------
async function refreshStatus() {
	let st;
	try { st = await api("/api/status"); } catch (e) { return; }
	$("#h-eq").textContent = inr(st.equity);
	$("#h-dd").textContent = (st.drawdown * 100).toFixed(2) + "%";
	$("#h-regime").textContent = st.regime || "—";
	$("#h-last").textContent = st.last_processed ? String(st.last_processed).slice(0, 10) : "never";
	$("#h-risk").textContent = st.kill_switch ? "KILL SWITCH" : st.halted ? "HALTED" : "normal";
	$("#h-risk").style.color = st.kill_switch || st.halted ? "var(--bad)" : "var(--good)";

	const pos = $("#positions");
	pos.textContent = "";
	if (!st.open.length) pos.appendChild(h("p", { class: "note" }, "No open positions."));
	else pos.appendChild(h("table", {}, h("tr", {}, h("th", {}, "Trade"), h("th", { class: "num" }, "P&L"), h("th", {}, "")),
		st.open.map((t) => h("tr", {},
			h("td", {}, h("b", {}, t.symbol), " ", t.strategy, h("div", { class: "note" }, (t.structure ? t.structure.split("|")[0] : "since " + t.opened) + (t.stop != null ? " · stop " + px(t.stop) : ""))),
			h("td", { class: "num " + (t.pnl >= 0 ? "pos" : "neg") }, inr(t.pnl)),
			h("td", {}, h("button", { onclick: () => closeTrade(t.id) }, "Close"))))));

	const q = $("#queued");
	q.textContent = "";
	if (!st.queued.length) q.appendChild(h("p", { class: "note" }, "Nothing queued."));
	st.queued.forEach((x) => q.appendChild(h("div", { class: "review" },
		h("b", {}, x.symbol), " ", x.strategy, " ×", x.units, " ",
		h("button", { onclick: () => post("/api/cancel", { orderId: x.key }).then(() => { toast("cancelled"); refreshStatus(); pushBroker(); }).catch((e) => toast(e.message, true)) }, "Cancel"),
		h("div", { class: "note" }, x.rationale))));

	const rv = $("#reviews");
	rv.textContent = "";
	if (!st.recent.length) rv.appendChild(h("p", { class: "note" }, "No closed trades yet."));
	st.recent.forEach((r) => {
		let lessons = [];
		try { lessons = JSON.parse(r.lessons || "[]"); } catch (e) { /* ignore */ }
		rv.appendChild(h("div", { class: "review" }, h("span", { class: "g" }, r.grade || "?"),
			h("b", {}, r.symbol), " ", r.strategy, " ", h("span", { class: r.pnl >= 0 ? "pos" : "neg" }, inr(r.pnl)),
			h("div", { class: "note" }, r.review || ""), lessons.length ? h("div", { class: "note" }, "Lesson: " + lessons[0]) : null));
	});

	const ck = $("#checks");
	ck.textContent = "";
	const latest = {};
	st.checks.forEach((c) => { if (!latest[c.name]) latest[c.name] = c; });
	const rows = Object.values(latest);
	if (!rows.length) ck.appendChild(h("p", { class: "note" }, "No routine has run yet (quantdesk paper premarket)."));
	else ck.appendChild(h("table", {}, rows.map((c) => h("tr", {}, h("td", {}, c.name), h("td", { class: "st-" + c.status }, c.status), h("td", { class: "note" }, c.detail)))));
}

async function closeTrade(id) {
	try {
		const r = await post("/api/close", { id });
		toast("closed, P&L " + inr(r.pnl));
		refreshStatus();
		pushBroker();
		if (mode === "fallback") drawFallback();
	} catch (e) { toast(e.message, true); }
}

async function ticket(side) {
	const num = (id) => { const v = $(id).value; return v === "" ? null : Number(v); };
	try {
		const r = await post("/api/order", { symbol: current, side, quantity: num("#t-qty"), stopLoss: num("#t-stop"), takeProfit: num("#t-tgt") });
		toast(`${side} ${r.qty} @ ${px(r.price)} ✓`);
		refreshStatus();
		pushBroker();
		if (mode === "fallback") drawFallback();
	} catch (e) { toast(e.message, true); }
}

// ---- built-in fallback chart ---------------------------------------------------------------------
function useFallback(why) {
	mode = "fallback";
	$("#mode").textContent = why;
	drawFallback();
}

function niceTicks(lo, hi, n) {
	const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0))), f = step0 / mag;
	const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag, out = [];
	for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
	return out;
}

async function drawFallback() {
	const sym = bare(current);
	const el = $("#chart");
	let H_, ov, mk;
	try {
		[H_, ov, mk] = await Promise.all([api("/api/history?symbol=" + sym), api("/api/overlays?symbol=" + sym), api("/api/markers?symbol=" + sym)]);
	} catch (e) { el.textContent = e.message; return; }
	if (H_.s !== "ok") { el.textContent = "no data"; return; }
	const n = Math.min(RANGE, H_.t.length), s0 = H_.t.length - n;
	const T = H_.t.slice(s0), O = H_.o.slice(s0), Hi = H_.h.slice(s0), Lo = H_.l.slice(s0), C = H_.c.slice(s0), V = H_.v.slice(s0);
	const oi = ov.t.length - n;
	const lines = { SMA50: ov.lines.SMA50.slice(oi), SMA200: ov.lines.SMA200.slice(oi) };
	const dateOf = (t) => new Date(t * 1000).toISOString().slice(0, 10);
	const idxByDate = new Map(T.map((t, i) => [dateOf(t), i]));
	const W = 1000, HH = 520, L = 8, R = 74, TOP = 10, PH = 390, VT = 410, VH = 70, B = 24;
	let lo = Math.min(...Lo), hi = Math.max(...Hi);
	ov.levels.forEach((l) => { if (l.price > lo * 0.85 && l.price < hi * 1.15) { lo = Math.min(lo, l.price); hi = Math.max(hi, l.price); } });
	const pad = (hi - lo) * 0.04;
	lo -= pad; hi += pad;
	const step = (W - L - R) / n, X = (i) => L + (i + 0.5) * step, Y = (v) => TOP + (hi - v) / (hi - lo) * PH;
	const vmax = Math.max(...V, 1), VY = (v) => VT + VH - (v / vmax) * VH;
	el.textContent = "";
	const g = svg("svg", { viewBox: `0 0 ${W} ${HH}`, role: "img", "aria-label": sym + " daily candlestick chart" }, el);
	niceTicks(lo, hi, 6).forEach((v) => {
		svg("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: "var(--grid)", "stroke-width": 1 }, g);
		svg("text", { x: W - R + 6, y: Y(v) + 4 }, g).textContent = px(v);
	});
	let lastM = "";
	T.forEach((t, i) => {
		const d = dateOf(t), m = d.slice(0, 7);
		if (m !== lastM && (n <= 130 || ["01", "04", "07", "10"].includes(d.slice(5, 7)))) {
			if (lastM) svg("text", { x: X(i), y: HH - 6, "text-anchor": "middle" }, g).textContent = n > 400 ? m.replace("-", "/") : d.slice(5, 7) === "01" ? d.slice(0, 4) : new Date(t * 1000).toLocaleString("en", { month: "short" });
		}
		lastM = m;
	});
	svg("line", { x1: L, x2: W - R, y1: VT + VH, y2: VT + VH, stroke: "var(--axis)", "stroke-width": 1 }, g);
	const bw = Math.max(1, Math.min(10, step * 0.7));
	for (let i = 0; i < n; i++) {
		const up = C[i] >= O[i], col = up ? "var(--up)" : "var(--down)";
		svg("line", { x1: X(i), x2: X(i), y1: Y(Hi[i]), y2: Y(Lo[i]), stroke: col, "stroke-width": 1 }, g);
		const top = Y(Math.max(O[i], C[i])), bh = Math.max(1, Math.abs(Y(O[i]) - Y(C[i])));
		svg("rect", { x: X(i) - bw / 2, y: top, width: bw, height: bh, fill: col, rx: bw > 4 ? 1 : 0 }, g);
		svg("rect", { x: X(i) - bw / 2, y: VY(V[i]), width: bw, height: VT + VH - VY(V[i]), fill: col, "fill-opacity": 0.35 }, g);
	}
	const lineStyle = { SMA50: ["var(--ink2)", 1.5], SMA200: ["var(--muted)", 2] };
	for (const [name, arr] of Object.entries(lines)) {
		let d = "";
		arr.forEach((v, i) => { if (v != null) d += (d ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1); });
		if (!d) continue;
		svg("path", { d, fill: "none", stroke: lineStyle[name][0], "stroke-width": lineStyle[name][1], "stroke-linejoin": "round" }, g);
	}
	const lvlColor = { stop: "var(--bad)", target: "var(--good)", entry: "var(--ink2)" };
	ov.levels.forEach((l) => {
		if (l.price < lo || l.price > hi) return;
		svg("line", { x1: L, x2: W - R, y1: Y(l.price), y2: Y(l.price), stroke: lvlColor[l.kind], "stroke-width": 1 }, g);
		const t = svg("text", { x: W - R - 4, y: Y(l.price) - 4, "text-anchor": "end" }, g);
		t.style.fill = "var(--ink2)";
		t.textContent = l.label;
	});
	const marks = new Map();
	mk.forEach((m) => {
		const i = idxByDate.get(m.time);
		if (i == null) return;
		(marks.get(i) || marks.set(i, []).get(i)).push(m);
		const entry = m.kind === "entry", long = m.side === "buy";
		const pointUp = entry ? long : !long;
		const y = pointUp ? Y(Lo[i]) + 12 : Y(Hi[i]) - 12, s = 6, x = X(i);
		const pts = pointUp ? `${x},${y - s} ${x - s},${y + s} ${x + s},${y + s}` : `${x},${y + s} ${x - s},${y - s} ${x + s},${y - s}`;
		svg("polygon", { points: pts, fill: entry ? "var(--ink)" : "var(--surface)", stroke: "var(--ink)", "stroke-width": 1.5 }, g);
	});
	const cross = svg("line", { y1: TOP, y2: VT + VH, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" }, g);
	const hit = svg("rect", { x: L, y: TOP, width: W - L - R, height: VT + VH - TOP, fill: "transparent" }, g);
	hit.addEventListener("pointermove", (e) => {
		const r = g.getBoundingClientRect(), xx = (e.clientX - r.left) / r.width * W;
		const i = Math.max(0, Math.min(n - 1, Math.round((xx - L) / step - 0.5)));
		cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("visibility", "visible");
		const chg = i > 0 ? (C[i] / C[i - 1] - 1) * 100 : 0;
		const rows = [dateOf(T[i]), `O ${px(O[i])}  H ${px(Hi[i])}  L ${px(Lo[i])}  C ${px(C[i])}  (${chg >= 0 ? "+" : ""}${chg.toFixed(2)}%)`,
			`Vol ${Math.round(V[i]).toLocaleString("en-IN")}` + (lines.SMA50[i] ? `  SMA50 ${px(lines.SMA50[i])}` : "") + (lines.SMA200[i] ? `  SMA200 ${px(lines.SMA200[i])}` : "")];
		(marks.get(i) || []).forEach((m) => rows.push((m.kind === "entry" ? "▲ " : "▼ ") + m.text));
		showTT(e, rows);
	});
	hit.addEventListener("pointerleave", () => { hideTT(); cross.setAttribute("visibility", "hidden"); });
	const lg = $("#legend");
	lg.textContent = "";
	lg.appendChild(h("span", {}, h("i", { style: "background:var(--up);height:8px" }), "up day"));
	lg.appendChild(h("span", {}, h("i", { style: "background:var(--down);height:8px" }), "down day"));
	lg.appendChild(h("span", {}, h("i", { style: "background:var(--ink2)" }), "SMA50"));
	lg.appendChild(h("span", {}, h("i", { style: "background:var(--muted)" }), "SMA200"));
	lg.appendChild(h("span", {}, "▲ entry  ▽ exit (from the journal)"));
	if (ov.levels.length) lg.appendChild(h("span", {}, h("i", { style: "background:var(--bad)" }), "stop  ", h("i", { style: "background:var(--good)" }), "target"));
}

boot().catch((e) => toast("boot failed: " + e.message, true));
