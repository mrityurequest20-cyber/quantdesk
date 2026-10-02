/* QuantDesk: the phone app for the intraday options desk.
 * Desk: account, the desk's stance per index, open positions, global pulse, headlines.
 * Chart: candles with VWAP, levels and trade markers; the desk's read, its evidence, key levels, quant.
 * Trades: positions, history by day, performance, session reviews. Brain: global regime, the influence
 * graph and the measured wiring. Feed: the headlines the desk reads and its own reasoning log.
 * Works against the desk's own server (/api/i/*) and, unchanged, as the read-only published site (a shim
 * answers the same calls from data.json). All server text goes in with textContent. */
"use strict";
const VERSION = "2.0";
const $ = (s, r) => (r || document).querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const S = {
	account: "live", tab: "desk", sub: { trades: "positions", feed: "news" }, sym: "NIFTY", interval: "5m", brainSym: "NIFTY",
	state: null, charts: {}, chartAt: {}, news: null, newsAt: 0, lw: null, eq: null, thBefore: null, thSym: "", newsF: "", brk: "by_day_type",
	accounts: [], installEvt: null,
};

// ---- icons (24px, stroked) -----------------------------------------------------------------------------------
const IC = {
	desk: "M3 12h4l3-7 4 14 3-7h4",
	chart: "M7 3v3M7 17v4M17 4v3M17 16v4M5 6h4v11H5zM15 7h4v9h-4z",
	trades: "M4 8h16v11a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1zM9 8V5.5A1.5 1.5 0 0 1 10.5 4h3A1.5 1.5 0 0 1 15 5.5V8M4 13h16",
	brain: "M5 6a2 2 0 1 0 0 .01M19 6a2 2 0 1 0 0 .01M12 12a2 2 0 1 0 0 .01M5 18a2 2 0 1 0 0 .01M19 18a2 2 0 1 0 0 .01M6.5 7.5l4 3M17.5 7.5l-4 3M6.5 16.5l4-3M17.5 16.5l-4-3",
	feed: "M4 5h13v14H6a2 2 0 0 1-2-2zM17 9h3v8a2 2 0 0 1-2 2M8 9h5M8 13h5M8 17h3",
	chev: "M9 6l6 6-6 6", x: "M6 6l12 12M18 6L6 18",
	aside: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM10 9v6M14 9v6",
	watch: "M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12zM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z",
	target: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10zM12 11a1 1 0 1 0 0 2 1 1 0 0 0 0-2z",
	alert: "M12 4l9 16H3zM12 10v4M12 17v.01", moon: "M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z",
	up: "M6 15l6-6 6 6", down: "M6 9l6 6 6-6", flat: "M5 12h14",
	ext: "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
	install: "M12 4v11M8 11l4 4 4-4M5 20h14", refresh: "M20 11a8 8 0 1 0-2.3 5.7M20 5v6h-6",
	pause: "M9 5v14M15 5v14", play: "M7 5l12 7-12 7z", stop: "M6 6h12v12H6z",
	globe: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM3 12h18M12 3c3 3.5 3 14.5 0 18M12 3c-3 3.5-3 14.5 0 18",
	news: "M4 5h13v14H6a2 2 0 0 1-2-2zM17 9h3v8a2 2 0 0 1-2 2M8 9h5M8 13h5",
	book: "M5 4h9a3 3 0 0 1 3 3v13H8a3 3 0 0 1-3-3zM17 20h2V6M9 8h5M9 12h5", info: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 11v6M12 7.5v.01",
	bolt: "M13 3L5 13h6l-1 8 8-10h-6z", clock: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 7v5l3 2",
};
function svg(tag, attrs, parent) {
	const e = document.createElementNS(NS, tag);
	for (const k in attrs) e.setAttribute(k, attrs[k]);
	if (parent) parent.appendChild(e);
	return e;
}
function icon(name) {
	const s = svg("svg", { viewBox: "0 0 24 24", class: "i", "aria-hidden": "true" });
	svg("path", { d: IC[name] || IC.info }, s);
	return s;
}

// ---- helpers ----------------------------------------------------------------------------------------------
function h(tag, attrs, ...kids) {
	const e = document.createElement(tag);
	for (const [k, v] of Object.entries(attrs || {})) {
		if (v == null || v === false) continue;
		if (k === "class") e.className = v;
		else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
		else e.setAttribute(k, v === true ? "" : v);
	}
	for (const k of kids.flat()) if (k != null && k !== false) e.appendChild(typeof k === "object" ? k : document.createTextNode(String(k)));
	return e;
}
// append children, skipping null/false (Element.append would print them as text)
function put(el, ...kids) { for (const k of kids.flat()) if (k != null && k !== false) el.append(k); return el; }
const fin = (v) => v != null && isFinite(v);
const inr = (v, sign) => {
	if (!fin(v)) return "—";
	const a = Math.abs(v), s = v < 0 ? "−" : sign && v > 0 ? "+" : "";
	return s + (a >= 1e7 ? "₹" + (a / 1e7).toFixed(2) + " Cr" : a >= 1e5 ? "₹" + (a / 1e5).toFixed(2) + " L" : "₹" + Math.round(a).toLocaleString("en-IN"));
};
const num = (v, d = 2) => (fin(v) ? Number(v).toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d }) : "—");
const signed = (v, d = 2) => (fin(v) ? (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d }) : "—");
const pct = (v, d = 2) => (fin(v) ? (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v * 100).toFixed(d) + "%" : "—");
const cls = (v) => (v > 0 ? "up" : v < 0 ? "dn" : "");
const cap = (s) => (s ? String(s).charAt(0).toUpperCase() + String(s).slice(1) : "");
const words = (s) => String(s || "").replace(/_/g, " ");
// "2026-09-29 09:27:04.466562+05:30" → ms. Safari won't parse >3 fractional digits or a space separator.
const tms = (t) => (typeof t === "number" ? t * 1000 : Date.parse(String(t).trim().replace(" ", "T").replace(/(\.\d{3})\d+/, "$1")));
const ist = (t, withDate) => {
	const ms = tms(t);
	if (!isFinite(ms)) return "—";
	return new Date(ms).toLocaleString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hour12: false,
		...(withDate ? { day: "2-digit", month: "short" } : {}) });
};
const istDay = (t) => new Date(tms(t)).toLocaleDateString("en-IN", { timeZone: "Asia/Kolkata", weekday: "short", day: "2-digit", month: "short" });
function ago(ts) {
	const m = Math.max(0, (Date.now() - tms(ts)) / 60000);
	if (!isFinite(m)) return "";
	return m < 1 ? "just now" : m < 60 ? `${Math.round(m)} min ago` : m < 1440 ? `${Math.round(m / 60)} h ago` : ist(ts, true);
}
function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } return null; }
function toast(msg) {
	const t = $("#toast");
	t.textContent = msg;
	t.classList.add("show");
	clearTimeout(toast.t);
	toast.t = setTimeout(() => t.classList.remove("show"), 3400);
}
async function api(path, opt) {
	const sep = path.includes("?") ? "&" : "?";
	const r = await fetch(path + (path.startsWith("/api/i/") ? sep + "account=" + encodeURIComponent(S.account) : ""), opt);
	const j = await r.json().catch(() => ({}));
	if (r.status === 401) { location.href = "/"; throw new Error("sign in again"); }
	if (!r.ok) throw new Error(j.error || "HTTP " + r.status);
	return j;
}
const post = (p, b) => api(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) });
const empty = (ic, text) => h("div", { class: "empty" }, icon(ic), text);
const SETUP = { orb: "Opening-range breakout", vwap_trend: "VWAP trend", trend_break: "Trend break", fade: "Fade", reversal: "Reversal", range: "Range" };
const setupName = (s) => SETUP[s] || cap(words(s));
const structName = (s) => cap(words(s || ""));

// ---- IST clock: is the market open, and when does it next open ------------------------------------------------------
function istNow() {
	const p = Object.fromEntries(new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false })
		.formatToParts(new Date()).map((x) => [x.type, x.value]));
	return { wd: p.weekday, min: Number(p.hour) * 60 + Number(p.minute) };
}
const inSession = () => { const n = istNow(); return !["Sat", "Sun"].includes(n.wd) && n.min >= 555 && n.min <= 930; };
function nextOpen() {
	const n = istNow(), days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
	let i = days.indexOf(n.wd), k = 0;
	if (i >= 1 && i <= 5 && n.min < 555) return "today 09:15";
	do { i = (i + 1) % 7; k++; } while (i === 0 || i === 6);
	return (k === 1 ? "tomorrow" : days[i]) + " 09:15";
}

// ---- what the desk said, in plain words -------------------------------------------------------------------------
function readAction(a) {
	a = String(a || "").trim();
	let m;
	if (/^ENTER/.test(a)) { m = a.match(/^ENTER (\S+) (\d+)×(.*?)(?: \| |$)/); return { kind: "enter", label: "Entered a trade", reason: m ? `${setupName(m[1])}: ${m[2]} × ${m[3]}` : a.slice(6) }; }
	if (/^EXIT/.test(a)) return { kind: "exit", label: "Closed a trade", reason: a.slice(5) };
	if ((m = a.match(/^standing aside[^:]*:\s*(.*)$/i))) return { kind: "aside", label: "Standing aside", reason: cap(m[1]) };
	if ((m = a.match(/^watching[^:]*:\s*(.*)$/i))) return { kind: "watch", label: "Watching", reason: cap(m[1]) };
	if (/sized to 0 lots|not worth it|\bEV\b/.test(a)) return { kind: "pass", label: "Passed on a setup", reason: cap(a) };
	return { kind: a ? "other" : "none", label: a ? cap(a) : "No read yet", reason: "" };
}
// what the desk is doing on one index: its last action, else (older desks) its first no-trade flag
function stance(v) {
	const a = readAction(v && v.action);
	if (a.kind === "none" && v && (v.vetoes || []).length) return { kind: "aside", label: "Standing aside", reason: cap(v.vetoes[0]) };
	return a;
}
const biasCls = (b) => (b === "bullish" ? "bull" : b === "bearish" ? "bear" : "flat");
function biasTag(bias, score) {
	const k = biasCls(bias);
	return h("span", { class: "tag " + k }, icon(k === "bull" ? "up" : k === "bear" ? "down" : "flat"),
		cap(bias || "neutral") + (fin(score) ? " " + signed(score) : ""));
}
function chgPill(v, pts) {
	return h("span", { class: "chg mono " + (v > 0 ? "up" : v < 0 ? "dn" : "flat") }, pts != null ? signed(pts) : pct(v));
}

// ---- glossary (the "?" buttons) ---------------------------------------------------------------------------------
const GLOSS = {
	bias: ["Bias score", "The desk's lean for the index, from −1 (strongly bearish) to +1 (strongly bullish): a weighted vote of every piece of evidence — trend, structure, momentum, options positioning, volatility, news, global markets, and the quant model when it has proven an edge."],
	conviction: ["Conviction", "How much the evidence agrees, 0 (split) to 1 (unanimous). The desk trades only with enough conviction and no no-trade flag up."],
	premium: ["Premium: rich, fair or cheap", "Implied volatility (IV, what option prices assume) against realised volatility (RV, how much the index is actually moving). Rich premium (IV well above RV) favours spreads that sell some of it back; cheap premium favours buying options outright."],
	aside: ["Standing aside", "A no-trade flag is up: the first minutes after the open, a move too stretched to chase, breaking news, a stale option chain, global stress, or an EV that doesn't clear costs. Not trading is a decision too."],
	r: ["R-multiple", "P&L divided by the risk planned at entry. +1R made what the trade risked; −1R lost exactly the planned amount. It makes trades of different sizes comparable."],
	grade: ["Trade grade", "A post-trade review of decision quality (entry, sizing, management, exit), A to F. A well-run loser can grade well; a lucky winner can grade badly."],
	quant: ["The quant layer", "A 30-minute volatility forecast (realised + implied) and a Monte Carlo that prices every candidate options structure after all costs. A direction model votes only if it beat a coin flip out of sample (AUC ≥ 0.53); research drifts come from years of real data and only count when they survive a holdout."],
	levels: ["Levels", "OR: opening range (first 15 min). IB: initial balance (first hour). PDH/PDL: prior day high/low. VAH/VAL/POC: value area and point of control. CPR: central pivot range. OI walls: the strikes with the most call / put open interest."],
	regime: ["Global regime", "A weighted read of US, Asian and European equities, US volatility, the dollar, the rupee, crude and rates. It's context: only links the weekly research has validated on real data actually vote in the bias."],
	stress: ["Global stress", "How unusual global moves are right now, in standard deviations. Above 2σ the desk cuts position size, down to half."],
	vwap: ["VWAP", "The session's volume-weighted average price (time-weighted for an index, which has no volume of its own). Above it buyers have had the upper hand today, below it sellers."],
	paper: ["Paper trading", "Every trade here is simulated against real prices and real option chains, with real costs (brokerage, STT, exchange fees, slippage). No real money moves and nothing here can place an order."],
};
function info(key) {
	return h("button", { class: "info", "aria-label": "What is " + (GLOSS[key] || [key])[0] + "?", onclick: (e) => { e.stopPropagation(); showGloss(key); } }, "?");
}
function showGloss(key) {
	const g = GLOSS[key];
	if (!g) return;
	openSheet(g[0], h("p", { style: "margin:0;font-size:14.5px;line-height:1.6;color:var(--fg2)" }, g[1]),
		h("button", { class: "btn block", style: "margin-top:18px", onclick: () => openSheet("How to read this app", glossaryList()) }, icon("book"), "All terms"));
}
function glossaryList() {
	return h("div", { class: "panel rows" }, Object.entries(GLOSS).map(([k, [t, d]]) =>
		h("div", { class: "pad" }, h("div", { style: "font-weight:600;margin-bottom:3px" }, t), h("div", { style: "font-size:13px;color:var(--fg2);line-height:1.55" }, d))));
}

// ---- shell: tabs, routing, sheets, theme ---------------------------------------------------------------------------
const TABS = [["desk", "Desk"], ["chart", "Chart"], ["trades", "Trades"], ["brain", "Brain"], ["feed", "Feed"]];
const SUBS = { trades: [["positions", "Positions"], ["history", "History"], ["performance", "Performance"], ["reviews", "Reviews"]],
	feed: [["news", "Headlines"], ["log", "Desk log"]] };
function buildTabs() {
	const nav = $("#tabs");
	for (const [id, label] of TABS)
		nav.appendChild(h("button", { role: "tab", "aria-selected": "false", "data-tab": id, onclick: () => go(id) }, icon(id), h("span", {}, label)));
}
function go(route) {
	const [tab, sub] = String(route).split("/");
	if (location.hash !== "#" + route) history.replaceState(null, "", "#" + route);
	show(tab, sub);
}
function show(tab, sub) {
	if (!TABS.some((t) => t[0] === tab)) tab = "desk";
	const changed = tab !== S.tab;
	S.tab = tab;
	if (sub && SUBS[tab] && SUBS[tab].some((s) => s[0] === sub)) S.sub[tab] = sub;
	store("qd.tab", tab);
	document.querySelectorAll("#tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
	document.querySelectorAll("main > section").forEach((s) => (s.hidden = s.dataset.view !== tab));
	$("#title").textContent = (TABS.find((t) => t[0] === tab) || ["", "Desk"])[1];
	if (changed) scrollTo(0, 0);
	if ($("#sheet").classList.contains("open")) closeSheet();          // navigating (or Back) closes a sheet
	if (tab !== "chart" && $("#c-box").classList.contains("fs")) fullscreen(false);
	refresh(true);
}
function openSheet(title, ...content) {
	const b = $("#sheetbody");
	b.textContent = "";
	put(b, h("div", { class: "grab", "aria-hidden": "true" }),
		h("div", { class: "sh-h" }, h("h3", {}, title), h("button", { class: "ib", "aria-label": "Close", onclick: closeSheet }, icon("x"))), ...content);
	$("#sheet").classList.add("open");
	b.scrollTop = 0;
	document.body.style.overflow = "hidden";
}
function closeSheet() { $("#sheet").classList.remove("open"); document.body.style.overflow = ""; }
function themeMode() { return store("qd.theme") || "auto"; }
function applyTheme(mode) {
	const r = document.documentElement;
	if (mode === "dark" || mode === "light") r.setAttribute("data-theme", mode); else r.removeAttribute("data-theme");
	store("qd.theme", mode);
	requestAnimationFrame(() => {
		$("#meta-theme").setAttribute("content", cssv("--bg") || "#08090b");
		restyleCharts();
	});
}
function cssv(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function seg(el, items, current, onpick, role) {
	el.textContent = "";
	for (const [val, label] of items)
		el.appendChild(h("button", { "aria-pressed": String(val === current), role: role || null, "aria-selected": role ? String(val === current) : null,
			onclick: () => { onpick(val); seg(el, items, val, onpick, role); } }, label));
}

// ---- status pill + banner --------------------------------------------------------------------------------------
function deskStatus(st) {
	const hb = (st && st.heartbeat) || {}, pub = !!window.QD_PUBLISHED;
	const stale = !st || st.age_sec == null || !(st.age_sec <= (pub ? 900 : 180));
	if (window.QD_DEMO && !pub) return { k: "off", t: "Snapshot", s: hb.ts ? ist(hb.ts, true) : "", stale: false };
	if (stale) return { k: "off", t: inSession() ? "Offline" : "Closed", s: hb.ts ? ist(hb.ts, true) : "", stale: true };
	if (st.paused) return { k: "paused", t: "Paused", s: ist(hb.ts), stale: false };
	if (hb.halted) return { k: "paused", t: "Done for day", s: ist(hb.ts), stale: false };
	return { k: "live", t: "Live", s: ist(hb.ts), stale: false };
}
function setStatus(st) {
	const d = deskStatus(st), p = $("#status");
	p.className = "pill " + d.k;
	$("#status-t").textContent = d.t;
	$("#status-s").textContent = d.s;
	return d;
}

// ---- data ------------------------------------------------------------------------------------------------------
async function loadState() {
	try { S.state = await api("/api/i/state"); } catch (e) { S.state = null; S.stateErr = e.message; }
	setStatus(S.state);
	return S.state;
}
async function loadChart(sym, iv, maxAge) {
	const k = sym + "|" + iv;
	if (S.charts[k] && Date.now() - (S.chartAt[k] || 0) < (maxAge == null ? 20000 : maxAge)) return S.charts[k];
	try { S.charts[k] = await api(`/api/i/chart?symbol=${encodeURIComponent(sym)}&interval=${iv}`); S.chartAt[k] = Date.now(); } catch (e) { /* keep the last */ }
	return S.charts[k];
}
async function loadNews(maxAge) {
	if (S.news && Date.now() - S.newsAt < (maxAge == null ? 60000 : maxAge)) return S.news;
	try { S.news = await api("/api/i/news?n=150"); S.newsAt = Date.now(); } catch (e) { S.news = S.news || []; }
	return S.news;
}
const views = () => ((S.state && S.state.heartbeat && S.state.heartbeat.views) || {});
const symbols = () => { const v = Object.keys(views()); return v.length ? v : ["NIFTY", "BANKNIFTY"]; };

// ==================================================================================================================
// DESK
// ==================================================================================================================
async function renderDesk() {
	const st = S.state;
	if (!st) {
		$("#d-hero").textContent = "";
		$("#d-hero").appendChild(empty("info", (S.stateErr || "No desk yet") + ". Start one with `quantdesk intraday live` (or `intraday replay --synthetic 5`)."));
		return;
	}
	const hb = st.heartbeat || {}, vs = hb.views || {}, status = deskStatus(st);
	// banner: why nothing is moving
	const bn = $("#d-banner");
	bn.textContent = "";
	if (status.stale && (window.QD_PUBLISHED || S.account === "live")) {
		const mins = st.age_sec != null ? Math.round(st.age_sec / 60) : null;
		bn.appendChild(h("div", { class: "banner" }, icon(inSession() ? "clock" : "moon"), h("div", {}, inSession()
			? `The desk hasn't reported for ${mins != null ? mins + " min" : "a while"}. It hands over to a fresh runner at 12:20, and a restart takes a few minutes; this page refreshes by itself.`
			: `Market closed. The desk trades every NSE session by itself, 09:15–15:30 IST; next session ${nextOpen()} (exchange holidays excepted). Showing its last state${hb.ts ? ", " + ist(hb.ts, true) : ""}.`)));
	}
	// account hero
	const lim = st.limits || {}, dse = hb.day_start_equity || st.equity || st.capital, dp = hb.day_pnl;
	const lossCap = lim.daily_loss_limit && dse ? lim.daily_loss_limit * dse : null;
	const used = lossCap && fin(dp) ? Math.max(0, -dp) / lossCap : 0;
	const hero = $("#d-hero");
	hero.textContent = "";
	put(hero, 
		h("div", { class: "lbl" }, "Paper account", info("paper"), h("span", { class: "tag line", style: "margin-left:auto" }, `Started ${inr(st.capital)}`)),
		h("div", { class: "eq num" }, inr(st.equity)),
		h("div", { class: "day" }, h("span", { class: "mono " + cls(dp) }, fin(dp) ? `${inr(dp, true)} (${pct(dp / (dse || 1))})` : "—"), h("span", { class: "f3" }, "today")),
		h("div", { class: "stats" },
			h("div", {}, h("span", {}, "All-time"), h("b", { class: "mono " + cls(st.total_pnl) }, inr(st.total_pnl, true))),
			h("div", {}, h("span", {}, "Trades"), h("b", { class: "mono" }, `${hb.trades_today ?? 0}${lim.max_trades_per_day ? " / " + lim.max_trades_per_day : ""}`)),
			h("div", {}, h("span", {}, "Open"), h("b", { class: "mono" }, String((hb.positions || []).length))),
			h("div", { title: lossCap ? `daily loss limit ${inr(lossCap)}` : null }, h("span", {}, "Loss cap"), h("b", { class: "mono" }, lossCap ? Math.round(used * 100) + "%" : "—"),
				h("div", { class: "meter" }, h("i", { style: `width:${Math.min(100, used * 100)}%;background:${used > 0.66 ? "var(--dn)" : "var(--acc)"}` })))));
	renderControls(st);
	// markets: the two indices, with the desk's stance on each
	const syms = Object.keys(vs).length ? Object.keys(vs) : ["NIFTY", "BANKNIFTY"];
	const mk = $("#d-markets");
	mk.textContent = "";
	for (const s of syms) {
		const v = vs[s], c = S.charts[s + "|5m"], a = stance(v);
		const last = v ? v.spot : c && c.bars ? c.bars.c[c.bars.c.length - 1] : null;
		const prev = v && fin(v.chg) && fin(v.spot) ? v.spot / (1 + v.chg) : null;
		mk.appendChild(h("button", { class: "mrow", onclick: () => { S.sym = s; store("qd.sym", s); go("chart"); } },
			h("div", { style: "min-width:0" }, h("div", { class: "nm" }, s), h("div", { class: "sub" },
				v ? `${v.expiry && v.expiry !== "None" ? "Exp " + fmtExpiry(v.expiry) + " · " : ""}IV ${num(v.iv, 1)} · ${v.vol_view || "—"}` : "no read yet")),
			spark(c, prev, v && v.chg),
			h("div", { class: "px" }, h("div", { class: "v" }, num(last)), v ? chgPill(v.chg) : h("span", { class: "chg flat" }, "—")),
			v ? h("div", { class: "desk" }, biasTag(v.bias, v.score), h("span", { class: "t" }, a.kind === "none" ? "" : a.label + (a.reason ? " · " + a.reason : ""))) : null));
	}
	$("#d-mkt-hint").textContent = hb.feed ? `${hb.feed} · chain ${hb.chain || "—"}` : "tap for the chart";
	renderNow(st);
	// open positions
	const pos = hb.positions || [];
	$("#d-pos-wrap").hidden = !pos.length;
	$("#d-pos-hint").textContent = pos.length ? `${inr(pos.reduce((a, p) => a + (p.pnl || 0), 0), true)} open P&L` : "";
	const pl = $("#d-positions");
	pl.textContent = "";
	pos.forEach((p) => pl.appendChild(positionCard(p)));
	renderGlobalStrip(hb.global);
	// closed today
	const ct = st.closed_today || [];
	$("#d-closed-wrap").hidden = !ct.length;
	const cl = $("#d-closed");
	cl.textContent = "";
	ct.forEach((t) => cl.appendChild(tradeRow(t)));
	// sparklines and headlines load after the first paint
	Promise.all(syms.map((s) => loadChart(s, "5m", 60000))).then(() => {
		if (S.tab !== "desk") return;
		document.querySelectorAll("#d-markets .mrow").forEach((row, i) => {
			const s = syms[i], v = vs[s], old = row.querySelector(".spark");
			if (old) old.replaceWith(spark(S.charts[s + "|5m"], v && fin(v.chg) ? v.spot / (1 + v.chg) : null, v && v.chg));
		});
	});
	loadNews().then((rows) => {
		if (S.tab !== "desk") return;
		const box = $("#d-news");
		box.textContent = "";
		if (!rows.length) box.appendChild(empty("news", "Headlines appear here while the desk runs."));
		rows.slice(0, 4).forEach((r) => box.appendChild(newsRow(r, true)));
	});
	const note = $("#qd-note");
	if (!note.dataset.done) {
		note.dataset.done = "1";
		put(note, h("b", {}, (window.QD_LABEL || (window.QD_PUBLISHED ? "Live paper desk" : "QuantDesk")) + ". "),
			window.QD_NOTE || "Paper trades only: simulated against real prices and option chains, with real costs. Nothing here can place an order.");
	}
	installCard();
}
function fmtExpiry(e) {
	const d = new Date(String(e).slice(0, 10) + "T00:00:00Z");
	return isFinite(d) ? d.toLocaleDateString("en-IN", { day: "2-digit", month: "short", timeZone: "UTC" }) : e;
}
function spark(c, prev, chg) {
	const W = 76, H = 34, s = svg("svg", { class: "spark", viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" });
	const ys = c && c.bars ? c.bars.c : null;
	if (!ys || ys.length < 2) return s;
	let lo = Math.min(...ys), hi = Math.max(...ys);
	if (fin(prev)) { lo = Math.min(lo, prev); hi = Math.max(hi, prev); }
	const span = hi - lo || 1, X = (i) => 1 + (i / (ys.length - 1)) * (W - 2), Y = (v) => 3 + (hi - v) / span * (H - 6);
	const col = (chg != null ? chg : ys[ys.length - 1] - ys[0]) >= 0 ? cssv("--up") : cssv("--dn");
	if (fin(prev)) svg("line", { x1: 0, x2: W, y1: Y(prev), y2: Y(prev), stroke: cssv("--line2"), "stroke-width": 1, "stroke-dasharray": "2 3" }, s);
	let d = "";
	ys.forEach((v, i) => (d += (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1)));
	const gid = "g" + Math.random().toString(36).slice(2, 8), defs = svg("defs", {}, s), lg = svg("linearGradient", { id: gid, x1: 0, x2: 0, y1: 0, y2: 1 }, defs);
	svg("stop", { offset: "0", "stop-color": col, "stop-opacity": ".22" }, lg);
	svg("stop", { offset: "1", "stop-color": col, "stop-opacity": "0" }, lg);
	svg("path", { d: d + `L${X(ys.length - 1).toFixed(1)} ${H}L${X(0)} ${H}Z`, fill: `url(#${gid})` }, s);
	svg("path", { d, fill: "none", stroke: col, "stroke-width": 1.5, "stroke-linejoin": "round" }, s);
	return s;
}
function renderControls(st) {
	const box = $("#d-controls");
	box.textContent = "";
	if (window.QD_PUBLISHED || window.QD_DEMO) return;
	box.appendChild(h("div", { class: "btns", style: "margin-top:12px" },
		h("button", { class: "btn", onclick: () => command(st.paused ? "resume" : "pause") }, icon(st.paused ? "play" : "pause"), st.paused ? "Resume entries" : "Pause entries"),
		h("button", { class: "btn danger", onclick: () => confirmSheet("Flatten everything?", "Close every open position at the next minute's prices and pause new entries.", "Flatten all", () => command("flatten")) },
			icon("stop"), "Flatten all")));
}
function renderNow(st) {
	const hb = st.heartbeat || {}, vs = hb.views || {}, box = $("#d-now"), pos = hb.positions || [];
	box.textContent = "";
	const syms = Object.keys(vs);
	if (!syms.length) { box.appendChild(empty("watch", "The desk's read appears here from 09:15 IST, updated every minute.")); return; }
	const reads = syms.map((s) => [s, vs[s], stance(vs[s])]);
	const biases = new Set(reads.map((r) => r[1].bias));
	const leanTxt = biases.size === 1
		? `${cap([...biases][0])} on ${syms.length > 1 ? "both indices" : syms[0]}`
		: reads.map(([s, v]) => `${cap(v.bias)} ${s}`).join(", ");
	let ic = "watch", icls = "", head, body;
	if (st.paused) { ic = "pause"; head = "Paused from the app"; body = "No new entries until it's resumed. Open positions are still managed to their exits."; }
	else if (hb.halted) { ic = "alert"; head = "Done for the day"; body = "The daily loss limit is hit: no new trades until tomorrow. Open positions are still managed."; }
	else if (pos.length) { ic = "target"; icls = "in"; head = `In ${pos.length === 1 ? "a trade" : pos.length + " trades"}`; body = pos.map((p) => `${p.symbol} ${setupName(p.setup)}, ${inr(p.pnl, true)} so far`).join("; ") + "."; }
	else if (reads.some((r) => r[2].kind === "enter")) { ic = "target"; icls = "in"; head = "Just entered a trade"; body = leanTxt + "."; }
	else if (reads.every((r) => r[2].kind === "aside")) { ic = "aside"; head = `${leanTxt}, but standing aside`; body = "A no-trade flag is up, so no entry yet. The desk re-checks every minute."; }
	else if (reads.some((r) => r[2].kind === "watch")) { ic = "watch"; head = `${leanTxt}, watching for a setup`; body = "No setup has triggered its entry rules yet."; }
	else { head = leanTxt; body = "Reading the market every minute."; }
	put(box, h("div", { class: "now" }, h("div", { class: "ic " + icls }, icon(ic)), h("div", {}, h("div", { class: "hd" }, head), h("p", {}, body))),
		h("div", { class: "why" }, reads.filter((r) => r[2].kind !== "none").map(([s, v, a]) => h("div", {}, h("b", {}, s), h("span", {}, a.label + (a.reason ? ": " + a.reason : ""))))));
	if (reads.some((r) => r[2].kind === "aside")) box.lastChild.appendChild(h("div", {}, h("b", {}, ""), h("button", { class: "link", onclick: () => showGloss("aside") }, "Why stand aside?")));
}
function positionCard(p) {
	const dir = fin(p.target) && fin(p.entry_underlying) ? Math.sign(p.target - p.entry_underlying) : 0;
	const card = h("div", { class: "panel" });
	const pnlPct = p.pnl != null && S.state && S.state.equity ? p.pnl / S.state.equity : null;
	card.appendChild(h("div", { class: "pos" },
		h("div", { class: "top" }, h("div", { style: "min-width:0" }, h("div", { class: "t1" }, `${p.symbol} · ${setupName(p.setup)}`),
			h("div", { class: "t2" }, `${structName(p.structure)} · ${p.lots} lot${p.lots === 1 ? "" : "s"} · since ${ist(p.opened)}`)),
			h("div", { class: "pnl " + cls(p.pnl) }, inr(p.pnl, true), h("small", {}, pnlPct != null ? pct(pnlPct) + " of equity" : ""))),
		fin(p.stop) && fin(p.target) ? track(p.stop, p.target, p.entry_underlying, p.spot, dir) : null,
		legsTable(p.legs.map((l) => ({ symbol: l.symbol, qty: l.qty, entry: l.entry, mark: l.mark }))),
		h("div", { class: "btns" }, h("button", { class: "btn", onclick: () => openTrade(p.id) }, icon("info"), "Why this trade"),
			window.QD_PUBLISHED || window.QD_DEMO ? h("span") : h("button", { class: "btn danger", onclick: () =>
				confirmSheet(`Close ${p.symbol} ${setupName(p.setup)}?`, "Square off this position at the next minute's prices.", "Close position", () => command("close", p.id)) }, icon("x"), "Close"))));
	return card;
}
function track(stop, target, entry, spot, dir) {
	const lo = Math.min(stop, target), hi = Math.max(stop, target), span = hi - lo || 1;
	const X = (v) => Math.max(0, Math.min(100, (v - lo) / span * 100));
	const targetRight = target > stop;
	return h("div", { class: "track", role: "img", "aria-label": `spot ${num(spot)} between stop ${num(stop)} and target ${num(target)}` },
		h("div", { class: "ln" + (targetRight ? "" : " rev") }),
		fin(entry) ? h("div", { class: "en", style: `left:${X(entry)}%`, title: "entry" }) : null,
		fin(spot) ? h("div", { class: "sp", style: `left:${X(spot)}%` }) : null,
		h("span", { class: "l" }, `${targetRight ? "Stop" : "Target"} ${num(lo, 0)}`), h("span", { class: "r" }, `${targetRight ? "Target" : "Stop"} ${num(hi, 0)}`));
}
function legsTable(legs, withExit) {
	return h("table", { class: "legs" }, h("tr", {}, h("th", {}, "Leg"), h("th", {}, "Qty"), h("th", {}, "Entry"), h("th", {}, withExit ? "Exit" : "Mark")),
		legs.map((l) => h("tr", {}, h("td", {}, l.symbol), h("td", { class: l.qty < 0 ? "dn" : "" }, (l.qty > 0 ? "+" : "") + l.qty), h("td", {}, num(l.entry)), h("td", {}, num(withExit ? l.exit : l.mark)))));
}
const GLOBAL_ORDER = ["ES", "NQ", "N225", "HSI", "KOSPI", "SSE", "STOXX", "DAX", "FTSE", "USDINR", "DXY", "BRENT", "GOLD", "UST10", "USVIX", "SPX", "NASDAQ", "DJI"];
const SHORT = { ES: "S&P fut", NQ: "Nasdaq fut", N225: "Nikkei", HSI: "Hang Seng", KOSPI: "Kospi", SSE: "Shanghai", STOXX: "Stoxx 50", DAX: "DAX", FTSE: "FTSE",
	USDINR: "USD/INR", DXY: "Dollar", BRENT: "Brent", GOLD: "Gold", UST10: "US 10Y", USVIX: "VIX", SPX: "S&P 500", NASDAQ: "Nasdaq", DJI: "Dow" };
function mchg(m) { return m.live && m.since_open != null ? m.since_open : m.prior_ret; }
function renderGlobalStrip(G) {
	const mk = (G && G.markets) || {}, box = $("#d-global");
	$("#d-global-wrap").hidden = !Object.keys(mk).length;
	box.textContent = "";
	for (const k of GLOBAL_ORDER) {
		const m = mk[k];
		if (!m || m.last == null) continue;
		const c = mchg(m);
		box.appendChild(h("button", { class: "tk", onclick: () => go("brain"), title: m.name },
			h("div", { class: "n" }, h("i", { class: m.live ? "on" : "" }), SHORT[k] || m.name),
			h("div", { class: "v" }, num(m.last, m.last > 1000 ? 0 : 2)),
			h("div", { class: "c " + cls(c) }, pct(c))));
	}
}
function installCard() {
	const box = $("#d-banner");
	if (store("qd.installed") || store("qd.install-dismissed") || matchMedia("(display-mode: standalone)").matches || navigator.standalone) return;
	const ios = /iphone|ipad|ipod/i.test(navigator.userAgent);
	if (!S.installEvt && !ios) return;
	if (box.querySelector(".install")) return;
	const card = h("div", { class: "banner install" }, icon("install"), h("div", { class: "grow" },
		h("div", { style: "font-weight:600" }, "Install QuantDesk"),
		h("div", { class: "f2", style: "margin-top:2px" }, ios ? "Tap Share, then “Add to Home Screen”: it opens full-screen like an app." : "Add it to your home screen: full-screen, one tap away, works offline."),
		S.installEvt ? h("button", { class: "btn primary", style: "margin-top:10px;height:36px", onclick: doInstall }, "Install app") : null),
		h("button", { class: "ib", "aria-label": "Dismiss", style: "margin:-8px -8px 0 0", onclick: () => { store("qd.install-dismissed", "1"); card.remove(); } }, icon("x")));
	box.appendChild(card);
}
async function doInstall() {
	if (!S.installEvt) return toast("Use your browser's menu: “Install app” or “Add to Home Screen”.");
	S.installEvt.prompt();
	const r = await S.installEvt.userChoice.catch(() => null);
	S.installEvt = null;
	if (r && r.outcome === "accepted") { store("qd.installed", "1"); toast("Installed. Open QuantDesk from your home screen."); }
	document.querySelectorAll(".install").forEach((e) => e.remove());
}

// ---- commands (own desk only) ------------------------------------------------------------------------------------
function confirmSheet(title, text, okLabel, fn) {
	openSheet(title, h("p", { style: "margin:0 0 18px;color:var(--fg2);font-size:14px;line-height:1.55" }, text),
		h("div", { class: "btns" }, h("button", { class: "btn", onclick: closeSheet }, "Cancel"),
			h("button", { class: "btn danger", onclick: () => { closeSheet(); fn(); } }, okLabel)));
}
async function command(cmd, arg) {
	if (window.QD_PUBLISHED) return toast("This site is read-only. Stop a run from the repo's Actions tab.");
	if (window.QD_DEMO) return toast("This is a snapshot: controls work on your own running desk.");
	try {
		await post("/api/i/command", { cmd, arg });
		toast(`${cap(cmd)} sent: applied on the engine's next minute`);
		refresh(true);
	} catch (e) { toast(e.message); }
}

// ==================================================================================================================
// CHART
// ==================================================================================================================
async function renderChart() {
	const syms = symbols();
	if (!syms.includes(S.sym)) S.sym = syms[0];
	seg($("#c-sym"), syms.map((s) => [s, s === "BANKNIFTY" ? "BANK" : s]), S.sym, (v) => { S.sym = v; store("qd.sym", v); renderChart(); });
	seg($("#c-iv"), [["1m", "1m"], ["5m", "5m"], ["15m", "15m"]], S.interval, (v) => { S.interval = v; store("qd.iv", v); renderChart(); });
	const v = views()[S.sym];
	renderQuoteHead(v);
	renderRead(v);
	await loadChart(S.sym, S.interval);
	if (S.tab !== "chart") return;
	drawChart(false);
	renderQuoteHead(v);
	renderLevelsTable(v);
	renderQuant(v);
}
function chartData() { return S.charts[S.sym + "|" + S.interval]; }
function renderQuoteHead(v) {
	const d = chartData(), B = d && d.bars, n = B ? B.t.length : 0;
	const last = v && fin(v.spot) ? v.spot : n ? B.c[n - 1] : null;
	const prev = v && fin(v.chg) && fin(v.spot) ? v.spot / (1 + v.chg) : null;
	const hi = n ? Math.max(...B.h) : null, lo = n ? Math.min(...B.l) : null;
	const box = $("#c-head");
	box.textContent = "";
	put(box, h("div", {}, h("div", { class: "lbl" }, S.sym + (d && d.day ? " · " + istDay(d.day + "T12:00:00+05:30") : "")),
		h("div", { class: "px" }, num(last)),
		h("div", { class: "ch " + cls(v && v.chg) }, prev != null ? `${signed(last - prev)} (${pct(v.chg)})` : "")),
		h("div", { class: "hl" }, h("div", {}, "H ", h("b", {}, num(hi))), h("div", {}, "L ", h("b", {}, num(lo))),
			h("div", {}, "VWAP ", h("b", {}, num(v ? v.vwap : d && d.vwap ? d.vwap[d.vwap.length - 1] : null)))));
	$("#c-fs-title").textContent = `${S.sym} · ${S.interval}`;
}
function renderRead(v) {
	const box = $("#c-read");
	box.textContent = "";
	$("#c-read-hint").textContent = v && S.state && S.state.heartbeat ? "as of " + ist(S.state.heartbeat.ts) : "";
	if (!v) { box.appendChild(empty("watch", `No read yet for ${S.sym}. It appears from 09:15 IST.`)); $("#c-evidence").textContent = ""; return; }
	const a = stance(v);
	const pos = Math.max(0, Math.min(100, (v.score + 1) * 50));
	put(box, h("div", { class: "pad" },
		h("div", { class: "row" }, biasTag(v.bias, v.score), info("bias"), h("span", { class: "grow" }),
			h("span", { class: "f3", style: "font-size:12px" }, "Conviction ", h("b", { class: "mono", style: "color:var(--fg)" }, num(v.conviction)), info("conviction"))),
		h("div", { class: "gauge", role: "meter", "aria-valuemin": "-1", "aria-valuemax": "1", "aria-valuenow": num(v.score), "aria-label": "bias score" },
			h("i", { style: `left:${pos}%` })),
		h("div", { class: "gscale" }, h("span", {}, "Bearish"), h("span", {}, "Neutral"), h("span", {}, "Bullish"))),
	h("div", { class: "facts" },
		fact("Doing now", a.label), fact("Day type", cap(words(v.day_type))),
		fact(h("span", {}, "Premium", info("premium")), `${cap(v.vol_view)} · IV ${num(v.iv, 1)} / RV ${num(v.rv, 1)}`), fact("Session", `${cap(v.phase)} · exp ${fmtExpiry(v.expiry)}`)),
	(v.vetoes || []).length ? h("div", { class: "flags" }, v.vetoes.map((x) => h("div", {}, icon("alert"), cap(x)))) : null,
	narrative(v.narrative));
	const ev = $("#c-evidence");
	ev.textContent = "";
	ev.appendChild(evidenceList(v.evidence || [], 6));
}
function fact(label, value) { return h("div", {}, h("span", {}, label), h("b", {}, value || "—")); }
function narrative(text) {
	if (!text) return null;
	const p = h("div", { class: "narr" }, text);
	return p;
}
const CAT = { trend: "Tape", structure: "Tape", momentum: "Tape", flow: "Flow", options: "Options", volatility: "Vol", news: "News", quant: "Quant", global: "Global" };
function evidenceList(list, limit) {
	const box = h("div", { class: "evl" });
	if (!list.length) { box.appendChild(empty("info", "No evidence yet.")); return box; }
	const sorted = [...list].sort((a, b) => Math.abs(b.direction * b.weight) - Math.abs(a.direction * a.weight));
	sorted.forEach((e, i) => {
		const w = Math.min(50, Math.abs(e.direction) * Math.min(e.weight, 1.2) / 1.2 * 50);
		box.appendChild(h("div", { class: "ev" },
			h("div", { class: "f" }, words(e.factor), h("small", {}, `${CAT[e.category] || cap(e.category)} · w ${num(e.weight, 1)}`)),
			h("div", { class: "dv", title: `${signed(e.direction)} × ${e.weight}` },
				h("i", { style: `${e.direction >= 0 ? "left:50%" : "right:50%"};width:${w}%;background:${e.direction >= 0 ? "var(--up)" : "var(--dn)"}` })),
			h("div", { class: "o" }, e.observation)));
		if (limit && i >= limit) box.lastChild.hidden = true;
	});
	if (limit && sorted.length > limit + 1) {
		const btn = h("button", { class: "link", style: "padding:8px 0 2px", onclick: () => { box.querySelectorAll(".ev[hidden]").forEach((x) => (x.hidden = false)); btn.remove(); } },
			`Show all ${sorted.length}`, icon("down"));
		box.appendChild(btn);
	} else box.querySelectorAll(".ev[hidden]").forEach((x) => (x.hidden = false));
	return box;
}
const LEVELS = {
	or_high: ["OR high", "or"], or_low: ["OR low", "or"], ib_high: ["IB high", "ib"], ib_low: ["IB low", "ib"], vah: ["VAH", "value"], val: ["VAL", "value"],
	poc: ["POC", "value"], pdh: ["PDH", "prior"], pdl: ["PDL", "prior"], cpr_tc: ["CPR top", "cpr"], cpr_bc: ["CPR bottom", "cpr"],
	call_wall: ["Call wall", "oi"], put_wall: ["Put wall", "oi"], day_high: ["Day high", "day"], day_low: ["Day low", "day"],
};
const LEVEL_GROUPS = [["prior", "Prior day", "--fg2"], ["or", "Opening range", "--acc"], ["value", "Value area", "--fg3"], ["oi", "OI walls", "--dn"], ["cpr", "CPR", "--fg3"], ["ib", "Initial balance", "--fg3"]];
function levelsOn() {
	let on = null;
	try { on = JSON.parse(store("qd.levels") || "null"); } catch (e) { on = null; }
	return Object.assign({ prior: true, or: true, value: false, oi: true, cpr: false, ib: false }, on || {});
}
function levelColor(k) {
	if (k === "call_wall") return cssv("--dn");
	if (k === "put_wall") return cssv("--up");
	const g = LEVELS[k][1];
	return cssv((LEVEL_GROUPS.find((x) => x[0] === g) || [0, 0, "--fg3"])[2]);
}
function renderLevelChips() {
	const d = chartData(), box = $("#c-levels"), on = levelsOn();
	const have = new Set(Object.keys((d && d.levels) || {}).map((k) => LEVELS[k] && LEVELS[k][1]));
	box.textContent = "";
	for (const [g, label, col] of LEVEL_GROUPS) {
		if (!have.has(g)) continue;
		box.appendChild(h("button", { class: "chip", "aria-pressed": String(!!on[g]), onclick: () => {
			const o = levelsOn(); o[g] = !o[g]; store("qd.levels", JSON.stringify(o)); drawChart(false); } },
			h("i", { style: `color:${cssv(col)}` }), label));
	}
	box.appendChild(info("levels"));
}
function renderLevelsTable(v) {
	const box = $("#c-levtbl"), d = chartData();
	box.textContent = "";
	const lv = Object.assign({}, (d && d.levels) || {}, (v && v.levels) || {});
	const spot = v && fin(v.spot) ? v.spot : d && d.bars ? d.bars.c[d.bars.c.length - 1] : null;
	const rows = Object.entries(lv).filter(([k, x]) => LEVELS[k] && fin(x) && x > 0).map(([k, x]) => [LEVELS[k][0], x, k]);
	if (!rows.length || !fin(spot)) { box.appendChild(empty("info", "Levels appear once the session has a few bars.")); return; }
	rows.push(["Spot", spot, "spot"]);
	rows.sort((a, b) => b[1] - a[1]);
	box.appendChild(h("div", { class: "scroll" }, h("table", { class: "tbl" },
		h("tr", {}, h("th", {}, "Level"), h("th", {}, "Price"), h("th", {}, "From spot")),
		rows.map(([n, x, k]) => h("tr", { class: k === "spot" ? "spot" : "" },
			h("td", {}, k !== "spot" && LEVELS[k] ? h("span", { class: "sw", style: `background:${levelColor(k)}` }) : null, n),
			h("td", {}, num(x)),
			h("td", { class: k === "spot" ? "" : cls(x - spot) }, k === "spot" ? "—" : [signed(x - spot), h("small", { class: "sub" }, pct((x - spot) / spot))]))))));
}
function renderQuant(v) {
	const box = $("#c-quant");
	box.textContent = "";
	const q = v && v.quant;
	if (!q) { box.appendChild(empty("info", "The quant layer's read appears while the desk runs.")); return; }
	const sig = q.sigma_30m_pct, spot = v.spot, band = fin(sig) && fin(spot) ? spot * sig / 100 : null;
	const drift = q.research_drift;
	put(box, h("div", { class: "facts", style: "border-top:0" },
		fact("30-min move, 1σ", fin(sig) ? `±${num(sig, 2)}%${band != null ? " · ±" + num(band, 0) + " pts" : ""}` : "—"),
		fact("Volatility (ann.)", fin(q.vol_ann) ? `${num(q.vol_ann, 1)}% · ${words(q.vol_source || "")}` : "—"),
		fact("Direction model", q.valid ? `Voting · AUC ${num(q.auc, 2)}` : `Off · AUC ${num(q.auc, 2)}`),
		fact("P(up) used", q.valid && fin(q.p_model) ? pct(q.p_model, 0).replace("+", "") : "coin flip + prior"),
		fact("Research drift", drift ? `${signed(drift.bps_day, 1)} bps/day · t ${num(drift.t, 1)}` : "none validated"),
		fact("Samples", fin(q.samples) ? num(q.samples, 0) : "—")),
	h("div", { class: "narr" }, q.valid
		? "The direction model beat a coin flip out of sample, so its probability feeds the EV of every candidate trade."
		: `The direction model scored AUC ${num(q.auc, 2)} out of sample: no better than a coin flip, so it doesn't vote. ${q.model_status ? cap(q.model_status) + "." : ""}`));
}

// ---- TradingView Lightweight Charts (vendored); IST on the axis by shifting times +5:30 ---------------------------------
const IST_OFF = 19800;
function lwTheme() {
	return {
		layout: { background: { type: "solid", color: cssv("--panel") }, textColor: cssv("--fg3"), fontSize: 11, fontFamily: cssv("--mono") || "monospace", attributionLogo: true },
		grid: { vertLines: { color: cssv("--grid") }, horzLines: { color: cssv("--grid") } },
		rightPriceScale: { borderColor: cssv("--line"), scaleMargins: { top: 0.12, bottom: 0.16 } },
		timeScale: { borderColor: cssv("--line"), timeVisible: true, secondsVisible: false, rightOffset: 5, minBarSpacing: 2 },
		crosshair: { mode: 0, vertLine: { color: cssv("--fg3"), width: 1, style: 3, labelBackgroundColor: cssv("--raise") },
			horzLine: { color: cssv("--fg3"), width: 1, style: 3, labelBackgroundColor: cssv("--raise") } },
	};
}
function candleColors() {
	const up = cssv("--cup"), dn = cssv("--cdn");
	return { upColor: up, downColor: dn, borderUpColor: up, borderDownColor: dn, wickUpColor: up, wickDownColor: dn };
}
function destroyLW() { if (S.lw) { try { S.lw.chart.remove(); } catch (e) { /* gone */ } } S.lw = null; }
function createLW(el) {
	const LW = window.LightweightCharts;
	el.textContent = "";
	const chart = LW.createChart(el, Object.assign(lwTheme(), {
		autoSize: true,
		handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
		handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
		localization: { priceFormatter: (p) => num(p, p > 1000 ? 1 : 2) },
	}));
	const candle = chart.addSeries(LW.CandlestickSeries, Object.assign({ priceLineVisible: true, lastValueVisible: true, priceLineStyle: 3 }, candleColors()));
	const vol = chart.addSeries(LW.HistogramSeries, { priceScaleId: "vol", priceFormat: { type: "volume" }, lastValueVisible: false, priceLineVisible: false });
	chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.86, bottom: 0 }, visible: false });
	const vwap = chart.addSeries(LW.LineSeries, { color: cssv("--vwap"), lineWidth: 2, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: false, title: "VWAP" });
	const markers = LW.createSeriesMarkers ? LW.createSeriesMarkers(candle, []) : null;
	S.lw = { chart, candle, vol, vwap, markers, lines: [], key: null, byTime: new Map(), marksAt: new Map() };
	chart.subscribeCrosshairMove((param) => ohlcLegend(param && param.time));
}
function ohlcLegend(time) {
	const d = chartData(), L = S.lw, el = $("#c-ohlc");
	if (!L || !d || !d.bars) { el.textContent = ""; return; }
	const B = d.bars, n = B.t.length;
	let i = n - 1;
	if (time != null) { const j = L.byTime.get(time); if (j != null) i = j; }
	const prev = i > 0 ? B.c[i - 1] : B.o[i], chg = (B.c[i] / prev - 1) * 100;
	const kv = (k, v) => [h("span", { class: "k" }, k), v, "  "];
	el.textContent = "";
	put(el, h("div", {}, h("b", {}, ist(B.t[i])), "  ", ...kv("O", num(B.o[i])), ...kv("H", num(B.h[i])), ...kv("L", num(B.l[i])),
		h("span", { class: "k" }, "C"), h("b", { class: cls(chg) }, `${num(B.c[i])} ${chg >= 0 ? "+" : "−"}${Math.abs(chg).toFixed(2)}%`)),
	h("div", { class: "r2" }, h("span", { style: "color:var(--vwap)" }, "VWAP " + num(d.vwap[i])), B.v[i] > 0 ? "  Vol " + num(B.v[i], 0) : "",
		...(L.marksAt.get(B.t[i]) || []).map((m) => `  ${m.kind === "entry" ? "▲" : "●"} ${m.text}`)));
}
function drawChart(reframe) {
	const el = $("#chart"), d = chartData();
	if (!window.LightweightCharts || !window.LightweightCharts.createChart) { el.textContent = ""; el.appendChild(empty("chart", "The chart library didn't load.")); return; }
	if (!d || !d.bars) {
		destroyLW();
		el.textContent = "";
		$("#c-ohlc").textContent = "";
		el.appendChild(empty("chart", "No bars recorded yet for this session."));
		$("#c-levels").textContent = "";
		return;
	}
	if (!S.lw) createLW(el);
	const L = S.lw, B = d.bars, n = B.t.length, up = cssv("--cup"), dn = cssv("--cdn");
	const T = (t) => t + IST_OFF;
	L.byTime = new Map(B.t.map((t, i) => [T(t), i]));
	L.candle.setData(B.t.map((t, i) => ({ time: T(t), open: B.o[i], high: B.h[i], low: B.l[i], close: B.c[i] })));
	const hasVol = B.v.some((x) => x > 0);
	L.vol.setData(hasVol ? B.t.map((t, i) => ({ time: T(t), value: B.v[i], color: (B.c[i] >= B.o[i] ? up : dn) + "44" })) : []);
	L.vwap.setData(B.t.map((t, i) => ({ time: T(t), value: d.vwap[i] })));
	L.lines.forEach((pl) => L.candle.removePriceLine(pl));
	L.lines = [];
	const on = levelsOn();
	for (const [k, x] of Object.entries(d.levels || {})) {
		if (!LEVELS[k] || !on[LEVELS[k][1]] || !(x > 0) || k === "vwap") continue;
		L.lines.push(L.candle.createPriceLine({ price: x, color: levelColor(k), lineWidth: k.includes("wall") ? 2 : 1, lineStyle: k === "poc" ? 0 : 2,
			axisLabelVisible: true, title: LEVELS[k][0] }));
	}
	renderLevelChips();
	const snap = (t) => { let lo = 0; for (let i = 0; i < n; i++) if (B.t[i] <= t) lo = i; return B.t[lo]; };
	L.marksAt = new Map();
	const mk = (d.markers || []).map((m) => {
		const bt = snap(m.t), long = m.dir >= 0, entry = m.kind === "entry";
		(L.marksAt.get(bt) || L.marksAt.set(bt, []).get(bt)).push(m);
		const loss = /₹-|₹−/.test(m.text);
		return { time: T(bt), position: entry ? (long ? "belowBar" : "aboveBar") : (long ? "aboveBar" : "belowBar"), shape: entry ? (long ? "arrowUp" : "arrowDown") : "circle",
			color: entry ? cssv("--acc-hi") : (loss ? dn : up), text: entry ? m.text.split(" · ")[0] : m.text.replace(/^\S+\s/, "") };
	}).sort((a, b) => a.time - b.time);
	if (L.markers) L.markers.setMarkers(mk); else if (L.candle.setMarkers) L.candle.setMarkers(mk);
	const key = `${d.symbol}|${d.interval}|${d.day}`;
	if (reframe || L.key !== key) {
		L.key = key;
		const narrow = el.clientWidth < 640, want = d.interval === "1m" ? (narrow ? 90 : 200) : d.interval === "5m" ? (narrow ? 60 : n) : n;
		if (n > want) L.chart.timeScale().setVisibleLogicalRange({ from: n - want, to: n + 4 });
		else L.chart.timeScale().fitContent();
	}
	ohlcLegend(null);
}
function restyleCharts() {
	if (S.lw) {
		S.lw.chart.applyOptions(lwTheme());
		S.lw.candle.applyOptions(candleColors());
		S.lw.vwap.applyOptions({ color: cssv("--vwap") });
		drawChart(false);
	}
	if (S.eq) { try { const t = lwTheme(); delete t.timeScale; S.eq.chart.applyOptions(t); S.eq.paint(); } catch (e) { /* gone */ } }
	if (S.tab === "desk" && S.state) renderDesk();
}
function fullscreen(on) {
	const box = $("#c-box");
	box.classList.toggle("fs", on);
	document.body.style.overflow = on ? "hidden" : "";
	if (S.lw) setTimeout(() => S.lw && S.lw.chart.timeScale().scrollToRealTime(), 60);
}

// ==================================================================================================================
// TRADES
// ==================================================================================================================
async function renderTrades(full) {
	seg($("#t-seg"), SUBS.trades, S.sub.trades, (v) => { S.sub.trades = v; history.replaceState(null, "", "#trades/" + v); renderTrades(true); }, "tab");
	const body = $("#t-body"), sub = S.sub.trades;
	if (sub === "positions") return renderPositionsTab(body);
	if (!full && sub !== "positions") return;
	body.textContent = "";
	body.appendChild(h("div", { class: "panel", style: "margin-top:14px" }, h("div", { class: "pad" }, h("span", { class: "skel", style: "width:70%" }), h("span", { class: "skel", style: "width:40%;margin-top:10px" }))));
	if (sub === "history") return renderHistory(body);
	if (sub === "performance") return renderPerformance(body);
	if (sub === "reviews") return renderReviews(body);
}
function renderPositionsTab(body) {
	const st = S.state || {}, hb = st.heartbeat || {}, pos = hb.positions || [], ct = st.closed_today || [];
	body.textContent = "";
	put(body, h("div", { class: "sec-h" }, h("h2", {}, "Open"), h("span", { class: "hint" }, pos.length ? inr(pos.reduce((a, p) => a + (p.pnl || 0), 0), true) + " open P&L" : "")));
	body.appendChild(pos.length ? h("div", { class: "stack" }, pos.map(positionCard)) : h("div", { class: "panel" }, empty("target", "Flat: no open positions. The desk takes at most one position at a time.")));
	put(body, h("div", { class: "sec-h" }, h("h2", {}, "Closed today"), h("span", { class: "hint mono " + cls(ct.reduce((a, t) => a + t.pnl, 0)) },
		ct.length ? inr(ct.reduce((a, t) => a + t.pnl, 0), true) : "")),
	h("div", { class: "panel rows" }, ct.length ? ct.map(tradeRow) : empty("trades", "No closed trades this session.")));
}
function tradeRow(t) {
	const open = t.status === "open";
	return h("button", { class: "trow", onclick: () => openTrade(t.id) },
		h("span", { class: "grade " + (t.grade || "") }, t.grade || "·"),
		h("div", { style: "min-width:0" }, h("div", { class: "t1" }, `${t.symbol} · ${setupName(t.strategy)}`),
			h("div", { class: "t2" }, `${ist(t.opened_at)}${t.closed_at ? "–" + ist(t.closed_at) : ""} · ${t.structure ? structName(t.structure) + " · " : ""}${open ? "open" : words(t.exit_reason || "")}`)),
		h("div", { class: "v " + (open ? "" : cls(t.pnl)) }, open ? "Open" : inr(t.pnl, true),
			h("small", {}, !open && fin(t.r_multiple) ? signed(t.r_multiple) + "R" : "")));
}
async function renderHistory(body) {
	let rows;
	try { rows = await api("/api/i/trades?n=300"); } catch (e) { body.textContent = ""; body.appendChild(empty("alert", e.message)); return; }
	if (S.tab !== "trades" || S.sub.trades !== "history") return;
	body.textContent = "";
	if (!rows.length) { body.appendChild(h("div", { class: "panel", style: "margin-top:14px" }, empty("trades", "No trades yet. Every trade the desk takes lands here with its full reasoning."))); return; }
	const days = new Map();
	for (const t of rows) { const d = String(t.opened_at).slice(0, 10); (days.get(d) || days.set(d, []).get(d)).push(t); }
	const closed = rows.filter((t) => t.status !== "open"), net = closed.reduce((a, t) => a + t.pnl, 0), wins = closed.filter((t) => t.pnl > 0).length;
	body.appendChild(h("div", { class: "kpis", style: "margin-top:14px" },
		h("div", { class: "kpi" }, h("span", {}, "Trades"), h("b", {}, String(rows.length))),
		h("div", { class: "kpi" }, h("span", {}, "Net P&L"), h("b", { class: cls(net) }, inr(net, true))),
		h("div", { class: "kpi" }, h("span", {}, "Win rate"), h("b", {}, closed.length ? Math.round(wins / closed.length * 100) + "%" : "—")),
		h("div", { class: "kpi" }, h("span", {}, "Sessions"), h("b", {}, String(days.size)))));
	for (const [d, ts] of days) {
		const dn = ts.filter((t) => t.status !== "open").reduce((a, t) => a + t.pnl, 0);
		put(body, h("div", { class: "panel rows", style: "margin-top:12px" },
			h("div", { class: "dayh" }, h("span", {}, istDay(d + "T12:00:00+05:30")), h("span", { class: "mono " + cls(dn) }, inr(dn, true))), ts.map(tradeRow)));
	}
}
async function openTrade(id) {
	let t;
	try { t = await api("/api/i/trade?id=" + encodeURIComponent(id)); } catch (e) { return toast(e.message); }
	const para = (label, text) => (text ? h("div", { class: "para" }, h("div", { class: "lbl" }, label), h("p", {}, text)) : null);
	const [why, ...rest] = String(t.rationale || "").split(" Market read: ");
	const open = t.status === "open", held = t.closed_at ? Math.round((tms(t.closed_at) - tms(t.opened_at)) / 60000) : null;
	const legs = (t.legs || []).map((l) => ({ symbol: (l.instrument || {}).symbol || l.symbol, qty: l.qty, entry: l.entry_price, exit: l.exit_price }));
	openSheet(`${t.symbol} · ${setupName(t.strategy)}`,
		h("div", { class: "row", style: "align-items:flex-end;margin-bottom:12px" },
			h("div", { class: "grow" }, h("div", { class: "lbl" }, open ? "Open P&L" : "Net P&L, after costs"),
				h("div", { class: "mono " + cls(t.pnl), style: "font-size:30px;font-weight:600;letter-spacing:-.02em" }, open ? "Open" : inr(t.pnl, true))),
			h("div", { style: "text-align:right" }, h("span", { class: "grade " + (t.grade || "") }, t.grade || "·"), info("grade"))),
		h("div", { class: "kpis" },
			h("div", { class: "kpi" }, h("span", {}, "R multiple"), h("b", { class: cls(t.r_multiple) }, fin(t.r_multiple) ? signed(t.r_multiple) + "R" : "—")),
			h("div", { class: "kpi" }, h("span", {}, "Risked"), h("b", {}, inr(t.initial_risk))),
			h("div", { class: "kpi" }, h("span", {}, "Lots · costs"), h("b", {}, `${t.units} · ${inr(t.fees)}`)),
			h("div", { class: "kpi" }, h("span", {}, "Held"), h("b", {}, held != null ? `${held} min` : "—"))),
		h("div", { class: "panel", style: "margin-top:12px" }, h("div", { class: "pad" },
			h("div", { class: "row", style: "font-size:13px" }, h("span", { class: "f3" }, "Entry"), h("b", { class: "mono" }, `${ist(t.opened_at, true)} @ ${num(t.entry_underlying)}`)),
			h("div", { class: "row", style: "font-size:13px;margin-top:6px" }, h("span", { class: "f3" }, "Exit"),
				h("b", { class: "mono" }, t.closed_at ? `${ist(t.closed_at)} @ ${num(t.exit_underlying)}` : "open")),
			fin(t.stop) && fin(t.target) ? h("div", { style: "margin-top:10px" }, track(t.stop, t.target, t.entry_underlying, open ? null : t.exit_underlying, t.direction)) : null,
			legs.length ? h("div", { style: "margin-top:12px" }, legsTable(legs, true)) : null)),
		para("Why it was taken", why), para("What the market looked like", rest.join(" Market read: ")),
		para("Sizing", (t.sizing || []).join(" · ")), para("Exit", t.exit_reason ? `${cap(words(t.exit_reason))}: ${t.exit_note || ""}` : "Still open"),
		para("Review", t.review), para("Lessons", (t.lessons || []).join(" ")),
		t.fills && t.fills.length ? h("div", { class: "para" }, h("div", { class: "lbl" }, "Fills"), h("div", { class: "panel scroll" }, h("table", { class: "tbl" },
			h("tr", {}, h("th", {}, "Time"), h("th", {}, "Contract"), h("th", {}, "Qty"), h("th", {}, "Price"), h("th", {}, "Fees")),
			t.fills.map((f) => h("tr", {}, h("td", {}, ist(f.ts)), h("td", {}, f.symbol), h("td", {}, f.qty), h("td", {}, num(f.price)), h("td", {}, num(f.fees))))))) : null);
}
async function renderPerformance(body) {
	let s;
	try { s = await api("/api/i/stats"); } catch (e) { body.textContent = ""; body.appendChild(empty("alert", e.message)); return; }
	if (S.tab !== "trades" || S.sub.trades !== "performance") return;
	body.textContent = "";
	if (!s.trades) { body.appendChild(h("div", { class: "panel", style: "margin-top:14px" }, empty("chart", "Performance appears after the first closed trade."))); return; }
	const kpi = (l, v, c) => h("div", { class: "kpi" }, h("span", {}, l), h("b", { class: c || "" }, v));
	put(body, h("div", { class: "kpis", style: "margin-top:14px" },
		kpi("Net P&L", inr(s.net, true), cls(s.net)), kpi("Return", pct(s.net / s.capital, 1), cls(s.net)),
		kpi("Win rate", Math.round(s.win_rate * 100) + "%"), kpi("Profit factor", s.profit_factor ? num(s.profit_factor) : "—"),
		kpi("Avg trade", signed(s.avg_r) + "R", cls(s.avg_r)), kpi("Max drawdown", pct(s.max_dd, 1), "dn"),
		kpi("Green days", Math.round(s.green_days * 100) + "%"), kpi("Costs paid", inr(s.fees))),
	h("div", { class: "sec-h" }, h("h2", {}, "Equity"), h("span", { class: "hint" }, `${s.trades} trades · ${s.sessions} sessions`)));
	const eq = h("div", { class: "panel" }, h("div", { class: "eqc" }));
	body.appendChild(eq);
	drawEquity(eq.firstChild, s);
	put(body, h("div", { class: "sec-h" }, h("h2", {}, "P&L by setup")), h("div", { class: "panel" }, hbars(s.by_setup, setupName)));
	const BRK = [["by_day_type", "Day type"], ["by_structure", "Structure"], ["by_exit", "Exit"], ["by_hour", "Hour"], ["by_symbol", "Index"]];
	const segEl = h("div", { class: "seg" }), tbl = h("div", { class: "panel scroll" });
	const paint = () => {
		tbl.textContent = "";
		tbl.appendChild(h("table", { class: "tbl" }, h("tr", {}, h("th", {}, ""), h("th", {}, "Trades"), h("th", {}, "Win"), h("th", {}, "Avg R"), h("th", {}, "Net")),
			(s[S.brk] || []).map((r) => h("tr", {}, h("td", {}, cap(words(r.key))), h("td", {}, r.trades), h("td", {}, Math.round(r.win * 100) + "%"),
				h("td", { class: cls(r.avg_r) }, signed(r.avg_r)), h("td", { class: cls(r.pnl) }, inr(r.pnl, true))))));
	};
	seg(segEl, BRK, S.brk, (v) => { S.brk = v; paint(); });
	paint();
	put(body, h("div", { class: "sec-h" }, h("h2", {}, "Breakdown")), h("div", { style: "margin-bottom:8px" }, segEl), tbl);
	if ((s.calibration || []).length)
		put(body, h("div", { class: "sec-h" }, h("h2", {}, "Calibration"), h("span", { class: "hint" }, "assumed vs realised")),
			h("div", { class: "panel" }, h("div", { class: "narr", style: "border-top:0" },
				"P(right direction) the quant layer priced each trade on, against how often the index actually went that way by the exit. Until the realised column tracks the assumed one over many trades, the edge is unproven."),
			h("div", { class: "scroll" }, h("table", { class: "tbl" }, h("tr", {}, h("th", {}, "Source"), h("th", {}, "Assumed"), h("th", {}, "Trades"), h("th", {}, "Realised"), h("th", {}, "Net")),
				s.calibration.map((r) => h("tr", {}, h("td", {}, cap(String(r.source).split(/ from | \(/)[0])), h("td", {}, Math.round(r.assumed * 100) + "%"), h("td", {}, r.trades),
					h("td", {}, Math.round(r.realised * 100) + "%"), h("td", { class: cls(r.pnl) }, inr(r.pnl, true))))))));
}
function drawEquity(el, s) {
	if (S.eq) { try { S.eq.chart.remove(); } catch (e) { /* gone */ } S.eq = null; }
	const LW = window.LightweightCharts;
	if (!LW || !s.equity || !s.equity.length) return;
	const chart = LW.createChart(el, Object.assign(lwTheme(), { autoSize: true, handleScroll: false, handleScale: false,
		localization: { priceFormatter: (p) => inr(p) } }));
	chart.applyOptions({ timeScale: { timeVisible: false, borderColor: cssv("--line"), rightOffset: 0, fixLeftEdge: true, fixRightEdge: true },
		rightPriceScale: { scaleMargins: { top: 0.15, bottom: 0.1 } } });
	const ser = chart.addSeries(LW.BaselineSeries, { baseValue: { type: "price", price: s.capital }, lineWidth: 2, priceLineVisible: false });
	const first = new Date(s.equity[0].day + "T00:00:00Z");
	first.setUTCDate(first.getUTCDate() - 1);
	const pts = [{ time: first.toISOString().slice(0, 10), value: s.capital }, ...s.equity.map((r) => ({ time: r.day, value: r.equity }))];
	const paint = () => {
		const up = cssv("--up"), dn = cssv("--dn");
		ser.applyOptions({ topLineColor: up, topFillColor1: up + "40", topFillColor2: up + "05", bottomLineColor: dn, bottomFillColor1: dn + "05", bottomFillColor2: dn + "40" });
	};
	paint();
	ser.setData(pts);
	ser.createPriceLine({ price: s.capital, color: cssv("--fg3"), lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "start" });
	chart.timeScale().fitContent();
	S.eq = { chart, paint };
}
function hbars(rows, name) {
	const box = h("div", { style: "padding:6px 0" });
	rows = [...rows].sort((a, b) => b.pnl - a.pnl);
	const max = Math.max(...rows.map((r) => Math.abs(r.pnl)), 1);
	for (const r of rows) {
		const w = Math.abs(r.pnl) / max * 50;
		box.appendChild(h("div", { class: "hb" }, h("span", {}, name ? name(r.key) : r.key, h("div", { class: "f3", style: "font-size:11px" }, `${r.trades} trade${r.trades === 1 ? "" : "s"} · ${Math.round(r.win * 100)}% win`)),
			h("div", { class: "bar2" }, h("i", { style: `${r.pnl >= 0 ? "left:50%" : "right:50%"};width:${w}%;background:${r.pnl >= 0 ? "var(--up)" : "var(--dn)"}` })),
			h("b", { class: cls(r.pnl) }, inr(r.pnl, true))));
	}
	return box;
}
async function renderReviews(body) {
	let dates;
	try { dates = await api("/api/i/reviews"); } catch (e) { body.textContent = ""; body.appendChild(empty("alert", e.message)); return; }
	if (S.tab !== "trades" || S.sub.trades !== "reviews") return;
	body.textContent = "";
	body.appendChild(h("div", { class: "panel rows", style: "margin-top:14px" }, dates.length ? dates.map((d) =>
		h("button", { class: "set", onclick: async () => {
			try { const r = await api("/api/i/review?date=" + encodeURIComponent(d)); openSheet("Session " + istDay(d + "T12:00:00+05:30"), markdown(r.markdown)); } catch (e) { toast(e.message); }
		} }, icon("book"), h("div", { class: "grow" }, istDay(d + "T12:00:00+05:30"), h("small", {}, "The desk's own post-session review")), icon("chev")))
		: empty("book", "Session reviews appear after each close.")));
}
function inline(text) {
	const out = [];
	String(text).split(/(\*\*[^*]+\*\*)/).forEach((p) => out.push(p.startsWith("**") && p.endsWith("**") ? h("b", {}, p.slice(2, -2)) : p));
	return out;
}
function markdown(md) {
	const root = h("div", { class: "md" }), lines = String(md || "").split("\n");
	for (let i = 0; i < lines.length; i++) {
		const l = lines[i];
		if (/^#{1,3} /.test(l)) root.appendChild(h(l.startsWith("## ") ? "h2" : "h1", {}, inline(l.replace(/^#+ /, ""))));
		else if (l.startsWith("|")) {
			const rows = [];
			while (i < lines.length && lines[i].startsWith("|")) { if (!/^\|[-:| ]+\|$/.test(lines[i])) rows.push(lines[i]); i++; }
			i--;
			const cells = (r) => r.slice(1, -1).split("|").map((c) => c.trim());
			root.appendChild(h("div", { class: "panel scroll" }, h("table", { class: "tbl" }, rows.map((r, k) => h("tr", {}, cells(r).map((c) => h(k ? "td" : "th", {}, inline(c))))))));
		} else if (l.startsWith("- ")) root.appendChild(h("li", {}, inline(l.slice(2))));
		else if (l.trim()) root.appendChild(h("p", {}, inline(l)));
	}
	return root;
}

// ==================================================================================================================
// BRAIN
// ==================================================================================================================
function renderBrain() {
	const syms = symbols();
	if (!syms.includes(S.brainSym)) S.brainSym = syms[0];
	seg($("#b-sym"), syms.map((s) => [s, s]), S.brainSym, (v) => { S.brainSym = v; renderBrain(); });
	const hb = (S.state && S.state.heartbeat) || {}, v = (hb.views || {})[S.brainSym] || {}, b = v.brain, body = $("#b-body");
	body.textContent = "";
	// regime
	const rp = h("div", { class: "panel", style: "margin-top:14px" });
	if (!b) rp.appendChild(empty("globe", "The brain's global read appears here while the desk runs: which world markets are moving, how they usually lean on India, and what that means for the bias."));
	else {
		const rc = b.regime === "risk-on" ? "on" : b.regime === "risk-off" ? "offr" : "mixed";
		put(rp, h("div", { class: "regime" }, h("div", { class: "lbl" }, "Global regime", info("regime")),
			h("div", { class: "big " + rc }, words(b.regime)),
			h("div", { class: "tags" }, h("span", { class: "tag line" }, `score ${signed(b.regime_score)}`),
				h("span", { class: "tag " + (b.stress >= 2 ? "bear" : "line") }, `stress ${num(b.stress, 1)}σ`, info("stress")),
				h("span", { class: "tag " + (b.size_mult < 1 ? "acc" : "line") }, b.size_mult < 1 ? `size ×${num(b.size_mult)}` : "full size"))),
		h("div", { class: "narr" }, b.narrative));
	}
	body.appendChild(rp);
	// the open, explained
	if (b && b.gap) {
		const g = b.gap, against = g.explained * g.gap < 0 && Math.abs(g.explained) > 0.0005;
		const txt = against
			? `${S.brainSym} opened ${pct(g.gap)}, against the global cue: what the world did while India was shut pointed to ${pct(g.explained)}. India shrugged it off.`
			: g.share != null && g.share > 1.2
				? `${S.brainSym} opened ${pct(g.gap)}, while what the world did overnight pointed to ${pct(g.explained)}: India moved less than the global cue.`
				: `${S.brainSym} opened ${pct(g.gap)}. What global markets did while India was shut accounts for ${pct(g.explained)}` +
					(g.share != null && g.share > 0 ? `, about ${Math.round(g.share * 100)}% of the gap.` : ".");
		put(body, h("div", { class: "sec-h" }, h("h2", {}, "Today's open, explained")),
			h("div", { class: "panel pad" }, h("div", { style: "font-size:14px;line-height:1.5" }, txt),
				(g.parts || []).length ? h("div", { class: "chips", style: "margin-top:10px;flex-wrap:wrap" }, g.parts.map(([n, x]) => h("span", { class: "tag " + (x > 0 ? "bull" : x < 0 ? "bear" : "flat") }, `${n} ${pct(x)}`))) : null));
	}
	// influence graph
	put(body, h("div", { class: "sec-h" }, h("h2", {}, "How everything connects")),
		h("div", { class: "panel pad" }, h("div", { class: "f3", style: "font-size:12px;margin-bottom:8px" }, "World → what the desk reads in India → the bias → the decision. Solid: research-validated leads that vote. Dashed: explains, doesn't vote."),
			h("div", { class: "scroll", id: "bgraph" })));
	drawBrainGraph($("#bgraph"), v, hb);
	// what's pushing the bias
	put(body, h("div", { class: "sec-h" }, h("h2", {}, "What's pushing the bias"), h("span", { class: "hint" }, v.bias ? cap(v.bias) + " " + signed(v.score) : "")),
		h("div", { class: "panel" }, evidenceList(v.evidence || [], 8)));
	// global markets board
	const G = hb.global || {}, mk = G.markets || {}, board = h("div", { class: "panel rows" });
	const regions = ["US", "Asia", "Europe", "FX", "Commodities", "Rates"];
	for (const r of regions) {
		const ms = Object.entries(mk).filter(([, m]) => m.region === r);
		if (!ms.length) continue;
		board.appendChild(h("div", { class: "reg-h" }, r));
		for (const [, m] of ms) {
			const c = mchg(m), lean = c == null || !m.india ? "" : (c * m.india > 0 ? "up" : c * m.india < 0 ? "dn" : "");
			board.appendChild(h("div", { class: "grow-row" },
				h("div", { class: "n" }, h("i", { class: m.live ? "on" : "" }), h("span", {}, m.name)),
				h("div", { class: "v" }, m.last != null ? num(m.last, m.last > 1000 ? 0 : 2) : "—"),
				h("span", { class: "chg mono " + (lean || "flat") }, pct(c)),
				h("div", { class: "s" }, `${m.live && m.since_open != null ? "since 09:15 IST" : "last session" + (m.prior_date ? " " + m.prior_date : "")}` +
					(m.z30 != null ? ` · 30m ${signed(m.z30, 1)}σ` : "") + (m.india ? ` · usually ${m.india > 0 ? "moves with" : "leans against"} India` : ""))));
		}
	}
	if (!Object.keys(mk).length) board.appendChild(empty("globe", "Global markets appear here while the desk runs."));
	put(body, h("div", { class: "sec-h" }, h("h2", {}, "Global markets"), h("span", { class: "hint" }, "colour: good or bad for India")), board);
	// the wiring
	const ds = (b && b.drivers) || [];
	put(body, h("div", { class: "sec-h" }, h("h2", {}, "The wiring, measured"), h("span", { class: "hint" }, "weekly research, real data")),
		h("div", { class: "panel scroll" }, ds.length ? h("table", { class: "tbl" },
			h("tr", {}, h("th", {}, "Driver"), h("th", {}, "Gap ρ"), h("th", {}, "Same-5m ρ"), h("th", {}, "Lead t"), h("th", {}, "Votes")),
			ds.map((d) => h("tr", {}, h("td", {}, d.name), h("td", {}, fin(d.gap_corr) ? num(d.gap_corr) : "—"), h("td", {}, fin(d.co_corr) ? num(d.co_corr) : "—"),
				h("td", {}, fin(d.lead_t) ? num(d.lead_t, 1) : "—"), h("td", { class: d.validated ? "up" : "f3" }, d.validated ? (d.lead_sign < 0 ? "yes, fades" : "yes") : "no"))))
			: empty("info", "Appears with the brain's first read.")));
}
function drawBrainGraph(el, v, hb) {
	el.textContent = "";
	const b = v.brain;
	if (!b && !(v.evidence || []).length) { el.appendChild(empty("brain", "Appears with the first market read.")); return; }
	const W = 340, colX = [2, 126, 236], nodeW = [112, 92, 102];
	const world = (b && b.drivers) || [];
	const agg = {};
	for (const e of v.evidence || []) {
		const n = CAT[e.category] || cap(e.category);
		const a = (agg[n] = agg[n] || { n, c: 0, w: 0 });
		a.c += e.direction * e.weight; a.w += e.weight;
	}
	const india = ["Tape", "Flow", "Options", "Vol", "News", "Quant", "Global"].filter((k) => agg[k]).map((k) => agg[k]);
	const rowH = 36, H = Math.max(world.length, india.length, 3) * rowH + 30;
	const g = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Influence graph from global markets to the decision" }, el);
	const yOf = (i, n) => 24 + (H - 30) * (i + 0.5) / n;
	const color = (x) => (x > 0.02 ? cssv("--up") : x < -0.02 ? cssv("--dn") : cssv("--line2"));
	[["World", colX[0]], ["India read", colX[1]], ["Bias → decision", colX[2]]].forEach(([t, x]) => { svg("text", { x, y: 12, class: "t" }, g).textContent = t; });
	const biasY = H / 2 - 22, decY = H / 2 + 26;
	const edge = (x1, y1, x2, y2, w, col, dash) => svg("path", { d: `M${x1} ${y1} C${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2} ${y2}`,
		fill: "none", stroke: col, "stroke-width": w, "stroke-dasharray": dash || "", "stroke-opacity": 0.8 }, g);
	const node = (x, y, w, t1, t2, col) => {
		svg("rect", { x, y: y - 15, width: w, height: 30, rx: 8, fill: cssv("--panel2"), stroke: col, "stroke-width": 1.5 }, g);
		svg("text", { x: x + 7, y: y - 2, class: "t" }, g).textContent = t1;
		svg("text", { x: x + 7, y: y + 10 }, g).textContent = t2;
	};
	const gNode = india.find((x) => x.n === "Global"), newsNode = india.find((x) => x.n === "News");
	world.forEach((d, i) => {
		const y = yOf(i, world.length), p = d.pressure || 0;
		const target = d.validated && gNode ? [colX[1], yOf(india.indexOf(gNode), india.length)] : [colX[2], biasY];
		const strength = d.validated ? 3 : Math.max(0.6, Math.abs(d.gap_corr || d.co_corr || 0) * 5);
		edge(colX[0] + nodeW[0], y, target[0], target[1], strength, color(p), d.validated ? "" : "3 3");
		if (d.news_n && newsNode) edge(colX[0] + nodeW[0], y, colX[1], yOf(india.indexOf(newsNode), india.length), 0.8, cssv("--line2"), "1 3");
		const mv = d.move || {};
		const txt = mv.r30 != null ? `${pct(mv.r30)} 30m` : mv.prior_ret != null ? `${pct(mv.prior_ret)} prev` : "—";
		node(colX[0], y, nodeW[0], d.name.length > 16 ? d.name.slice(0, 15) + "…" : d.name, txt, color(p));
	});
	const tot = india.reduce((a, x) => a + x.w, 0) || 1;
	india.forEach((x, i) => {
		const y = yOf(i, india.length), share = x.c / tot;
		edge(colX[1] + nodeW[1], y, colX[2], biasY, Math.max(0.8, Math.abs(share) * 9), color(share));
		node(colX[1], y, nodeW[1], x.n, signed(share), color(share));
	});
	node(colX[2], biasY, nodeW[2], `${S.brainSym} ${v.bias || ""}`, `score ${fin(v.score) ? signed(v.score) : "—"}`, color(v.score || 0));
	edge(colX[2] + nodeW[2] / 2, biasY + 15, colX[2] + nodeW[2] / 2, decY - 15, 1.5, cssv("--line2"));
	const a = readAction(v.action), holding = (hb.positions || []).some((p) => p.symbol === S.brainSym);
	node(colX[2], decY, nodeW[2], "Decision", holding ? "holding" : a.kind === "none" ? "—" : a.label.toLowerCase().slice(0, 16), cssv("--acc-hi"));
}

// ==================================================================================================================
// FEED
// ==================================================================================================================
async function renderFeed(full) {
	seg($("#f-seg"), SUBS.feed, S.sub.feed, (v) => { S.sub.feed = v; history.replaceState(null, "", "#feed/" + v); renderFeed(true); }, "tab");
	if (S.sub.feed === "log") { if (full) await renderLog(true); return; }
	const body = $("#f-body");
	const rows = await loadNews(full ? 30000 : 60000);
	if (S.tab !== "feed" || S.sub.feed !== "news") return;
	body.textContent = "";
	const hb = (S.state && S.state.heartbeat) || {}, vs = Object.entries(hb.views || {});
	const tone = h("div", { class: "panel rows", style: "margin-top:14px" });
	if (!vs.length) tone.appendChild(empty("news", "The desk's read of the news appears here while it runs."));
	for (const [u, v] of vs) {
		const n = v.news;
		tone.appendChild(h("div", { class: "pad", style: "display:grid;gap:8px" },
			h("div", { class: "row" }, h("b", {}, u), h("span", { class: "grow" }),
				n ? h("span", { class: "tag " + (n.tone > 0.1 ? "bull" : n.tone < -0.1 ? "bear" : "flat") }, `tone ${signed(n.tone)}`) : h("span", { class: "f3", style: "font-size:12px" }, "no relevant stories in 2 h")),
			n ? h("div", { class: "dv", style: "height:8px" }, h("i", { style: `${n.tone >= 0 ? "left:50%" : "right:50%"};width:${Math.min(Math.abs(n.tone), 1) * 50}%;background:${n.tone >= 0 ? "var(--up)" : "var(--dn)"}` })) : null,
			n ? h("div", { class: "f3", style: "font-size:12px" }, `${n.n} ${n.n === 1 ? "story" : "stories"} in the last 2 h · weighs in the bias as “news” evidence`) : null,
			n && n.breaking ? h("div", { class: "banner", style: "margin:0;background:var(--dn-bg)" }, icon("bolt"),
				h("div", {}, h("b", {}, `Breaking, ${Math.round(n.breaking.age_min)} min ago: `), n.breaking.title, ". No new entries until it settles.")) : null));
	}
	const hl = hb.news_health || {}, names = Object.keys(hl), ok = names.filter((k) => String(hl[k]).startsWith("ok"));
	put(body, tone, names.length ? h("div", { class: "note", style: "padding:8px 4px 0" }, `${ok.length} of ${names.length} feeds live` +
		(ok.length < names.length ? ` · down: ${names.filter((k) => !ok.includes(k)).join(", ")}` : "")) : null);
	const chips = h("div", { class: "chips", style: "margin:16px 0 10px" });
	const paint = () => {
		chips.textContent = "";
		for (const [val, label] of [["", "All"], ["NIFTY", "NIFTY"], ["BANKNIFTY", "BANKNIFTY"], ["high", "High impact"], ["bear", "Bearish"], ["bull", "Bullish"]])
			chips.appendChild(h("button", { class: "chip", "aria-pressed": String(S.newsF === val), onclick: () => { S.newsF = val; paint(); } }, label));
		const f = S.newsF;
		const shown = rows.filter((r) => !f || (f === "high" ? r.impact === "high" : f === "bear" ? r.sentiment < -0.15 : f === "bull" ? r.sentiment > 0.15 : ((r.about || {})[f] || 0) >= 2));
		list.textContent = "";
		if (!shown.length) list.appendChild(empty("news", rows.length ? "Nothing matches this filter." : "No headlines yet. The desk reads 10 feeds every few minutes while it runs."));
		shown.slice(0, 120).forEach((r) => list.appendChild(newsRow(r)));
	};
	const list = h("div", { class: "panel rows" });
	put(body, chips, list);
	paint();
}
function newsRow(r, compact) {
	const k = r.sentiment > 0.15 ? "bull" : r.sentiment < -0.15 ? "bear" : "";
	const tags = Object.entries(r.about || {}).filter(([t, x]) => t !== "macro" && x >= 2).map(([t]) => t);
	if ((r.about || {}).macro >= 2) tags.push("macro");
	const src = (r.sources || [r.source]).filter(Boolean);
	return h("div", { class: "li" },
		h("div", { class: "meta" }, h("span", { class: "sent " + k, title: `sentiment ${signed(r.sentiment)}` }),
			r.impact && r.impact !== "low" ? h("span", { class: "imp " + r.impact }, r.impact) : null,
			h("span", {}, `${src.slice(0, compact ? 1 : 3).join(", ")}${src.length > (compact ? 1 : 3) ? " +" + (src.length - (compact ? 1 : 3)) : ""}`), h("span", {}, "·"), h("span", {}, ago(r.ts)),
			compact ? null : tags.map((t) => h("span", { class: "tag line", style: "height:18px;font-size:10.5px" }, t))),
		r.link ? h("a", { class: "ttl", href: r.link, target: "_blank", rel: "noopener noreferrer" }, r.title) : h("div", { class: "ttl" }, r.title));
}
async function renderLog(reset) {
	const body = $("#f-body");
	if (reset) {
		S.thBefore = null;
		body.textContent = "";
		const chips = h("div", { class: "chips", style: "margin:14px 0 10px" });
		for (const [val, label] of [["", "All"], ...symbols().map((s) => [s, s]), ["trades", "Trades only"]])
			chips.appendChild(h("button", { class: "chip", "aria-pressed": String(S.thSym === val), onclick: () => { S.thSym = val; renderLog(true); } }, label));
		put(body, chips, h("div", { class: "panel rows", id: "f-log" }), h("button", { class: "btn block", id: "f-more", style: "margin-top:10px", onclick: () => renderLog(false) }, "Load earlier"));
	}
	let rows;
	const sym = S.thSym && S.thSym !== "trades" ? "&symbol=" + S.thSym : "";
	try { rows = await api(`/api/i/thoughts?n=40${sym}${S.thBefore ? "&before=" + S.thBefore : ""}`); } catch (e) { rows = []; }
	if (S.tab !== "feed" || S.sub.feed !== "log") return;
	const list = $("#f-log");
	if (!list) return;
	const shown = S.thSym === "trades" ? rows.filter((r) => /^(ENTER|EXIT)/.test(r.action || "")) : rows;
	if (reset && !rows.length) list.appendChild(empty("watch", "The desk's reasoning appears here, one entry every few minutes and on every trade."));
	let lastDay = list.dataset.day || "";
	for (const r of shown) {
		const day = String(r.ts).slice(0, 10);
		if (day !== lastDay) { list.appendChild(h("div", { class: "dayh" }, h("span", {}, istDay(r.ts)), h("span", {}, ""))); lastDay = day; }
		list.appendChild(thoughtRow(r));
	}
	list.dataset.day = lastDay;
	if (rows.length) S.thBefore = rows[rows.length - 1].id;
	$("#f-more").hidden = rows.length < 40;
}
function thoughtRow(r) {
	const a = readAction(r.action), trade = a.kind === "enter" || a.kind === "exit";
	const more = h("div", { hidden: true, style: "margin-top:8px" }, r.evidence ? evidenceList(r.evidence) : null);
	const row = h("button", { class: "th", "aria-expanded": "false", onclick: () => {
		more.hidden = !more.hidden;
		row.classList.toggle("open", !more.hidden);
		row.setAttribute("aria-expanded", String(!more.hidden));
	} },
	h("div", { class: "tm" }, ist(r.ts)),
	h("div", { style: "min-width:0" },
		h("div", { class: "row", style: "flex-wrap:wrap;gap:6px" }, h("b", { style: "font-size:13px" }, r.symbol), biasTag(r.bias, Number(r.score)),
			h("span", { class: "f3", style: "font-size:12px" }, `${words(r.day_type)} · c ${num(r.conviction)}`)),
		h("div", { class: "act" + (trade ? " trade" : "") }, a.label + (a.reason ? ": " + a.reason : "")),
		h("div", { class: "nar" }, r.narrative), more));
	return row;
}

// ==================================================================================================================
// settings
// ==================================================================================================================
function openSettings() {
	const mode = themeMode(), themeSeg = h("div", { class: "seg full" });
	seg(themeSeg, [["auto", "Auto"], ["dark", "Dark"], ["light", "Light"]], mode, (v) => applyTheme(v));
	const acct = S.accounts.length > 1 ? h("select", { class: "btn block", style: "margin-top:8px", "aria-label": "Account", onchange: (e) => {
		S.account = e.target.value; store("qd.account", S.account); S.charts = {}; S.news = null; closeSheet(); refresh(true); } },
		S.accounts.map((a) => h("option", Object.assign({ value: a.id }, a.id === S.account ? { selected: true } : {}), a.label))) : null;
	const standalone = matchMedia("(display-mode: standalone)").matches || navigator.standalone;
	const ios = /iphone|ipad|ipod/i.test(navigator.userAgent);
	openSheet("Settings",
		h("div", { class: "lbl", style: "margin-bottom:8px" }, "Appearance"), themeSeg,
		acct ? h("div", { class: "lbl", style: "margin:18px 0 0" }, "Account") : null, acct,
		h("div", { class: "lbl", style: "margin:18px 0 8px" }, "App"),
		h("div", { class: "panel rows" },
			standalone ? null : h("button", { class: "set", onclick: () => { closeSheet(); doInstall(); } }, icon("install"),
				h("div", { class: "grow" }, "Install on this phone", h("small", {}, ios ? "Share → Add to Home Screen" : "Full-screen, one tap away, works offline")), icon("chev")),
			h("button", { class: "set", onclick: () => { closeSheet(); S.charts = {}; S.news = null; refresh(true, true); } }, icon("refresh"),
				h("div", { class: "grow" }, "Refresh now", h("small", {}, window.QD_PUBLISHED ? "The desk republishes every ~6 min while it runs" : "Updates every 15 s by itself")), icon("chev")),
			h("button", { class: "set", onclick: () => openSheet("How to read this app", glossaryList()) }, icon("book"),
				h("div", { class: "grow" }, "How to read this app", h("small", {}, "Bias, conviction, premium, R, grades, levels…")), icon("chev"))),
		h("div", { class: "note" }, h("b", {}, `QuantDesk ${VERSION}. `),
			"An automated intraday options desk on NIFTY and BANKNIFTY, paper trading against live prices, option chains and news. Not investment advice."));
}

// ==================================================================================================================
// refresh loop, pull to refresh, boot
// ==================================================================================================================
let tick = 0, busy = false, queued = null;
async function refresh(full, force) {
	if (document.hidden) return;
	// one refresh at a time; a tab switch during a slow one runs right after it instead of being dropped
	if (busy) { queued = { full: !!(full || (queued && queued.full)), force: !!(force || (queued && queued.force)) }; return; }
	busy = true;
	try {
		if (force && window.QD_PUBLISHED && window.qdReload) await window.qdReload();
		await loadState();
		const t = S.tab;
		if (t === "desk") await renderDesk();
		else if (t === "chart") await renderChart();
		else if (t === "trades") await renderTrades(full);
		else if (t === "brain") renderBrain();
		else if (t === "feed") await renderFeed(full || tick % 4 === 0);
	} catch (e) {
		console.error(e);
	} finally {
		busy = false;
		if (queued) { const q = queued; queued = null; refresh(q.full, q.force); }
	}
}
function pullToRefresh() {
	const ptr = $("#ptr");
	let y0 = null, dy = 0;
	addEventListener("touchstart", (e) => {
		y0 = scrollY <= 0 && !e.target.closest(".chartbox,.sheet,.ticker,.scroll") ? e.touches[0].clientY : null;
		dy = 0;
	}, { passive: true });
	addEventListener("touchmove", (e) => {
		if (y0 == null) return;
		dy = Math.max(0, e.touches[0].clientY - y0);
		const p = Math.min(1, dy / 80);
		ptr.style.opacity = String(p);
		ptr.style.transform = `translateY(${-20 + p * 34}px) rotate(${p * 270}deg)`;
	}, { passive: true });
	addEventListener("touchend", async () => {
		if (y0 == null) return;
		y0 = null;
		if (dy > 80) {
			ptr.classList.add("spin");
			S.charts = {}; S.news = null;
			await refresh(true, true);
			ptr.classList.remove("spin");
		}
		ptr.style.opacity = "0";
		ptr.style.transform = "";
	});
}
async function boot() {
	applyTheme(themeMode());
	matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme(themeMode()));
	S.sym = store("qd.sym") || S.sym;
	S.interval = store("qd.iv") || S.interval;
	buildTabs();
	document.addEventListener("click", (e) => { const g = e.target.closest("[data-go]"); if (g) go(g.dataset.go); });
	document.querySelectorAll("[data-info]").forEach((b) => b.addEventListener("click", () => showGloss(b.dataset.info)));
	$("#btn-settings").addEventListener("click", openSettings);
	$("#status").addEventListener("click", () => { S.charts = {}; S.news = null; refresh(true, true); toast("Refreshed"); });
	$("#c-fs").addEventListener("click", () => fullscreen(true));
	$("#c-fs-x").addEventListener("click", () => fullscreen(false));
	$("#sheet").addEventListener("click", (e) => { if (e.target.id === "sheet") closeSheet(); });
	document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeSheet(); fullscreen(false); } });
	addEventListener("hashchange", () => { const [t, s] = location.hash.slice(1).split("/"); show(t || "desk", s); });
	addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); S.installEvt = e; if (S.tab === "desk") installCard(); });
	addEventListener("appinstalled", () => { store("qd.installed", "1"); document.querySelectorAll(".install").forEach((x) => x.remove()); });
	try { S.accounts = await api("/api/i/accounts"); } catch (e) { S.accounts = []; }
	if (!S.accounts.length) S.accounts = [{ id: "live", label: "Live paper" }];
	const saved = store("qd.account");
	S.account = S.accounts.some((a) => a.id === saved) ? saved : S.accounts[0].id;
	const [t, s] = location.hash.slice(1).split("/");
	show(t || store("qd.tab") || "desk", s);
	pullToRefresh();
	// the chart draws text on a canvas: repaint once the app's own fonts have loaded
	if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => restyleCharts());
	setInterval(() => { tick++; refresh(false); }, 15000);
	document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(true); });
	if ("serviceWorker" in navigator && window.isSecureContext && window.QD_PUBLISHED)
		navigator.serviceWorker.register("sw.js").catch(() => { /* offline support is a bonus */ });
}
boot().catch((e) => toast("Failed to start: " + e.message));
