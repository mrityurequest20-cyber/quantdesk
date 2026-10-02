/* QuantDesk → GoCharting SDK datafeed.
 *
 * Implements the datafeed contract used by @gocharting/chart-sdk (see GoCharting's
 * reference `chart-datafeed.ts`): getBars returns UDF arrays with unix-second times,
 * resolveSymbol supplies `segment` and `exchange_info` (the SDK rebuilds the
 * EXCHANGE:SEGMENT:SYMBOL key from them), and subscribeTicks pushes "trade" ticks.
 * NSE daily bars come from the local QuantDesk API, so the feed is end-of-day: ticks
 * are polled and only emitted when the last price actually changes.
 */
(function (root) {
	"use strict";
	const INTERVALS = ["1D"];

	function toSec(v) {
		if (v == null) return null;
		if (v instanceof Date) return Math.floor(v.getTime() / 1000);
		const n = Number(v);
		if (!Number.isFinite(n)) return null;
		return n >= 1e11 ? Math.floor(n / 1000) : n; // ms since 1973 vs seconds before year 5138
	}

	/* GoCharting passes resolutions as strings ("5m") or objects ({units, scale}); normalise. */
	function toInterval(res) {
		if (res == null) return "1D";
		if (typeof res === "string") return res;
		if (res.type) return res.type;
		if (res.baseType) return res.baseType;
		if (res.scale === "minutes") return (res.units || 1) + "m";
		if (res.scale === "hours") return (res.units || 1) * 60 + "m";
		return "1D";
	}

	/* opts.udf(symbol, interval, from, to, countBack) -> URL switches getBars to intraday bars
	   (the intraday desk's recorded sessions); opts.resolutions lists what the chart may offer. */
	function createQuantDeskDatafeed(apiBase, opts) {
		const base = apiBase || "";
		const o = opts || {};
		const resolutions = o.resolutions || INTERVALS;
		const fetchFn = o.fetch || root.fetch.bind(root);
		const timers = new Map();
		let symbolsPromise = null;

		async function getJSON(path) {
			const r = await fetchFn(base + path);
			if (!r.ok) throw new Error(path + " → HTTP " + r.status);
			return r.json();
		}
		const symbols = () => (symbolsPromise = symbolsPromise || getJSON("/api/symbols"));
		const keyOf = (x) => (typeof x === "string" ? x : (x && (x.full_name || x.key || x.symbol)) || "");
		const bare = (x) => keyOf(x).split(":").pop().toUpperCase();

		function symbolInfo(s) {
			const index = s.segment === "INDEX";
			return {
				exchange: "NSE",
				segment: s.segment,
				symbol: s.symbol,
				name: s.description,
				ticker: s.symbol,
				full_name: s.key,
				description: s.description + " · NSE " + s.segment.toLowerCase(),
				type: index ? "index" : "stock",
				asset_type: index ? "INDEX" : "EQUITY",
				session: "0915-1530",
				timezone: "Asia/Kolkata",
				has_intraday: !!o.udf,
				has_daily: true,
				supported_resolutions: resolutions,
				tick_size: 0.05,
				display_tick_size: 0.05,
				max_tick_precision: 2,
				data_status: o.udf ? "streaming" : "endofday",
				delay_seconds: 0,
				tradeable: !!s.tradeable,
				quote_currency: "INR",
				lot_size: s.lot_size,
				exchange_info: {
					name: "nse",
					code: "NSE",
					zone: "Asia/Kolkata",
					hours: [0, 1, 2, 3, 4, 5, 6].map((d) => ({ open: d >= 1 && d <= 5 })),
					valid_intervals: resolutions,
				},
			};
		}

		return {
			async getBars(info, resolution, periodParams) {
				const p = periodParams || {};
				if (o.udf) {
					try {
						return await getJSON(o.udf(bare(info), toInterval(resolution), toSec(p.from), toSec(p.to), p.countBack || p.rows));
					} catch (e) {
						return { s: "error", errmsg: String((e && e.message) || e) };
					}
				}
				const q = new URLSearchParams({ symbol: bare(info) });
				const from = toSec(p.from), to = toSec(p.to);
				if (from != null) q.set("from", String(from));
				if (to != null) q.set("to", String(to));
				const cb = p.countBack || p.rows;
				if (cb) q.set("countback", String(cb));
				try {
					return await getJSON("/api/history?" + q.toString());
				} catch (e) {
					return { s: "error", errmsg: String((e && e.message) || e) };
				}
			},

			resolveSymbol(name, onResolve, onError) {
				symbols()
					.then((list) => {
						const s = list.find((x) => x.key === keyOf(name) || x.symbol === bare(name));
						if (!s) return onError && onError("QuantDesk has no symbol " + keyOf(name));
						onResolve(symbolInfo(s));
					})
					.catch((e) => onError && onError(String(e)));
			},

			searchSymbols(input, _exchange, _type, onResult) {
				symbols()
					.then((list) => {
						const q = String(input || "").toUpperCase();
						onResult({
							searchInProgress: false,
							items: list
								.filter((s) => s.symbol.includes(q))
								.map((s) => ({
									symbol: s.symbol, key: s.key, full_name: s.key, description: s.description,
									exchange: "NSE", segment: s.segment, ticker: s.symbol,
									type: s.segment === "INDEX" ? "index" : "stock",
								})),
						});
					})
					.catch(() => onResult({ searchInProgress: false, items: [] }));
			},

			subscribeTicks(info, _resolution, onTick, uid) {
				const sym = bare(info);
				let lastPrice = null;
				const poll = async () => {
					try {
						const l = await getJSON("/api/last?symbol=" + encodeURIComponent(sym));
						if (l.price === lastPrice) return;
						lastPrice = l.price;
						onTick({
							type: "trade", productId: keyOf(info), symbol: sym, exchange: "NSE",
							segment: info && info.segment, timeStamp: new Date(l.time * 1000),
							tradeID: sym + "-" + l.time, price: l.price, quantity: 0, amount: 0, side: "BUY",
						});
					} catch (e) {
						/* transient: next poll retries */
					}
				};
				timers.set(uid, setInterval(poll, o.pollMs || 15000));
			},

			unsubscribeTicks(uid) {
				clearInterval(timers.get(uid));
				timers.delete(uid);
			},

			destroy() {
				for (const t of timers.values()) clearInterval(t);
				timers.clear();
				symbolsPromise = null;
			},
		};
	}

	if (typeof module !== "undefined" && module.exports) module.exports = { createQuantDeskDatafeed, toSec, toInterval };
	else root.createQuantDeskDatafeed = createQuantDeskDatafeed;
})(typeof window !== "undefined" ? window : globalThis);
