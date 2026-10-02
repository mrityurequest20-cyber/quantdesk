"""Self-contained HTML reports (no external assets): a backtest tearsheet and the daily
desk report. Charts are inline SVG drawn by a small script with a hover layer; every
chart also has a table view. Colours follow a validated categorical palette with
separate light and dark steps."""
from __future__ import annotations

import html
import json
import math

import numpy as np
import pandas as pd

from ..risk import metrics as M

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--pos:#2a78d6;--neg:#e34948;
--mid:#f0efec;--good:#006300;--bad:#d03b3b;--warn:#8a5a00}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;
--s2:#d95926;--pos:#3987e5;--neg:#e66767;--mid:#383835;--good:#0ca30c;--bad:#e66767;--warn:#fab219}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--pos:#3987e5;--neg:#e66767;
--mid:#383835;--good:#0ca30c;--bad:#e66767;--warn:#fab219}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 10px}
.sub{color:var(--ink2);margin:0 0 16px}.note{color:var(--ink2);font-size:13px}
.banner{border:1px solid var(--ring);background:var(--surface);border-left:4px solid var(--warn);padding:10px 14px;border-radius:8px;margin:12px 0 4px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:12px 14px}
.tile .l{color:var(--ink2);font-size:12px}.tile .v{font-size:22px;font-weight:600}.tile .d{font-size:12px;color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:14px 16px;margin-top:10px}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink2);font-size:12px;margin-bottom:6px}
.legend i{display:inline-block;width:14px;height:2px;vertical-align:middle;margin-right:6px}
svg{display:block;width:100%;height:auto;overflow:visible}svg text{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.tt{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:8px;
padding:8px 10px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:9;min-width:140px}
.tt b{font-size:13px}.tt .row{display:flex;align-items:center;gap:8px}.tt .row i{width:12px;height:2px;display:inline-block}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;
font-variant-numeric:tabular-nums;white-space:nowrap}th{color:var(--ink2);font-weight:600}td:first-child,th:first-child{text-align:left}
.scroll{overflow-x:auto}.pos{color:var(--good)}.neg{color:var(--bad)}details summary{cursor:pointer;color:var(--ink2);font-size:12px;margin-top:8px}
.heat td{text-align:center;color:var(--ink)}.heat td.hot{color:#fff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .heat td.hot{color:#fff}}
.jr{border-top:1px solid var(--grid);padding:10px 0}.jr .h{font-weight:600}.jr .why{color:var(--ink2);margin:4px 0}
.grade{display:inline-block;min-width:22px;text-align:center;border:1px solid var(--ring);border-radius:6px;padding:0 6px;font-weight:600}
pre{white-space:pre-wrap;background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:12px;font-size:12.5px}
.ok{color:var(--good)}.warnc{color:var(--warn)}.fail{color:var(--bad)}
"""

JS = r"""
const tt=document.createElement('div');tt.className='tt';document.body.appendChild(tt);
function showTT(e,title,rows){tt.textContent='';const t=document.createElement('div');t.style.color='var(--ink2)';t.textContent=title;tt.appendChild(t);
 rows.forEach(r=>{const d=document.createElement('div');d.className='row';if(r.c){const i=document.createElement('i');i.style.background=r.c;d.appendChild(i);}
 const b=document.createElement('b');b.textContent=r.v;d.appendChild(b);if(r.n){const s=document.createElement('span');s.style.color='var(--ink2)';s.textContent=r.n;d.appendChild(s);}tt.appendChild(d);});
 tt.style.display='block';const x=Math.min(e.clientX+14,innerWidth-tt.offsetWidth-8);tt.style.left=x+'px';tt.style.top=(e.clientY+14)+'px';}
function hideTT(){tt.style.display='none';}
const NS='http://www.w3.org/2000/svg';function el(n,a,p){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
function niceTicks(lo,hi,n){const span=hi-lo||1;const step0=span/n;const mag=Math.pow(10,Math.floor(Math.log10(step0)));const f=step0/mag;
 const step=(f<1.5?1:f<3?2:f<7?5:10)*mag;const out=[];for(let v=Math.ceil(lo/step)*step;v<=hi+1e-9;v+=step)out.push(v);return out;}
function lineChart(id,cfg){const root=document.getElementById(id);const W=1100,H=cfg.h||300,L=72,R=90,T=10,B=26;
 const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':cfg.label},root);const xs=cfg.x.map(d=>new Date(d).getTime());
 let lo=Infinity,hi=-Infinity;cfg.series.forEach(s=>s.v.forEach(v=>{if(v!=null){lo=Math.min(lo,v);hi=Math.max(hi,v);}}));
 if(cfg.zero){hi=Math.max(hi,0);lo=Math.min(lo,0);}const pad=(hi-lo)*0.05;lo-=cfg.zero&&lo===0?0:pad;hi+=cfg.zero&&hi===0?0:pad;
 const X=t=>L+(t-xs[0])/(xs[xs.length-1]-xs[0]||1)*(W-L-R),Y=v=>T+(hi-v)/(hi-lo||1)*(H-T-B);
 niceTicks(lo,hi,5).forEach(v=>{el('line',{x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:'var(--grid)','stroke-width':1},svg);
  const t=el('text',{x:L-8,y:Y(v)+4,'text-anchor':'end'},svg);t.textContent=cfg.fmt(v);});
 const y0=new Date(xs[0]).getFullYear(),y1=new Date(xs[xs.length-1]).getFullYear();const every=Math.max(1,Math.ceil((y1-y0+1)/10));
 for(let y=y0+1;y<=y1;y+=every){const t=new Date(y,0,1).getTime();if(t<xs[0])continue;const x=X(t);
  const tx=el('text',{x:x,y:H-6,'text-anchor':'middle'},svg);tx.textContent=y;}
 el('line',{x1:L,x2:W-R,y1:Y(cfg.zero?0:lo),y2:Y(cfg.zero?0:lo),stroke:'var(--axis)','stroke-width':1},svg);
 cfg.series.forEach(s=>{let d='';s.v.forEach((v,i)=>{if(v==null)return;d+=(d?'L':'M')+X(xs[i]).toFixed(1)+' '+Y(v).toFixed(1);});
  if(s.area){el('path',{d:d+`L${X(xs[xs.length-1])} ${Y(0)}L${X(xs[0])} ${Y(0)}Z`,fill:s.c,'fill-opacity':0.1,stroke:'none'},svg);}
  el('path',{d:d,fill:'none',stroke:s.c,'stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round'},svg);
  const lv=s.v[s.v.length-1];if(lv!=null&&cfg.endLabels){el('circle',{cx:X(xs[xs.length-1]),cy:Y(lv),r:4,fill:s.c,stroke:'var(--surface)','stroke-width':2},svg);
   const t=el('text',{x:X(xs[xs.length-1])+8,y:Y(lv)+4},svg);t.style.fill='var(--ink2)';t.textContent=s.n+' '+cfg.fmt(lv);}});
 const cross=el('line',{y1:T,y2:H-B,stroke:'var(--axis)','stroke-width':1,visibility:'hidden'},svg);
 const dots=cfg.series.map(s=>el('circle',{r:4,fill:s.c,stroke:'var(--surface)','stroke-width':2,visibility:'hidden'},svg));
 const hit=el('rect',{x:L,y:T,width:W-L-R,height:H-T-B,fill:'transparent'},svg);
 hit.addEventListener('pointermove',e=>{const r=svg.getBoundingClientRect();const px=(e.clientX-r.left)/r.width*W;
  const t=xs[0]+(px-L)/(W-L-R)*(xs[xs.length-1]-xs[0]);let i=0,b=Infinity;for(let k=0;k<xs.length;k++){const dd=Math.abs(xs[k]-t);if(dd<b){b=dd;i=k;}}
  cross.setAttribute('x1',X(xs[i]));cross.setAttribute('x2',X(xs[i]));cross.setAttribute('visibility','visible');
  cfg.series.forEach((s,k)=>{const v=s.v[i];if(v==null){dots[k].setAttribute('visibility','hidden');return;}dots[k].setAttribute('cx',X(xs[i]));dots[k].setAttribute('cy',Y(v));dots[k].setAttribute('visibility','visible');});
  showTT(e,cfg.x[i],cfg.series.map(s=>({c:s.c,v:s.v[i]==null?'—':cfg.fmt(s.v[i]),n:s.n})));});
 hit.addEventListener('pointerleave',()=>{hideTT();cross.setAttribute('visibility','hidden');dots.forEach(d=>d.setAttribute('visibility','hidden'));});}
function barChart(id,cfg){const root=document.getElementById(id);const n=cfg.labels.length,rowH=30,W=1100,L=150,R=110,H=n*rowH+24;
 const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':cfg.label},root);
 const lo=Math.min(0,...cfg.values),hi=Math.max(0,...cfg.values);const gut=lo<0?90:0;const X=v=>L+gut+(v-lo)/(hi-lo||1)*(W-L-R-gut);
 el('line',{x1:X(0),x2:X(0),y1:0,y2:n*rowH,stroke:'var(--axis)','stroke-width':1},svg);
 cfg.labels.forEach((lab,i)=>{const v=cfg.values[i],y=i*rowH+(rowH-18)/2,x0=X(Math.min(0,v)),w=Math.max(1,Math.abs(X(v)-X(0)));
  const t=el('text',{x:L-10,y:y+13,'text-anchor':'end'},svg);t.style.fill='var(--ink2)';t.textContent=lab;
  const r=el('rect',{x:x0,y:y,width:w,height:18,rx:4,fill:'var(--s1)'},svg);
  const vt=el('text',{x:v>=0?X(v)+6:X(v)-6,y:y+13,'text-anchor':v>=0?'start':'end'},svg);vt.style.fill='var(--ink2)';vt.textContent=cfg.fmt(v);
  const hit=el('rect',{x:L,y:i*rowH,width:W-L-R,height:rowH,fill:'transparent'},svg);
  hit.addEventListener('pointermove',e=>{r.setAttribute('fill-opacity',0.8);showTT(e,lab,[{c:'var(--s1)',v:cfg.fmt(v),n:cfg.unit||''}].concat((cfg.extra||[])[i]||[]));});
  hit.addEventListener('pointerleave',()=>{r.setAttribute('fill-opacity',1);hideTT();});});}
function histChart(id,cfg){const root=document.getElementById(id);const W=1100,H=220,L=50,R=20,T=10,B=26,n=cfg.counts.length;
 const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':cfg.label},root);const mx=Math.max(...cfg.counts);
 const bw=(W-L-R)/n;niceTicks(0,mx,4).forEach(v=>{const y=T+(1-v/mx)*(H-T-B);el('line',{x1:L,x2:W-R,y1:y,y2:y,stroke:'var(--grid)'},svg);
  const t=el('text',{x:L-6,y:y+4,'text-anchor':'end'},svg);t.textContent=v;});
 cfg.counts.forEach((c,i)=>{const h=c/mx*(H-T-B),x=L+i*bw+1,w=Math.min(24,bw-2);const r=el('rect',{x:x+(bw-2-w)/2,y:H-B-h,width:w,height:Math.max(h,0),rx:Math.min(4,w/2),fill:'var(--s1)'},svg);
  const hit=el('rect',{x:L+i*bw,y:T,width:bw,height:H-T-B,fill:'transparent'},svg);
  hit.addEventListener('pointermove',e=>{r.setAttribute('fill-opacity',0.8);showTT(e,cfg.edges[i],[{c:'var(--s1)',v:String(c),n:'simulations'}]);});
  hit.addEventListener('pointerleave',()=>{r.setAttribute('fill-opacity',1);hideTT();});
  if(i%Math.ceil(n/8)===0){const t=el('text',{x:L+i*bw+bw/2,y:H-8,'text-anchor':'middle'},svg);t.textContent=cfg.ticks[i];}});}
document.querySelectorAll('.heat td[data-v]').forEach(td=>{td.addEventListener('pointermove',e=>showTT(e,td.dataset.k,[{v:td.dataset.v}]));td.addEventListener('pointerleave',hideTT);});
"""


def _e(x) -> str:
    return html.escape(str(x))


def _pct(x, d=1) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100:+.{d}f}%"


def _num(x, d=2) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:,.{d}f}"


def _inr(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    a = abs(x)
    s = f"₹{a / 1e7:,.2f} Cr" if a >= 1e7 else f"₹{a / 1e5:,.2f} L" if a >= 1e5 else f"₹{a:,.0f}"
    return ("-" if x < 0 else "") + s


def _tile(label, value, delta="") -> str:
    return f'<div class="tile"><div class="l">{_e(label)}</div><div class="v">{_e(value)}</div><div class="d">{_e(delta)}</div></div>'


def _table(df: pd.DataFrame, fmts: dict | None = None, index=True) -> str:
    fmts = fmts or {}
    cols = list(df.columns)
    head = ("<th></th>" if index else "") + "".join(f"<th>{_e(c)}</th>" for c in cols)
    rows = []
    for idx, r in df.iterrows():
        cells = [f"<td>{_e(idx)}</td>"] if index else []
        for c in cols:
            v = r[c]
            f = fmts.get(c)
            s = f(v) if f else (_num(v) if isinstance(v, (float, np.floating)) else _e(v))
            cls = ""
            if c in fmts and isinstance(v, (int, float, np.floating)) and v == v and fmts.get(c) in (_pct, _inr):
                cls = ' class="pos"' if v > 0 else ' class="neg"' if v < 0 else ""
            cells.append(f"<td{cls}>{s}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _heatmap(tab: pd.DataFrame) -> str:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    vals = tab.drop(columns=["year_total"]).to_numpy(dtype=float)
    scale = np.nanpercentile(np.abs(vals), 95) if np.isfinite(vals).any() else 0.05
    scale = max(scale, 1e-4)
    head = "<th>Year</th>" + "".join(f"<th>{m}</th>" for m in months) + "<th>Year</th>"
    rows = []
    for y, r in tab.iterrows():
        cells = [f"<td>{y}</td>"]
        for m in range(1, 13):
            v = r.get(m, np.nan)
            if v != v:
                cells.append("<td></td>")
                continue
            k = min(1.0, abs(v) / scale)
            pole = "var(--pos)" if v >= 0 else "var(--neg)"
            hot = " hot" if k > 0.6 else ""
            cells.append(f'<td class="{hot.strip()}" style="background:color-mix(in oklab,{pole} {k * 85:.0f}%,var(--mid))" '
                         f'data-k="{months[m - 1]} {y}" data-v="{v * 100:+.2f}%">{v * 100:+.1f}</td>')
        tv = r["year_total"]
        cells.append(f'<td class="{"pos" if tv > 0 else "neg"}"><b>{tv * 100:+.1f}%</b></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="scroll"><table class="heat"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            '<p class="note">Monthly returns, %. Blue up, red down, gray ≈ flat; intensity capped at the 95th percentile move.</p>')


def _downsample(s: pd.Series, n: int = 700) -> pd.Series:
    if len(s) <= n:
        return s
    step = int(np.ceil(len(s) / n))
    return s.iloc[::step].combine_first(s.iloc[[-1]])


def _page(title: str, body: str, data: dict, script: str) -> str:
    payload = json.dumps(data, default=float).replace("</", "<\\/")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_e(title)}</title><style>{CSS}</style></head><body><main>{body}</main>"
            f"<script id='data' type='application/json'>{payload}</script>"
            f"<script>{JS}\nconst D=JSON.parse(document.getElementById('data').textContent);\n{script}</script></body></html>")


def backtest_report(res, title: str = "QuantDesk backtest", synthetic: bool = False, mc: dict | None = None,
                    wf=None) -> str:
    s, t = res.stats, res.trade_stats
    eq = res.equity["equity"]
    eqd = _downsample(eq)
    bench = res.benchmark.reindex(eqd.index) if res.benchmark is not None else None
    dd = _downsample(M.drawdown_series(eq))
    ps = res.per_strategy.copy()
    body = [f"<h1>{_e(title)}</h1><p class='sub'>{_e(s.get('start'))} → {_e(s.get('end'))} · starting capital "
            f"{_inr(eq.iloc[0])} · strategies: {_e(', '.join(res.meta.get('strategies', [])))}</p>"]
    if synthetic:
        body.append("<div class='banner'><b>Synthetic data.</b> This run uses the built-in market simulator, so it "
                    "demonstrates the machinery, not an edge. Results on real NSE data will differ.</div>")
    body.append("<div class='tiles'>" + "".join([
        _tile("Final equity", _inr(eq.iloc[-1]), f"total {_pct(s['total_return'])}"),
        _tile("CAGR", _pct(s["cagr"]), f"benchmark {_pct(M.returns_stats(res.benchmark)['cagr'])}" if res.benchmark is not None else ""),
        _tile("Sharpe (vs risk-free)", _num(s["sharpe"]), f"Sortino {_num(s['sortino'])}"),
        _tile("Max drawdown", _pct(s["max_drawdown"]), f"{s['max_dd_days']} days under water"),
        _tile("Probabilistic Sharpe", f"{s['psr_vs_0'] * 100:.0f}%", "P(true Sharpe > 0)"),
        _tile("Trades", f"{t.get('trades', 0):,}", f"win {t.get('win_rate', 0) * 100:.0f}% · PF {_num(t.get('profit_factor', float('nan')))}"),
        _tile("Avg trade", f"{t.get('avg_r', 0):+.2f}R", f"expectancy {_inr(t.get('expectancy', 0))}"),
        _tile("Costs paid", _inr(res.fees.get("total", 0)), f"STT {_inr(res.fees.get('stt', 0))}"),
    ]) + "</div>")
    body.append("<h2>Equity vs buy-and-hold</h2><div class='card'><div class='legend'><span><i style='background:var(--s1)'></i>Portfolio</span>"
                + ("<span><i style='background:var(--s2)'></i>NIFTY buy &amp; hold (same capital)</span>" if bench is not None else "")
                + "</div><div id='eq'></div></div>")
    body.append("<h2>Drawdown</h2><div class='card'><div id='dd'></div></div>")
    body.append("<h2>Monthly returns</h2><div class='card'>" + _heatmap(M.monthly_returns(eq)) + "</div>")
    if not ps.empty:
        ps = ps.sort_values("total_pnl", ascending=False)
        body.append("<h2>P&amp;L by strategy</h2><div class='card'><div id='strat'></div><details><summary>Table view</summary>"
                    + _table(ps[["trades", "win_rate", "profit_factor", "avg_r", "total_pnl", "total_fees", "avg_bars", "max_consec_losses"]]
                             .rename(columns={"win_rate": "win %", "profit_factor": "PF", "avg_r": "avg R", "total_pnl": "P&L",
                                              "total_fees": "fees", "avg_bars": "avg bars", "max_consec_losses": "max losing streak"}),
                             {"trades": lambda v: f"{int(v)}", "win %": lambda v: f"{v * 100:.0f}%", "PF": lambda v: _num(v),
                              "avg R": lambda v: f"{v:+.2f}", "P&L": _inr, "fees": _inr, "avg bars": lambda v: f"{v:.1f}",
                              "max losing streak": lambda v: f"{int(v)}"}) + "</details></div>")
    rs = pd.DataFrame({"metric": ["Annual volatility", "Sortino", "Calmar", "VaR 95% (1d)", "CVaR 95% (1d)", "Cornish-Fisher VaR 95%",
                                  "Skew", "Excess kurtosis", "Best day", "Worst day", "% up days", "Tail ratio"],
                       "value": [_pct(s["ann_vol"]), _num(s["sortino"]), _num(s["calmar"]), _pct(-s["var95_1d"], 2), _pct(-s["cvar95_1d"], 2),
                                 _pct(-s["cf_var95_1d"], 2), _num(s["skew"]), _num(s["kurtosis"]), _pct(s["best_day"], 2),
                                 _pct(s["worst_day"], 2), f"{s['pct_up_days'] * 100:.0f}%", _num(s["tail_ratio"])]}).set_index("metric")
    body.append("<h2>Risk statistics</h2><div class='card'>" + _table(rs) + "</div>")
    if mc:
        body.append(f"<h2>Monte Carlo: trade-order bootstrap</h2><div class='card'><p class='note'>{len(mc['dd_dist']):,} resampled "
                    f"sequences of the same trades. Median max drawdown {_pct(mc['maxdd_p50'])}, 95th percentile "
                    f"{_pct(mc['maxdd_p95'])}, worst {_pct(mc['maxdd_worst'])}; P(losing money) {mc['p_loss'] * 100:.1f}%, "
                    f"P(drawdown &gt; 20%) {mc['p_dd_gt_20'] * 100:.1f}%.</p><div id='mc'></div></div>")
    if wf is not None and not wf.folds.empty:
        body.append("<h2>Walk-forward</h2><div class='card'>" + _table(wf.folds.set_index("test")) +
                    f"<p class='note'>Stitched out-of-sample Sharpe {_num(wf.oos_stats.get('sharpe', float('nan')))}, "
                    f"Deflated Sharpe (vs {wf.n_trials} variants) {wf.dsr * 100:.0f}%.</p></div>")
    if not res.trades.empty:
        tr = res.trades
        ex = tr.groupby("exit_reason").agg(trades=("pnl", "size"), pnl=("pnl", "sum"), avg_r=("r_multiple", "mean")).sort_values("trades", ascending=False)
        gr = tr["grade"].value_counts().sort_index()
        body.append("<h2>How trades ended</h2><div class='card'>" + _table(ex, {"trades": lambda v: f"{int(v)}", "pnl": _inr,
                                                                              "avg_r": lambda v: f"{v:+.2f}"}) +
                    "<p class='note'>Journal grades (process 60%, outcome 40%): " + ", ".join(f"{k}: {v}" for k, v in gr.items()) + "</p></div>")
        body.append("<h2>From the journal</h2><div class='card'>" + _journal_samples(tr) + "</div>")
        last = tr.tail(60).iloc[::-1][["opened_at", "closed_at", "strategy", "symbol", "exit_reason", "pnl", "r_multiple", "grade"]].copy()
        last["opened_at"], last["closed_at"] = last["opened_at"].str[:10], last["closed_at"].str[:10]
        body.append("<h2>Latest trades</h2><div class='card'>" + _table(last.set_index("opened_at"),
                    {"pnl": _inr, "r_multiple": lambda v: f"{v:+.2f}"}) + "</div>")
    fees = pd.DataFrame({"INR": {k: v for k, v in res.fees.items()}}).rename(index=str.upper)
    body.append("<h2>Transaction costs</h2><div class='card'>" + _table(fees, {"INR": _inr}) +
                "<p class='note'>Statutory charges per the config (STT, exchange, SEBI, stamp duty, GST) plus brokerage. "
                "Slippage is inside fill prices, not listed here.</p></div>")
    data = {"x": [str(d.date()) for d in eqd.index], "eq": [float(v) for v in eqd.values],
            "bench": [float(v) if v == v else None for v in bench.values] if bench is not None else None,
            "ddx": [str(d.date()) for d in dd.index], "dd": [float(v) for v in dd.values],
            "strat": {"labels": list(ps.index), "values": [float(v) for v in ps["total_pnl"]]} if not ps.empty else None}
    script = ("const fmtINR=v=>{const a=Math.abs(v);return (v<0?'-':'')+(a>=1e7?'₹'+(a/1e7).toFixed(2)+' Cr':a>=1e5?'₹'+(a/1e5).toFixed(1)+' L':'₹'+Math.round(a).toLocaleString('en-IN'));};"
              "lineChart('eq',{label:'Equity curve',x:D.x,fmt:fmtINR,endLabels:true,series:[{n:'Portfolio',c:'var(--s1)',v:D.eq}]"
              ".concat(D.bench?[{n:'NIFTY',c:'var(--s2)',v:D.bench}]:[])});"
              "lineChart('dd',{label:'Drawdown',x:D.ddx,h:200,zero:true,fmt:v=>(v*100).toFixed(1)+'%',series:[{n:'Drawdown',c:'var(--s1)',v:D.dd,area:true}]});"
              "if(D.strat)barChart('strat',{label:'P&L by strategy',labels:D.strat.labels,values:D.strat.values,fmt:fmtINR,unit:'net P&L'});")
    if mc:
        h, edges = np.histogram(mc["dd_dist"] * 100, bins=30)
        data["mc"] = {"counts": [int(c) for c in h], "edges": [f"{edges[i]:.1f}% to {edges[i + 1]:.1f}%" for i in range(len(h))],
                      "ticks": [f"{edges[i]:.0f}%" for i in range(len(h))]}
        script += "histChart('mc',{label:'Max drawdown distribution',counts:D.mc.counts,edges:D.mc.edges,ticks:D.mc.ticks});"
    return _page(title, "".join(body), data, script)


def _journal_samples(tr: pd.DataFrame, n: int = 6) -> str:
    picks = []
    for s, g in tr.groupby("strategy"):
        g = g.dropna(subset=["review"])
        if len(g):
            picks.append(g.loc[g["pnl"].abs().idxmax()])
    out = []
    for r in picks[:n]:
        lessons = json.loads(r["lessons"]) if isinstance(r["lessons"], str) else []
        out.append(f"<div class='jr'><div class='h'><span class='grade'>{_e(r['grade'])}</span> {_e(r['strategy'])} · {_e(r['symbol'])} · "
                   f"{_e(str(r['opened_at'])[:10])} → {_e(str(r['closed_at'])[:10])} · {_inr(r['pnl'])} ({r['r_multiple']:+.2f}R)</div>"
                   f"<div class='why'><b>Plan:</b> {_e(r['rationale'])}</div><div class='why'><b>Review:</b> {_e(r['review'])}</div>"
                   + ("<div class='why'><b>Lessons:</b> " + _e(" ".join(lessons)) + "</div>" if lessons else "") + "</div>")
    return "".join(out)


def desk_report(title: str, sections: list[tuple[str, str]], checks=None) -> str:
    """Daily desk report: market analysis narrative, options view, checks, journal review."""
    body = [f"<h1>{_e(title)}</h1>"]
    if checks:
        cls = {"PASS": "ok", "WARN": "warnc", "FAIL": "fail", "INFO": ""}
        rows = "".join(f"<tr><td>{_e(c.name)}</td><td class='{cls.get(c.status, '')}'>{_e(c.status)}</td>"
                       f"<td style='text-align:left;white-space:normal'>{_e(c.detail)}</td></tr>" for c in checks)
        body.append(f"<h2>Routine checks</h2><div class='card'><table><thead><tr><th>Check</th><th>Status</th>"
                    f"<th style='text-align:left'>Detail</th></tr></thead><tbody>{rows}</tbody></table></div>")
    for h, txt in sections:
        body.append(f"<h2>{_e(h)}</h2><pre>{_e(txt)}</pre>")
    return _page(title, "".join(body), {}, "")
