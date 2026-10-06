#!/usr/bin/env python3
"""Nightly market dashboard: 200-day checks for the S&P 500 + your watchlist.

Usage:
    python build_dashboard.py            # live data (needs internet)
    python build_dashboard.py --demo     # synthetic data, to test the page
Output: docs/index.html  (+ data/history.csv, data/state.json)
"""
import argparse, io, json, os, sys, html
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
ACCOUNT_SIZE = float(os.environ.get("ACCOUNT_SIZE", 15700))  # Roth IRA, approx
RISK_PCT = 0.02                    # rule 1: max risk per trade
DIP_LOW, DIP_HIGH = -0.15, -0.08   # trend-dip window (drop from 52-week high)
BREADTH_BULL, BREADTH_BEAR = 50.0, 40.0
CASH_RESERVE = {"bull": 15, "caution": 30, "bear": 40}   # bear value = draft
MIN_BARS = 210
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs")
DATA = os.path.join(HERE, "data")


# ---------------------------------------------------------------- universe
def load_watchlist():
    df = pd.read_csv(os.path.join(HERE, "watchlist.csv")).fillna("")
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["stop_pct"] = pd.to_numeric(df["stop_pct"], errors="coerce").fillna(18)
    return df


def load_sp500():
    """S&P 500 members from Wikipedia. Returns (list, warning)."""
    try:
        import requests
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (dashboard script)"}, timeout=30)
        r.raise_for_status()
        t = pd.read_html(io.StringIO(r.text))[0]
        syms = t["Symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
        syms = sorted(set(syms))
        if len(syms) < 400:
            raise ValueError(f"only {len(syms)} symbols parsed")
        names = dict(zip(syms, t["Security"].astype(str)))
        return syms, names, None
    except Exception as e:  # noqa
        return [], {}, f"S&P 500 list could not be loaded ({e}); showing watchlist only."


# ---------------------------------------------------------------- prices
def fetch_prices(symbols):
    import yfinance as yf
    frames, failed = [], []
    syms = list(dict.fromkeys(symbols))
    for i in range(0, len(syms), 100):
        chunk = syms[i:i + 100]
        try:
            d = yf.download(chunk, period="16mo", interval="1d", auto_adjust=True,
                            group_by="column", threads=True, progress=False)
            close = d["Close"] if "Close" in d else d
            if isinstance(close, pd.Series):
                close = close.to_frame(chunk[0])
            frames.append(close)
        except Exception as e:  # noqa
            failed += chunk
            print("chunk failed:", e, file=sys.stderr)
    if not frames:
        raise RuntimeError("no price data downloaded")
    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()]
    return px.sort_index(), failed


def demo_prices(symbols, n=330):
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    out = {}
    for s in symbols:
        drift = rng.normal(0.0004, 0.0004)
        vol = rng.uniform(0.01, 0.028)
        r = rng.normal(drift, vol, n)
        # some names get a recent pullback so every setup shows up in the demo
        if rng.random() < 0.3:
            r[-25:] += rng.normal(-0.004, 0.003, 25)
        out[s] = 50 * rng.uniform(0.5, 6) * np.exp(np.cumsum(r))
    return pd.DataFrame(out, index=idx)


# ---------------------------------------------------------------- math
def floor_touches(c126):
    """How many separate times the 6-month floor held (within 3%)."""
    floor = c126.min()
    near = (c126 <= floor * 1.03).to_numpy()
    touches, last = 0, -99
    for i, v in enumerate(near):
        if v and i - last >= 10:
            touches += 1
        if v:
            last = i
    return floor, touches


def analyze(s):
    s = s.dropna()
    if len(s) < MIN_BARS:
        return None
    last = float(s.iloc[-1])
    sma200 = s.rolling(200).mean()
    sma50 = s.rolling(50).mean()
    hi = float(s.iloc[-252:].max())
    lo = float(s.iloc[-252:].min())
    above = last > sma200.iloc[-1]
    above_5ago = s.iloc[-6] > sma200.iloc[-6]
    slope_up = sma200.iloc[-1] > sma200.iloc[-21]
    dd = last / hi - 1
    floor, touches = floor_touches(s.iloc[-126:])
    near_floor = last <= floor * 1.05 and last > floor
    hi126 = float(s.iloc[-126:].max())
    range_ok = touches >= 3 and near_floor and hi126 / floor - 1 >= 0.10
    trend_dip = bool(above and slope_up and DIP_LOW <= dd <= DIP_HIGH)
    return dict(
        last=last, sma200=float(sma200.iloc[-1]), sma50=float(sma50.iloc[-1]),
        pct200=last / sma200.iloc[-1] - 1, dd=dd, hi=hi, lo=lo,
        above200=bool(above), above50=bool(last > sma50.iloc[-1]),
        cross=("up" if above and not above_5ago else "down" if (not above) and above_5ago else ""),
        ret1=last / s.iloc[-2] - 1, ret20=last / s.iloc[-21] - 1,
        ret126=last / s.iloc[-127] - 1,
        trend_dip=trend_dip, range_ok=bool(range_ok), floor=float(floor),
        touches=int(touches), slope_up=bool(slope_up),
        new_hi=bool(last >= hi * 0.999), new_lo=bool(last <= lo * 1.001),
        asof=str(s.index[-1].date()),
    )


def regime(spy, breadth200, tnx_change):
    spy_above = spy["above200"]
    if spy_above and breadth200 >= BREADTH_BULL:
        r, why = "bull", "S&P 500 is above its 200-day average and most stocks agree."
    elif (not spy_above) and breadth200 < BREADTH_BEAR:
        r, why = "bear", "S&P 500 is below its 200-day average and most stocks are too."
    else:
        r = "caution"
        why = ("Mixed signals: the index and the broad market of stocks disagree." if spy_above
               else "S&P 500 is below its 200-day average, but not everything is broken.")
    if r == "bull" and tnx_change is not None and tnx_change >= 0.25:
        r, why = "caution", "Trend is up, but the 10-year yield has jumped over the last month."
    return r, why


# ---------------------------------------------------------------- html
def pct(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x*100:+.{d}f}%" if sign else f"{x*100:.{d}f}%"


def money(x):
    return f"${x:,.2f}" if x < 1000 else f"${x:,.0f}"


def sparkline(vals, w=260, h=48):
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    pts = " ".join(f"{i*(w-4)/(len(vals)-1)+2:.1f},{h-4-(v-lo)/rng*(h-8):.1f}" for i, v in enumerate(vals))
    y40 = h - 4 - (40 - lo) / rng * (h - 8) if lo <= 40 <= hi else None
    ref = f'<line x1="0" x2="{w}" y1="{y40:.1f}" y2="{y40:.1f}" class="ref"/>' if y40 else ""
    return (f'<svg viewBox="0 0 {w} {h}" class="spark" role="img" aria-label="Breadth trend">'
            f'{ref}<polyline points="{pts}" fill="none" class="line"/></svg>')


def setup_label(a):
    tags = []
    if a["trend_dip"]:
        tags.append("Trend dip")
    if a["range_ok"]:
        tags.append("Range floor")
    return " + ".join(tags)


def row_html(sym, a, meta, account=ACCOUNT_SIZE):
    stop = meta.get("stop_pct", 18) / 100
    max_sh = int((account * RISK_PCT) / (a["last"] * stop)) if a["last"] > 0 else 0
    cls = "up" if a["above200"] else "dn"
    state = "Above 200d ▲" if a["above200"] else "Below 200d ▼"
    cross = {"up": " · crossed UP", "down": " · crossed DOWN"}.get(a["cross"], "")
    setup = setup_label(a) or "–"
    zone = f'{money(a["hi"]*(1+DIP_HIGH))} – {money(a["hi"]*(1+DIP_LOW))}'
    return (f'<tr><td class="sym">{html.escape(sym)}</td>'
            f'<td>{html.escape(meta.get("theme",""))}</td>'
            f'<td data-v="{a["last"]:.4f}">{money(a["last"])}</td>'
            f'<td data-v="{a["pct200"]:.4f}" class="{cls}">{pct(a["pct200"])}<span class="sub">{state}{cross}</span></td>'
            f'<td data-v="{a["dd"]:.4f}">{pct(a["dd"])}</td>'
            f'<td data-v="{a["ret20"]:.4f}">{pct(a["ret20"])}</td>'
            f'<td>{setup}</td><td>{zone}</td>'
            f'<td data-v="{max_sh}">{max_sh:,} <span class="sub">@ {int(stop*100)}% stop</span></td></tr>')


TABLE_HEAD = ('<thead><tr><th>Symbol</th><th>Theme</th><th>Price</th><th>vs 200-day</th>'
              '<th>From high</th><th>1-month</th><th>Setup</th><th>Dip buy zone</th>'
              '<th>Max shares (2% risk)</th></tr></thead>')


def build_page(ctx):
    r = ctx["regime"]
    labels = {"bull": "BULL", "caution": "CAUTION", "bear": "BEAR"}
    todo = {
        "bull": "Normal rules. Buy only valid setups; keep at least 15% cash.",
        "caution": "Be selective: smaller sizes, keep about 30% cash, no chasing.",
        "bear": "Protect capital: keep about 40% cash, only the best setups, tight risk.",
    }[r]
    alerts = "".join(f"<li>{html.escape(a)}</li>" for a in ctx["alerts"]) or "<li>No changes since the last run.</li>"
    warn = "".join(f'<p class="warn">{html.escape(w)}</p>' for w in ctx["warnings"])
    demo = '<p class="warn">DEMO DATA – synthetic prices for testing the layout. Not real.</p>' if ctx["demo"] else ""

    def tbl(rows):
        return f'<div class="tw"><table class="sortable">{TABLE_HEAD}<tbody>{"".join(rows)}</tbody></table></div>' if rows else '<p class="muted">None right now.</p>'

    idx_rows = "".join(
        f'<div class="chip"><b>{html.escape(s)}</b><span>{pct(a["pct200"])} vs 200d</span>'
        f'<span class="{"up" if a["above200"] else "dn"}">{"▲ above" if a["above200"] else "▼ below"}</span></div>'
        for s, a in ctx["indexes"])
    tnx = f'{ctx["tnx"]:.2f}%' if ctx["tnx"] is not None else "–"
    tnxc = f'{ctx["tnx_change"]:+.2f} pts / 20d' if ctx["tnx_change"] is not None else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Dashboard</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#14171c;--mut:#5d6675;--line:#e1e5eb;--up:#0b7a43;--dn:#b3261e;--warn:#8a5a00;
--bull:#0b7a43;--caution:#b86e00;--bear:#b3261e;--acc:#2b5cd6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f1217;--card:#181c24;--ink:#e8ebf0;--mut:#98a2b3;--line:#2a303b;--up:#4cc38a;--dn:#ff8a80;--warn:#f5c26b;
--bull:#2fa56b;--caution:#d68a1c;--bear:#e5534b;--acc:#7da2ff}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:16px}}
h1{{font-size:20px;margin:4px 0}}h2{{font-size:16px;margin:28px 0 8px}}
.muted,.sub{{color:var(--mut)}}.sub{{display:block;font-size:12px}}
.banner{{border-radius:12px;padding:16px 18px;color:#fff;background:var(--{r});margin:12px 0}}
.banner .big{{font-size:28px;font-weight:700;letter-spacing:.04em}}.banner p{{margin:4px 0 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}}
.card .k{{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}}
.card .v{{font-size:26px;font-weight:650}}
.chips{{display:flex;flex-wrap:wrap;gap:8px}}.chip{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px;display:flex;flex-direction:column;font-size:13px;min-width:120px}}
.up{{color:var(--up)}}.dn{{color:var(--dn)}}
.tw{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:12px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}}th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap;vertical-align:top}}
th{{font-size:12px;color:var(--mut);cursor:pointer;user-select:none}}tr:last-child td{{border-bottom:0}}.sym{{font-weight:650}}
ul{{margin:6px 0 0 18px;padding:0}}.warn{{background:rgba(245,194,107,.18);border:1px solid var(--warn);color:var(--warn);padding:8px 12px;border-radius:8px}}
.spark{{width:100%;height:48px}}.spark .line{{stroke:var(--acc);stroke-width:2}}.spark .ref{{stroke:var(--mut);stroke-dasharray:3 3;stroke-width:1}}
footer{{margin:28px 0 8px;color:var(--mut);font-size:12.5px}}
</style></head><body><main>
<h1>Market dashboard</h1>
<div class="muted">Prices as of close {ctx["asof"]} · built {ctx["built"]}</div>
{demo}{warn}
<div class="banner"><div class="big">{labels[r]}</div><p>{html.escape(ctx["why"])}</p><p><b>What to do:</b> {todo}</p></div>

<div class="grid">
<div class="card"><div class="k">S&amp;P 500 stocks above 200-day</div><div class="v">{ctx["b200"]:.0f}%</div><div class="muted">Bull ≥ {BREADTH_BULL:.0f}% · Bear &lt; {BREADTH_BEAR:.0f}%</div>{ctx["spark"]}</div>
<div class="card"><div class="k">Above 50-day</div><div class="v">{ctx["b50"]:.0f}%</div><div class="muted">New 52-week highs: {ctx["nh"]} · lows: {ctx["nl"]}</div></div>
<div class="card"><div class="k">10-year yield</div><div class="v">{tnx}</div><div class="muted">{tnxc}</div></div>
<div class="card"><div class="k">Cash to keep</div><div class="v">{CASH_RESERVE[r]}%</div><div class="muted">Of the account, in this regime</div></div>
</div>

<h2>Markets vs their 200-day average</h2><div class="chips">{idx_rows}</div>

<h2>What changed since last run</h2><div class="card"><ul>{alerts}</ul></div>

<h2>Buy-zone shortlist · your watchlist</h2>
<p class="muted">Trend dip = above a rising 200-day and 8–15% off its 52-week high. Range floor = the 6-month floor held at least 3 times and price is within 5% above it.</p>
{tbl(ctx["short_rows"])}

<h2>S&amp;P 500 trend dips · top 15 by 6-month strength</h2>
{tbl(ctx["sp_rows"])}

<h2>Whole watchlist</h2><p class="muted">Click a column title to sort.</p>
{tbl(ctx["all_rows"])}

<footer>Rules recap: max 2% account risk per trade · 15% position cap · semis/AI cap 40% · stop widths 12/18/25% by risk class · pre-trade checker must pass.
Data: Yahoo Finance via yfinance (adjusted closes), refreshed after the US close. Information for your own tracking only, not financial advice.</footer>
</main>
<script>
document.querySelectorAll('table.sortable').forEach(t=>{{t.querySelectorAll('th').forEach((th,i)=>{{let asc=true;th.addEventListener('click',()=>{{
const b=t.tBodies[0];const rows=[...b.rows];rows.sort((x,y)=>{{const a=x.cells[i],c=y.cells[i];const av=a.dataset.v!==undefined?parseFloat(a.dataset.v):a.textContent.trim(),cv=c.dataset.v!==undefined?parseFloat(c.dataset.v):c.textContent.trim();
return (av>cv?1:av<cv?-1:0)*(asc?1:-1)}});asc=!asc;rows.forEach(r=>b.appendChild(r))}})}})}});
</script></body></html>"""


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)

    wl = load_watchlist()
    meta = {r.symbol: dict(theme=r.theme, stop_pct=r.stop_pct) for r in wl.itertuples()}
    warnings = []

    if args.demo:
        sp = [f"SP{i:03d}" for i in range(120)]
        names = {}
        px = demo_prices(sorted(set(sp + list(wl.symbol) + ["^TNX"])))
        px["^TNX"] = 4.2 + (px["^TNX"] / px["^TNX"].iloc[0] - 1) * 2
        px["SPY"] = px["SPY"]
        failed = []
    else:
        sp, names, w = load_sp500()
        if w:
            warnings.append(w)
        px, failed = fetch_prices(sorted(set(sp + list(wl.symbol) + ["SPY", "^TNX"])))

    res = {}
    for s in px.columns:
        a = analyze(px[s])
        if a:
            res[s] = a
    missing = [s for s in wl.symbol if s not in res]
    if missing:
        warnings.append("No usable price history for: " + ", ".join(missing))
    if "SPY" not in res:
        raise SystemExit("SPY data missing; cannot compute regime")

    members = [s for s in sp if s in res]
    if members:
        b200 = 100 * np.mean([res[s]["above200"] for s in members])
        b50 = 100 * np.mean([res[s]["above50"] for s in members])
        nh = sum(res[s]["new_hi"] for s in members)
        nl = sum(res[s]["new_lo"] for s in members)
    else:
        b200 = b50 = float("nan"); nh = nl = 0
        warnings.append("Breadth could not be computed (no S&P 500 list).")

    tnx_series = px["^TNX"].dropna() if "^TNX" in px else pd.Series(dtype=float)
    tnx = float(tnx_series.iloc[-1]) if len(tnx_series) else None
    tnx_change = float(tnx_series.iloc[-1] - tnx_series.iloc[-21]) if len(tnx_series) > 21 else None

    reg, why = regime(res["SPY"], b200 if members else 50.0, tnx_change)

    # history + state (for the trend line and "what changed")
    hpath, spath = os.path.join(DATA, "history.csv"), os.path.join(DATA, "state.json")
    asof = res["SPY"]["asof"]
    hist = pd.read_csv(hpath) if os.path.exists(hpath) else pd.DataFrame(columns=["date", "regime", "b200", "b50", "spy_vs200"])
    prev_regime = hist["regime"].iloc[-1] if len(hist) else None
    if not args.demo and members:
        hist = hist[hist["date"] != asof]
        hist.loc[len(hist)] = [asof, reg, round(b200, 1), round(b50, 1), round(res["SPY"]["pct200"], 4)]
        hist.to_csv(hpath, index=False)
    prev = json.load(open(spath)) if os.path.exists(spath) else {}

    alerts = []
    if prev_regime and prev_regime != reg:
        alerts.append(f"REGIME CHANGED: {prev_regime.upper()} → {reg.upper()}")
    wl_syms = [s for s in wl.symbol if s in res]
    for s in wl_syms:
        a, p = res[s], prev.get(s, {})
        if a["cross"] == "up":
            alerts.append(f"{s} crossed ABOVE its 200-day average (this week).")
        if a["cross"] == "down":
            alerts.append(f"{s} crossed BELOW its 200-day average (this week).")
        if p and a["trend_dip"] and not p.get("trend_dip"):
            alerts.append(f"{s} entered the trend-dip buy window ({pct(a['dd'])} from high).")
        if p and a["range_ok"] and not p.get("range_ok"):
            alerts.append(f"{s} is at a range floor ({money(a['floor'])}).")
    if not args.demo:
        json.dump({s: dict(trend_dip=res[s]["trend_dip"], range_ok=res[s]["range_ok"]) for s in wl_syms},
                  open(spath, "w"))

    short = [s for s in wl_syms if res[s]["trend_dip"] or res[s]["range_ok"]]
    short.sort(key=lambda s: res[s]["dd"])
    sp_dips = sorted([s for s in members if res[s]["trend_dip"] and s not in meta],
                     key=lambda s: -res[s]["ret126"])[:15]
    for s in sp_dips:
        meta.setdefault(s, dict(theme=names.get(s, "S&P 500"), stop_pct=18))
    all_sorted = sorted(wl_syms, key=lambda s: res[s]["pct200"])

    idx_syms = [s for s in ["SPY", "QQQ", "DIA", "IWM", "SMH", "GLD", "TLT"] if s in res]
    spark_vals = list(hist["b200"].astype(float).tail(60)) if len(hist) > 1 else []
    ctx = dict(
        regime=reg, why=why, b200=b200 if members else 0, b50=b50 if members else 0, nh=nh, nl=nl,
        tnx=tnx, tnx_change=tnx_change, asof=asof, demo=args.demo,
        built=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        warnings=warnings, alerts=alerts, spark=sparkline(spark_vals),
        indexes=[(s, res[s]) for s in idx_syms],
        short_rows=[row_html(s, res[s], meta[s]) for s in short],
        sp_rows=[row_html(s, res[s], meta[s]) for s in sp_dips],
        all_rows=[row_html(s, res[s], meta[s]) for s in all_sorted],
    )
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(build_page(ctx))
    print(f"OK regime={reg} breadth200={b200:.1f} watch={len(wl_syms)} short={len(short)} failed={len(failed)}")


if __name__ == "__main__":
    main()
