#!/usr/bin/env python3
"""Cross-exchange arbitrage scanner (Binance / Bybit / Bitget USDT perpetuals).

READ-ONLY research tool: it uses only public market-data endpoints (no API
keys) and never places orders. Runs as its OWN process with its own small
Dash page, so it never slows down the main dashboard (main.py).

What it measures, every scan (default 120 s):
  * Funding gap  - short the perp with the higher funding, long the one with
                   the lower funding; price moves cancel, you earn the
                   difference. Normalised to %/day (exchanges use 1h/4h/8h).
  * Price gap    - buy the cheaper exchange's ask, sell the dearer one's bid.
Both are shown net of taker fees (4 fills: open + close on two exchanges),
and the top opportunities are re-checked against the order book at your trade
size. Every opportunity above a loose threshold is saved to SQLite so you can
see how LONG gaps persist - which is what decides whether they are tradeable.

Usage:
    .venv/bin/python arb_scanner.py                 # UI on :8081 + collector
    .venv/bin/python arb_scanner.py --port 8090
    .venv/bin/python arb_scanner.py --collect-only  # no UI, just record
"""
import argparse
import logging
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

import arb_backtest as bt

logger = logging.getLogger("arb_scanner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_PATH = "arb_snapshots.db"
TIMEOUT = 15
EXCHANGES = ["binance", "bybit", "bitget"]
# Default taker fees in % (VIP 0). Editable in the UI.
DEFAULT_FEES = {"binance": 0.05, "bybit": 0.055, "bitget": 0.06}
# Opportunities above these loose thresholds are stored for persistence stats.
STORE_MIN_FUNDING_DAILY = 0.03   # % per day funding spread
STORE_MIN_PRICE_GAP = 0.10       # % raw price gap
KEEP_DAYS = 30
DEPTH_TOP_N = 8                  # order-book check for top N of each type

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "bince-arb-scanner/1.0"})

# Shared state between collector thread and UI.
STATE = {
    "rows": {},            # exchange -> DataFrame of normalised tickers
    "errors": {},          # exchange -> last error text
    "last_scan": None,
    "scan_seconds": None,
    "depth": {},           # (kind, base, ex_long, ex_short) -> effective gap %
    "settings": {"size_usdt": 500.0, "interval": 120, "fees": dict(DEFAULT_FEES)},
}
LOCK = threading.Lock()


def _get(url, params=None):
    r = SESSION.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _f(x, default=None):
    try:
        v = float(x)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


_MULT_RE = re.compile(r"^(1000000|100000|10000|1000|1M)(?=[A-Z])")


def normalise(symbol):
    """'1000PEPEUSDT' -> ('PEPE', 1000). Returns (None, None) for non-USDT."""
    s = symbol.upper().replace("_UMCBL", "").replace("-", "")
    if not s.endswith("USDT"):
        return None, None
    base = s[:-4]
    mult = 1
    m = _MULT_RE.match(base)
    if m:
        tok = m.group(1)
        mult = 1_000_000 if tok == "1M" else int(tok)
        base = base[len(tok):]
    return base, mult


# ---------------------------------------------------------------- fetchers --
# Each returns a DataFrame: base, symbol, mult, bid, ask, last (all per 1 coin),
# funding (fraction per interval), interval_h, next_funding (ms or None),
# volume_usdt (24h quote volume).

def fetch_binance():
    book = {d["symbol"]: d for d in _get("https://fapi.binance.com/fapi/v1/ticker/bookTicker")}
    prem = {d["symbol"]: d for d in _get("https://fapi.binance.com/fapi/v1/premiumIndex")}
    vol = {d["symbol"]: d for d in _get("https://fapi.binance.com/fapi/v1/ticker/24hr")}
    intervals = {}
    try:  # only symbols with a non-default interval are listed
        for d in _get("https://fapi.binance.com/fapi/v1/fundingInfo"):
            intervals[d["symbol"]] = _f(d.get("fundingIntervalHours"), 8)
    except Exception:
        pass
    rows = []
    for sym, b in book.items():
        base, mult = normalise(sym)
        p = prem.get(sym)
        if not base or not p:
            continue
        rows.append(dict(base=base, symbol=sym, mult=mult,
                         bid=_f(b.get("bidPrice")), ask=_f(b.get("askPrice")),
                         last=_f(p.get("markPrice")),
                         funding=_f(p.get("lastFundingRate"), 0.0),
                         interval_h=intervals.get(sym, 8.0),
                         next_funding=_f(p.get("nextFundingTime")),
                         volume_usdt=_f(vol.get(sym, {}).get("quoteVolume"), 0.0)))
    return pd.DataFrame(rows)


_BYBIT_INTERVALS = {}


def fetch_bybit():
    if not _BYBIT_INTERVALS:
        cursor = ""
        for _ in range(10):
            j = _get("https://api.bybit.com/v5/market/instruments-info",
                     {"category": "linear", "limit": 1000, "cursor": cursor})
            res = j.get("result", {})
            for d in res.get("list", []):
                _BYBIT_INTERVALS[d["symbol"]] = (_f(d.get("fundingInterval"), 480) or 480) / 60.0
            cursor = res.get("nextPageCursor") or ""
            if not cursor:
                break
    j = _get("https://api.bybit.com/v5/market/tickers", {"category": "linear"})
    rows = []
    for d in j.get("result", {}).get("list", []):
        base, mult = normalise(d["symbol"])
        if not base or d.get("fundingRate") in (None, ""):
            continue
        rows.append(dict(base=base, symbol=d["symbol"], mult=mult,
                         bid=_f(d.get("bid1Price")), ask=_f(d.get("ask1Price")),
                         last=_f(d.get("lastPrice")),
                         funding=_f(d.get("fundingRate"), 0.0),
                         interval_h=_BYBIT_INTERVALS.get(d["symbol"], 8.0),
                         next_funding=_f(d.get("nextFundingTime")),
                         volume_usdt=_f(d.get("turnover24h"), 0.0)))
    return pd.DataFrame(rows)


_BITGET_INTERVALS = {}


def fetch_bitget():
    if not _BITGET_INTERVALS:
        j = _get("https://api.bitget.com/api/v2/mix/market/contracts",
                 {"productType": "USDT-FUTURES"})
        for d in j.get("data", []) or []:
            _BITGET_INTERVALS[d["symbol"]] = _f(d.get("fundInterval"), 8) or 8.0
    j = _get("https://api.bitget.com/api/v2/mix/market/tickers", {"productType": "USDT-FUTURES"})
    rows = []
    for d in j.get("data", []) or []:
        base, mult = normalise(d["symbol"])
        if not base:
            continue
        rows.append(dict(base=base, symbol=d["symbol"], mult=mult,
                         bid=_f(d.get("bidPr")), ask=_f(d.get("askPr")),
                         last=_f(d.get("lastPr")),
                         funding=_f(d.get("fundingRate"), 0.0),
                         interval_h=_BITGET_INTERVALS.get(d["symbol"], 8.0),
                         next_funding=None,
                         volume_usdt=_f(d.get("usdtVolume") or d.get("quoteVolume"), 0.0)))
    return pd.DataFrame(rows)


FETCHERS = {"binance": fetch_binance, "bybit": fetch_bybit, "bitget": fetch_bitget}


def fetch_book(exchange, symbol, limit=50):
    """Return (bids, asks) as lists of (price, qty) in contract units."""
    if exchange == "binance":
        j = _get("https://fapi.binance.com/fapi/v1/depth", {"symbol": symbol, "limit": limit})
        bids, asks = j["bids"], j["asks"]
    elif exchange == "bybit":
        j = _get("https://api.bybit.com/v5/market/orderbook",
                 {"category": "linear", "symbol": symbol, "limit": limit})
        bids, asks = j["result"]["b"], j["result"]["a"]
    else:
        j = _get("https://api.bitget.com/api/v2/mix/market/merge-depth",
                 {"symbol": symbol, "productType": "USDT-FUTURES", "limit": limit})
        bids, asks = j["data"]["bids"], j["data"]["asks"]
    return ([(float(p), float(q)) for p, q in bids], [(float(p), float(q)) for p, q in asks])


def fill_price(levels, size_usdt):
    """Average fill price for a market order of size_usdt; None if book too thin."""
    left, cost, qty = size_usdt, 0.0, 0.0
    for p, q in levels:
        take = min(left, p * q)
        cost += take
        qty += take / p
        left -= take
        if left <= 1e-9:
            return cost / qty
    return None


# ------------------------------------------------------------ calculations --

def _per_coin(df):
    df = df.copy()
    for c in ("bid", "ask", "last"):
        df[c] = df[c] / df["mult"]
    df["funding_daily_pct"] = df["funding"] * 100 * 24 / df["interval_h"].clip(lower=0.25)
    return df


def compute_opportunities(frames, fees, exchanges=None, hold_days=3.0):
    """All cross-exchange pairs per coin -> (funding_df, price_df)."""
    exchanges = exchanges or EXCHANGES
    parts = []
    for ex in exchanges:
        df = frames.get(ex)
        if df is None or df.empty:
            continue
        d = _per_coin(df)
        d["exchange"] = ex
        parts.append(d)
    if len(parts) < 2:
        return pd.DataFrame(), pd.DataFrame()
    allx = pd.concat(parts, ignore_index=True).dropna(subset=["bid", "ask"])
    allx = allx[(allx.bid > 0) & (allx.ask > 0)]
    allx = allx.drop_duplicates(subset=["base", "exchange"], keep="first")
    m = allx.merge(allx, on="base", suffixes=("_a", "_b"))
    m = m[m.exchange_a < m.exchange_b]
    if m.empty:
        return pd.DataFrame(), pd.DataFrame()
    m["fee4"] = 2 * (m.exchange_a.map(fees) + m.exchange_b.map(fees))
    m["min_vol"] = m[["volume_usdt_a", "volume_usdt_b"]].min(axis=1)

    # Funding: short the higher-funding leg, long the lower one.
    a_short = m.funding_daily_pct_a >= m.funding_daily_pct_b
    f = pd.DataFrame({
        "coin": m.base,
        "long_ex": m.exchange_b.where(a_short, m.exchange_a),
        "short_ex": m.exchange_a.where(a_short, m.exchange_b),
        "long_sym": m.symbol_b.where(a_short, m.symbol_a),
        "short_sym": m.symbol_a.where(a_short, m.symbol_b),
        "long_funding_day": m.funding_daily_pct_b.where(a_short, m.funding_daily_pct_a),
        "short_funding_day": m.funding_daily_pct_a.where(a_short, m.funding_daily_pct_b),
        "long_interval_h": m.interval_h_b.where(a_short, m.interval_h_a),
        "short_interval_h": m.interval_h_a.where(a_short, m.interval_h_b),
        "long_ask": m.ask_b.where(a_short, m.ask_a),
        "short_bid": m.bid_a.where(a_short, m.bid_b),
        "fee4": m.fee4, "min_vol": m.min_vol,
    })
    f["spread_day"] = f.short_funding_day - f.long_funding_day
    # Entry basis: + means you open with the short leg priced above the long leg.
    f["entry_basis"] = (f.short_bid - f.long_ask) / f.long_ask * 100
    f["net_hold"] = f.spread_day * hold_days + f.entry_basis - f.fee4
    f["breakeven_days"] = ((f.fee4 - f.entry_basis) / f.spread_day.where(f.spread_day > 0)).clip(lower=0)

    # Price: buy the cheap ask, sell the dear bid.
    gap_ab = (m.bid_b - m.ask_a) / m.ask_a * 100   # buy a, sell b
    gap_ba = (m.bid_a - m.ask_b) / m.ask_b * 100   # buy b, sell a
    buy_a = gap_ab >= gap_ba
    p = pd.DataFrame({
        "coin": m.base,
        "buy_ex": m.exchange_a.where(buy_a, m.exchange_b),
        "sell_ex": m.exchange_b.where(buy_a, m.exchange_a),
        "buy_sym": m.symbol_a.where(buy_a, m.symbol_b),
        "sell_sym": m.symbol_b.where(buy_a, m.symbol_a),
        "buy_ask": m.ask_a.where(buy_a, m.ask_b),
        "sell_bid": m.bid_b.where(buy_a, m.bid_a),
        "gap": gap_ab.where(buy_a, gap_ba),
        "fee4": m.fee4, "min_vol": m.min_vol,
        # Funding you'd pay/earn per day holding that hedge (long buy_ex, short sell_ex)
        "funding_day": (m.funding_daily_pct_b.where(buy_a, m.funding_daily_pct_a)
                        - m.funding_daily_pct_a.where(buy_a, m.funding_daily_pct_b)),
    })
    p["net"] = p.gap - p.fee4
    return (f.sort_values("net_hold", ascending=False).reset_index(drop=True),
            p.sort_values("net", ascending=False).reset_index(drop=True))


# ----------------------------------------------------------------- storage --

def db():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.execute("""CREATE TABLE IF NOT EXISTS opps (
        ts TEXT, kind TEXT, coin TEXT, long_ex TEXT, short_ex TEXT,
        value REAL, basis REAL, min_vol REAL)""")
    con.execute("CREATE INDEX IF NOT EXISTS ix_opps ON opps(kind, coin, long_ex, short_ex, ts)")
    con.execute("CREATE TABLE IF NOT EXISTS scans (ts TEXT PRIMARY KEY, n_coins INTEGER, seconds REAL)")
    return con


def store(ts, f, p, n_coins, seconds):
    rows = []
    for r in f[f.spread_day >= STORE_MIN_FUNDING_DAILY].itertuples():
        rows.append((ts, "funding", r.coin, r.long_ex, r.short_ex, r.spread_day, r.entry_basis, r.min_vol))
    for r in p[p.gap >= STORE_MIN_PRICE_GAP].itertuples():
        rows.append((ts, "price", r.coin, r.buy_ex, r.sell_ex, r.gap, r.funding_day, r.min_vol))
    con = db()
    with con:
        con.executemany("INSERT INTO opps VALUES (?,?,?,?,?,?,?,?)", rows)
        con.execute("INSERT OR REPLACE INTO scans VALUES (?,?,?)", (ts, n_coins, seconds))
        cutoff = (datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)).isoformat()
        con.execute("DELETE FROM opps WHERE ts < ?", (cutoff,))
        con.execute("DELETE FROM scans WHERE ts < ?", (cutoff,))
    con.close()


def persistence(kind, threshold, since_hours=24 * 7):
    """For every pair: how many of the most recent consecutive scans were at or
    above `threshold`, and total hours above it in the window."""
    con = db()
    since = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()
    scans = [r[0] for r in con.execute("SELECT ts FROM scans WHERE ts >= ? ORDER BY ts DESC", (since,))]
    hits = pd.read_sql_query(
        "SELECT ts, coin, long_ex, short_ex FROM opps WHERE kind=? AND ts>=? AND value>=?",
        con, params=(kind, since, threshold))
    con.close()
    if hits.empty or not scans:
        return {}
    interval_h = STATE["settings"]["interval"] / 3600.0
    out = {}
    for key, g in hits.groupby(["coin", "long_ex", "short_ex"]):
        tset = set(g.ts)
        streak = 0
        for ts in scans:
            if ts in tset:
                streak += 1
            else:
                break
        out[key] = (streak, streak * interval_h, len(tset) * interval_h)
    return out


# --------------------------------------------------------------- collector --

def depth_check(f, p, size):
    res = {}
    jobs = []
    for r in f[f.spread_day > 0].head(DEPTH_TOP_N).itertuples():
        jobs.append(("funding", r.coin, r.long_ex, r.short_ex, r.long_sym, r.short_sym))
    for r in p[p.gap > 0].head(DEPTH_TOP_N).itertuples():
        jobs.append(("price", r.coin, r.buy_ex, r.sell_ex, r.buy_sym, r.sell_sym))

    def one(job):
        kind, coin, lex, sex, lsym, ssym = job
        try:
            _, asks = fetch_book(lex, lsym)
            bids, _ = fetch_book(sex, ssym)
            buy = fill_price(asks, size)
            sell = fill_price(bids, size)
            if buy is None or sell is None:
                return (kind, coin, lex, sex), None   # book too thin for size
            return (kind, coin, lex, sex), (sell - buy) / buy * 100
        except Exception:
            return (kind, coin, lex, sex), "err"

    with ThreadPoolExecutor(max_workers=6) as ex:
        for k, v in ex.map(one, jobs):
            res[k] = v
    return res


def scan_once():
    t0 = time.time()
    frames, errors = {}, {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {name: ex.submit(fn) for name, fn in FETCHERS.items()}
        for name, fut in futs.items():
            try:
                frames[name] = fut.result()
            except Exception as e:
                errors[name] = str(e)[:200]
                logger.warning(f"{name} fetch failed: {e}")
    with LOCK:
        fees = dict(STATE["settings"]["fees"])
        size = STATE["settings"]["size_usdt"]
    f, p = compute_opportunities(frames, fees)
    depth = depth_check(f, p, size) if not f.empty else {}
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    secs = time.time() - t0
    n_coins = len(set(f.coin)) if not f.empty else 0
    try:
        store(ts, f, p, n_coins, secs)
    except Exception as e:
        logger.error(f"store failed: {e}")
    with LOCK:
        STATE.update(rows=frames, errors=errors, last_scan=ts, scan_seconds=secs, depth=depth)
    logger.info(f"scan {ts}: {', '.join(f'{k}={len(v)}' for k, v in frames.items())} "
                f"coins-on-2+={n_coins} errors={list(errors)} ({secs:.1f}s)")


def collector_loop():
    while True:
        try:
            scan_once()
        except Exception as e:
            logger.exception(f"scan failed: {e}")
        time.sleep(max(int(STATE["settings"]["interval"]), 30))


# ---------------------------------------------------------------------- UI --

BT_JOB = {}

DEFAULT_EXCLUDE = ("ANTHROPIC, OPENAI, SPACEX, XAI, SOXL, SOXS, MSTR, SKHYNIX, SNXX, TSLA, NVDA, AAPL, "
                   "AAPU, AAL, COIN, HOOD, AMZN, GOOGL, META, MSFT, XAU, XAG, PAXG, USDC, FDUSD")


def config_warnings(p, min_vol):
    """Plain-language warnings for settings that don't make sense."""
    w = []
    if min_vol < 1_000_000:
        w.append(f"Min 24h vol is {min_vol:,.0f}$ — thin coins with fake gaps/spikes; use ≥ 5 (million)")
    if p["gap_exit"] >= p["gap_entry"]:
        w.append("Gap exit ≥ Gap entry: trades would close instantly")
    cost = 2 * 2 * max(p["fees"].values()) + 4 * p["slippage"]
    if p["gap_entry"] - p["gap_exit"] <= cost:
        w.append(f"Gap entry − Gap exit ({p['gap_entry'] - p['gap_exit']:.2f}%) ≤ round-trip costs (~{cost:.2f}%): "
                 f"trades lose money even when the gap closes")
    if p.get("max_gap", 3) > 5:
        w.append(f"Max gap {p['max_gap']:g}%: gaps that big are almost always bad data or a different "
                 f"token with the same name; use ≤ 3")
    if p.get("gap_minutes", 3) < p.get("tf_min", 1):
        w.append("Gap must last is shorter than one candle: it is rounded up to 1 candle")
    if p["delay_min"] < 1:
        w.append("Bot delay 0: fills at the same minute the gap is seen — optimistic")
    if p["usage"] > 0.9:
        w.append("Margin usage > 90%: no buffer against liquidation")
    return w


def _bt_worker(params, exchanges, min_vol, top_n, exclude=(), warns=()):
    """Background backtest: load 1m history once, then compare leverages and
    'gap must last' settings on the same data."""
    BT_JOB.update(running=True, result=None, msg="Preparing coins…",
                  job_id=BT_JOB.get("job_id", 0) + 1)
    try:
        with LOCK:
            frames = dict(STATE["rows"])
        if len(frames) < 2:
            frames = {}
            for name in exchanges:
                try:
                    frames[name] = FETCHERS[name]()
                except Exception as e:
                    logger.warning(f"backtest universe: {name} failed: {e}")
        universe = [u for u in bt.build_universe(frames, exchanges, min_vol, top_n * 2)
                    if u["coin"] not in exclude][:top_n]
        if not universe:
            BT_JOB["result"] = {"error": "No coins on 2+ exchanges pass the volume filter."}
            return
        t0 = time.time()

        def say(m):
            BT_JOB["msg"] = f"{m}  ({time.time() - t0:.0f}s, {len(universe)} coins)"

        main = bt.run_price_backtest(universe, params, say)
        if "error" in main:
            BT_JOB["result"] = main
            return
        data = main.pop("data")
        compare, equities = [], {}
        sel_lev, sel_n = float(params["leverage"]), int(params["gap_minutes"])
        for lev in sorted({2.0, 3.0, 5.0, 10.0, sel_lev}):
            say(f"Comparing leverage {lev:g}x…")
            r = main if lev == sel_lev else bt.run_price_backtest(universe, {**params, "leverage": lev}, data=data)
            r.pop("data", None)
            compare.append({"Selected": "◀" if lev == sel_lev else "", **r["summary"]})
            equities[f"{lev:g}x, last {sel_n}m"] = r["equity"]
        tf = int(params.get("tf_min", 1))
        for n in sorted({tf * k for k in (1, 2, 3, 5, 10)} - {sel_n}):
            say(f"Comparing 'gap must last' {n} min…")
            r = bt.run_price_backtest(universe, {**params, "gap_minutes": n}, data=data)
            r.pop("data", None)
            compare.append({"Selected": "", **r["summary"]})
            equities[f"{sel_lev:g}x, last {n}m"] = r["equity"]
        BT_JOB["result"] = {"main": main, "compare": compare, "equities": equities}
        days = (params["end_ms"] - params["start_ms"]) / 86_400_000
        BT_JOB["msg"] = (f"✅ Done in {time.time() - t0:.0f}s — {len(universe)} coins, {days:.0f} days of 1m data. "
                         f"Bias: coin list/volume use TODAY's listings."
                         + ((" ⚠ " + " | ".join(warns)) if warns else ""))
    except Exception as e:
        logger.exception("backtest failed")
        BT_JOB["result"] = {"error": str(e)}
    finally:
        BT_JOB["running"] = False


def build_app():
    from dash import Dash, html, dcc, dash_table, Input, Output, State, no_update
    import plotly.graph_objects as go

    app = Dash(__name__, title="Arb Scanner")
    inp = {"backgroundColor": "#222", "color": "#eee", "border": "1px solid #555",
           "padding": "4px", "width": "90px"}
    lbl = {"color": "#ccc", "fontSize": "13px", "display": "block"}

    def field(label, comp):
        return html.Div([html.Label(label, style=lbl), comp])

    table_style = dict(
        style_cell={"backgroundColor": "#2c2c2c", "color": "#f0f0f0", "border": "1px solid #444",
                    "fontSize": "13px", "padding": "4px"},
        style_header={"backgroundColor": "#00BFFF", "color": "#000", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"filter_query": "{Net %} > 0"}, "color": "#4caf50"},
            {"if": {"filter_query": "{Net %} < 0"}, "color": "#f44336"},
        ],
        sort_action="native", filter_action="native", page_size=25,
        style_table={"overflowX": "auto"},
    )

    live_tab = html.Div(children=[
        html.Div(id="status", style={"fontSize": "13px", "marginBottom": "10px"}),
        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "10px 16px",
                        "alignItems": "flex-end", "marginBottom": "12px"}, children=[
            html.Div(dcc.RadioItems(id="kind", value="price", options=[{"label": "Price gap", "value": "price"}]),
                     style={"display": "none"}),
            field("Exchanges", dcc.Checklist(id="exchanges", value=EXCHANGES, inline=True,
                                             options=[{"label": f" {e} ", "value": e} for e in EXCHANGES],
                                             style={"color": "#eee"})),
            field("Min 24h vol (USDT, both)", dcc.Input(id="min-vol", type="number", value=1_000_000, style=inp)),
            field("Min net %", dcc.Input(id="min-net", type="number", value=0, step=0.05, style=inp)),
            html.Div(dcc.Input(id="hold-days", type="number", value=3), style={"display": "none"}),
            field("Trade size USDT (depth)", dcc.Input(id="size", type="number", value=500, style=inp)),
            field("Fee % Binance", dcc.Input(id="fee-binance", type="number", value=DEFAULT_FEES["binance"], step=0.005, style=inp)),
            field("Fee % Bybit", dcc.Input(id="fee-bybit", type="number", value=DEFAULT_FEES["bybit"], step=0.005, style=inp)),
            field("Fee % Bitget", dcc.Input(id="fee-bitget", type="number", value=DEFAULT_FEES["bitget"], step=0.005, style=inp)),
            field("Persistence threshold", dcc.Input(id="persist-th", type="number", value=0.7, step=0.05, style=inp)),
        ]),
        html.Div(id="explain", style={"color": "#999", "fontSize": "12px", "marginBottom": "8px"}),
        dash_table.DataTable(id="opps", **table_style),
        html.H3("History of selected pair", style={"marginTop": "18px"}),
        html.Div("Click a row above to see how its gap evolved (stored scans above the loose threshold).",
                 style={"color": "#888", "fontSize": "12px"}),
        dcc.Graph(id="history", figure=go.Figure().update_layout(template="plotly_dark", height=320)),
        html.H3("Last 7 days summary"),
        html.Div(id="summary", style={"fontSize": "13px"}),
        dcc.Interval(id="tick", interval=30_000),
    ])

    today = datetime.now(timezone.utc).date()

    def num(id_, value, **kw):
        return dcc.Input(id=id_, type="number", value=value, style=inp, **kw)

    bt_tab = html.Div(children=[
        html.Div("Minute-by-minute replay of 1m closes on all exchanges. A trade starts when one exchange "
                 "stays ≥ Gap entry % above another, in the SAME direction, for 'Gap must last' minutes; the bot "
                 "fills 'Bot delay' minutes later (short the dear exchange, long the cheap one). It closes "
                 "when the gap is ≤ Gap exit % (again after the delay), after Max hold, or on liquidation "
                 "(1m highs/lows). Fees + slippage on 4 fills; funding actually paid while open is included. "
                 "First run downloads 1m history (can take several minutes); re-runs use the local cache.",
                 style={"color": "#999", "fontSize": "12px", "margin": "6px 0 10px 0"}),
        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "10px 16px",
                        "alignItems": "flex-end", "marginBottom": "12px"}, children=[
            field("Period", dcc.Dropdown(id="bt-preset", value=7, clearable=False,
                                         options=[{"label": "Last 3 days", "value": 3},
                                                  {"label": "Last 7 days", "value": 7},
                                                  {"label": "Last 14 days", "value": 14},
                                                  {"label": "Last 30 days", "value": 30},
                                                  {"label": "Custom (pick dates)", "value": 0}],
                                         style={"width": "170px", "color": "#000"})),
            field("Dates", dcc.DatePickerRange(id="bt-dates", display_format="YYYY-MM-DD",
                                               start_date=str(today - timedelta(days=7)),
                                               end_date=str(today), max_date_allowed=str(today))),
            field("Exchanges", dcc.Checklist(id="bt-exchanges", value=EXCHANGES, inline=True,
                                             options=[{"label": f" {e} ", "value": e} for e in EXCHANGES],
                                             style={"color": "#eee"})),
            field("Timeframe", dcc.Dropdown(id="bt-tf", value=1, clearable=False,
                                            options=[{"label": l, "value": v} for l, v in
                                                     [("1m", 1), ("3m", 3), ("5m", 5), ("15m", 15),
                                                      ("30m", 30), ("1h", 60)]],
                                            style={"width": "90px", "color": "#000"})),
            field("Gap entry %", num("bt-gap-entry", 0.7, step=0.05)),
            field("Gap exit %", num("bt-gap-exit", 0.1, step=0.05)),
            field("Gap must last (min)", num("bt-gap-min", 3, min=1)),
            field("Bot delay (min)", num("bt-delay", 1, min=0)),
            field("Max gap % (glitch)", num("bt-max-gap", 3, min=0, step=0.5)),
            field("Max hold (min)", num("bt-hold", 1440, min=1)),
            field("Cooldown same coin (min)", num("bt-cooldown", 30, min=0)),
            field("Max positions", num("bt-pairs", 2, min=1)),
            field("Capital per exchange $", num("bt-capital", 500)),
            field("Leverage", num("bt-lev", 10, min=1)),
            field("Margin usage %", num("bt-usage", 80, min=1, max=100)),
            field("Slippage % / fill", num("bt-slip", 0.05, step=0.01)),
            field("Liq. fee %", num("bt-liqfee", 1.0, step=0.25)),
            field("Margin mode", dcc.RadioItems(id="bt-cross", value="cross", inline=True,
                                                options=[{"label": " Cross ", "value": "cross"},
                                                         {"label": " Isolated", "value": "isolated"}],
                                                style={"color": "#eee"})),
            field("Min 24h vol (million $)", num("bt-minvol", 5, min=0, step=0.5)),
            field("Coins (top by volume)", num("bt-topn", 40, min=2)),
            field("Exclude coins", dcc.Input(id="bt-exclude", type="text", value=DEFAULT_EXCLUDE,
                                             style={**inp, "width": "330px"})),
            html.Button("Run backtest", id="bt-run", n_clicks=0,
                        style={"backgroundColor": "#28a745", "color": "#fff", "border": "none",
                               "padding": "8px 16px", "borderRadius": "4px", "cursor": "pointer"}),
        ]),
        html.Div(id="bt-progress", style={"fontSize": "13px", "color": "#ffcc66", "marginBottom": "8px"}),
        html.H3("Comparison: leverages (selected 'gap must last') and 'gap must last' (selected leverage)"),
        dash_table.DataTable(id="bt-compare", **{**table_style, "page_size": 12,
                                                 "style_data_conditional": [
                                                     {"if": {"filter_query": "{Net $} > 0"}, "color": "#4caf50"},
                                                     {"if": {"filter_query": "{Net $} < 0"}, "color": "#f44336"},
                                                     {"if": {"filter_query": "{Liquidations} > 0",
                                                             "column_id": "Liquidations"},
                                                      "backgroundColor": "#5c1e1e"}]}),
        dcc.Graph(id="bt-equity", figure=go.Figure().update_layout(template="plotly_dark", height=340)),
        html.H3("Per coin"),
        dash_table.DataTable(id="bt-coins", **{**table_style, "page_size": 15,
                                               "style_data_conditional": [
                                                   {"if": {"filter_query": "{Net $} > 0"}, "color": "#4caf50"},
                                                   {"if": {"filter_query": "{Net $} < 0"}, "color": "#f44336"}]}),
        html.H3("Trades (selected settings)"),
        dash_table.DataTable(id="bt-trades", **{**table_style,
                                                "style_data_conditional": [
                                                    {"if": {"filter_query": "{Net $} > 0"}, "color": "#4caf50"},
                                                    {"if": {"filter_query": "{Net $} < 0"}, "color": "#f44336"}]}),
        dcc.Interval(id="bt-tick", interval=2000),
    ])

    app.layout = html.Div(style={"backgroundColor": "#1a1a1a", "minHeight": "100vh",
                                 "padding": "12px", "fontFamily": "sans-serif", "color": "#eee"}, children=[
        html.H2("Cross-Exchange Price-Gap Scanner (read-only)", style={"margin": "0 0 4px 0"}),
        html.Div("Binance · Bybit · Bitget USDT perpetuals — public data only, no orders are placed.",
                 style={"color": "#888", "fontSize": "13px", "marginBottom": "10px"}),
        dcc.Tabs(id="tabs", value="live", colors={"border": "#444", "primary": "#00BFFF", "background": "#2c2c2c"},
                 children=[
                     dcc.Tab(label="Live scanner", value="live", children=live_tab,
                             style={"backgroundColor": "#2c2c2c", "color": "#ccc"},
                             selected_style={"backgroundColor": "#1a1a1a", "color": "#00BFFF"}),
                     dcc.Tab(label="Price-gap backtest (1m)", value="bt", children=bt_tab,
                             style={"backgroundColor": "#2c2c2c", "color": "#ccc"},
                             selected_style={"backgroundColor": "#1a1a1a", "color": "#00BFFF"}),
                 ]),
    ])

    @app.callback([Output("bt-dates", "start_date"), Output("bt-dates", "end_date")],
                  Input("bt-preset", "value"), prevent_initial_call=True)
    def apply_preset(days):
        if not days:
            return no_update, no_update
        now = datetime.now(timezone.utc).date()
        return str(now - timedelta(days=int(days))), str(now)

    @app.callback(Output("bt-preset", "value"),
                  [Input("bt-dates", "start_date"), Input("bt-dates", "end_date")],
                  State("bt-preset", "value"), prevent_initial_call=True)
    def dates_to_custom(start, end, preset):
        if not preset or not start or not end:
            return no_update
        now = datetime.now(timezone.utc).date()
        if start[:10] == str(now - timedelta(days=int(preset))) and end[:10] == str(now):
            return no_update
        return 0

    @app.callback(Output("bt-progress", "children"),
                  Input("bt-run", "n_clicks"),
                  [State("bt-dates", "start_date"), State("bt-dates", "end_date"), State("bt-exchanges", "value"),
                   State("bt-gap-entry", "value"), State("bt-gap-exit", "value"), State("bt-gap-min", "value"),
                   State("bt-delay", "value"), State("bt-max-gap", "value"), State("bt-hold", "value"),
                   State("bt-cooldown", "value"), State("bt-pairs", "value"), State("bt-capital", "value"),
                   State("bt-lev", "value"), State("bt-usage", "value"), State("bt-slip", "value"),
                   State("bt-liqfee", "value"), State("bt-cross", "value"), State("bt-minvol", "value"),
                   State("bt-topn", "value"), State("bt-exclude", "value"),
                   State("fee-binance", "value"), State("fee-bybit", "value"), State("fee-bitget", "value"),
                   State("bt-tf", "value")],
                  prevent_initial_call=True)
    def start_bt(_n, start, end, exs, g_in, g_out, g_min, delay, max_gap, hold, cooldown, pairs, cap,
                 lev, usage, slip, liqfee, cross, minvol, topn, exclude, fb, fy, fg, tf):
        if BT_JOB.get("running"):
            return "A backtest is already running…"
        if not start or not end:
            return "❌ Pick a start and end date."
        val = lambda v, d: d if v is None else v
        try:
            params = dict(
                start_ms=bt.ms(start),
                end_ms=min(bt.ms(end, end_of_day=True), int(time.time() * 1000) - 600_000),
                tf_min=int(val(tf, 1)),
                gap_entry=float(val(g_in, 0.7)), gap_exit=float(val(g_out, 0.1)),
                gap_minutes=int(val(g_min, 3)), delay_min=int(val(delay, 1)),
                max_gap=float(val(max_gap, 3)), max_hold_min=int(val(hold, 1440)),
                cooldown_min=int(val(cooldown, 30)), max_pairs=int(val(pairs, 2)),
                capital=float(val(cap, 500)), leverage=float(val(lev, 10)),
                usage=float(val(usage, 80)) / 100, slippage=float(val(slip, 0.05)),
                liq_fee=float(val(liqfee, 1.0)), cross=(cross == "cross"), mmr=0.5,
                fees={"binance": val(fb, DEFAULT_FEES["binance"]), "bybit": val(fy, DEFAULT_FEES["bybit"]),
                      "bitget": val(fg, DEFAULT_FEES["bitget"])})
        except Exception as e:
            return f"❌ Bad input: {e}"
        if params["end_ms"] <= params["start_ms"]:
            return "❌ End must be after start."
        minvol = float(val(minvol, 5)) * 1_000_000
        excl = {c.strip().upper() for c in (exclude or "").replace(";", ",").split(",") if c.strip()}
        warns = config_warnings(params, minvol)
        threading.Thread(target=_bt_worker, args=(params, exs or EXCHANGES, minvol, int(val(topn, 40)),
                                                  excl, warns), daemon=True).start()
        return "Starting…" + ((" ⚠ " + " | ".join(warns)) if warns else "")

    @app.callback([Output("bt-progress", "children", allow_duplicate=True),
                   Output("bt-compare", "data"), Output("bt-compare", "columns"),
                   Output("bt-equity", "figure"), Output("bt-coins", "data"), Output("bt-coins", "columns"),
                   Output("bt-trades", "data"), Output("bt-trades", "columns")],
                  Input("bt-tick", "n_intervals"), State("tabs", "value"),
                  prevent_initial_call=True)
    def poll_bt(_n, tab):
        if tab != "bt":
            return (no_update,) * 8
        msg = BT_JOB.get("msg", "")
        res = BT_JOB.get("result")
        if res is None or BT_JOB.get("shown") == BT_JOB.get("job_id"):
            return (msg,) + (no_update,) * 7
        BT_JOB["shown"] = BT_JOB.get("job_id")
        if "error" in res:
            return (f"❌ {res['error']}",) + (no_update,) * 7
        comp, main = res["compare"], res["main"]
        fig = go.Figure()
        for name, r in res["equities"].items():
            fig.add_trace(go.Scatter(x=r.index, y=r.values, mode="lines", name=name))
        fig.update_layout(template="plotly_dark", height=340, title="Equity (net $, both exchanges)",
                          margin=dict(l=40, r=20, t=40, b=30))
        tr = main["trades"]
        coins = []
        if not tr.empty:
            g = tr.groupby("Coin").agg(Trades=("Net $", "size"), Price=("Price PnL $", "sum"),
                                       Funding=("Funding $", "sum"), Fees=("Fees $", "sum"),
                                       Net=("Net $", "sum"), Minutes=("Minutes", "sum")).reset_index()
            g = g.rename(columns={"Price": "Price PnL $", "Funding": "Funding $", "Fees": "Fees $",
                                  "Net": "Net $", "Minutes": "Minutes held"}).round(2)
            coins = g.sort_values("Net $", ascending=False).to_dict("records")
        cols = lambda rows: [{"name": c, "id": c} for c in (rows[0].keys() if rows else ["—"])]
        trows = tr.to_dict("records") if not tr.empty else []
        return (msg, comp, cols(comp), fig, coins, cols(coins), trows, cols(trows))

    @app.callback(
        [Output("opps", "data"), Output("opps", "columns"), Output("status", "children"),
         Output("explain", "children"), Output("summary", "children")],
        [Input("tick", "n_intervals"), Input("kind", "value"), Input("exchanges", "value"),
         Input("min-vol", "value"), Input("min-net", "value"), Input("hold-days", "value"),
         Input("size", "value"), Input("fee-binance", "value"), Input("fee-bybit", "value"),
         Input("fee-bitget", "value"), Input("persist-th", "value")])
    def refresh(_n, kind, exchanges, min_vol, min_net, hold_days, size, fb, fy, fg, persist_th):
        fees = {"binance": fb if fb is not None else DEFAULT_FEES["binance"],
                "bybit": fy if fy is not None else DEFAULT_FEES["bybit"],
                "bitget": fg if fg is not None else DEFAULT_FEES["bitget"]}
        with LOCK:
            STATE["settings"]["fees"] = fees
            if size:
                STATE["settings"]["size_usdt"] = float(size)
            frames = dict(STATE["rows"])
            errors = dict(STATE["errors"])
            last, secs, depth = STATE["last_scan"], STATE["scan_seconds"], dict(STATE["depth"])

        counts = " · ".join(f"{e}: {len(frames[e]) if e in frames else '—'}" for e in EXCHANGES)
        err = (" | ❌ " + "; ".join(f"{k}: {v}" for k, v in errors.items())) if errors else ""
        status = (f"Last scan: {last or 'running first scan…'}"
                  + (f" ({secs:.1f}s)" if secs else "") + f" | contracts: {counts}{err}")

        f, p = compute_opportunities(frames, fees, exchanges or EXCHANGES, float(hold_days or 3))
        mv, mn = float(min_vol or 0), float(min_net or 0)
        th = float(persist_th if persist_th is not None else 0.05)
        try:
            pers = persistence(kind, th)
        except Exception:
            pers = {}

        def fmt_pers(key):
            v = pers.get(key)
            return (f"{_dur(v[1])} now / {_dur(v[2])} in 7d") if v else "—"

        def fmt_depth(key):
            v = depth.get(key, "")
            if v is None:
                return "book too thin"
            if v == "err":
                return "error"
            return f"{v:.3f}" if v != "" else ""

        if kind == "funding":
            d = f[(f.min_vol >= mv) & (f.net_hold >= mn)] if not f.empty else f
            data = [{
                "Coin": r.coin, "Long @": r.long_ex, "Short @": r.short_ex,
                "Long fund %/day": round(r.long_funding_day, 4),
                "Short fund %/day": round(r.short_funding_day, 4),
                "Intervals (L/S h)": f"{r.long_interval_h:g}/{r.short_interval_h:g}",
                "Spread %/day": round(r.spread_day, 4),
                "Entry basis %": round(r.entry_basis, 4),
                "Fees (4 fills) %": round(r.fee4, 3),
                "Net %": round(r.net_hold, 4),
                "Break-even days": None if pd.isna(r.breakeven_days) else round(r.breakeven_days, 2),
                "Basis @size %": fmt_depth(("funding", r.coin, r.long_ex, r.short_ex)),
                "Above threshold": fmt_pers((r.coin, r.long_ex, r.short_ex)),
                "Min 24h vol": f"{r.min_vol / 1e6:.1f}M",
            } for r in d.head(300).itertuples()]
            explain = (f"Funding gap: long on 'Long @', short on 'Short @' (same size). Net % = spread × "
                       f"{hold_days or 3} days + entry basis − fees for 4 fills. Rates are the CURRENT "
                       f"funding; they change every interval, so check 'Above threshold' (consecutive "
                       f"scans with spread ≥ {th}%/day) before trusting a row.")
        else:
            d = p[(p.min_vol >= mv) & (p.net >= mn)] if not p.empty else p
            data = [{
                "Coin": r.coin, "Buy @": r.buy_ex, "Sell @": r.sell_ex,
                "Buy ask": r.buy_ask, "Sell bid": r.sell_bid,
                "Gap %": round(r.gap, 4), "Fees (4 fills) %": round(r.fee4, 3),
                "Net %": round(r.net, 4),
                "Gap @size %": fmt_depth(("price", r.coin, r.buy_ex, r.sell_ex)),
                "Hedge funding %/day": round(r.funding_day, 4),
                "Above threshold": fmt_pers((r.coin, r.buy_ex, r.sell_ex)),
                "Min 24h vol": f"{r.min_vol / 1e6:.1f}M",
            } for r in d.head(300).itertuples()]
            explain = (f"Price gap: buy perp on 'Buy @', sell perp on 'Sell @', close both when prices "
                       f"converge. Net % = gap − fees for 4 fills. 'Gap @size' re-prices the top "
                       f"{DEPTH_TOP_N} with the order book at your size. 'Hedge funding' is what holding "
                       f"the hedge earns(+)/costs(−) per day. Large gaps on tiny coins are often "
                       f"halted deposits/withdrawals or different tokens with the same ticker.")
        cols = [{"name": c, "id": c} for c in (data[0].keys() if data else ["Coin"])]

        try:
            summary = summarise(kind, th)
        except Exception as e:
            summary = f"No history yet ({e})."
        return data, cols, status, explain, summary

    @app.callback(Output("history", "figure"),
                  Input("opps", "active_cell"),
                  [State("opps", "derived_viewport_data"), State("kind", "value")],
                  prevent_initial_call=True)
    def history(cell, rows, kind):
        if not cell or not rows:
            return no_update
        r = rows[cell["row"]]
        lex = r.get("Long @") or r.get("Buy @")
        sex = r.get("Short @") or r.get("Sell @")
        con = db()
        h = pd.read_sql_query(
            "SELECT ts, value, basis FROM opps WHERE kind=? AND coin=? AND long_ex=? AND short_ex=? ORDER BY ts",
            con, params=(kind, r["Coin"], lex, sex))
        con.close()
        fig = go.Figure()
        name = "Funding spread %/day" if kind == "funding" else "Price gap %"
        if not h.empty:
            h["ts"] = pd.to_datetime(h.ts)
            fig.add_trace(go.Scatter(x=h.ts, y=h.value, mode="lines+markers", name=name))
            fig.add_trace(go.Scatter(x=h.ts, y=h.basis, mode="lines", name=(
                "Entry basis %" if kind == "funding" else "Hedge funding %/day"), yaxis="y2"))
        fig.update_layout(template="plotly_dark", height=320,
                          title=f"{r['Coin']}: {lex} → {sex}" + ("" if not h.empty else " (no stored history)"),
                          yaxis=dict(title=name), yaxis2=dict(overlaying="y", side="right"),
                          margin=dict(l=40, r=40, t=40, b=30))
        return fig

    return app


def _dur(hours):
    """0.4 -> '24m', 5.5 -> '5.5h', 50 -> '2.1d'."""
    if hours < 1:
        return f"{hours * 60:.0f}m"
    return f"{hours:.1f}h" if hours < 48 else f"{hours / 24:.1f}d"


def summarise(kind, threshold):
    """Pairs that stayed above threshold the longest over the last 7 days."""
    pers = persistence(kind, threshold)
    con = db()
    n_scans = con.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    first = con.execute("SELECT MIN(ts) FROM scans").fetchone()[0]
    con.close()
    if not pers:
        return f"{n_scans} scans recorded since {first or '—'}; no pair above {threshold} yet."
    top = sorted(pers.items(), key=lambda kv: kv[1][2], reverse=True)[:15]
    unit = "%/day spread" if kind == "funding" else "% gap"
    from dash import html
    return html.Div([
        html.Div(f"{n_scans} scans recorded since {first}. Pairs with the most hours ≥ {threshold}{unit}:"),
        html.Ul([html.Li(f"{c}  {a} → {b}: {_dur(tot)} total, current streak {_dur(cur)}")
                 for (c, a, b), (_s, cur, tot) in top]),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--interval", type=int, default=120, help="seconds between scans (min 30)")
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()
    STATE["settings"]["interval"] = max(args.interval, 30)
    db().close()
    if args.collect_only:
        collector_loop()
        return
    threading.Thread(target=collector_loop, daemon=True).start()
    build_app().run(debug=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
