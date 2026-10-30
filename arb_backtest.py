"""Historical data layer + minute-by-minute price-gap backtest
(Binance / Bybit / Bitget USDT perpetuals, public endpoints only).

1-minute candles have no bid/ask, so gaps come from 1m closes; the slippage
input stands in for the bid/ask spread. Gaps shorter than a minute are not
visible, and a polling bot could not take them anyway.
"""
import sqlite3
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

CACHE_DB = "arb_history_cache.db"
HOUR_MS = 3_600_000
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "bince-arb-backtest/1.0"})
_DB_LOCK = threading.Lock()


def _get(url, params=None, retries=3):
    for i in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == retries - 1:
                raise
            time.sleep(1 + i)
    return None


# ------------------------------------------------------------------ cache --

def _db():
    con = sqlite3.connect(CACHE_DB, timeout=30)
    con.execute("CREATE TABLE IF NOT EXISTS funding (ex TEXT, sym TEXT, ts INTEGER, rate REAL, "
                "PRIMARY KEY (ex, sym, ts))")
    con.execute("CREATE TABLE IF NOT EXISTS klines (ex TEXT, sym TEXT, ts INTEGER, high REAL, low REAL, "
                "close REAL, PRIMARY KEY (ex, sym, ts))")
    con.execute("CREATE TABLE IF NOT EXISTS fetched (kind TEXT, ex TEXT, sym TEXT, start INTEGER, "
                "end INTEGER, PRIMARY KEY (kind, ex, sym))")
    return con


def _covered(kind, ex, sym, start, end):
    with _DB_LOCK:
        con = _db()
        r = con.execute("SELECT start, end FROM fetched WHERE kind=? AND ex=? AND sym=?",
                        (kind, ex, sym)).fetchone()
        con.close()
    # Allow the most recent few hours to be missing (they will be refetched next run).
    return bool(r) and r[0] <= start and r[1] >= end - 6 * HOUR_MS


def _save(kind, ex, sym, rows, start, end):
    with _DB_LOCK:
        con = _db()
        with con:
            if kind == "funding":
                con.executemany("INSERT OR REPLACE INTO funding VALUES (?,?,?,?)",
                                [(ex, sym, int(t), float(v)) for t, v in rows])
            else:
                con.executemany("INSERT OR REPLACE INTO klines VALUES (?,?,?,?,?,?)",
                                [(ex, sym, int(t), float(h), float(l), float(c)) for t, h, l, c in rows])
            r = con.execute("SELECT start, end FROM fetched WHERE kind=? AND ex=? AND sym=?",
                            (kind, ex, sym)).fetchone()
            if r:
                start, end = min(start, r[0]), max(end, r[1])
            con.execute("INSERT OR REPLACE INTO fetched VALUES (?,?,?,?,?)", (kind, ex, sym, start, end))
        con.close()


def _load(kind, ex, sym, start, end):
    with _DB_LOCK:
        con = _db()
        if kind == "funding":
            df = pd.read_sql_query("SELECT ts, rate FROM funding WHERE ex=? AND sym=? AND ts BETWEEN ? AND ? "
                                   "ORDER BY ts", con, params=(ex, sym, start, end))
        else:
            df = pd.read_sql_query("SELECT ts, high, low, close FROM klines WHERE ex=? AND sym=? AND "
                                   "ts BETWEEN ? AND ? ORDER BY ts", con, params=(ex, sym, start, end))
        con.close()
    return df


# --------------------------------------------------------------- fetchers --

def _fund_binance(sym, start, end):
    out, cur = [], start
    while cur < end:
        j = _get("https://fapi.binance.com/fapi/v1/fundingRate",
                 {"symbol": sym, "startTime": cur, "endTime": end, "limit": 1000})
        if not j:
            break
        out += [(d["fundingTime"], float(d["fundingRate"])) for d in j]
        if len(j) < 1000:
            break
        cur = j[-1]["fundingTime"] + 1
    return out


def _fund_bybit(sym, start, end):
    out, cur_end = [], end
    for _ in range(200):
        j = _get("https://api.bybit.com/v5/market/funding/history",
                 {"category": "linear", "symbol": sym, "startTime": start, "endTime": cur_end, "limit": 200})
        lst = (j or {}).get("result", {}).get("list", [])
        if not lst:
            break
        out += [(int(d["fundingRateTimestamp"]), float(d["fundingRate"])) for d in lst]
        oldest = min(int(d["fundingRateTimestamp"]) for d in lst)
        if len(lst) < 200 or oldest <= start:
            break
        cur_end = oldest - 1
    return out


def _fund_bitget(sym, start, end):
    out = []
    for page in range(1, 200):
        j = _get("https://api.bitget.com/api/v2/mix/market/history-fund-rate",
                 {"symbol": sym, "productType": "USDT-FUTURES", "pageSize": 100, "pageNo": page})
        lst = (j or {}).get("data") or []
        if not lst:
            break
        out += [(int(d["fundingTime"]), float(d["fundingRate"])) for d in lst]
        if min(int(d["fundingTime"]) for d in lst) <= start or len(lst) < 100:
            break
    return [(t, r) for t, r in out if start <= t <= end]


def _kl_binance(sym, start, end):
    out, cur = [], start
    while cur < end:
        j = _get("https://fapi.binance.com/fapi/v1/klines",
                 {"symbol": sym, "interval": "1h", "startTime": cur, "endTime": end, "limit": 1500})
        if not j:
            break
        out += [(k[0], k[2], k[3], k[4]) for k in j]
        if len(j) < 1500:
            break
        cur = j[-1][0] + HOUR_MS
    return out


def _kl_bybit(sym, start, end):
    out, cur_end = [], end
    for _ in range(100):
        j = _get("https://api.bybit.com/v5/market/kline",
                 {"category": "linear", "symbol": sym, "interval": "60", "start": start,
                  "end": cur_end, "limit": 1000})
        lst = (j or {}).get("result", {}).get("list", [])
        if not lst:
            break
        out += [(int(k[0]), k[2], k[3], k[4]) for k in lst]
        oldest = min(int(k[0]) for k in lst)
        if len(lst) < 1000 or oldest <= start:
            break
        cur_end = oldest - 1
    return out


def _kl_bitget(sym, start, end):
    out, cur_end = [], end
    for _ in range(200):
        j = _get("https://api.bitget.com/api/v2/mix/market/history-candles",
                 {"symbol": sym, "productType": "USDT-FUTURES", "granularity": "1H",
                  "endTime": cur_end, "limit": 200})
        lst = (j or {}).get("data") or []
        if not lst:
            break
        out += [(int(k[0]), k[2], k[3], k[4]) for k in lst]
        oldest = min(int(k[0]) for k in lst)
        if oldest <= start or len(lst) < 200:
            break
        cur_end = oldest - 1
    return [r for r in out if start <= r[0] <= end]


FUND_FETCH = {"binance": _fund_binance, "bybit": _fund_bybit, "bitget": _fund_bitget}
KLINE_FETCH = {"binance": _kl_binance, "bybit": _kl_bybit, "bitget": _kl_bitget}


def get_funding(ex, sym, start, end):
    if not _covered("funding", ex, sym, start, end):
        _save("funding", ex, sym, FUND_FETCH[ex](sym, start, end), start, end)
    return _load("funding", ex, sym, start, end)


def get_klines(ex, sym, start, end):
    if not _covered("klines", ex, sym, start, end):
        _save("klines", ex, sym, KLINE_FETCH[ex](sym, start, end), start, end)
    return _load("klines", ex, sym, start, end)


# --------------------------------------------------- 1-minute price data --
# Stored as one .npz file per exchange/symbol/UTC day (complete days only), so
# re-runs and overlapping periods never download the same minutes twice.
MIN_MS = 60_000
DAY_MS = 86_400_000
CACHE_DIR = "arb_cache_1m"


def _m1_binance(sym, start, end):
    out, cur = [], start
    while cur < end:
        j = _get("https://fapi.binance.com/fapi/v1/klines",
                 {"symbol": sym, "interval": "1m", "startTime": cur, "endTime": end, "limit": 1500})
        if not j:
            break
        out += [(k[0], k[2], k[3], k[4]) for k in j]
        if len(j) < 1500:
            break
        cur = j[-1][0] + MIN_MS
    return out


def _m1_bybit(sym, start, end):
    out, cur_end = [], end
    for _ in range(20):
        j = _get("https://api.bybit.com/v5/market/kline",
                 {"category": "linear", "symbol": sym, "interval": "1", "start": start,
                  "end": cur_end, "limit": 1000})
        lst = (j or {}).get("result", {}).get("list", [])
        if not lst:
            break
        out += [(int(k[0]), k[2], k[3], k[4]) for k in lst]
        oldest = min(int(k[0]) for k in lst)
        if len(lst) < 1000 or oldest <= start:
            break
        cur_end = oldest - 1
    return out


def _m1_bitget(sym, start, end):
    out, cur_end = [], end
    for _ in range(40):
        lst = []
        try:   # recent data: up to 1000 per call
            j = _get("https://api.bitget.com/api/v2/mix/market/candles",
                     {"symbol": sym, "productType": "USDT-FUTURES", "granularity": "1m",
                      "startTime": start, "endTime": cur_end, "limit": 1000})
            lst = (j or {}).get("data") or []
        except Exception:
            lst = []
        if not lst:  # older data: history endpoint, 200 per call
            j = _get("https://api.bitget.com/api/v2/mix/market/history-candles",
                     {"symbol": sym, "productType": "USDT-FUTURES", "granularity": "1m",
                      "endTime": cur_end, "limit": 200})
            lst = (j or {}).get("data") or []
        if not lst:
            break
        out += [(int(k[0]), k[2], k[3], k[4]) for k in lst]
        oldest = min(int(k[0]) for k in lst)
        if oldest <= start:
            break
        cur_end = oldest - 1
    return [r for r in out if start <= r[0] <= end]


M1_FETCH = {"binance": _m1_binance, "bybit": _m1_bybit, "bitget": _m1_bitget}


def get_1m(ex, sym, start, end):
    """DataFrame ts/high/low/close of 1-minute candles, cached per UTC day."""
    import os
    os.makedirs(CACHE_DIR, exist_ok=True)
    frames = []
    now = int(time.time() * 1000)
    day = start // DAY_MS * DAY_MS
    while day < end:
        d_end = day + DAY_MS - 1
        path = os.path.join(CACHE_DIR, f"{ex}_{sym}_{day // DAY_MS}.npz")
        if os.path.exists(path):
            z = np.load(path)
            arr = z["a"]
        else:
            rows = M1_FETCH[ex](sym, day, min(d_end, now))
            arr = np.array(sorted({int(r[0]): r for r in rows}.values()), dtype=float) if rows \
                else np.zeros((0, 4))
            if d_end < now - 5 * MIN_MS:          # only complete days are cached
                np.savez_compressed(path, a=arr)
        if len(arr):
            frames.append(arr)
        day += DAY_MS
    if not frames:
        return pd.DataFrame(columns=["ts", "high", "low", "close"])
    a = np.vstack(frames)
    df = pd.DataFrame(a, columns=["ts", "high", "low", "close"])
    df["ts"] = df.ts.astype(np.int64)
    return df[(df.ts >= start) & (df.ts <= end)].drop_duplicates("ts")


# --------------------------------------------------------- price-gap test --

def run_price_backtest(universe, p, progress=None, data=None):
    """Minute-by-minute price-gap backtest.

    Each minute, for every coin on 2+ exchanges: gap = (dear close - cheap close)
    / cheap close. ENTRY when the gap in the SAME direction stayed >= gap_entry
    for `gap_minutes` consecutive minutes; orders are filled `delay_min` minutes
    later at that minute's close (bot reaction time) +/- slippage. EXIT when the
    gap is <= gap_exit (filled `delay_min` later), after max_hold_min, or on
    liquidation (1m highs/lows). Fees on 4 fills. Funding actually paid is
    charged/credited while a position is open.
    `data` can carry already-loaded arrays so leverage/setting comparisons
    don't reload anything.
    """
    say = progress or (lambda *_: None)
    start, end = int(p["start_ms"]), int(p["end_ms"])
    grid = np.arange((start // MIN_MS + 1) * MIN_MS, end, MIN_MS)
    T = len(grid)
    if data is None:
        data = {"px": {}, "fund": {}}
        n = len(universe)
        for i, u in enumerate(universe):
            say(f"1m prices {i + 1}/{n}: {u['coin']}")
            for ex, sym in u["symbols"].items():
                try:
                    k = get_1m(ex, sym, start, end)
                except Exception:
                    continue
                if len(k) < 60:
                    continue
                k = k.set_index("ts").reindex(grid)
                stale = k.close.isna().values
                k = k.ffill()
                m = u["mult"].get(ex, 1) or 1
                data["px"][(u["coin"], ex)] = (k.high.values.astype(np.float32) / m,
                                               k.low.values.astype(np.float32) / m,
                                               k.close.values.astype(np.float32) / m, stale)
                try:
                    f = get_funding(ex, sym, start, end)
                    paid = np.zeros(T, dtype=np.float32)
                    idx = np.searchsorted(grid, f.ts.values)
                    ok = idx < T
                    np.add.at(paid, idx[ok], f.rate.values[ok])
                    data["fund"][(u["coin"], ex)] = paid
                except Exception:
                    pass
    px, fund = data["px"], data["fund"]

    # pairs + oriented gap matrix (row r: short a / long b; negative = other way)
    pairs = []
    for u in universe:
        exs = sorted(ex for ex in u["symbols"] if (u["coin"], ex) in px)
        for i in range(len(exs)):
            for j in range(i + 1, len(exs)):
                pairs.append((u["coin"], exs[i], exs[j]))
    if not pairs:
        return {"error": "No 1-minute price history could be loaded for 2+ exchanges."}
    say("Finding gaps…")
    G = np.empty((len(pairs), T), dtype=np.float32)
    for r, (coin, a, b) in enumerate(pairs):
        ca, cb = px[(coin, a)][2], px[(coin, b)][2]
        g = (ca - cb) / cb * 100
        # Ignore minutes where either exchange had no trade (stale, ffilled price).
        g[px[(coin, a)][3] | px[(coin, b)][3]] = np.nan
        G[r] = g

    n_need = max(int(p.get("gap_minutes", 3)), 1)
    ge, gx, max_gap = p["gap_entry"], p["gap_exit"], p.get("max_gap", 3.0)

    def persistent(M):
        """True where M >= ge for the last n_need minutes (same direction)."""
        ok = (M >= ge) & (M <= max_gap if max_gap else True)
        ok = np.nan_to_num(ok.astype(np.int16))
        c = np.cumsum(ok, axis=1)
        run = c.copy()
        run[:, n_need:] = c[:, n_need:] - c[:, :-n_need]
        run[:, :n_need - 1] = 0
        return run >= n_need

    sig_pos = persistent(G)        # short a, long b
    sig_neg = persistent(-G)       # short b, long a
    any_sig = np.flatnonzero(sig_pos.any(axis=0) | sig_neg.any(axis=0))

    slots = max(int(p["max_pairs"]), 1)
    notional = p["capital"] * p["usage"] * p["leverage"] / slots
    margin = p["capital"] / slots if p["cross"] else notional / p["leverage"]
    liq_move = max(margin / notional - p["mmr"] / 100.0, 0.001)
    fees, slip = p["fees"], p["slippage"] / 100.0
    delay = max(int(p.get("delay_min", 1)), 0)
    max_hold = int(p.get("max_hold_min", 24 * 60))
    cooldown = int(p.get("cooldown_min", 30))

    say("Simulating…")
    open_pos, trades, last_close = [], [], {}
    equity = np.zeros(T)
    realised = 0.0
    sig_set = set(any_sig.tolist())
    t = 0
    while t < T:
        still = []
        for pos in open_pos:
            lh, ll, lc, _ = px[(pos["coin"], pos["long"])]
            sh, sl, sc, _ = px[(pos["coin"], pos["short"])]
            if t > pos["t0"]:
                fs = fund.get((pos["coin"], pos["short"]))
                fl = fund.get((pos["coin"], pos["long"]))
                pos["funding"] += notional * ((fs[t] if fs is not None else 0) - (fl[t] if fl is not None else 0))
            liq = None
            if t > pos["t0"]:
                if ll[t] <= pos["l0"] * (1 - liq_move):
                    liq = "long"
                elif sh[t] >= pos["s0"] * (1 + liq_move):
                    liq = "short"
            r, sgn = pos["row"], pos["sgn"]
            g_now = sgn * G[r, t]
            reason = None
            if liq:
                reason = f"LIQUIDATED {liq}"
            elif pos["exit_at"] is None and t > pos["t0"] and not np.isnan(g_now) and g_now <= gx:
                pos["exit_at"] = min(t + delay, T - 1)       # bot reacts next minute
            elif pos["exit_at"] is None and t - pos["t0"] >= max_hold:
                pos["exit_at"], pos["why"] = t, "max hold"
            if pos["exit_at"] is not None and t >= pos["exit_at"] and not liq:
                reason = pos.get("why", "price gap closed")
            if not reason and t == T - 1:
                reason = "end of test"
            if reason:
                if liq == "long":
                    l1, s1 = pos["l0"] * (1 - liq_move), sc[t]
                elif liq == "short":
                    l1, s1 = lc[t], pos["s0"] * (1 + liq_move)
                else:
                    l1, s1 = lc[t] * (1 - slip), sc[t] * (1 + slip)
                price_pnl = notional * ((l1 - pos["l0"]) / pos["l0"] + (pos["s0"] - s1) / pos["s0"])
                exit_fee = notional * (fees[pos["long"]] + fees[pos["short"]]) / 100
                liq_cost = notional * p["liq_fee"] / 100 if liq else 0.0
                net = pos["funding"] + price_pnl - pos["entry_fee"] - exit_fee - liq_cost
                realised += net
                last_close[pos["coin"]] = t
                trades.append({
                    "Coin": pos["coin"], "Long @": pos["long"], "Short @": pos["short"],
                    "Signal": pd.to_datetime(grid[pos["t_sig"]], unit="ms", utc=True).strftime("%m-%d %H:%M"),
                    "Exit": pd.to_datetime(grid[t], unit="ms", utc=True).strftime("%m-%d %H:%M"),
                    "Minutes": int(t - pos["t0"]),
                    "Gap at signal %": round(float(pos["g_sig"]), 3),
                    "Gap at fill %": round(float(pos["g_fill"]), 3),
                    "Funding $": round(pos["funding"], 2), "Price PnL $": round(price_pnl, 2),
                    "Fees $": round(pos["entry_fee"] + exit_fee + liq_cost, 2),
                    "Net $": round(net, 2), "Exit reason": reason,
                })
            else:
                still.append(pos)
        open_pos = still

        # entries: detected at t, filled at t + delay
        if t in sig_set and len(open_pos) < slots and t + delay < T - 1:
            busy = {x["coin"] for x in open_pos}
            legs = [x["long"] for x in open_pos] + [x["short"] for x in open_pos]
            cands = []
            for r in np.flatnonzero(sig_pos[:, t]):
                cands.append((G[r, t], r, 1))
            for r in np.flatnonzero(sig_neg[:, t]):
                cands.append((-G[r, t], r, -1))
            for g, r, sgn in sorted(cands, key=lambda c: c[0], reverse=True):
                if len(open_pos) >= slots:
                    break
                coin, a, b = pairs[r]
                if coin in busy or t - last_close.get(coin, -10 ** 9) < cooldown:
                    continue
                sex, lex = (a, b) if sgn == 1 else (b, a)
                if legs.count(lex) >= slots or legs.count(sex) >= slots:
                    continue
                tf = t + delay
                lc, sc = px[(coin, lex)][2], px[(coin, sex)][2]
                l0, s0 = lc[tf] * (1 + slip), sc[tf] * (1 - slip)
                open_pos.append({
                    "coin": coin, "long": lex, "short": sex, "row": r, "sgn": sgn,
                    "t_sig": t, "t0": tf, "g_sig": g, "g_fill": sgn * G[r, tf] if not np.isnan(G[r, tf]) else np.nan,
                    "l0": l0, "s0": s0, "funding": 0.0, "exit_at": None,
                    "entry_fee": notional * (fees[lex] + fees[sex]) / 100,
                })
                busy.add(coin)
                legs += [lex, sex]

        unreal = 0.0
        for pos in open_pos:
            if t >= pos["t0"]:
                lc = px[(pos["coin"], pos["long"])][2][t]
                sc = px[(pos["coin"], pos["short"])][2][t]
                unreal += pos["funding"] - pos["entry_fee"] + notional * (
                    (lc - pos["l0"]) / pos["l0"] + (pos["s0"] - sc) / pos["s0"])
        equity[t] = realised + unreal
        # fast-forward through quiet minutes (no positions, no signals)
        if not open_pos:
            nxt = np.searchsorted(any_sig, t + 1)
            if nxt >= len(any_sig):
                equity[t:] = realised
                break
            t2 = int(any_sig[nxt])
            equity[t:t2] = realised
            t = t2
            continue
        t += 1

    tr = pd.DataFrame(trades)
    total_cap = p["capital"] * 2
    days = max((end - start) / DAY_MS, 1)
    eq = pd.Series(equity, index=pd.to_datetime(grid, unit="ms", utc=True))
    eq = eq.resample("15min").last()
    peak = (eq + total_cap).cummax()
    dd = float(((eq + total_cap) / peak - 1).min() * 100)
    net_total = float(tr["Net $"].sum()) if not tr.empty else 0.0
    summary = {
        "Leverage": p["leverage"], "Gap must last (min)": n_need,
        "Notional per leg $": round(notional, 0), "Liq. move per leg %": round(liq_move * 100, 1),
        "Trades": len(tr),
        "Liquidations": int(tr["Exit reason"].str.startswith("LIQ").sum()) if not tr.empty else 0,
        "Price PnL $": round(float(tr["Price PnL $"].sum()), 2) if not tr.empty else 0,
        "Funding $": round(float(tr["Funding $"].sum()), 2) if not tr.empty else 0,
        "Fees $": round(float(tr["Fees $"].sum()), 2) if not tr.empty else 0,
        "Net $": round(net_total, 2),
        "Return % (on 2x capital)": round(net_total / total_cap * 100, 2),
        "Per month %": round(net_total / total_cap * 100 / days * 30, 2),
        "Max drawdown %": round(dd, 2),
        "Win rate %": round(float((tr["Net $"] > 0).mean() * 100), 1) if not tr.empty else 0,
        "Avg minutes held": round(float(tr["Minutes"].mean()), 1) if not tr.empty else 0,
    }
    return {"summary": summary, "trades": tr, "equity": eq, "data": data}


def build_universe(frames, exchanges, min_vol, top_n):
    """Pick coins on 2+ exchanges from a live ticker snapshot (arb_scanner STATE)."""
    rows = []
    for ex in exchanges:
        df = frames.get(ex)
        if df is None or df.empty:
            continue
        d = df[["base", "symbol", "mult", "volume_usdt"]].copy()
        d["ex"] = ex
        rows.append(d.drop_duplicates("base"))
    if not rows:
        return []
    allx = pd.concat(rows)
    out = []
    for coin, g in allx.groupby("base"):
        g = g[g.volume_usdt >= min_vol]
        if len(g) < 2:
            continue
        out.append({"coin": coin, "symbols": dict(zip(g.ex, g.symbol)),
                    "mult": dict(zip(g.ex, g["mult"].astype(int))), "vol": float(g.volume_usdt.min())})
    out.sort(key=lambda u: u["vol"], reverse=True)
    return out[:top_n]


def ms(date_str, end_of_day=False):
    dt = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000) + (24 * HOUR_MS - 1 if end_of_day else 0)
