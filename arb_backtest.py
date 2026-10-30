"""Historical backtest of cross-exchange funding carry, price-gap convergence,
or both as independent engines (mode = funding | price | both).

Price engine: enter when the 1h-close price on one exchange is >= gap_entry %
above the other (short the dear one, long the cheap one), exit when the gap is
<= gap_exit % or after price_max_hold_h. Funding engine: the rules below.
"both" runs the two engines INDEPENDENTLY with their own slots; one coin can
only be in one position, each exchange holds at most (total slots) legs, and a
closed coin has a per-engine cooldown. Funding is accrued on every hedge.
NOTE: history has no bid/ask, so gaps come from 1h closes; the slippage input
stands in for the bid/ask spread, and gaps lasting < 1h are not seen.

Strategy simulated (hourly steps):
  * For every coin listed on 2+ exchanges, take each exchange's ACTUALLY PAID
    funding history and turn the last settled rate into a %/day figure.
  * ENTER a hedge (long on the low-funding exchange, short on the high one,
    same notional) when the spread has stayed >= entry %/day for `confirm_h`
    hours and a slot is free (max_pairs).
  * Each settlement: the short leg receives rate x notional, the long leg pays
    rate x notional (negative rates reverse that).
  * EXIT when the spread falls below exit %/day, flips, or max_hold_days pass.
  * Costs: taker fee + slippage on all 4 fills; price PnL of both legs from 1h
    closes (captures entry/exit basis between exchanges).
  * Liquidation check per leg on 1h highs/lows. Cross margin: each leg is
    backed by (capital_per_exchange / max_pairs); isolated: notional/leverage.
    A liquidated leg loses its margin + liquidation fee; the other leg is
    closed at that hour's close.

Decisions only use rates settled BEFORE the step, so there is no look-ahead.
Known biases: universe = coins listed today (delisted coins are missing) and
the volume filter uses today's 24h volume.
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


# ------------------------------------------------------------- simulation --

def _daily_rate_series(fund, grid):
    """Hourly forward-filled %/day from the last SETTLED rate, plus the rate
    paid in each hour bucket (fraction, 0 when no settlement)."""
    if fund is None or fund.empty:
        return None, None
    f = fund.drop_duplicates("ts").sort_values("ts").copy()
    gaps = f.ts.diff().fillna(8 * HOUR_MS) / HOUR_MS
    f["interval_h"] = gaps.clip(lower=1, upper=8).round()
    f["daily"] = f.rate * 100 * 24 / f.interval_h
    f["bucket"] = (f.ts // HOUR_MS) * HOUR_MS
    paid = f.groupby("bucket").rate.sum().reindex(grid, fill_value=0.0)
    # Known at the END of the settlement hour -> usable from the next step on.
    daily = f.groupby("bucket").daily.last().reindex(grid).ffill().shift(1)
    return daily, paid


def run_backtest(universe, params, progress=None):
    """universe: list of dicts {coin, symbols: {ex: symbol}, mult: {ex: m}}.
    params: start_ms, end_ms, capital, leverage, usage, max_pairs, entry, exit,
            confirm_h, max_hold_days, fees{ex:%}, slippage, cross, mmr, liq_fee,
            mode (funding|price|both), gap_entry, gap_exit, price_slots, funding_slots,
            cooldown_price_h, cooldown_funding_h, price_max_hold_h.
    Returns dict with summary, trades (DataFrame), equity (Series)."""
    p = params
    start, end = int(p["start_ms"]), int(p["end_ms"])
    grid = np.arange((start // HOUR_MS + 1) * HOUR_MS, end, HOUR_MS)
    say = progress or (lambda *_: None)

    # 1) funding for every leg
    series = {}   # (coin, ex) -> (daily, paid)
    n = len(universe)
    for i, u in enumerate(universe):
        say(f"Funding history {i + 1}/{n}: {u['coin']}")
        for ex, sym in u["symbols"].items():
            try:
                d, pd_ = _daily_rate_series(get_funding(ex, sym, start - 2 * 24 * HOUR_MS, end), grid)
            except Exception:
                d = None
            if d is not None and d.notna().any():
                series[(u["coin"], ex)] = (d.values, pd_.values)

    # 2) candidate pairs
    pairs = []
    for u in universe:
        exs = sorted(ex for ex in u["symbols"] if (u["coin"], ex) in series)
        for a in range(len(exs)):
            for b in range(a + 1, len(exs)):
                pairs.append((u["coin"], exs[a], exs[b], u))
    if not pairs:
        return {"error": "No funding history could be loaded for 2+ exchanges."}

    mode = p.get("mode", "both")             # funding | price | both (independent engines)
    if mode == "both":
        slots = {"price": int(p.get("price_slots", 1)), "funding": int(p.get("funding_slots", 1))}
    else:
        slots = {"price": 0, "funding": 0}
        slots[mode] = int(p["max_pairs"])
    total_slots = max(sum(slots.values()), 1)
    p = {**p, "max_pairs": total_slots}
    cooldown = {"price": float(p.get("cooldown_price_h", 0.5)), "funding": float(p.get("cooldown_funding_h", 4))}
    max_hold_h = {"price": float(p.get("price_max_hold_h", 24)), "funding": float(p["max_hold_days"]) * 24}

    notional = p["capital"] * p["usage"] * p["leverage"] / p["max_pairs"]
    if p["cross"]:
        margin = p["capital"] / p["max_pairs"]      # balance share backing each leg
    else:
        margin = notional / p["leverage"]
    liq_move = max(margin / notional - p["mmr"] / 100.0, 0.001)
    confirm = max(int(p["confirm_h"]), 1)
    fees, slip = p["fees"], p["slippage"] / 100.0

    # oriented spread arrays: spread[k] = daily_a - daily_b (NaN-safe)
    sp = {}
    for coin, a, b, _u in pairs:
        da, db = series[(coin, a)][0], series[(coin, b)][0]
        sp[(coin, a, b)] = da - db

    kl_cache = {}

    def closes(coin, ex, u):
        key = (coin, ex)
        if key not in kl_cache:
            try:
                k = get_klines(ex, u["symbols"][ex], start, end)
                k = k.drop_duplicates("ts").set_index("ts").reindex(grid).ffill()
                m = u["mult"].get(ex, 1) or 1
                kl_cache[key] = (k.high.values / m, k.low.values / m, k.close.values / m)
            except Exception:
                kl_cache[key] = None
        return kl_cache[key]

    gap_entry, gap_exit = p.get("gap_entry", 0.5), p.get("gap_exit", 0.1)
    min_settle = int(p.get("min_settlements", 1))      # funding: hold through N payments
    exit_confirm = max(int(p.get("exit_confirm_h", 4)), 1)
    max_move = float(p.get("max_1h_move", 8.0)) / 100
    gap_confirm = max(int(p.get("gap_confirm_h", 2)), 1)    # gap must show on N hourly closes in a row
    max_gap = float(p.get("max_gap", 3.0))                 # bigger = data glitch / different token
    max_spread = float(p.get("max_entry_spread", 1.5))     # %/day; above = short-lived spike
    max_adverse = float(p.get("max_adverse_basis", 0.2))   # % price gap against the hedge at entry   # skip coins that jumped this much in 1h

    def too_wild(k, t):
        """True if this leg moved more than max_move in any hour of the last 24h."""
        if not max_move:
            return False
        c = k[2][max(t - 24, 0):t + 1]
        if len(c) < 2:
            return False
        with np.errstate(invalid="ignore", divide="ignore"):
            mv = np.nanmax(np.abs(c[1:] / c[:-1] - 1))
        return bool(mv > max_move)
    # Price basis per pair from 1h closes: (close_a - close_b) / close_b in %.
    basis = {}
    if slots["price"] or p.get("max_adverse_basis", 0.2) is not None:
        for i, (coin, a, b, u) in enumerate(pairs):
            say(f"Price history {i + 1}/{len(pairs)}: {coin} {a}/{b}")
            ka, kb = closes(coin, a, u), closes(coin, b, u)
            if ka is not None and kb is not None:
                basis[(coin, a, b)] = (ka[2] - kb[2]) / kb[2] * 100

    def oriented(coin, lex, sex, t):
        """(funding spread short-long %/day, price basis short-long %) at step t."""
        a, b = sorted((lex, sex))
        sgn = 1 if a == sex else -1
        fs = sgn * sp[(coin, a, b)][t]
        bs = basis.get((coin, a, b))
        return fs, (sgn * bs[t] if bs is not None else np.nan)

    open_pos, trades = [], []
    last_close = {}                           # (engine, coin) -> step closed
    equity = np.zeros(len(grid))
    realised = 0.0
    say("Simulating…")
    for t in range(len(grid)):
        # --- accrue funding + check liquidation / exit for open positions
        still = []
        for pos in open_pos:
            coin, lex, sex, u, eng = pos["coin"], pos["long"], pos["short"], pos["u"], pos["engine"]
            # Funding is real money on every hedge, whichever engine opened it.
            paid_s, paid_l = series[(coin, sex)][1][t], series[(coin, lex)][1][t]
            pos["funding"] += notional * (paid_s - paid_l)
            if paid_s != 0 or paid_l != 0:
                pos["settlements"] += 1
            lk, sk = pos["lk"], pos["sk"]
            liq = None
            if lk[1][t] <= pos["l0"] * (1 - liq_move):
                liq = "long"
            elif sk[0][t] >= pos["s0"] * (1 + liq_move):
                liq = "short"
            s_now, b_now = oriented(coin, lex, sex, t)
            if eng == "price":
                done = np.isnan(b_now) or b_now <= gap_exit
            else:
                # Exit only after the spread stays below Exit for exit_confirm hours AND
                # at least min_settle funding payments were collected (no 1-hour flips).
                below = np.isnan(s_now) or s_now < p["exit"]
                pos["below_h"] = pos["below_h"] + 1 if below else 0
                done = pos["below_h"] >= exit_confirm and pos["settlements"] >= min_settle
            held_h = t - pos["t0"]
            reason = None
            if liq:
                reason = f"LIQUIDATED {liq}"
            elif done:
                reason = "price gap closed" if eng == "price" else "funding spread below exit"
            elif held_h >= max_hold_h[eng]:
                reason = "max hold"
            elif t == len(grid) - 1:
                reason = "end of test"
            if reason:
                l1, s1 = lk[2][t], sk[2][t]
                if liq == "long":
                    l1 = pos["l0"] * (1 - liq_move)
                if liq == "short":
                    s1 = pos["s0"] * (1 + liq_move)
                price_pnl = notional * ((l1 - pos["l0"]) / pos["l0"] + (pos["s0"] - s1) / pos["s0"])
                exit_fee = notional * ((fees[lex] + fees[sex]) / 100 + 2 * slip)
                liq_cost = notional * p["liq_fee"] / 100 if liq else 0.0
                net = pos["funding"] + price_pnl - pos["entry_fee"] - exit_fee - liq_cost
                realised += net
                last_close[(eng, coin)] = t
                trades.append({
                    "Engine": eng, "Coin": coin, "Long @": lex, "Short @": sex,
                    "Entry": pd.to_datetime(grid[pos["t0"]], unit="ms", utc=True).strftime("%Y-%m-%d %H:%M"),
                    "Exit": pd.to_datetime(grid[t], unit="ms", utc=True).strftime("%Y-%m-%d %H:%M"),
                    "Hours": held_h,
                    "Entry gap %": None if np.isnan(pos["g_entry"]) else round(pos["g_entry"], 4),
                    "Entry funding %/day": None if np.isnan(pos["f_entry"]) else round(pos["f_entry"], 4),
                    "Funding $": round(pos["funding"], 2), "Price PnL $": round(price_pnl, 2),
                    "Fees $": round(pos["entry_fee"] + exit_fee + liq_cost, 2),
                    "Net $": round(net, 2), "Exit reason": reason,
                })
            else:
                still.append(pos)
        open_pos = still

        # --- entries: each engine fills ONLY its own slots, independently
        if t >= confirm and t < len(grid) - 1:
            for eng in ("price", "funding"):
                if sum(1 for x in open_pos if x["engine"] == eng) >= slots[eng]:
                    continue
                busy = {x["coin"] for x in open_pos}          # one coin = one position
                cands = []
                for (coin, a, b), arr in sp.items():
                    if coin in busy:
                        continue
                    lc = last_close.get((eng, coin))
                    if lc is not None and t - lc < cooldown[eng]:
                        continue
                    bs = basis.get((coin, a, b))
                    w = arr[t - confirm + 1:t + 1]
                    for sgn, lex, sex in ((1, b, a), (-1, a, b)):   # sgn=1: short a, long b
                        if eng == "price":
                            if bs is None:
                                continue
                            gw = sgn * bs[max(t - gap_confirm + 1, 0):t + 1]
                            gap = gw[-1]
                            if (len(gw) == gap_confirm and not np.isnan(gw).any()
                                    and gw.min() >= gap_entry and (not max_gap or gw.max() <= max_gap)):
                                cands.append((gap, coin, lex, sex))
                        elif not np.isnan(w).any() and (sgn * w).min() >= p["entry"]:
                            if max_spread and sgn * arr[t] > max_spread:
                                continue                      # spike: rarely lasts
                            gap = sgn * bs[t] if bs is not None else 0.0
                            if not np.isnan(gap) and gap < -max_adverse:
                                continue                      # would start with a price loss
                            cands.append((sgn * arr[t], coin, lex, sex))
                for score, coin, lex, sex in sorted(cands, reverse=True):
                    if sum(1 for x in open_pos if x["engine"] == eng) >= slots[eng]:
                        break
                    if coin in busy:
                        continue
                    # Each exchange can host at most `total_slots` legs (margin limit).
                    legs = [x["long"] for x in open_pos] + [x["short"] for x in open_pos]
                    if legs.count(lex) >= total_slots or legs.count(sex) >= total_slots:
                        continue
                    u = next(x for x in universe if x["coin"] == coin)
                    lk, sk = closes(coin, lex, u), closes(coin, sex, u)
                    if lk is None or sk is None or np.isnan(lk[2][t]) or np.isnan(sk[2][t]):
                        continue
                    if too_wild(lk, t) or too_wild(sk, t):
                        continue
                    f0, g0 = oriented(coin, lex, sex, t)
                    open_pos.append({
                        "engine": eng, "coin": coin, "long": lex, "short": sex, "u": u, "t0": t,
                        "f_entry": f0, "g_entry": g0,
                        "lk": lk, "sk": sk, "l0": lk[2][t] * (1 + slip), "s0": sk[2][t] * (1 - slip),
                        "funding": 0.0, "settlements": 0, "below_h": 0,
                        "entry_fee": notional * (fees[lex] + fees[sex]) / 100,
                    })
                    busy.add(coin)

        # --- mark to market (funding accrued so far; price legs roughly offset)
        unreal = 0.0
        for pos in open_pos:
            lk, sk = pos["lk"], pos["sk"]
            pp = notional * ((lk[2][t] - pos["l0"]) / pos["l0"] + (pos["s0"] - sk[2][t]) / pos["s0"])
            unreal += pos["funding"] + (0 if np.isnan(pp) else pp) - pos["entry_fee"]
        equity[t] = realised + unreal

    tr = pd.DataFrame(trades)
    total_cap = p["capital"] * 2
    days = max((end - start) / (24 * HOUR_MS), 1)
    eq = pd.Series(equity, index=pd.to_datetime(grid, unit="ms", utc=True))
    peak = (eq + total_cap).cummax()
    dd = ((eq + total_cap) / peak - 1).min() * 100
    net_total = float(tr["Net $"].sum()) if not tr.empty else 0.0
    summary = {
        "Mode": mode, "Leverage": p["leverage"],
        "Price trades": int((tr["Engine"] == "price").sum()) if not tr.empty else 0,
        "Price net $": round(float(tr.loc[tr["Engine"] == "price", "Net $"].sum()), 2) if not tr.empty else 0,
        "Funding trades": int((tr["Engine"] == "funding").sum()) if not tr.empty else 0,
        "Funding net $": round(float(tr.loc[tr["Engine"] == "funding", "Net $"].sum()), 2) if not tr.empty else 0, "Notional per leg $": round(notional, 0),
        "Liq. move per leg %": round(liq_move * 100, 1),
        "Trades": len(tr), "Liquidations": int(tr["Exit reason"].str.startswith("LIQ").sum()) if not tr.empty else 0,
        "Funding $": round(float(tr["Funding $"].sum()), 2) if not tr.empty else 0,
        "Price PnL $": round(float(tr["Price PnL $"].sum()), 2) if not tr.empty else 0,
        "Fees $": round(float(tr["Fees $"].sum()), 2) if not tr.empty else 0,
        "Net $": round(net_total, 2),
        "Return % (on 2x capital)": round(net_total / total_cap * 100, 2),
        "Per month %": round(net_total / total_cap * 100 / days * 30, 2),
        "Max drawdown %": round(float(dd), 2),
        "Win rate %": round(float((tr["Net $"] > 0).mean() * 100), 1) if not tr.empty else 0,
    }
    return {"summary": summary, "trades": tr, "equity": eq}


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
