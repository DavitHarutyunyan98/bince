#!/usr/bin/env python3
"""Event study on the collected social data (social.db): what did prices do AFTER posts?

Three views:
  1) SPIKES   a coin's mentions in one hour jump to >= --min-mentions and >= --spike x its
              normal hourly level (normal = its average before that hour). One event per coin per 6 h.
  2) POSTS    every single post mentioning a coin, grouped by source and by tone
              (positive > 0.3, negative < -0.3, neutral in between).
  3) BINANCE  Binance announcements (listings / delistings / news) for coins already trading.

For each event: price change from the next 5-minute candle after the event to +1h, +2h, +4h,
+8h, +24h. "excess" = that change minus the coin's AVERAGE change over the same horizon during
the whole period (so a coin that rose all week does not look like a signal). Only events whose
horizon has fully passed are counted. Prices: Binance futures, else Bybit, else Bitget (5m).

Usage:
  .venv/bin/python event_study.py                    # all views, defaults
  .venv/bin/python event_study.py --spike 3 --min-mentions 3
  .venv/bin/python event_study.py --exclude BTC,ETH,USDC   # leave out big/stable coins
"""
import argparse
import sqlite3
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

HORIZONS = [1, 2, 4, 8, 24]          # hours
STABLES = {"USDC", "USDT", "FDUSD", "DAI", "USDE", "RLUSD", "PYUSD", "TUSD", "USD1", "PAXG", "XAUT"}
_cache = {}


def _binance(sym, start, end):
    rows, cur = [], start
    while cur < end:
        j = requests.get("https://fapi.binance.com/fapi/v1/klines", timeout=15,
                         params={"symbol": sym, "interval": "5m", "startTime": cur, "limit": 1500}).json()
        if not isinstance(j, list) or not j:
            break
        rows += [(int(k[0]), float(k[4])) for k in j]
        cur = int(j[-1][0]) + 300_000
        if len(j) < 1500:
            break
        time.sleep(0.1)
    return rows


def _bybit(sym, start, end):
    rows, cur_end = [], end
    while cur_end > start:      # Bybit returns newest first: page backwards
        j = requests.get("https://api.bybit.com/v5/market/kline", timeout=15, params={
            "category": "linear", "symbol": sym, "interval": "5", "start": start, "end": cur_end,
            "limit": 1000}).json().get("result", {}).get("list") or []
        if not j:
            break
        rows += [(int(k[0]), float(k[4])) for k in j]
        oldest = min(int(k[0]) for k in j)
        if len(j) < 1000 or oldest <= start:
            break
        cur_end = oldest - 1
        time.sleep(0.1)
    return rows


def _bitget(sym, start, end):
    rows, cur = [], start
    while cur < end:
        j = requests.get("https://api.bitget.com/api/v2/mix/market/candles", timeout=15, params={
            "symbol": sym, "productType": "USDT-FUTURES", "granularity": "5m", "startTime": cur,
            "endTime": min(cur + 999 * 300_000, end), "limit": 1000}).json().get("data") or []
        if j:
            rows += [(int(k[0]), float(k[4])) for k in j]
        cur += 999 * 300_000
        time.sleep(0.1)
    return rows


def klines(coin, start_ms, end_ms):
    """5-minute closes for COINUSDT as a pandas Series indexed by ms (Binance -> Bybit -> Bitget)."""
    key = (coin, start_ms // 3_600_000, end_ms // 3_600_000)
    if key in _cache:
        return _cache[key]
    out = None
    for fetch in (_binance, _bybit, _bitget):
        try:
            rows = fetch(coin + "USDT", start_ms, end_ms)
        except Exception:
            rows = []
        if len(rows) > 12:
            s = pd.Series(dict(rows)).sort_index()
            out = s[(s.index >= start_ms) & (s.index <= end_ms)]
            break
    _cache[key] = out
    return out


def forward(series, t_ms):
    """% change from the first candle after t to t+h for each horizon (None if not yet passed)."""
    idx = series.index.values
    i = np.searchsorted(idx, t_ms + 1)
    if i >= len(idx):
        return None
    p0 = series.iloc[i]
    res = {}
    for h in HORIZONS:
        j = np.searchsorted(idx, idx[i] + h * 3_600_000)
        res[h] = (series.iloc[j] / p0 - 1) * 100 if j < len(idx) and idx[j] - idx[i] <= h * 3_600_000 + 600_000 \
            else None
    return res


def baseline(series):
    """Average % change over each horizon across the whole series (the coin's drift)."""
    out = {}
    for h in HORIZONS:
        step = h * 12
        if len(series) > step:
            v = series.values
            out[h] = float(np.mean(v[step:] / v[:-step] - 1) * 100)
        else:
            out[h] = 0.0
    return out


def summarize(title, events):
    """events: list of dicts with 'ret' {h: %} and 'base' {h: %}."""
    print(f"\n{title}")
    if not events:
        print("  (no events with enough price history yet)")
        return
    print(f"  {'horizon':>8} {'events':>7} {'avg move %':>11} {'excess %':>9} {'median ex %':>12} {'% up':>6}")
    for h in HORIZONS:
        vals = [(e["ret"][h], e["ret"][h] - e["base"][h]) for e in events if e["ret"] and e["ret"].get(h) is not None]
        if not vals:
            continue
        r = np.array([v[0] for v in vals])
        ex = np.array([v[1] for v in vals])
        print(f"  {str(h) + 'h':>8} {len(vals):7d} {r.mean():11.2f} {ex.mean():9.2f} {np.median(ex):12.2f} "
              f"{(r > 0).mean() * 100:6.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="social.db")
    ap.add_argument("--spike", type=float, default=3.0, help="mentions >= this x normal hourly level")
    ap.add_argument("--min-mentions", type=int, default=3, help="and at least this many in the hour")
    ap.add_argument("--exclude", default="", help="comma list of coins to leave out")
    ap.add_argument("--max-coins", type=int, default=60, help="price downloads are limited to this many coins")
    a = ap.parse_args()
    excl = STABLES | {c.strip().upper() for c in a.exclude.split(",") if c.strip()}

    con = sqlite3.connect(a.db)
    m = pd.read_sql_query("SELECT m.coin, m.created, m.source, p.sentiment, p.title, p.channel "
                          "FROM mentions m JOIN posts p ON p.uid = m.uid", con)
    con.close()
    m = m[~m.coin.isin(excl)]
    if m.empty:
        print("No mentions yet.")
        return
    now_ms = int(time.time() * 1000)
    t0 = int(m.created.min()) * 1000
    print(f"Mentions: {len(m)} of {m.coin.nunique()} coins, from "
          f"{pd.to_datetime(t0, unit='ms'):%Y-%m-%d %H:%M} to now (UTC). Excluded: {', '.join(sorted(excl))}")

    # Coins to price: the most mentioned ones (downloads take time).
    top = m.coin.value_counts().head(a.max_coins).index.tolist()
    m = m[m.coin.isin(top)]
    prices, bases = {}, {}
    for i, coin in enumerate(top, 1):
        print(f"\rprices {i}/{len(top)}: {coin:12}", end="", flush=True)
        first = int(m[m.coin == coin].created.min()) * 1000
        s = klines(coin, max(first - 6 * 3_600_000, now_ms - 30 * 86_400_000), now_ms)
        if s is not None:
            prices[coin], bases[coin] = s, baseline(s)
    print(f"\rprices: {len(prices)} of {len(top)} coins found on Binance/Bybit/Bitget        ")

    # ---- 1) spikes
    m["hour"] = (m.created // 3600) * 3600
    spikes = []
    for coin, g in m.groupby("coin"):
        if coin not in prices:
            continue
        per_h = g.groupby("hour").size().sort_index()
        last_ev = -1e18
        for hour, n in per_h.items():
            prev = per_h[per_h.index < hour]
            span_h = max((hour - int(g.created.min())) / 3600, 1)
            normal = prev.sum() / span_h                     # average mentions per hour before this hour
            if n >= a.min_mentions and (normal == 0 or n >= a.spike * normal) and hour - last_ev >= 6 * 3600:
                last_ev = hour
                tone = g[g.hour == hour].sentiment.mean()
                spikes.append({"coin": coin, "t": (hour + 3600) * 1000, "n": int(n), "tone": tone,
                               "ret": forward(prices[coin], (hour + 3600) * 1000), "base": bases[coin]})
    summarize(f"1) MENTION SPIKES (≥ {a.min_mentions} mentions in an hour and ≥ {a.spike:g}× normal; "
              f"price measured from the END of that hour)", spikes)
    pos = [e for e in spikes if e["tone"] is not None and e["tone"] > 0.3]
    neg = [e for e in spikes if e["tone"] is not None and e["tone"] < -0.3]
    summarize("   … spikes with POSITIVE tone (> 0.3)", pos)
    summarize("   … spikes with NEGATIVE tone (< -0.3)", neg)
    if spikes:
        print("\n   latest spikes:")
        for e in sorted(spikes, key=lambda e: -e["t"])[:12]:
            r = e["ret"] or {}
            print(f"   {pd.to_datetime(e['t'], unit='ms'):%m-%d %H:%M} {e['coin']:10} {e['n']:3d} mentions "
                  f"tone {e['tone'] if e['tone'] == e['tone'] else 0:+.2f} · "
                  + " ".join(f"{h}h {r[h]:+.2f}%" if r.get(h) is not None else f"{h}h …" for h in HORIZONS))

    # ---- 2) every post, by source and tone
    posts = defaultdict(list)
    for row in m.itertuples():
        if row.coin not in prices:
            continue
        ret = forward(prices[row.coin], int(row.created) * 1000)
        e = {"ret": ret, "base": bases[row.coin]}
        posts[f"source: {row.source}"].append(e)
        tone = row.sentiment
        if tone is not None and tone == tone:
            posts["tone: positive" if tone > 0.3 else "tone: negative" if tone < -0.3 else "tone: neutral"].append(e)
    for k in sorted(posts):
        summarize(f"2) EVERY POST — {k}", posts[k])

    # ---- 3) Binance announcements
    b = m[m.source == "binance"]
    for kind in ("new listings", "delistings", "latest news"):
        ev = []
        for row in b[b.channel == kind].itertuples():
            if row.coin in prices:
                ev.append({"ret": forward(prices[row.coin], int(row.created) * 1000), "base": bases[row.coin]})
        summarize(f"3) BINANCE ANNOUNCEMENTS — {kind}", ev)

    print("\nHow to read: 'excess %' > 0 means the price did better than that coin's normal drift after the "
          "event; '% up' is how often it rose. Few events = not reliable yet; it improves every day.")


if __name__ == "__main__":
    main()
