#!/usr/bin/env python3
"""Replay the REAL price gaps recorded by arb_scanner.py (arb_snapshots.db).

The scanner stored every scan (about every 2 minutes) of real bid/ask gaps
between Binance, Bybit and Bitget: a row exists when a coin's gap (sell bid on
the dear exchange vs buy ask on the cheap one) was >= 0.1%; no row = below 0.1%.

For each coin and exchange pair it measures, with the gap_trader rules:
  * episodes: the gap reached >= ENTRY%
  * whether it came back to <= EXIT% within MAX_HOLD minutes ("closed")
  * result per episode: entry gap - exit gap - COST  (in % of position size)
and compares the excluded coins (OPENAI, ANTHROPIC, ...) with all the others.
Resolution is the scan interval (~2 min): gaps that lived only seconds are not here.

Result per episode = entry gap - exit gap - COST + funding earned while holding
(funding from the scanner's recorded hedge funding %/day). "widest gap %" is the
largest gap seen while the position was open (how far it went against you).

Usage:
  .venv/bin/python analyze_gaps.py                      # defaults below
  .venv/bin/python analyze_gaps.py --entry 0.8 --exit 0.1 --hold 240 --cost 0.35 --fresh 0.3
  .venv/bin/python analyze_gaps.py --coins OPENAI,ANTHROPIC   # only these
"""
import argparse
import sqlite3
from collections import defaultdict
from datetime import datetime

import pandas as pd

EXCLUDED = ["ANTHROPIC", "OPENAI", "SPACEX", "XAI", "USDC", "FDUSD", "XAU", "XAG", "PAXG"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="arb_snapshots.db")
    ap.add_argument("--entry", type=float, default=0.8, help="open when gap >= this %%")
    ap.add_argument("--exit", type=float, default=0.1, help="closed when gap <= this %%")
    ap.add_argument("--hold", type=float, default=240, help="max hold minutes")
    ap.add_argument("--cost", type=float, default=0.35, help="fees+slippage for 4 fills, %%")
    ap.add_argument("--fresh", type=float, default=None,
                    help="optional: only count gaps that were <= this %% on the previous scan")
    ap.add_argument("--exchanges", default="binance,bitget,bybit")
    ap.add_argument("--coins", default="", help="comma list to limit the report")
    a = ap.parse_args()

    con = sqlite3.connect(a.db)
    scans = [r[0] for r in con.execute("SELECT ts FROM scans ORDER BY ts")]
    df = pd.read_sql_query("SELECT ts, coin, long_ex AS buy, short_ex AS sell, value AS gap, basis AS fund, min_vol "
                           "FROM opps WHERE kind='price'", con)
    con.close()
    if not scans or df.empty:
        print("No price-gap history in", a.db)
        return
    exs = set(a.exchanges.split(","))
    df = df[df.buy.isin(exs) & df.sell.isin(exs)]
    if a.coins:
        df = df[df.coin.isin({c.strip().upper() for c in a.coins.split(",")})]
    idx = {ts: i for i, ts in enumerate(scans)}
    t0 = datetime.fromisoformat(scans[0])
    mins = [(datetime.fromisoformat(s) - t0).total_seconds() / 60 for s in scans]
    print(f"Data: {len(scans)} scans from {scans[0][:16]} to {scans[-1][:16]} UTC "
          f"(~{(mins[-1]) / 60:.0f} h), every ~{mins[-1] / max(len(scans) - 1, 1):.1f} min")
    print(f"Rules: open at gap ≥ {a.entry}%, closed at ≤ {a.exit}%, max hold {a.hold:g} min, "
          f"cost {a.cost}%" + (f", fresh = previous scan ≤ {a.fresh}%" if a.fresh is not None else "") + "\n")

    stats = defaultdict(lambda: {"episodes": 0, "closed": 0, "net": [], "minutes": [], "above_min": 0.0,
                                 "vol": 0.0, "pairs": set(), "worst": [], "funding": []})
    for (coin, buy, sell), g in df.groupby(["coin", "buy", "sell"]):
        gap = [0.0] * len(scans)                    # missing row = gap < 0.1% (treated as 0)
        fund = [None] * len(scans)                  # hedge funding %/day (long buy_ex, short sell_ex)
        for ts, v, f in zip(g.ts, g.gap, g.fund):
            if ts in idx:
                gap[idx[ts]] = v
                fund[idx[ts]] = f
        st = stats[coin]
        st["vol"] = max(st["vol"], float(g.min_vol.max() or 0))
        st["above_min"] += sum(mins[i + 1] - mins[i] for i in range(len(scans) - 1) if gap[i] >= a.entry)
        i = 0
        while i < len(scans):
            fresh_ok = a.fresh is None or i == 0 or gap[i - 1] <= a.fresh
            if gap[i] >= a.entry and fresh_ok:
                st["pairs"].add(f"{buy}->{sell}")
                j = i + 1
                while j < len(scans) and gap[j] > a.exit and mins[j] - mins[i] < a.hold:
                    j += 1
                if j >= len(scans):
                    break                            # still open at the end of the data: not counted
                closed = gap[j] <= a.exit
                # Funding earned (+) / paid (-) while holding, from the recorded %/day.
                last_f, f_sum = 0.0, 0.0
                for k in range(i, j):
                    if fund[k] is not None:
                        last_f = fund[k]
                    f_sum += last_f * (mins[k + 1] - mins[k]) / 1440
                st["episodes"] += 1
                st["closed"] += closed
                st["worst"].append(max(gap[i:j + 1]))
                st["funding"].append(f_sum)
                st["net"].append(gap[i] - gap[j] - a.cost + f_sum)
                st["minutes"].append(mins[j] - mins[i])
                i = j + 1
            else:
                i += 1

    rows = []
    for coin, st in stats.items():
        if not st["episodes"] and not st["above_min"]:
            continue
        n = st["episodes"]
        rows.append({
            "coin": coin, "excluded": "YES" if coin in EXCLUDED else "",
            "episodes": n, "closed %": round(100 * st["closed"] / n) if n else None,
            "avg minutes": round(sum(st["minutes"]) / n) if n else None,
            "avg net %": round(sum(st["net"]) / n, 3) if n else None,
            "avg funding %": round(sum(st["funding"]) / n, 3) if n else None,
            "widest gap %": round(max(st["worst"]), 2) if n else None,
            "sum net %": round(sum(st["net"]), 2),
            f"hours ≥ {a.entry}%": round(st["above_min"] / 60, 1),
            "max 24h vol M": round(st["vol"] / 1e6, 1),
            "pairs": ", ".join(sorted(st["pairs"]))[:60],
        })
    if not rows:
        print("No gaps reached the entry level in this data.")
        return
    out = pd.DataFrame(rows).sort_values(["excluded", "sum net %"], ascending=[False, False])
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(out.to_string(index=False))

    print("\n=== Summary ===")
    for label, part in (("Excluded coins", out[out.excluded == "YES"]), ("All other coins", out[out.excluded == ""])):
        ep = int(part.episodes.sum())
        if not ep:
            print(f"{label}: no episodes")
            continue
        closed = sum(stats[c]["closed"] for c in part.coin)
        nets = [x for c in part.coin for x in stats[c]["net"]]
        print(f"{label}: {ep} episodes · closed within {a.hold:g} min: {100 * closed / ep:.0f}% · "
              f"avg net {sum(nets) / ep:+.3f}% per trade · wins {100 * sum(x > 0 for x in nets) / ep:.0f}%")


if __name__ == "__main__":
    main()
