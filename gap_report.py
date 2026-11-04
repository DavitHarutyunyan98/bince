#!/usr/bin/env python3
"""One report of the gap trader for review: settings, bot state, REAL exchange
positions and balances, and the important log lines. Contains no keys/secrets.

Run:  .venv/bin/python gap_report.py            (prints the report)
      .venv/bin/python gap_report.py > report.txt
"""
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone

import gap_trader as gt

LOG = "gap_trader.log"


def section(title):
    print(f"\n===== {title} =====")


def main():
    print(f"Gap trader report · {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC")

    section("PROCESS")
    out = subprocess.run(["pgrep", "-af", "gap_trader.py"], capture_output=True, text=True).stdout.strip()
    print(out or "NOT RUNNING")
    for f in (gt.STOP_FILE, gt.CLOSE_ALL_FILE):
        print(f"{f}: {'PRESENT' if os.path.exists(f) else 'not present'}")

    section("SETTINGS (keys never shown)")
    cfg = gt.load_json(gt.CONFIG_FILE, {}) or {}
    print(json.dumps(cfg, indent=1))

    section("BOT STATE (what the bot thinks is open)")
    st = gt.load_json(gt.STATE_FILE, {}) or {}
    now = time.time()
    for coin, p in (st.get("positions") or {}).items():
        print(f"{coin}: long {p['long']} @ {p['long_px']:.8g} / short {p['short']} @ {p['short_px']:.8g} · "
              f"qty {p['qty']:g} · captured {p['captured_gap']:.3f}% (signal {p['signal_gap']:.3f}%) · "
              f"open fees {p['open_fees']:.4f} · open for {gt.dur(now - p['opened'])}")
    if not st.get("positions"):
        print("no open positions")
    print("today:", json.dumps(st.get("day", {})))

    section("REAL POSITIONS & BALANCES ON THE EXCHANGES")
    try:
        bot = gt.GapTrader()
        ours = {p["symbol"]: p for p in (st.get("positions") or {}).values()}
        for ex in bot.exchanges:
            api = bot.api[ex]
            try:
                bal = api.balance()
            except Exception as e:
                bal = f"error: {str(e)[:80]}"
            try:
                pos = api.positions()
            except Exception as e:
                pos = {}
                print(f"{gt.NAMES[ex]} positions error: {str(e)[:100]}")
            print(f"{gt.NAMES[ex]}: balance {bal if isinstance(bal, str) else f'{bal:,.2f} USDT'} · "
                  f"positions: " + (", ".join(f"{s} {a:+g}" for s, a in pos.items()) or "none"))
        # Cross-check: every bot position must exist on both exchanges with the right side and size.
        for sym, p in ours.items():
            la = bot.api[p["long"]].position(sym)
            sa = bot.api[p["short"]].position(sym)
            ok = la >= p["qty"] * 0.99 and -sa >= p["qty"] * 0.99
            print(f"CHECK {p['coin']}: {gt.NAMES[p['long']]} long {la:+g}, {gt.NAMES[p['short']]} short {sa:+g} "
                  f"(expected ±{p['qty']:g}) -> {'OK' if ok else 'MISMATCH!'}")
    except SystemExit as e:
        print(f"could not log in: {e}")
    except Exception as e:
        print(f"exchange check failed: {str(e)[:200]}")

    section("TRADES, ERRORS AND WARNINGS (last 60)")
    keep = re.compile(r"OPEN |CLOSED|LOST LEG|depth ok|ERROR|WARNING|failed|Universe|TELEGRAM")
    try:
        lines = [ln.rstrip() for ln in open(LOG, encoding="utf-8", errors="replace") if keep.search(ln)]
        print("\n".join(lines[-60:]) or "(none)")
    except FileNotFoundError:
        print("no log file")

    section("GAP REPORTS (last 40 lines)")
    try:
        lines = [ln.rstrip() for ln in open(LOG, encoding="utf-8", errors="replace") if "HOUR" in ln]
        print("\n".join(lines[-40:]) or "(none yet — create REPORT_GAPS to get one now)")
    except FileNotFoundError:
        pass

    section("LAST 15 LOG LINES")
    try:
        print("".join(open(LOG, encoding="utf-8", errors="replace").readlines()[-15:]).rstrip())
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
