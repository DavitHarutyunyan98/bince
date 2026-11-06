#!/usr/bin/env python3
"""One-command health check of everything running on the server (read-only).

  .venv/bin/python health_check.py              # status of all programs and data
  .venv/bin/python health_check.py --telegram   # also send a test message to Telegram
"""
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone

import requests

OK, BAD, WARN = "✅", "❌", "⚠️ "
now = time.time()


def ago(ts):
    if not ts:
        return "never"
    s = int(now - ts)
    return f"{s}s ago" if s < 120 else f"{s // 60} min ago" if s < 7200 else f"{s // 3600} h ago"


def line(status, text):
    print(f"{status} {text}")


def process(name):
    out = subprocess.run(["pgrep", "-af", name], capture_output=True, text=True).stdout.strip().splitlines()
    out = [o for o in out if "pgrep" not in o and "health_check" not in o]
    return out


def last_log(path, words, n=1):
    try:
        lines = [ln.rstrip() for ln in open(path, encoding="utf-8", errors="replace") if any(w in ln for w in words)]
        return lines[-n:]
    except FileNotFoundError:
        return []


def main():
    print(f"Health check · {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC\n")

    print("=== Programs ===")
    for name, why in (("gap_trader.py", "trading bot"), ("social_collector.py", "collects posts"),
                      ("signal_engine.py", "learns + signals"), ("arb_scanner.py", "price-gap scanner (optional)"),
                      ("hybrid_trader.py", "directional bot (optional)")):
        p = process(name)
        optional = "optional" in why
        line(OK if p else (WARN if optional else BAD), f"{name:22} {why}: " + (f"running (PID {p[0].split()[0]})" if p else "NOT running"))

    print("\n=== Data collection (social.db) ===")
    if os.path.exists("social.db"):
        con = sqlite3.connect("social.db")
        last = con.execute("SELECT MAX(collected) FROM posts").fetchone()[0]
        line(OK if last and now - last < 900 else BAD, f"last new post saved: {ago(last)}")
        for hours in (1, 24):
            rows = con.execute("SELECT source, COUNT(*) FROM posts WHERE collected >= ? GROUP BY source",
                               (now - hours * 3600,)).fetchall()
            total = sum(n for _, n in rows)
            line(OK if total else BAD, f"new posts in last {hours:2d}h: {total} (" +
                 ", ".join(f"{s}: {n}" for s, n in rows) + ")")
        m = con.execute("SELECT COUNT(*) FROM mentions WHERE created >= ?", (now - 24 * 3600,)).fetchone()[0]
        line(OK if m else WARN, f"coin mentions in last 24h: {m}")
        tot = con.execute("SELECT COUNT(*), MIN(created) FROM posts").fetchone()
        print(f"   total stored: {tot[0]} posts since {datetime.fromtimestamp(tot[1] or now, timezone.utc):%Y-%m-%d}")
        con.close()
    else:
        line(BAD, "social.db not found (collector never ran here?)")
    for ln in last_log("social_collector.log", ["failed", "ERROR"], 3):
        print("   recent problem:", ln)

    print("\n=== Learning (signals.db) ===")
    if os.path.exists("signals.db"):
        con = sqlite3.connect("signals.db")
        for hz in (4, 24):
            r = con.execute("SELECT version, trained_at, n_train, oos FROM models WHERE horizon=? ORDER BY version "
                            "DESC LIMIT 1", (hz,)).fetchone()
            if not r:
                line(WARN, f"{hz}h model: none yet")
                continue
            o = json.loads(r[3] or "{}")
            fresh = now - r[1] < 2 * 3600
            right = o.get("dir_right", o.get("win_rate"))
            line(OK if fresh else BAD, f"{hz}h model v{r[0]} trained {ago(r[1])} on {r[2]} coin-hours · past tests: "
                 + (f"{o.get('events')} tests, {right * 100:.0f}% right, avg {o.get('avg_net'):+.2f}% after costs"
                    if o.get("events") else "none yet"))
        n_models = con.execute("SELECT COUNT(*) FROM models WHERE trained_at >= ?", (now - 24 * 3600,)).fetchone()[0]
        print(f"   versions trained in last 24h: {n_models} (expected ≈ 48: 2 models × 24 hours)")
        rows = con.execute("SELECT status, COUNT(*) FROM signals WHERE ts >= ? GROUP BY status",
                           ((now - 24 * 3600) * 1000,)).fetchall()
        d = dict(rows)
        print(f"   signals last 24h: sent to Telegram {d.get('sent', 0)} · held back {d.get('logged', 0)}"
              " (held back = that model's past tests are not profitable after costs)")
        con.close()
    else:
        line(BAD, "signals.db not found (signal engine never ran here?)")
    for ln in last_log("signal_engine.log", ["hourly run", "signals:"], 2):
        print("   last run:", ln)
    for ln in last_log("signal_engine.log", ["ERROR", "failed", "Traceback"], 3):
        print("   recent problem:", ln)

    print("\n=== Gap trader ===")
    st = {}
    try:
        st = json.load(open("gap_trader_state.json"))
    except Exception:
        pass
    pos = st.get("positions") or {}
    print(f"   open positions: {', '.join(pos) or 'none'} · today: {st.get('day', {}).get('trades', 0)} trades, "
          f"{st.get('day', {}).get('net', 0):+.2f} USDT")
    for ln in last_log("gap_trader.log", ["status:"], 1):
        print("   last status:", ln)

    if "--telegram" in sys.argv:
        print("\n=== Telegram ===")
        try:
            c = json.load(open("config.json"))
            r = requests.post(f"https://api.telegram.org/bot{c['telegram_bot_token']}/sendMessage", timeout=10,
                              data={"chat_id": c["telegram_chat_id"], "text": "🩺 Health check: Telegram works."}).json()
            line(OK if r.get("ok") else BAD, "test message " + ("sent — check Telegram" if r.get("ok") else str(r)))
        except Exception as e:
            line(BAD, f"Telegram failed: {e}")


if __name__ == "__main__":
    main()
