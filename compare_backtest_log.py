#!/usr/bin/env python3
"""Compare live-trader log trades against a backtest over the same period.

Run this on a host that can reach Binance (the sandbox used to develop it
cannot). It:

  1. Parses opens/closes (symbol, side, price, time, realized PnL) from a
     hybrid_trader log file.
  2. For each symbol in the trade config, fetches 15m futures klines covering
     the log window (plus warm-up), runs the exact same strategy + Backtester
     the dashboard uses, and lists the backtest trades.
  3. Prints a side-by-side summary so you can see where live and backtest agree
     or diverge.

Usage:
    python3 compare_backtest_log.py <log_file> [trade_config.json]
"""
import sys
import re
import json
import collections
from datetime import datetime, timezone, timedelta

import pandas as pd
from binance.client import Client
from strategy_utils import STRATEGY_REGISTRY, Backtester

OPEN_RE = re.compile(
    r'([\d\-: ,]+) - INFO - \[(\w+)\] .*?(LONG|SHORT) position opened successfully at ([\d.]+)')
CLOSE_RE = re.compile(
    r'([\d\-: ,]+) - INFO - \[(\w+)\] (LONG|SHORT) position closed\. PnL: ([+\-][\d.]+)')

INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
               "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def parse_log(path):
    """Return {symbol: {'opens': [...], 'closes': [...]}} from a log file."""
    trades = collections.defaultdict(lambda: {"opens": [], "closes": []})
    window = [None, None]
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = OPEN_RE.search(ln)
        if m:
            ts = m.group(1).strip()
            trades[m.group(2)]["opens"].append(
                {"time": ts, "side": m.group(3), "price": float(m.group(4))})
            window[0] = window[0] or ts
            window[1] = ts
        m = CLOSE_RE.search(ln)
        if m:
            ts = m.group(1).strip()
            trades[m.group(2)]["closes"].append(
                {"time": ts, "side": m.group(3), "pnl": float(m.group(4))})
            window[1] = ts
    return trades, window


def fetch_klines(client, symbol, interval, start_ms, end_ms):
    rows = []
    cur = start_ms
    step = INTERVAL_MS.get(interval, 900_000)
    while cur < end_ms:
        batch = client.futures_klines(symbol=symbol, interval=interval,
                                      startTime=cur, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + step
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows, columns=[
        "OpenTime", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "qav", "trades", "tbav", "tqav", "ignore"])
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
    return df


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    log_path = sys.argv[1]
    cfg_path = sys.argv[2] if len(sys.argv) > 2 else "trade_config.json"

    configs = json.load(open(cfg_path))
    if isinstance(configs, dict):
        configs = [configs]
    log_trades, window = parse_log(log_path)
    print(f"Log window: {window[0]}  ->  {window[1]}\n")

    fmt = "%Y-%m-%d %H:%M:%S,%f"
    start_dt = datetime.strptime(window[0], fmt).replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(window[1], fmt).replace(tzinfo=timezone.utc)
    # Warm up two days before the window so signal windows are populated.
    warm_start_ms = int((start_dt - timedelta(days=2)).timestamp() * 1000)
    end_ms = int((end_dt + timedelta(hours=1)).timestamp() * 1000)

    client = Client()
    for cfg in configs:
        if not cfg.get("enabled", True):
            continue
        sym = cfg["symbol"]
        strat_cls = STRATEGY_REGISTRY.get(cfg["strategy_name"])
        if not strat_cls:
            print(f"[{sym}] unknown strategy {cfg['strategy_name']!r}, skip")
            continue
        interval = cfg.get("bar_length", "15m")
        df = fetch_klines(client, sym, interval, warm_start_ms, end_ms)
        if df.empty:
            print(f"[{sym}] no klines, skip\n")
            continue

        sig = strat_cls().generate_signals(df, cfg)
        # Only evaluate trades inside the log window (warm-up is context only).
        sig_window = sig[sig.index >= start_dt]
        bt = Backtester(initial_capital=cfg.get("units_usdt", 100.0),
                        fee_percent=0.05,
                        sizing_mode=cfg.get("sizing_mode", "fixed"))
        trades_df, _ = bt.run_backtest(sig_window)
        n_bt = 0 if trades_df is None else len(trades_df)
        bt_pnl = 0.0 if trades_df is None else trades_df["PnL"].sum()

        lg = log_trades.get(sym, {"opens": [], "closes": []})
        lg_pnl = sum(c["pnl"] for c in lg["closes"])
        print(f"=== {sym} ===")
        print(f"  LIVE log : {len(lg['opens'])} opens, {len(lg['closes'])} closes, "
              f"realized PnL {lg_pnl:+.2f}")
        print(f"  BACKTEST : {n_bt} trades, PnL {bt_pnl:+.2f} "
              f"(base {cfg.get('units_usdt')}, {cfg.get('sizing_mode')})")
        if trades_df is not None and not trades_df.empty:
            for _, t in trades_df.iterrows():
                print(f"      {t['Entry_Date']}  {t['Position']:5}  "
                      f"in {t['Entry_Price']:.6g} out {t['Exit_Price']:.6g}  "
                      f"PnL {t['PnL']:+.3f}  ({t['Exit_Reason']})")
        print()


if __name__ == "__main__":
    main()
