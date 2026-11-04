#!/usr/bin/env python3
"""Read-only check of everything gap_trader.py needs. Places NO orders.

Run:  .venv/bin/python check_apis.py
"""
import asyncio
import json
import time

import requests
import websockets

import gap_trader as gt

OK, BAD, WARN = "✅", "❌", "⚠️ "
results = []


def report(status, what, detail=""):
    results.append(status)
    print(f"{status} {what}" + (f" — {detail}" if detail else ""))


def main():
    print("\n=== Files ===")
    cfg = gt.load_json("config.json", {}) or {}
    keys = gt.load_json(gt.KEYS_FILE, {}) or {}
    for k in ("api_key", "secret_key"):
        report(OK if cfg.get(k) else BAD, f"config.json {k}", "present" if cfg.get(k) else "missing")
    for k in ("telegram_bot_token", "telegram_chat_id"):
        report(OK if cfg.get(k) else WARN, f"config.json {k}", "present" if cfg.get(k) else "missing (no Telegram)")
    for k in ("bitget_api_key", "bitget_secret_key", "bitget_passphrase"):
        report(OK if keys.get(k) else BAD, f"{gt.KEYS_FILE} {k}", "present" if keys.get(k) else "missing")
    tcfg = gt.load_json(gt.CONFIG_FILE)
    report(OK if tcfg is not None else WARN, gt.CONFIG_FILE,
           "valid JSON" if tcfg is not None else "missing or invalid JSON — defaults will be used")

    print("\n=== Binance ===")
    bn = None
    try:
        bn = gt.BinanceAPI(cfg.get("api_key"), cfg.get("secret_key"))
        b = bn.balance()
        report(OK if b >= 1 else BAD, "Binance futures balance",
               f"{b:,.2f} USDT" + ("" if b >= 1 else " — no USDT in USD-M Futures: transfer some (or wrong account)"))
    except Exception as e:
        report(BAD, "Binance login / futures balance", str(e)[:200])
    if bn:
        try:
            p = bn.c.get_account_api_permissions()
            report(OK if p.get("enableFutures") else BAD, "Binance key: futures trading",
                   "enabled" if p.get("enableFutures") else "NOT enabled — turn on 'Enable Futures'")
            report(OK if not p.get("enableWithdrawals") else WARN, "Binance key: withdrawals",
                   "disabled (good)" if not p.get("enableWithdrawals") else "ENABLED — safer to turn off")
            report(OK if p.get("ipRestrict") else WARN, "Binance key: IP restriction",
                   "on" if p.get("ipRestrict") else "off — recommended to restrict to your server IP")
        except Exception as e:
            report(WARN, "Binance key permissions", f"could not read: {str(e)[:150]}")
        try:
            pos = bn.positions()
            report(OK, "Binance open positions", ", ".join(f"{s} {a:g}" for s, a in pos.items()) or "none")
        except Exception as e:
            report(BAD, "Binance positions", str(e)[:200])
        try:
            mode = bn.c.futures_get_position_mode()
            dual = mode.get("dualSidePosition")
            report(OK if not dual else BAD, "Binance position mode",
                   "one-way (good)" if not dual else "HEDGE mode — switch to One-way in Binance futures settings")
        except Exception as e:
            report(WARN, "Binance position mode", str(e)[:150])

    print("\n=== Bitget ===")
    bg = gt.BitgetAPI(keys.get("bitget_api_key", ""), keys.get("bitget_secret_key", ""),
                      keys.get("bitget_passphrase", ""))
    try:
        b = bg.balance()
        report(OK if b >= 1 else BAD, "Bitget USDT-M futures balance",
               f"{b:,.2f} USDT" + ("" if b >= 1 else " — transfer USDT from Spot to USDT-M Futures"))
    except Exception as e:
        report(BAD, "Bitget login / futures balance", str(e)[:200])
    try:
        pos = bg.positions()
        report(OK, "Bitget open positions", ", ".join(f"{s} {a:g}" for s, a in pos.items()) or "none")
    except Exception as e:
        report(BAD, "Bitget positions (needs futures read permission)", str(e)[:200])
    try:
        info = bg._req("GET", "/api/v2/spot/account/info")
        auth = info.get("authorities") or []
        report(OK, "Bitget key permissions", ", ".join(auth) or "(not listed)")
    except Exception as e:
        report(WARN, "Bitget key permissions", f"could not read: {str(e)[:150]}")
    try:
        d = bg._req("GET", "/api/v2/mix/account/account",
                    {"symbol": "BTCUSDT", "productType": gt.PRODUCT, "marginCoin": "USDT"})
        mode = (d or {}).get("posMode")
        report(OK if mode == "one_way_mode" else WARN, "Bitget position mode",
               "one-way (good)" if mode == "one_way_mode"
               else f"{mode} — the bot switches it to one-way at start (only possible with no open positions)")
    except Exception as e:
        report(WARN, "Bitget position mode", str(e)[:150])

    print("\n=== Live price feeds (5 seconds) ===")

    async def feeds():
        got = {"binance": 0, "bitget": 0}

        async def b():
            async with websockets.connect("wss://fstream.binance.com/stream?streams=btcusdt@bookTicker/"
                                          "ethusdt@bookTicker") as ws:
                end = time.time() + 5
                while time.time() < end:
                    await asyncio.wait_for(ws.recv(), 5)
                    got["binance"] += 1

        async def g():
            async with websockets.connect("wss://ws.bitget.com/v2/ws/public") as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": [
                    {"instType": gt.PRODUCT, "channel": "books1", "instId": s} for s in ("BTCUSDT", "ETHUSDT")]}))
                end = time.time() + 5
                while time.time() < end:
                    m = await asyncio.wait_for(ws.recv(), 5)
                    if '"data"' in m:
                        got["bitget"] += 1

        res = await asyncio.gather(b(), g(), return_exceptions=True)
        for ex, r in zip(("binance", "bitget"), res):
            if isinstance(r, Exception) and not got[ex]:
                report(BAD, f"{ex.title()} websocket", str(r)[:150])
            else:
                report(OK if got[ex] else BAD, f"{ex.title()} websocket", f"{got[ex]} price updates in 5 s")

    asyncio.run(feeds())

    print("\n=== Coins both exchanges list ===")
    try:
        bot = object.__new__(gt.GapTrader)
        bot.cfg = {**gt.DEFAULT_CONFIG, **(tcfg or {})}
        bot.bn = bn
        bot.build_universe()
        report(OK if bot.meta else BAD, "Tradable coins", f"{len(bot.meta)} pass the volume filter")
    except Exception as e:
        report(BAD, "Coin list", str(e)[:200])

    print("\n=== Telegram ===")
    if cfg.get("telegram_bot_token") and cfg.get("telegram_chat_id"):
        try:
            r = requests.post(f"https://api.telegram.org/bot{cfg['telegram_bot_token']}/sendMessage",
                              data={"chat_id": cfg["telegram_chat_id"],
                                    "text": "🔧 Gap trader API check: Telegram works."}, timeout=10).json()
            report(OK if r.get("ok") else BAD, "Telegram test message",
                   "sent — check your Telegram" if r.get("ok") else r.get("description", ""))
        except Exception as e:
            report(BAD, "Telegram", str(e)[:150])

    bad = results.count(BAD)
    print(f"\n{'All good — ready to run gap_trader.py' if not bad else f'{bad} problem(s) to fix (marked ❌)'}\n")


if __name__ == "__main__":
    main()
