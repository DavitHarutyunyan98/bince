#!/usr/bin/env python3
"""Live cross-exchange price-gap trader: Binance + Bitget + Bybit USDT perpetuals
(any pair of the exchanges listed in the config's "exchanges").

How it works
  * Websockets stream the best bid/ask of every selected coin on both
    exchanges (updates many times per second).
  * Gap = what you could really sell at on the dear exchange (its BID) vs. what
    you could really buy at on the cheap one (its ASK). When it is >= entry_gap_pct
    in the same direction for persist_seconds, and the top of both books holds
    our size, the bot sends BOTH market orders at the same moment: short on the
    dear exchange, long on the cheap one.
  * Right after the fills it checks the gap actually captured from the real fill
    prices. Below min_captured_gap_pct -> it closes both legs immediately
    ("opened without a real gap") and goes back to scanning.
  * It closes both legs when the gap to close them (dear ASK vs cheap BID) is
    <= exit_gap_pct, or after max_hold_minutes.
  * If one leg fails to open, the other leg is closed at once (never one-sided).
  * Telegram messages use REAL fill prices, quantities and fees from the
    exchanges.

Files
  gap_trader_config.json  settings (created with defaults on first start)
  gap_trader_keys.json    Bitget + Bybit API keys (never committed)
  config.json             Binance keys + Telegram (same file the other bot uses)
  gap_trader_state.json   open positions + today's stats (survives restarts)
  STOP_GAP_TRADER         create this file -> no new entries (open ones still managed)
  CLOSE_ALL_GAPS          create this file -> close every open position now
  REPORT_GAPS             create this file -> write the gap report to the log now (normally hourly)

Run:  .venv/bin/python gap_trader.py
"""
import asyncio
import base64
import hashlib
import hmac
import html
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

import requests
import websockets
from binance.client import Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("gap_trader.log"), logging.StreamHandler()])
log = logging.getLogger("gap_trader")

CONFIG_FILE = "gap_trader_config.json"
KEYS_FILE = "gap_trader_keys.json"
STATE_FILE = "gap_trader_state.json"
STOP_FILE = "STOP_GAP_TRADER"
CLOSE_ALL_FILE = "CLOSE_ALL_GAPS"
REPORT_FILE = "REPORT_GAPS"      # create it -> the hourly report is written right away

DEFAULT_CONFIG = {
    "live_trading": True,
    "size_usdt_per_side": 50,
    "leverage": 10,
    "entry_gap_pct": 1.0,
    "persist_seconds": 1,
    "min_captured_gap_pct": 0.3,
    "min_gap_at_size_pct": 0.6,     # when the best price is too small: gap at the AVERAGE fill for our size
    "exit_gap_pct": 0.1,
    "max_gap_pct": 5.0,
    "max_hold_minutes": 240,
    "max_positions": 2,
    "cooldown_minutes": 10,
    "fresh_gap_minutes": 30,
    "fresh_gap_below_pct": 0.3,
    "min_24h_volume_usdt": 5000000,
    "max_quote_age_seconds": 5,
    "daily_loss_limit_usdt": 30,
    "exchanges": ["binance", "bitget", "bybit"],
    # Slow mode: coins whose gaps close over hours/days (e.g. pre-IPO contracts). Own rules, own slot.
    "slow_mode": {
        "enabled": True,
        "coins": ["OPENAI", "ANTHROPIC"],
        "size_usdt_per_side": 500,
        "leverage": 10,
        "entry_gap_pct": 1.0,
        "min_captured_gap_pct": 0.7,
        "exit_gap_pct": 0.3,
        "max_hold_minutes": 7200,
        "max_positions": 1,
        "persist_seconds": 5,
        "min_gap_at_size_pct": 0.9,
        "cooldown_minutes": 30,
    },
    "taker_fee_pct": {"binance": 0.05, "bitget": 0.06, "bybit": 0.055},
    "exclude": ["ANTHROPIC", "OPENAI", "SPACEX", "XAI", "USDC", "FDUSD", "XAU", "XAG", "PAXG"],
}

BITGET = "https://api.bitget.com"
PRODUCT = "USDT-FUTURES"
BYBIT = "https://api.bybit.com"
NAMES = {"binance": "Binance", "bitget": "Bitget", "bybit": "Bybit"}


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def fmt_px(x):
    if x is None:
        return "—"
    if x >= 100:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6g}"


def money(x):
    return f"{'+' if x >= 0 else '−'}{abs(x):.2f}"


def dur(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


# ======================================================================== exchanges

class BitgetAPI:
    def __init__(self, key, secret, passphrase):
        self.key, self.secret, self.passphrase = key, secret, passphrase
        self.s = requests.Session()
        self.prepared = set()

    def _req(self, method, path, params=None, body=None):
        ts = str(int(time.time() * 1000))
        query = ""
        if params:
            query = "?" + "&".join(f"{k}={v}" for k, v in params.items())
        payload = json.dumps(body) if body else ""
        msg = ts + method.upper() + path + query + payload
        sign = base64.b64encode(hmac.new(self.secret.encode(), msg.encode(), hashlib.sha256).digest()).decode()
        headers = {"ACCESS-KEY": self.key, "ACCESS-SIGN": sign, "ACCESS-TIMESTAMP": ts,
                   "ACCESS-PASSPHRASE": self.passphrase, "Content-Type": "application/json", "locale": "en-US"}
        r = self.s.request(method, BITGET + path + query, headers=headers, data=payload or None, timeout=10)
        j = r.json()
        if str(j.get("code")) != "00000":
            raise RuntimeError(f"Bitget {path}: {j.get('code')} {j.get('msg')}")
        return j.get("data")

    def balance(self):
        d = self._req("GET", "/api/v2/mix/account/accounts", {"productType": PRODUCT})
        for a in d or []:
            if a.get("marginCoin") == "USDT":
                return float(a.get("usdtEquity") or a.get("accountEquity") or 0)
        return 0.0

    def positions(self):
        d = self._req("GET", "/api/v2/mix/position/all-position", {"productType": PRODUCT, "marginCoin": "USDT"})
        return {p["symbol"]: float(p.get("total") or 0) * (1 if p.get("holdSide") == "long" else -1)
                for p in (d or []) if float(p.get("total") or 0) != 0}

    def position(self, symbol):
        d = self._req("GET", "/api/v2/mix/position/single-position",
                      {"symbol": symbol, "productType": PRODUCT, "marginCoin": "USDT"})
        return sum(float(p.get("total") or 0) * (1 if p.get("holdSide") == "long" else -1) for p in (d or []))

    def one_way_mode(self):
        try:
            self._req("POST", "/api/v2/mix/account/set-position-mode", body={"productType": PRODUCT,
                                                                             "posMode": "one_way_mode"})
        except Exception as e:
            log.warning(f"Bitget position mode not changed ({e}); it must be one-way for this bot.")

    def prepare_symbol(self, symbol, leverage):
        if symbol in self.prepared:
            return
        self.prepared.add(symbol)
        for path, body in (("/api/v2/mix/account/set-margin-mode", {"marginMode": "crossed"}),
                           ("/api/v2/mix/account/set-leverage", {"leverage": str(int(leverage))})):
            try:
                self._req("POST", path, body={"symbol": symbol, "productType": PRODUCT, "marginCoin": "USDT", **body})
            except Exception as e:
                log.warning(f"Bitget {symbol} {path.rsplit('/', 1)[-1]}: {e}")

    def market(self, symbol, side, qty, reduce_only=False):
        """side: 'buy'/'sell'. Returns dict(price, qty, fee, id)."""
        d = self._req("POST", "/api/v2/mix/order/place-order", body={
            "symbol": symbol, "productType": PRODUCT, "marginMode": "crossed", "marginCoin": "USDT",
            "size": str(qty), "side": side, "orderType": "market", "reduceOnly": "YES" if reduce_only else "NO",
            "clientOid": "gap" + uuid.uuid4().hex[:20]})
        oid = d["orderId"]
        for _ in range(20):
            time.sleep(0.15)
            o = self._req("GET", "/api/v2/mix/order/detail", {"symbol": symbol, "productType": PRODUCT,
                                                              "orderId": oid})
            if o and o.get("state") == "filled":
                return {"price": float(o["priceAvg"]), "qty": float(o["baseVolume"]),
                        "fee": abs(float(o.get("fee") or 0)), "id": oid}
        raise RuntimeError(f"Bitget order {oid} not filled in time")


class BybitAPI:
    RECV = "5000"

    def __init__(self, key, secret):
        self.key, self.secret = key, secret
        self.s = requests.Session()
        self.prepared = set()

    def _req(self, method, path, params=None, body=None):
        ts = str(int(time.time() * 1000))
        if method == "GET":
            payload = "&".join(f"{k}={v}" for k, v in (params or {}).items())
        else:
            payload = json.dumps(body or {})
        sign = hmac.new(self.secret.encode(), (ts + self.key + self.RECV + payload).encode(),
                        hashlib.sha256).hexdigest()
        headers = {"X-BAPI-API-KEY": self.key, "X-BAPI-SIGN": sign, "X-BAPI-TIMESTAMP": ts,
                   "X-BAPI-RECV-WINDOW": self.RECV, "Content-Type": "application/json"}
        url = BYBIT + path + (("?" + payload) if method == "GET" and payload else "")
        r = self.s.request(method, url, headers=headers, data=None if method == "GET" else payload, timeout=10)
        j = r.json()
        if j.get("retCode") != 0:
            raise RuntimeError(f"Bybit {path}: {j.get('retCode')} {j.get('retMsg')}")
        return j.get("result") or {}

    def balance(self):
        for acct in ("UNIFIED", "CONTRACT"):
            try:
                d = self._req("GET", "/v5/account/wallet-balance", {"accountType": acct, "coin": "USDT"})
            except Exception:
                continue
            for a in d.get("list") or []:
                for c in a.get("coin") or []:
                    if c.get("coin") == "USDT":
                        return float(c.get("equity") or c.get("walletBalance") or 0)
        return 0.0

    def positions(self):
        d = self._req("GET", "/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
        return {p["symbol"]: float(p["size"]) * (1 if p.get("side") == "Buy" else -1)
                for p in d.get("list") or [] if float(p.get("size") or 0) != 0}

    def position(self, symbol):
        d = self._req("GET", "/v5/position/list", {"category": "linear", "symbol": symbol})
        return sum(float(p.get("size") or 0) * (1 if p.get("side") == "Buy" else -1) for p in d.get("list") or [])

    def one_way_mode(self):
        try:
            self._req("POST", "/v5/position/switch-mode", body={"category": "linear", "coin": "USDT", "mode": 0})
        except Exception as e:
            if "110025" not in str(e):          # 110025 = already one-way
                log.warning(f"Bybit position mode not changed ({e}); it must be one-way for this bot.")

    def prepare_symbol(self, symbol, leverage):
        if symbol in self.prepared:
            return
        self.prepared.add(symbol)
        try:
            self._req("POST", "/v5/position/set-leverage", body={
                "category": "linear", "symbol": symbol,
                "buyLeverage": str(int(leverage)), "sellLeverage": str(int(leverage))})
        except Exception as e:
            if "110043" not in str(e):          # 110043 = leverage not modified
                log.warning(f"Bybit {symbol} leverage: {e}")

    def market(self, symbol, side, qty, reduce_only=False):
        """side: 'buy'/'sell'. Returns dict(price, qty, fee, id)."""
        d = self._req("POST", "/v5/order/create", body={
            "category": "linear", "symbol": symbol, "side": side.capitalize(), "orderType": "Market",
            "qty": str(qty), "reduceOnly": bool(reduce_only), "orderLinkId": "gap" + uuid.uuid4().hex[:20]})
        oid = d["orderId"]
        for _ in range(20):
            time.sleep(0.15)
            for path in ("/v5/order/realtime", "/v5/order/history"):
                try:
                    lst = self._req("GET", path, {"category": "linear", "symbol": symbol, "orderId": oid}).get("list")
                except Exception:
                    lst = None
                if lst and lst[0].get("orderStatus") == "Filled":
                    o = lst[0]
                    return {"price": float(o["avgPrice"]), "qty": float(o["cumExecQty"]),
                            "fee": abs(float(o.get("cumExecFee") or 0)), "id": oid}
        raise RuntimeError(f"Bybit order {oid} not filled in time")


class BinanceAPI:
    def __init__(self, key, secret):
        self.c = Client(api_key=key, api_secret=secret)
        self.lev_done = set()

    def balance(self):
        for a in self.c.futures_account_balance():
            if a["asset"] == "USDT":
                return float(a["balance"])
        return 0.0

    def position(self, symbol):
        for p in self.c.futures_position_information(symbol=symbol):
            return float(p["positionAmt"])
        return 0.0

    def positions(self):
        return {p["symbol"]: float(p["positionAmt"]) for p in self.c.futures_position_information()
                if float(p["positionAmt"]) != 0}

    def prepare_symbol(self, symbol, leverage):
        if symbol in self.lev_done:
            return
        try:
            self.c.futures_change_leverage(symbol=symbol, leverage=int(leverage))
        except Exception as e:
            log.warning(f"Binance {symbol} leverage: {e}")
        self.lev_done.add(symbol)

    def market(self, symbol, side, qty, reduce_only=False):
        """side: 'buy'/'sell'. Returns dict(price, qty, fee, id)."""
        kw = dict(symbol=symbol, side=side.upper(), type="MARKET", quantity=qty, newOrderRespType="RESULT")
        if reduce_only:
            kw["reduceOnly"] = "true"
        o = self.c.futures_create_order(**kw)
        oid = o["orderId"]
        price, filled = float(o.get("avgPrice") or 0), float(o.get("executedQty") or 0)
        fee = None
        for _ in range(6):
            try:
                tr = self.c.futures_account_trades(symbol=symbol, orderId=oid)
                if tr:
                    usdt = [float(t["commission"]) for t in tr if t.get("commissionAsset") == "USDT"]
                    fee = sum(usdt) if len(usdt) == len(tr) else None   # paid in BNB -> estimate
                    if not price:
                        q = sum(float(t["qty"]) for t in tr)
                        price = sum(float(t["price"]) * float(t["qty"]) for t in tr) / q
                        filled = q
                    break
            except Exception:
                pass
            time.sleep(0.3)
        return {"price": price, "qty": filled, "fee": fee, "id": oid}


# ======================================================================== bot

class GapTrader:
    def __init__(self):
        cfg = load_json(CONFIG_FILE)
        if cfg is None:
            save_json(CONFIG_FILE, DEFAULT_CONFIG)
            cfg = {}
        self.cfg = {**DEFAULT_CONFIG, **cfg}
        # Nested dict: fill in exchanges missing from the user's file (e.g. a newly added Bybit).
        self.cfg["taker_fee_pct"] = {**DEFAULT_CONFIG["taker_fee_pct"], **(cfg.get("taker_fee_pct") or {})}
        self.cfg["slow_mode"] = {**DEFAULT_CONFIG["slow_mode"], **(cfg.get("slow_mode") or {})}
        sm = self.cfg["slow_mode"]
        self.slow_coins = {c.upper() for c in sm["coins"]} if sm.get("enabled") else set()
        main_cfg = load_json("config.json", {}) or {}
        keys = load_json(KEYS_FILE, {}) or {}
        self.exchanges = [e for e in self.cfg["exchanges"] if e in NAMES]
        need = {"bitget": ("bitget_api_key", "bitget_secret_key", "bitget_passphrase"),
                "bybit": ("bybit_api_key", "bybit_secret_key")}
        missing = [k for ex in self.exchanges for k in need.get(ex, ()) if not keys.get(k)]
        if missing:
            save_json(KEYS_FILE, {**{k: "" for ex in need for k in need[ex]}, **keys})
            raise SystemExit(f"Fill in {', '.join(missing)} in {KEYS_FILE} (or remove that exchange from "
                             f"\"exchanges\" in {CONFIG_FILE}), then start again.")
        if len(self.exchanges) < 2:
            raise SystemExit("At least two exchanges are needed in \"exchanges\".")
        self.tg_token, self.tg_chat = main_cfg.get("telegram_bot_token"), main_cfg.get("telegram_chat_id")
        self.bn = BinanceAPI(main_cfg.get("api_key"), main_cfg.get("secret_key"))
        self.api = {"binance": self.bn}
        if "bitget" in self.exchanges:
            self.api["bitget"] = BitgetAPI(keys["bitget_api_key"], keys["bitget_secret_key"],
                                           keys["bitget_passphrase"])
        if "bybit" in self.exchanges:
            self.api["bybit"] = BybitAPI(keys["bybit_api_key"], keys["bybit_secret_key"])
        self.quotes = {}            # (ex, coin) -> [bid, bid_qty, ask, ask_qty, t]
        self.feed_seen = {ex: 0.0 for ex in self.exchanges}
        self.pending = {}           # coin -> (direction, first_seen)
        self.busy = set()
        self.cooldown = {}
        self.pool = ThreadPoolExecutor(max_workers=8)
        st = load_json(STATE_FILE, {}) or {}
        self.positions = st.get("positions", {})
        self.day = st.get("day", {})
        self.lock = threading.Lock()
        self.started = time.time()
        self.funded = {}
        self.depth_checked = {}
        self.blocked_legs = {}      # (coin, exchange) -> until time: that exchange rejected this coin     # coin -> last time deeper order-book levels were fetched            # exchange -> True if its balance can carry one more position
        self.last_small = {}        # coin -> last time its gap was "normal" (<= fresh_gap_below_pct)
        self.hour = self._new_hour()

    # ---------------------------------------------------------- setup
    def _listings(self, ex):
        """{symbol: {step, min, min_usdt, vol}} of tradable USDT perpetuals on one exchange."""
        out = {}
        if ex == "binance":
            vol = {t["symbol"]: float(t["quoteVolume"]) for t in self.bn.c.futures_ticker()}
            for s in self.bn.c.futures_exchange_info()["symbols"]:
                # Any perpetual type (normal coins, TradFi / pre-IPO perpetuals use other type names).
                if "PERPETUAL" not in str(s.get("contractType", "")).upper() or s.get("quoteAsset") != "USDT" \
                        or s.get("status") != "TRADING":
                    continue
                f = {x["filterType"]: x for x in s["filters"]}
                lot = f.get("MARKET_LOT_SIZE") or f.get("LOT_SIZE")
                out[s["symbol"]] = {"step": Decimal(lot["stepSize"]).normalize(), "min": float(lot["minQty"]),
                                    "min_usdt": float(f.get("MIN_NOTIONAL", {}).get("notional", 5)),
                                    "vol": vol.get(s["symbol"], 0)}
        elif ex == "bitget":
            t = requests.get(BITGET + "/api/v2/mix/market/tickers", params={"productType": PRODUCT}, timeout=15).json()
            vol = {x["symbol"]: float(x.get("usdtVolume") or x.get("quoteVolume") or 0) for x in t.get("data") or []}
            c = requests.get(BITGET + "/api/v2/mix/market/contracts", params={"productType": PRODUCT}, timeout=15).json()
            for s in c.get("data") or []:
                if s.get("symbolStatus") == "normal":
                    out[s["symbol"]] = {"step": Decimal(s["sizeMultiplier"]).normalize(), "min": float(s["minTradeNum"]),
                                        "min_usdt": float(s.get("minTradeUSDT") or 5), "vol": vol.get(s["symbol"], 0)}
        elif ex == "bybit":
            t = requests.get(BYBIT + "/v5/market/tickers", params={"category": "linear"}, timeout=15).json()
            vol = {x["symbol"]: float(x.get("turnover24h") or 0) for x in t.get("result", {}).get("list") or []}
            cursor = ""
            for _ in range(10):
                j = requests.get(BYBIT + "/v5/market/instruments-info", timeout=15, params={
                    "category": "linear", "limit": 1000, "cursor": cursor}).json().get("result", {})
                for s in j.get("list") or []:
                    if s.get("status") != "Trading" or s.get("quoteCoin") != "USDT" \
                            or "Perpetual" not in str(s.get("contractType", "")):
                        continue
                    lot = s.get("lotSizeFilter") or {}
                    out[s["symbol"]] = {"step": Decimal(lot.get("qtyStep", "1")).normalize(),
                                        "min": float(lot.get("minOrderQty") or 0),
                                        "min_usdt": float(lot.get("minNotionalValue") or 5),
                                        "vol": vol.get(s["symbol"], 0)}
                cursor = j.get("nextPageCursor") or ""
                if not cursor:
                    break
        return out

    def build_universe(self):
        """Coins listed under the SAME symbol on 2+ exchanges, each with enough volume."""
        lists = {ex: self._listings(ex) for ex in self.exchanges}
        # Never trade symbols the other bot (hybrid_trader) uses: one-way positions would net together.
        other = {c.get("symbol") for c in (load_json("trade_config.json", []) or []) if isinstance(c, dict)}
        excl = {x.upper() for x in self.cfg["exclude"]} - getattr(self, "slow_coins", set())
        self.meta = {}
        for sym in set().union(*lists.values()):
            coin = sym[:-4]
            if not sym.endswith("USDT") or coin in excl or sym in other:
                continue
            exs = {ex: lists[ex][sym] for ex in self.exchanges
                   if sym in lists[ex] and lists[ex][sym]["vol"] >= self.cfg["min_24h_volume_usdt"]}
            if len(exs) >= 2:
                self.meta[coin] = {"symbol": sym, "ex": exs}
        per = {ex: sum(1 for m in self.meta.values() if ex in m["ex"]) for ex in self.exchanges}
        log.info(f"Universe: {len(self.meta)} coins on 2+ exchanges with ≥ {self.cfg['min_24h_volume_usdt']:,.0f} "
                 f"USDT 24h volume ({', '.join(f'{NAMES[e]} {n}' for e, n in per.items())})")

    # ---------------------------------------------------------- feeds
    def coins_on(self, ex):
        return sorted(c for c, m in self.meta.items() if ex in m["ex"])

    async def bybit_feed(self, coins):
        chunks = [coins[i:i + 200] for i in range(0, len(coins), 200)]
        await asyncio.gather(*(self._bybit_conn(c) for c in chunks))

    async def _bybit_conn(self, coins):
        while True:
            try:
                async with websockets.connect("wss://stream.bybit.com/v5/public/linear", ping_interval=None,
                                              max_size=2 ** 22) as ws:
                    for i in range(0, len(coins), 10):
                        await ws.send(json.dumps({"op": "subscribe", "args": [
                            f"orderbook.1.{self.meta[c]['symbol']}" for c in coins[i:i + 10]]}))
                        await asyncio.sleep(0.1)
                    log.info(f"Bybit websocket connected ({len(coins)} coins)")
                    book = {}

                    async def pinger():
                        while True:
                            await asyncio.sleep(20)
                            await ws.send(json.dumps({"op": "ping"}))
                    ping_task = asyncio.create_task(pinger())
                    try:
                        async for raw in ws:
                            m = json.loads(raw)
                            if not str(m.get("topic", "")).startswith("orderbook.1."):
                                continue
                            d = m.get("data") or {}
                            coin = d.get("s", "")[:-4]
                            if coin not in self.meta:
                                continue
                            bk = book.setdefault(coin, [None, None])
                            for i, side in ((0, "b"), (1, "a")):
                                lvl = d.get(side)
                                if lvl:   # empty list = side unchanged
                                    p, q = float(lvl[0][0]), float(lvl[0][1])
                                    bk[i] = (p, q) if q > 0 else None
                            if bk[0] and bk[1]:
                                self.quotes[("bybit", coin)] = [bk[0][0], bk[0][1], bk[1][0], bk[1][1], time.time()]
                                self.feed_seen["bybit"] = time.time()
                                self.on_quote(coin)
                    finally:
                        ping_task.cancel()
            except Exception as e:
                log.warning(f"Bybit websocket dropped: {e}; reconnecting")
                await asyncio.sleep(2)

    async def binance_feed(self, coins):
        chunks = [coins[i:i + 150] for i in range(0, len(coins), 150)]
        await asyncio.gather(*(self._binance_conn(c) for c in chunks))

    async def _binance_conn(self, coins):
        url = "wss://fstream.binance.com/stream?streams=" + "/".join(
            f"{self.meta[c]['symbol'].lower()}@bookTicker" for c in coins)
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, max_size=2 ** 22) as ws:
                    log.info(f"Binance websocket connected ({len(coins)} coins)")
                    async for raw in ws:
                        d = json.loads(raw).get("data") or {}
                        coin = d.get("s", "")[:-4]
                        if coin in self.meta:
                            self.quotes[("binance", coin)] = [float(d["b"]), float(d["B"]), float(d["a"]),
                                                              float(d["A"]), time.time()]
                            self.feed_seen["binance"] = time.time()
                            self.on_quote(coin)
            except Exception as e:
                log.warning(f"Binance websocket dropped: {e}; reconnecting")
                await asyncio.sleep(2)

    async def bitget_feed(self, coins):
        chunks = [coins[i:i + 200] for i in range(0, len(coins), 200)]
        await asyncio.gather(*(self._bitget_conn(c) for c in chunks))

    async def _bitget_conn(self, coins):
        while True:
            try:
                async with websockets.connect("wss://ws.bitget.com/v2/ws/public", ping_interval=None,
                                              max_size=2 ** 22) as ws:
                    for i in range(0, len(coins), 50):
                        await ws.send(json.dumps({"op": "subscribe", "args": [
                            {"instType": PRODUCT, "channel": "books1", "instId": self.meta[c]["symbol"]}
                            for c in coins[i:i + 50]]}))
                        await asyncio.sleep(0.2)
                    log.info(f"Bitget websocket connected ({len(coins)} coins)")

                    async def pinger():
                        while True:
                            await asyncio.sleep(25)
                            await ws.send("ping")
                    ping_task = asyncio.create_task(pinger())
                    try:
                        async for raw in ws:
                            if raw == "pong":
                                continue
                            m = json.loads(raw)
                            if not m.get("data"):
                                continue
                            coin = m["arg"]["instId"][:-4]
                            d = m["data"][0]
                            if coin in self.meta and d.get("bids") and d.get("asks"):
                                self.quotes[("bitget", coin)] = [float(d["bids"][0][0]), float(d["bids"][0][1]),
                                                                 float(d["asks"][0][0]), float(d["asks"][0][1]),
                                                                 time.time()]
                                self.feed_seen["bitget"] = time.time()
                                self.on_quote(coin)
                    finally:
                        ping_task.cancel()
            except Exception as e:
                log.warning(f"Bitget websocket dropped: {e}; reconnecting")
                await asyncio.sleep(2)

    # ---------------------------------------------------------- signal logic
    def R(self, coin):
        """Rules for this coin: slow-mode overrides for slow coins, the normal settings otherwise."""
        if coin in self.slow_coins:
            return {**self.cfg, **self.cfg["slow_mode"], "mode": "slow", "fresh_gap_minutes": 0}
        return {**self.cfg, "mode": "fast"}

    def mode_of(self, coin):
        return "slow" if coin in self.slow_coins else "fast"

    def gaps(self, coin):
        """{(long_ex, short_ex): gap%} for every pair of exchanges with FRESH quotes.
        gap = (bid on the short exchange - ask on the long exchange) / that ask."""
        age, now, q = self.cfg["max_quote_age_seconds"], time.time(), {}
        for ex in self.meta[coin]["ex"]:
            if self.blocked_legs.get((coin, ex), 0) > now:
                continue
            x = self.quotes.get((ex, coin))
            if x and now - x[4] <= age:
                q[ex] = x
        return {(lo, sh): (q[sh][0] - q[lo][2]) / q[lo][2] * 100
                for lo in q for sh in q if lo != sh}

    def close_gap(self, pos):
        """Gap left to close a position: buy back the short at its ASK, sell the long at its BID."""
        ql, qs = self.quotes.get((pos["long"], pos["coin"])), self.quotes.get((pos["short"], pos["coin"]))
        if not ql or not qs:
            return None
        age = self.cfg["max_quote_age_seconds"]
        if time.time() - ql[4] > age or time.time() - qs[4] > age:
            return None          # never decide on stale prices
        return (qs[2] - ql[0]) / ql[0] * 100

    def refresh_balances(self, announce=False):
        """An exchange is usable only if it holds 1.5x the margin one position needs."""
        modes = [self.cfg] + ([self.cfg["slow_mode"]] if self.slow_coins else [])
        need = min(m["size_usdt_per_side"] / max(m["leverage"], 1) for m in modes) * 1.5
        lines = []
        for ex in self.exchanges:
            try:
                bal = self.api[ex].balance()
            except Exception as e:
                log.warning(f"{ex} balance check failed: {e}")
                continue
            ok = bal >= need
            if self.funded.get(ex) is not None and self.funded[ex] != ok or (announce and not ok):
                lines.append(f"{NAMES[ex]}: {bal:,.2f} USDT — " + ("usable again ✅" if ok else
                             f"too low (needs ≥ {need:,.2f}), skipped ⛔"))
            self.funded[ex] = ok
        if lines:
            self.telegram("💰 <b>Balance check</b>\n" + "\n".join(lines))

    def open_blocker(self, mode="fast", ignore=None):
        """Why no new position may open right now ('' = allowed). Each mode has its own slots."""
        if not self.cfg["live_trading"]:
            return "live_trading is false"
        if os.path.exists(STOP_FILE):
            return "STOP_GAP_TRADER file present"
        limit = self.cfg["slow_mode"]["max_positions"] if mode == "slow" else self.cfg["max_positions"]
        used = sum(1 for c in (set(self.positions) | self.busy) - {ignore} if self.mode_of(c) == mode)
        if used >= limit:
            return f"{mode} max_positions ({limit}) already open"
        if self.today()["net"] <= -abs(self.cfg["daily_loss_limit_usdt"]):
            return "daily loss limit reached"
        return ""

    def on_quote(self, coin):
        if coin in self.positions:
            self.check_exit(coin)
            return
        r = self.R(coin)
        g = self.gaps(coin)
        if not g:
            self.pending.pop(coin, None)
            return
        now = time.time()
        # Ignore pairs that use an exchange without enough balance.
        g = {k: v for k, v in g.items() if self.funded.get(k[0], True) and self.funded.get(k[1], True)} or g
        direction, gap = max(g.items(), key=lambda kv: kv[1])
        if not (self.funded.get(direction[0], True) and self.funded.get(direction[1], True)):
            if gap >= r["entry_gap_pct"]:
                self.hour["why"][coin] = "exchange balance too low"
            self.pending.pop(coin, None)
            return
        if gap <= r["fresh_gap_below_pct"]:
            self.last_small[coin] = now                  # gap is "normal" right now
        h = self.hour
        if gap > h["max"][0]:
            h["max"] = (gap, coin)
        if coin in self.busy or now < self.cooldown.get(coin, 0):
            self.pending.pop(coin, None)
            return
        if not (r["entry_gap_pct"] <= gap <= r["max_gap_pct"]):
            if coin in self.pending and gap < r["entry_gap_pct"]:
                h["why"].setdefault(coin, f"gone before {r['persist_seconds']}s")
            elif gap > r["max_gap_pct"]:
                h["why"][coin] = f"above max_gap {r['max_gap_pct']}%"
            self.pending.pop(coin, None)
            return
        h["seen"].add(coin)
        h["best"][coin] = max(h["best"].get(coin, 0), gap)
        mins = r["fresh_gap_minutes"]
        if mins and now - self.last_small.get(coin, -1e18) > mins * 60:
            # Wide for longer than fresh_gap_minutes (or since the bot started): a stuck gap, skip.
            h["stale"].add(coin)
            self.pending.pop(coin, None)
            return
        first = self.pending.get(coin)
        if not first or first[0] != direction:
            self.pending[coin] = (direction, now)        # new gap or flipped direction: restart the clock
            return
        if now - first[1] < r["persist_seconds"]:
            return
        blocked = self.open_blocker(r["mode"])
        if blocked:
            h["why"][coin] = blocked
            return
        long_ex, short_ex = direction
        ql, qs = self.quotes[(long_ex, coin)], self.quotes[(short_ex, coin)]
        size = r["size_usdt_per_side"]
        if ql[2] * ql[3] < size or qs[0] * qs[1] < size:
            # Best price can't hold our size: check deeper levels (REST, at most every 10 s per coin).
            if now - self.depth_checked.get(coin, 0) < 10:
                return
            self.depth_checked[coin] = now
            self.pending.pop(coin, None)
            self.busy.add(coin)
            self.pool.submit(self.depth_then_open, coin, long_ex, short_ex, gap, size, r)
            return
        self.pending.pop(coin, None)
        self.busy.add(coin)
        h["traded"].add(coin)
        self.pool.submit(self.open_position, coin, long_ex, short_ex, gap, ql[2], qs[0])

    # ---------------------------------------------------------- order-book depth
    def book(self, ex, symbol):
        """Public order book: (bids, asks) as [(price, qty)], best first."""
        if ex == "binance":
            d = requests.get("https://fapi.binance.com/fapi/v1/depth", params={"symbol": symbol, "limit": 20},
                             timeout=5).json()
            b, a = d["bids"], d["asks"]
        elif ex == "bitget":
            d = requests.get(BITGET + "/api/v2/mix/market/merge-depth", timeout=5, params={
                "symbol": symbol, "productType": PRODUCT, "limit": "50"}).json()["data"]
            b, a = d["bids"], d["asks"]
        else:
            d = requests.get(BYBIT + "/v5/market/orderbook", timeout=5, params={
                "category": "linear", "symbol": symbol, "limit": 50}).json()["result"]
            b, a = d["b"], d["a"]
        return [(float(p), float(q)) for p, q in b], [(float(p), float(q)) for p, q in a]

    @staticmethod
    def avg_fill(levels, usdt):
        """Average price to fill `usdt` walking the book; None if the book is too thin."""
        left, cost, qty = usdt, 0.0, 0.0
        for p, q in levels:
            take = min(left, p * q)
            cost += take
            qty += take / p
            left -= take
            if left <= 1e-9:
                return cost / qty
        return None

    def depth_then_open(self, coin, long_ex, short_ex, top_gap, size, r):
        """Open only if the gap still pays at the AVERAGE prices our full size would get."""
        h = self.hour
        try:
            sym = self.meta[coin]["symbol"]
            (_, asks), (bids, _) = self.book(long_ex, sym), self.book(short_ex, sym)
            buy, sell = self.avg_fill(asks, size), self.avg_fill(bids, size)
            if not buy or not sell:
                h["why"][coin] = f"book too thin for {size:g} USDT even across 20+ levels"
                return
            gap_at_size = (sell - buy) / buy * 100
            if gap_at_size < r["min_gap_at_size_pct"]:
                h["why"][coin] = (f"gap at full {size:g} USDT only {gap_at_size:.2f}% "
                                  f"(needs {r['min_gap_at_size_pct']}%; best-price gap {top_gap:.2f}%)")
                return
            blocked = self.open_blocker(r["mode"], ignore=coin)
            if blocked:
                h["why"][coin] = blocked
                return
            h["traded"].add(coin)
            log.info(f"[{coin}] depth ok: gap {gap_at_size:.3f}% at {size:g} USDT (best-price gap {top_gap:.3f}%)")
            self.open_position(coin, long_ex, short_ex, gap_at_size, buy, sell)   # clears busy itself
            return
        except Exception as e:
            h["why"][coin] = f"order-book check failed: {str(e)[:80]}"
        finally:
            if coin not in self.positions:
                self.busy.discard(coin)

    @staticmethod
    def _new_hour():
        return {"start": time.time(), "max": (0.0, "—"), "seen": set(), "stale": set(), "traded": set(),
                "why": {}, "best": {}}

    def roll_hour(self):
        """Hourly log line (and daily totals) of what the bot saw, traded and skipped."""
        h, self.hour = self.hour, self._new_hour()
        stale = sorted(h["stale"] - h["traded"])
        log.info(f"HOUR: biggest gap {h['max'][0]:.2f}% ({h['max'][1]}) · coins with gap ≥ "
                 f"{self.cfg['entry_gap_pct']}%: {len(h['seen'])} · traded: {len(h['traded'])} · "
                 f"skipped as old (wide > {self.cfg['fresh_gap_minutes']} min): {len(stale)}"
                 + (f" [{', '.join(stale)}]" if stale else ""))
        with self.lock:
            d = self.today()
            d["gaps_seen"] = d.get("gaps_seen", 0) + len(h["seen"])
            d["gaps_traded"] = d.get("gaps_traded", 0) + len(h["traded"])
            d["gaps_stale"] = d.get("gaps_stale", 0) + len(stale)
            d["stale_coins"] = sorted(set(d.get("stale_coins", [])) | set(stale))
            if h["max"][0] > d.get("max_gap", 0):
                d["max_gap"], d["max_gap_coin"] = h["max"][0], h["max"][1]
            self.save()
        # One line per coin that crossed the entry gap but was not traded, with the reason.
        for coin, best in sorted(h["best"].items(), key=lambda kv: -kv[1])[:15]:
            if coin in h["traded"]:
                continue
            why = "stuck gap (old)" if coin in h["stale"] else h["why"].get(coin, "gone before it could be checked")
            log.info(f"HOUR   not traded: {coin} max {best:.2f}% — {why}")

    # ---------------------------------------------------------- trading
    def leg_rules(self, coin, legs):
        """Common quantity step / minimums for the two exchanges of a trade (None if incompatible)."""
        ex = [self.meta[coin]["ex"][e] for e in legs]
        step = max(x["step"] for x in ex)
        if any((step / x["step"]) % 1 != 0 for x in ex):
            return None
        return step, max(x["min"] for x in ex), max(x["min_usdt"] for x in ex)

    def qty_for(self, coin, price, legs, size=None):
        rules = self.leg_rules(coin, legs)
        if not rules:
            return None
        step, min_q, min_usdt = rules
        size = size or self.R(coin)["size_usdt_per_side"]
        q = (Decimal(str(size / price)) / step).to_integral_value(rounding=ROUND_DOWN) * step
        if float(q) < min_q or float(q) * price < min_usdt:
            return None
        return float(q)

    def fee_or_est(self, fill, ex):
        """Real fee from the exchange, or taker-rate estimate if it isn't in USDT."""
        if fill.get("fee") is not None:
            return fill["fee"], False
        return fill["price"] * fill["qty"] * self.cfg["taker_fee_pct"][ex] / 100, True

    def qstr(self, coin, q, legs):
        step = max(self.meta[coin]["ex"][e]["step"] for e in legs)
        return format((Decimal(str(q)) / step).to_integral_value(rounding=ROUND_DOWN) * step, "f")

    def _both(self, a, b):
        fa, fb = self.pool.submit(*a), self.pool.submit(*b)
        out = []
        for f in (fa, fb):
            try:
                out.append((f.result(timeout=15), None))
            except Exception as e:
                out.append((None, e))
        return out

    def open_position(self, coin, long_ex, short_ex, signal_gap, ask_long, bid_short):
        sym = self.meta[coin]["symbol"]
        try:
            legs = (long_ex, short_ex)
            qty = self.qty_for(coin, ask_long, legs)
            if not qty:
                log.info(f"[{coin}] size too small for exchange minimums, skipped")
                return
            if "binance" in legs and abs(self.bn.position(sym)) > 0:
                log.info(f"[{coin}] Binance already has a position in {sym} (not ours), skipped")
                self.cooldown[coin] = time.time() + 3600
                return
            for ex in (long_ex, short_ex):
                self.api[ex].prepare_symbol(sym, self.R(coin)["leverage"])
            t0 = time.time()
            q = self.qstr(coin, qty, legs)
            (lf, le), (sf, se) = self._both((self.api[long_ex].market, sym, "buy", q),
                                            (self.api[short_ex].market, sym, "sell", q))
            if le or se:
                # One leg failed: close the leg that did fill, never stay one-sided.
                msg = [f"⚠️ <b>{coin}: one side failed to open</b>"]
                if le:
                    msg.append(f"Long on {NAMES[long_ex]} failed: {html.escape(str(le))[:200]}")
                if se:
                    msg.append(f"Short on {NAMES[short_ex]} failed: {html.escape(str(se))[:200]}")
                for fill, ex, side in ((lf, long_ex, "sell"), (sf, short_ex, "buy")):
                    if fill and fill["qty"] > 0:
                        try:
                            c = self.api[ex].market(sym, side, self.qstr(coin, fill["qty"], legs), reduce_only=True)
                            pnl = (c["price"] - fill["price"]) * fill["qty"] * (1 if side == "sell" else -1)
                            fees = self.fee_or_est(fill, ex)[0] + self.fee_or_est(c, ex)[0]
                            self.add_day(pnl - fees, 1)
                            msg.append(f"Closed the filled side on {NAMES[ex]}: {money(pnl - fees)} USDT "
                                       f"(incl. {fees:.2f} fees)")
                        except Exception as e:
                            msg.append(f"🚨 Could NOT close {NAMES[ex]} side: {html.escape(str(e))[:200]} — "
                                       f"check the exchange manually!")
                self.cooldown[coin] = time.time() + self.R(coin)["cooldown_minutes"] * 60
                for err, ex in ((le, long_ex), (se, short_ex)):
                    if err:
                        # Don't retry the same rejection every cooldown (each try costs fees).
                        self.blocked_legs[(coin, ex)] = time.time() + 24 * 3600
                        msg.append(f"⛔ {coin} will not be traded on {NAMES[ex]} for 24h "
                                   f"(fix the cause, then restart the bot to retry sooner).")
                self.telegram("\n".join(msg))
                return
            captured = (sf["price"] - lf["price"]) / lf["price"] * 100
            (fl, el), (fs, es) = self.fee_or_est(lf, long_ex), self.fee_or_est(sf, short_ex)
            pos = {"coin": coin, "symbol": sym, "long": long_ex, "short": short_ex, "qty": min(lf["qty"], sf["qty"]),
                   "long_px": lf["price"], "short_px": sf["price"], "open_fees": fl + fs,
                   "fees_estimated": el or es,
                   "signal_gap": signal_gap, "captured_gap": captured, "opened": time.time(),
                   "exec_ms": int((time.time() - t0) * 1000)}
            with self.lock:
                self.positions[coin] = pos
                self.save()
            log.info(f"[{coin}] OPEN long {long_ex} @ {lf['price']} / short {short_ex} @ {sf['price']} "
                     f"qty {pos['qty']} captured {captured:.3f}% (signal {signal_gap:.3f}%)")
            if captured < self.R(coin)["min_captured_gap_pct"]:
                self.telegram(self.open_message(pos) + f"\n\n⚠️ Real gap at fill is only {captured:.2f}% "
                              f"(minimum {self.R(coin)['min_captured_gap_pct']:.2f}%) — closing right away.")
                self.close_position(coin, "opened without a real gap")
            else:
                self.telegram(self.open_message(pos))
        except Exception as e:
            log.exception(f"[{coin}] open failed")
            self.telegram(f"⚠️ <b>{coin}</b>: open failed: {html.escape(str(e))[:300]}")
            self.cooldown[coin] = time.time() + self.R(coin)["cooldown_minutes"] * 60
        finally:
            self.busy.discard(coin)

    def check_exit(self, coin):
        pos = self.positions.get(coin)
        if not pos or coin in self.busy or time.time() - pos.get("close_failed_at", 0) < 60:
            return
        g = self.close_gap(pos)
        held = time.time() - pos["opened"]
        reason = None
        if g is not None and g <= self.R(coin)["exit_gap_pct"]:
            reason = "gap closed"
        elif held >= self.R(coin)["max_hold_minutes"] * 60:
            reason = "max hold time"
        if reason:
            self.busy.add(coin)
            self.pool.submit(self.close_position, coin, reason)

    def leg_amounts(self, pos):
        """What is REALLY still open on each exchange: (long coins left, short coins left)."""
        sym = pos["symbol"]
        la = self.api[pos["long"]].position(sym)
        sa = self.api[pos["short"]].position(sym)
        return max(la, 0.0), max(-sa, 0.0)

    def close_position(self, coin, reason):
        self.busy.add(coin)
        pos = self.positions.get(coin)
        try:
            if not pos:
                return
            sym, qty = pos["symbol"], pos["qty"]
            # Close only what really exists: a side may already be gone (liquidated / closed by hand).
            long_left, short_left = self.leg_amounts(pos)
            long_left, short_left = min(long_left, qty), min(short_left, qty)
            lost = {"long": qty - long_left, "short": qty - short_left}
            if lost["long"] > qty * 0.01 or lost["short"] > qty * 0.01:
                reason = "LIQUIDATED or closed outside the bot"
            no_fill = {"price": None, "qty": 0.0, "fee": 0.0}
            jobs = {}
            if long_left > 0:
                jobs["long"] = (self.api[pos["long"]].market, sym, "sell", self.qstr(coin, long_left, (pos["long"], pos["short"])), True)
            if short_left > 0:
                jobs["short"] = (self.api[pos["short"]].market, sym, "buy", self.qstr(coin, short_left, (pos["long"], pos["short"])), True)
            fills, errs = {"long": no_fill, "short": no_fill}, {}
            futs = {k: self.pool.submit(*v) for k, v in jobs.items()}
            for k, f in futs.items():
                try:
                    fills[k] = f.result(timeout=15)
                except Exception as e:
                    errs[k] = e
            for _ in range(3):   # retry any leg that failed to close
                for k in list(errs):
                    try:
                        fills[k] = jobs[k][0](*jobs[k][1:])
                        errs.pop(k)
                    except Exception as e:
                        errs[k] = e
                if not errs:
                    break
                time.sleep(1)
            if errs:
                pos["close_failed_at"] = time.time()     # back off 60 s instead of retrying every second
                self.telegram(f"🚨 <b>{coin}: could not close</b> "
                              + ", ".join(f"{k} on {NAMES[pos[k]]}" for k in errs)
                              + f" — check the exchange manually! {html.escape(str(next(iter(errs.values()))))[:200]}")
                return
            lf, sf = fills["long"], fills["short"]
            if lost["long"] > qty * 0.01 or lost["short"] > qty * 0.01:
                self.report_lost_leg(pos, lost, lf, sf)
                return
            long_pnl = (lf["price"] - pos["long_px"]) * qty
            short_pnl = (pos["short_px"] - sf["price"]) * qty
            (cl, e1), (cs, e2) = self.fee_or_est(lf, pos["long"]), self.fee_or_est(sf, pos["short"])
            close_fees = cl + cs
            est = pos["fees_estimated"] or e1 or e2
            fees = pos["open_fees"] + close_fees
            net = long_pnl + short_pnl - fees
            self.add_day(net, 1)
            with self.lock:
                self.positions.pop(coin, None)
                self.save()
            self.cooldown[coin] = time.time() + self.R(coin)["cooldown_minutes"] * 60
            held = time.time() - pos["opened"]
            d = self.today()
            icon = "✅" if net >= 0 else "🔻"
            self.telegram(
                f"{icon} <b>{'🐢 SLOW · ' if self.mode_of(coin) == 'slow' else ''}CLOSED {coin}: "
                f"{money(net)} USDT</b>\n"
                f"Reason: {reason} · held {dur(held)}\n\n"
                f"Long  {NAMES[pos['long']]}: {fmt_px(pos['long_px'])} → {fmt_px(lf['price'])} = {money(long_pnl)}\n"
                f"Short {NAMES[pos['short']]}: {fmt_px(pos['short_px'])} → {fmt_px(sf['price'])} = {money(short_pnl)}\n"
                f"Fees: open {pos['open_fees']:.2f} + close {close_fees:.2f} = {fees:.2f}"
                f"{' (some estimated)' if est else ''}\n"
                f"Quantity: {qty:g} {coin} per side\n\n"
                f"<b>Net: {money(net)} USDT</b>\n"
                f"Today: {d['trades']} trades · {money(d['net'])} USDT")
            log.info(f"[{coin}] CLOSED ({reason}) net {net:.4f} long {long_pnl:.4f} short {short_pnl:.4f} fees {fees:.4f}")
        except Exception as e:
            log.exception(f"[{coin}] close failed")
            self.telegram(f"🚨 <b>{coin}: close failed</b>: {html.escape(str(e))[:300]} — check manually!")
        finally:
            self.busy.discard(coin)

    def report_lost_leg(self, pos, lost, lf, sf):
        """A side vanished (liquidation or manual close). The rest was just closed; report it plainly."""
        coin, qty = pos["coin"], pos["qty"]
        lines = [f"🚨 <b>{coin}: a side was LIQUIDATED or closed outside the bot</b>"]
        net, approx = 0.0, False
        for side, fill, entry, sign in (("long", lf, pos["long_px"], 1), ("short", sf, pos["short_px"], -1)):
            ex = pos[side]
            closed = qty - lost[side]
            if closed > 0 and fill["price"]:
                pnl = (fill["price"] - entry) * closed * sign
                fee = self.fee_or_est(fill, ex)[0]
                net += pnl - fee
                lines.append(f"{side.title()} {NAMES[ex]}: closed by bot {fmt_px(entry)} → {fmt_px(fill['price'])} "
                             f"= {money(pnl)} (fee {fee:.2f})")
            if lost[side] > 0:
                q = self.quotes.get((ex, coin))
                now_px = (q[0] if side == "long" else q[2]) if q else entry
                pnl = (now_px - entry) * lost[side] * sign
                net += pnl
                approx = True
                lines.append(f"{side.title()} {NAMES[ex]}: <b>gone on the exchange</b> ({lost[side]:g} {coin}); "
                             f"approx {money(pnl)} at today's price — liquidation is usually worse, check "
                             f"{NAMES[ex]} for the exact loss")
        net -= pos["open_fees"]
        self.add_day(net, 1)
        with self.lock:
            self.positions.pop(coin, None)
            self.save()
        self.cooldown[coin] = time.time() + 6 * 3600
        lines.append(f"\nNet ≈ <b>{money(net)} USDT</b>{' (approx.)' if approx else ''} · coin paused 6h")
        self.telegram("\n".join(lines))
        log.warning(f"[{coin}] LOST LEG {lost} net≈{net:.4f}")

    def sync_legs(self):
        """Every few seconds: compare what we think is open with what the exchanges really hold."""
        for coin, pos in list(self.positions.items()):
            if coin in self.busy or time.time() - pos["opened"] < 15:
                continue
            try:
                long_left, short_left = self.leg_amounts(pos)
            except Exception as e:
                log.warning(f"[{coin}] position check failed: {e}")
                continue
            if long_left < pos["qty"] * 0.99 or short_left < pos["qty"] * 0.99:
                self.busy.add(coin)
                self.close_position(coin, "side missing on exchange")

    def open_message(self, pos):
        notional = pos["qty"] * pos["long_px"]
        r = self.R(pos["coin"])
        tag = "🐢 SLOW · " if r["mode"] == "slow" else ""
        return (f"🟢 <b>{tag}OPENED {pos['coin']}</b>\n"
                f"Long  {NAMES[pos['long']]}: {pos['qty']:g} @ {fmt_px(pos['long_px'])}\n"
                f"Short {NAMES[pos['short']]}: {pos['qty']:g} @ {fmt_px(pos['short_px'])}\n\n"
                f"Gap seen: {pos['signal_gap']:.2f}% → real gap at fill: <b>{pos['captured_gap']:.2f}%</b>\n"
                f"Size: {notional:,.0f} USDT per side · {r['leverage']}x\n"
                f"Fees paid: {pos['open_fees']:.2f} USDT{' (estimated)' if pos['fees_estimated'] else ''}\n"
                f"Both orders filled in {pos['exec_ms']} ms\n"
                f"Will close when the gap is ≤ {r['exit_gap_pct']}% "
                f"(or after {dur(r['max_hold_minutes'] * 60)})")

    # ---------------------------------------------------------- bookkeeping
    def today(self):
        key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.day.get("date") != key:
            if self.day.get("date"):
                d = self.day
                stale = d.get("stale_coins", [])
                self.telegram(f"📊 <b>Daily summary {d['date']} (UTC)</b>\n"
                              f"Trades: {d.get('trades', 0)}\n"
                              f"Net: <b>{money(d.get('net', 0))} USDT</b>\n\n"
                              f"Biggest gap seen: {d.get('max_gap', 0):.2f}% ({d.get('max_gap_coin', '—')})\n"
                              f"Gaps ≥ entry (coin-hours): {d.get('gaps_seen', 0)} · traded {d.get('gaps_traded', 0)}"
                              f" · skipped as old {d.get('gaps_stale', 0)}"
                              + (f"\nSkipped coins: {', '.join(stale[:15])}" if stale else ""))
            self.day = {"date": key, "trades": 0, "net": 0.0}
        return self.day

    def add_day(self, net, trades):
        with self.lock:
            d = self.today()
            d["net"] += net
            d["trades"] += trades
            self.save()

    def save(self):
        save_json(STATE_FILE, {"positions": self.positions, "day": self.day})

    def telegram(self, text):
        if not self.tg_token or not self.tg_chat:
            log.info("TELEGRAM: " + text.replace("\n", " | "))
            return
        try:
            requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage",
                          data={"chat_id": self.tg_chat, "text": text[:4000], "parse_mode": "HTML"}, timeout=10)
        except Exception as e:
            log.warning(f"Telegram failed: {e}")

    # ---------------------------------------------------------- housekeeping loop
    async def housekeeping(self):
        last_status = last_sync = 0
        sync_job = None
        while True:
            await asyncio.sleep(1)
            now = time.time()
            if self.positions and now - last_sync >= 10 and (sync_job is None or sync_job.done()):
                last_sync = now
                sync_job = self.pool.submit(self.sync_legs)
            for coin in list(self.pending):              # re-check gaps waiting for persistence
                self.on_quote(coin)
            for coin in list(self.positions):
                self.check_exit(coin)
            if os.path.exists(CLOSE_ALL_FILE):
                os.remove(CLOSE_ALL_FILE)
                self.telegram(f"🛑 CLOSE_ALL requested: closing {len(self.positions)} position(s)")
                for coin in list(self.positions):
                    if coin not in self.busy:
                        self.pool.submit(self.close_position, coin, "manual close-all")
            if now - getattr(self, "_bal_t", 0) >= 300:
                self._bal_t = now
                self.pool.submit(self.refresh_balances)
            if os.path.exists(REPORT_FILE):
                os.remove(REPORT_FILE)
                self.roll_hour()
            elif now - self.hour["start"] >= 3600:
                self.roll_hour()
            if now - last_status > 600:
                last_status = now
                stale = [ex for ex, t in self.feed_seen.items() if now - t > 10]
                log.info(f"status: {len(self.positions)} open, today {self.today()['trades']} trades "
                         f"{self.today()['net']:+.2f} USDT, feeds {'OK' if not stale else 'STALE: ' + ','.join(stale)}")

    def startup_message(self):
        bal = []
        for ex in self.exchanges:
            try:
                bal.append(f"{NAMES[ex]} {self.api[ex].balance():,.2f}")
            except Exception as e:
                bal.append(f"{NAMES[ex]} n/a")
                log.warning(f"{ex} balance check failed: {e}")
        c = self.cfg
        lines = [
            "🚀 <b>Gap trader started</b>",
            f"Mode: {'LIVE TRADING' if c['live_trading'] else 'watching only (live_trading = false)'}",
            f"Balances (USDT): {' · '.join(bal)}",
            f"Coins watched: {len(self.meta)}",
            f"Size: {c['size_usdt_per_side']} USDT per side · {c['leverage']}x · max {c['max_positions']} positions",
            f"Open at gap ≥ {c['entry_gap_pct']}% held {c['persist_seconds']}s · keep only if real gap ≥ "
            f"{c['min_captured_gap_pct']}% · close at ≤ {c['exit_gap_pct']}%",
            (f"Only fresh gaps: must have been ≤ {c['fresh_gap_below_pct']}% within {c['fresh_gap_minutes']} min"
             if c['fresh_gap_minutes'] else "Fresh-gap rule: off"),
            (f"🐢 Slow mode: {', '.join(sorted(self.slow_coins))} · {c['slow_mode']['size_usdt_per_side']} USDT/side "
             f"· {c['slow_mode']['leverage']}x · open ≥ {c['slow_mode']['entry_gap_pct']}% · close ≤ "
             f"{c['slow_mode']['exit_gap_pct']}% · max {dur(c['slow_mode']['max_hold_minutes'] * 60)} · "
             f"{c['slow_mode']['max_positions']} slot(s)" if self.slow_coins else "🐢 Slow mode: off"),
            f"Daily loss limit: {c['daily_loss_limit_usdt']} USDT",
        ]
        if self.positions:
            lines.append(f"Resumed open positions: {', '.join(self.positions)}")
        self.telegram("\n".join(lines))

    async def run(self):
        self.build_universe()
        for ex in self.exchanges:
            if ex != "binance":
                self.api[ex].one_way_mode()
        # Positions this bot doesn't know about (Binance is shared with hybrid_trader and checked per trade).
        ours = {p["symbol"] for p in self.positions.values()}
        for ex in self.exchanges:
            if ex == "binance":
                continue
            try:
                foreign = [x for x in set(self.api[ex].positions()) - ours if x[:-4] in self.meta]
            except Exception as e:
                log.warning(f"{ex} position check failed: {e}")
                continue
            if foreign:
                self.telegram(f"⚠️ {NAMES[ex]} has open positions not opened by this bot: " + ", ".join(foreign)
                              + " — those coins are skipped.")
                for x in foreign:
                    self.cooldown[x[:-4]] = float("inf")
        self.refresh_balances(announce=True)
        self._bal_t = time.time()
        self.startup_message()
        feeds = {"binance": self.binance_feed, "bitget": self.bitget_feed, "bybit": self.bybit_feed}
        await asyncio.gather(*(feeds[ex](self.coins_on(ex)) for ex in self.exchanges), self.housekeeping())


if __name__ == "__main__":
    try:
        asyncio.run(GapTrader().run())
    except KeyboardInterrupt:
        pass
