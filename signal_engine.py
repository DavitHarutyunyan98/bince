#!/usr/bin/env python3
"""Social signal engine: learns from collected posts (social.db) + prices, retrains every day,
and sends transparent signals (alerts only — it never trades).

Everything is visible and stored in signals.db:
  * models    every daily version: its weights (how much each input pushes the forecast),
              what it was trained on, and its score on days it had NOT seen (walk-forward test)
  * changes   what changed between consecutive versions
  * signals   every signal with the inputs, each input's contribution, the model version,
              and later the real outcome

Model: for each coin and hour, inputs (attention, tone, engagement, recent price moves) predict
the coin's own price move over the next 4h and 24h -> LONG (expected up) or SHORT (expected down),
with entry price, target price and deadline. A ridge regression (a linear model) is used so
every forecast is a plain sum of "weight x input" that can be shown in full.

Every alert and daily update shows the plain record: how the model's forecasts did on days it
had not seen (walk-forward test) and how its live signals turned out. No pass/fail labels.

Run:      .venv/bin/python signal_engine.py            (runs forever: hourly signals, daily retrain)
Reports:  .venv/bin/python signal_engine.py --model    (current version, weights, unseen-days test)
          .venv/bin/python signal_engine.py --changes  (what changed between versions)
          .venv/bin/python signal_engine.py --signals 48   (signals of the last 48 h + outcomes)
          .venv/bin/python signal_engine.py --retrain  (retrain now and print the result)
"""
import argparse
import json
import logging
import math
import re
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("signal_engine.log"), logging.StreamHandler()])
log = logging.getLogger("signals")

SOCIAL_DB, DB, CONFIG = "social.db", "signals.db", "signal_config.json"
H = 3_600_000
DEFAULTS = {
    "horizons_hours": [4, 24],
    "cost_pct": 0.2,              # fees + slippage for one round trip, in %
    "signal_threshold_pct": 0.8,  # only signal when |forecast| >= this + costs (i.e. >= 1.0%)
    "min_train_days": 5,          # no model before this many days of data
    "ridge_alpha": 5.0,           # higher = more conservative weights
    "max_coins": 80,              # coins priced/considered (most mentioned)
    "exclude": ["USDC", "USDT", "FDUSD", "DAI", "USDE", "RLUSD", "PYUSD", "TUSD", "USD1", "PAXG", "XAUT"],
    "retrain_every_hours": 1,        # new model version this often (before that hour's signals)
    "notify_weight_change": 0.05,    # Telegram model update only if a weight moves at least this much
    "notify_record_change": 0.1,     # … or the unseen-days avg result moves at least this much (%)
    "telegram": True,
    "crypto_only": True,             # no stock / ETF / pre-IPO perpetuals
    "min_sources": 2,                # a signal needs >= 2 sources …
    "min_authors": 3,                # … or >= 3 different authors in that hour
    "max_signals_per_hour": 3,
    "send_only_if_tests_positive": True,   # send a horizon's signals only if its past tests made money after costs
    "daily_summary_hour_utc": 0,
}
FEATURES = [
    ("mentions", "mentions this hour (log)"),
    ("spike", "× above normal mentions (log)"),
    ("accel", "mentions vs previous hour (log ratio)"),
    ("authors", "different authors (log)"),
    ("sources", "different sources"),
    ("tone", "average tone (-1..+1)"),
    ("pos_share", "share of very positive posts"),
    ("neg_share", "share of very negative posts"),
    ("engagement", "votes + comments (log)"),
    ("ret_1h", "coin move last 1h vs BTC %"),
    ("ret_6h", "coin move last 6h vs BTC %"),
    ("ret_24h", "coin move last 24h vs BTC %"),
    ("btc_6h", "BTC move last 6h %"),
]
FNAMES = [f for f, _ in FEATURES]
LABEL = dict(FEATURES)


def load_cfg():
    try:
        cfg = json.load(open(CONFIG))
    except FileNotFoundError:
        cfg = {}
        json.dump(DEFAULTS, open(CONFIG, "w"), indent=2)
    return {**DEFAULTS, **cfg}


def db():
    con = sqlite3.connect(DB, timeout=30)
    con.execute("CREATE TABLE IF NOT EXISTS prices (coin TEXT, ts INTEGER, close REAL, PRIMARY KEY (coin, ts))")
    con.execute("""CREATE TABLE IF NOT EXISTS models (version INTEGER PRIMARY KEY, trained_at INTEGER,
        horizon INTEGER, data_from INTEGER, data_to INTEGER, n_train INTEGER, weights TEXT, mean TEXT, std TEXT,
        intercept REAL, oos TEXT, status TEXT)""")
    con.execute("CREATE TABLE IF NOT EXISTS changes (version INTEGER, horizon INTEGER, text TEXT)")
    con.execute("""CREATE TABLE IF NOT EXISTS signals (ts INTEGER, coin TEXT, horizon INTEGER, version INTEGER,
        forecast REAL, direction TEXT, status TEXT, inputs TEXT, contributions TEXT, outcome REAL,
        PRIMARY KEY (ts, coin, horizon))""")
    return con


# ============================================================ prices (hourly closes, cached)

def fetch_hourly(coin, start_ms, end_ms):
    sym = coin + "USDT"
    for src in ("binance", "bybit", "bitget"):
        rows = []
        try:
            if src == "binance":
                cur = start_ms
                while cur < end_ms:
                    j = requests.get("https://fapi.binance.com/fapi/v1/klines", timeout=15, params={
                        "symbol": sym, "interval": "1h", "startTime": cur, "limit": 1500}).json()
                    if not isinstance(j, list) or not j:
                        break
                    rows += [(int(k[0]), float(k[4])) for k in j]
                    cur = int(j[-1][0]) + H
                    if len(j) < 1500:
                        break
            elif src == "bybit":
                cur_end = end_ms
                while cur_end > start_ms:
                    j = requests.get("https://api.bybit.com/v5/market/kline", timeout=15, params={
                        "category": "linear", "symbol": sym, "interval": "60", "start": start_ms,
                        "end": cur_end, "limit": 1000}).json().get("result", {}).get("list") or []
                    if not j:
                        break
                    rows += [(int(k[0]), float(k[4])) for k in j]
                    oldest = min(int(k[0]) for k in j)
                    if len(j) < 1000 or oldest <= start_ms:
                        break
                    cur_end = oldest - 1
            else:
                cur = start_ms
                while cur < end_ms:
                    j = requests.get("https://api.bitget.com/api/v2/mix/market/candles", timeout=15, params={
                        "symbol": sym, "productType": "USDT-FUTURES", "granularity": "1H", "startTime": cur,
                        "endTime": min(cur + 199 * H, end_ms), "limit": 200}).json().get("data") or []
                    rows += [(int(k[0]), float(k[4])) for k in j]
                    cur += 199 * H
        except Exception:
            rows = []
        if len(rows) > 3:
            return rows
    return []


def hourly_prices(coins, start_ms, end_ms):
    """{coin: Series(close at the END of each hour, indexed by hour start ms)} using the cache."""
    con = db()
    out = {}
    for coin in coins:
        have = pd.read_sql_query("SELECT ts, close FROM prices WHERE coin=? AND ts>=? ORDER BY ts", con,
                                 params=(coin, start_ms))
        last = int(have.ts.max()) if not have.empty else start_ms - H
        if last < end_ms - 2 * H:
            rows = fetch_hourly(coin, last + H, end_ms)
            # Store only FINISHED hours: an unfinished candle's close is not final (no look-ahead).
            done = [(coin, t, c) for t, c in rows if t + H <= end_ms]
            if done:
                with con:
                    con.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?)", done)
                have = pd.read_sql_query("SELECT ts, close FROM prices WHERE coin=? AND ts>=? ORDER BY ts", con,
                                         params=(coin, start_ms))
        if len(have) > 3:
            out[coin] = pd.Series(have.close.values, index=have.ts.values)
        time.sleep(0.05)
    con.close()
    return out


# ============================================================ features & labels

def build_dataset(cfg, now_ms=None):
    """One row per coin-hour that had mentions: inputs known at the END of that hour, and the
    coin's own price move over the next N hours (label; NaN while not known yet)."""
    now_ms = now_ms or int(time.time() * 1000)
    s = sqlite3.connect(SOCIAL_DB)
    m = pd.read_sql_query("""SELECT m.coin, m.created, m.source, p.author, p.sentiment, p.score, p.comments
                             FROM mentions m JOIN posts p ON p.uid = m.uid WHERE m.created >= ?""", s,
                          params=((now_ms - 60 * 86_400_000) // 1000,))
    s.close()
    m = m[~m.coin.isin(set(cfg["exclude"]))]
    if m.empty:
        return pd.DataFrame(), {}
    m["hour"] = (m.created * 1000 // H) * H
    coins = m.coin.value_counts().head(cfg["max_coins"]).index.tolist()
    m = m[m.coin.isin(coins)]
    start = int(m.hour.min()) - 30 * H
    prices = hourly_prices(sorted(set(coins) | {"BTC"}), start, now_ms)
    btc = prices.get("BTC")
    if btc is None:
        return pd.DataFrame(), prices

    def move(series, t_end, hours):
        """% move of a series from (t_end - hours) to t_end, using hour-start-indexed closes."""
        a, b = series.get(t_end - hours * H - H), series.get(t_end - H)
        return (b / a - 1) * 100 if a and b else np.nan

    rows = []
    for coin, g in m.groupby("coin"):
        px = prices.get(coin)
        if px is None or coin == "BTC":     # BTC is the benchmark itself (its move vs BTC is always 0)
            continue
        per_h = g.groupby("hour").size()
        first_h = int(g.hour.min())
        for hour, gh in g.groupby("hour"):
            t_end = int(hour) + H                       # inputs are known at the end of the hour
            n = len(gh)
            prev = per_h[(per_h.index < hour) & (per_h.index >= hour - 72 * H)]
            span = max(min((hour - first_h) / H, 72), 1)
            normal = prev.sum() / span
            prev1 = per_h.get(hour - H, 0)
            tone = gh.sentiment.dropna()
            row = {
                "coin": coin, "t": t_end,
                "mentions": math.log1p(n),
                "spike": math.log((n + 0.5) / (normal + 0.5)),
                "accel": math.log((n + 0.5) / (prev1 + 0.5)),
                "authors": math.log1p(gh.author.nunique()),
                "sources": gh.source.nunique(),
                "tone": tone.mean() if len(tone) else 0.0,
                "pos_share": (tone > 0.5).mean() if len(tone) else 0.0,
                "neg_share": (tone < -0.5).mean() if len(tone) else 0.0,
                "engagement": math.log1p(max(gh.score.clip(lower=0).sum() + gh.comments.sum(), 0)),
                "btc_6h": move(btc, t_end, 6),
            }
            for hh in (1, 6, 24):
                row[f"ret_{hh}h"] = move(px, t_end, hh) - move(btc, t_end, hh)
            row["p0"] = px.get(t_end - H, np.nan)         # entry price: close of the finished hour
            for hz in cfg["horizons_hours"]:
                c0, c1 = px.get(t_end - H), px.get(t_end + hz * H - H)
                row[f"p1_{hz}"] = c1 if c1 else np.nan  # exit price hz hours later
                row[f"y{hz}"] = (c1 / c0 - 1) * 100 if c0 and c1 else np.nan
            rows.append(row)
    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    df[FNAMES] = df[FNAMES].fillna(0.0)
    return df, prices


# ============================================================ model (transparent ridge regression)

def fit(X, y, alpha):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    b0 = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - b0))
    return w, mu, sd, b0


def contributions(x, w, mu, sd):
    return (x - mu) / sd * w


def walk_forward(df, hz, cfg):
    """Each day: train only on rows whose outcome was known BEFORE that day, then forecast that day."""
    y = f"y{hz}"
    d = df.dropna(subset=[y]).copy()
    if d.empty:
        return []
    d["day"] = d.t // 86_400_000
    days = sorted(d.day.unique())
    res = []
    for day in days[cfg["min_train_days"]:]:
        known_by = day * 86_400_000
        train = d[d.t + hz * H <= known_by]
        test = d[d.day == day]
        if len(train) < 50 or test.empty:
            continue
        w, mu, sd, b0 = fit(train[FNAMES].values, train[y].values, cfg["ridge_alpha"])
        pred = b0 + ((test[FNAMES].values - mu) / sd) @ w
        for p, r in zip(pred, test.itertuples()):
            if abs(p) >= cfg["signal_threshold_pct"] + cfg["cost_pct"]:
                actual = getattr(r, y)
                res.append({"t": int(r.t), "coin": r.coin, "side": "LONG" if p > 0 else "SHORT",
                            "forecast": round(float(p), 3), "entry": float(r.p0), "exit": float(getattr(r, f"p1_{hz}")),
                            "actual": round(float(actual), 3),
                            "net": round(float(np.sign(p) * actual - cfg["cost_pct"]), 3)})
    return res


def oos_summary(res, cfg):
    if not res:
        return {"events": 0, "avg_net": None, "win_rate": None, "good_weeks": 0, "weeks": 0}
    r = pd.DataFrame(res)
    r["week"] = pd.to_datetime(r.t, unit="ms").dt.strftime("%G-W%V")
    wk = r.groupby("week").net.agg(["mean", "size"])
    return {"events": len(r), "long": int((r.side == "LONG").sum()), "short": int((r.side == "SHORT").sum()),
            "test_from": int(r.t.min()), "test_to": int(r.t.max()),
            "dir_right": round(float((np.sign(r.forecast) == np.sign(r.actual)).mean()), 3),
            "win_rate": round(float((r.net > 0).mean()), 3), "avg_net": round(float(r.net.mean()), 3),
            "good_weeks": int((wk["mean"] > 0).sum()), "weeks": int(len(wk)),
            "by_week": {k: f"{v['mean']:+.2f}% ({int(v['size'])} signals)" for k, v in wk.iterrows()},
            "list": r.drop(columns=["week"]).tail(500).to_dict("records")}


def latest_model(con, hz):
    r = con.execute("SELECT version, weights, mean, std, intercept, oos, status, n_train, data_from, data_to "
                    "FROM models WHERE horizon=? ORDER BY version DESC LIMIT 1", (hz,)).fetchone()
    if not r:
        return None
    return {"version": r[0], "w": np.array(json.loads(r[1])), "mu": np.array(json.loads(r[2])),
            "sd": np.array(json.loads(r[3])), "b0": r[4], "oos": json.loads(r[5]), "status": r[6],
            "n_train": r[7], "from": r[8], "to": r[9]}


def describe_change(old, new_w, new_oos, new_status, hz):
    lines = []
    if old is None:
        lines.append("first version")
    else:
        moves = sorted(zip(FNAMES, old["w"], new_w), key=lambda x: -abs(x[2] - x[1]))
        for f, a, b in moves[:5]:
            if abs(b - a) >= 0.005:
                lines.append(f"{LABEL[f]}: {a:+.3f} → {b:+.3f}")
        oa, ob = old["oos"].get("avg_net"), new_oos.get("avg_net")
        lines.append(f"unseen-days test: {old['oos'].get('events', 0)} → {new_oos.get('events', 0)} signals, "
                     f"avg after costs {oa if oa is not None else '—'}% → {ob if ob is not None else '—'}%")
    return lines


def meaningful(old, new_w, new_oos, cfg):
    if old is None:
        return True
    if np.max(np.abs(np.array(new_w) - old["w"])) >= cfg["notify_weight_change"]:
        return True
    oa, ob = old["oos"].get("avg_net"), new_oos.get("avg_net")
    return (oa is None) != (ob is None) or (oa is not None and abs(ob - oa) >= cfg["notify_record_change"])


def retrain(cfg, notify=True):
    df, _ = build_dataset(cfg)
    con = db()
    msgs = []
    for hz in cfg["horizons_hours"]:
        d = df.dropna(subset=[f"y{hz}"]) if not df.empty else df
        days = d.t.nunique() and (d.t.max() - d.t.min()) / 86_400_000 if not d.empty else 0
        if d.empty or days < cfg["min_train_days"] or len(d) < 50:
            msgs.append(f"{hz}h: not enough data yet ({len(d)} coin-hours with known outcome, "
                        f"{days:.1f} days; need {cfg['min_train_days']} days and 50 rows)")
            continue
        w, mu, sd, b0 = fit(d[FNAMES].values, d[f"y{hz}"].values, cfg["ridge_alpha"])
        oos = oos_summary(walk_forward(df, hz, cfg), cfg)
        status = ""
        old = latest_model(con, hz)
        version = (con.execute("SELECT MAX(version) FROM models").fetchone()[0] or 0) + 1
        with con:
            con.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                version, int(time.time()), hz, int(d.t.min()), int(d.t.max()), len(d), json.dumps(w.tolist()),
                json.dumps(mu.tolist()), json.dumps(sd.tolist()), float(b0), json.dumps(oos), status))
            change = describe_change(old, w, oos, status, hz)
            con.execute("INSERT INTO changes VALUES (?,?,?)", (version, hz, "\n".join(change)))
        if notify == "changes" and not meaningful(old, w, oos, cfg):
            continue           # saved and visible in --changes, just not sent to Telegram
        top_changes = [c for c in change if "→" in c and "test" not in c][:3]
        msgs.append(f"🔄 <b>{hz}h model updated → v{version}</b>\n"
                    f"Data: {len(d)} coin-hours, {pd.to_datetime(d.t.min(), unit='ms'):%m-%d} → "
                    f"{pd.to_datetime(d.t.max(), unit='ms'):%m-%d %H:%M} UTC\n"
                    + (("Biggest changes: " + "; ".join(top_changes) + "\n") if top_changes else "")
                    + f"Past tests: {short_record(oos)}")
    con.close()
    text = "\n\n".join(msgs)
    log.info("retrain: " + (text.replace("\n", " | ") or "no meaningful change"))
    if notify and text:
        telegram(cfg, text)
    return text


def record_text(oos, live=None):
    """Plain facts: how this model's forecasts did on days it had not seen, and live so far."""
    if not oos or not oos.get("events"):
        txt = "test on unseen days: no simulated signals yet"
    else:
        span = ""
        if oos.get("test_from"):
            span = (f" {pd.to_datetime(oos['test_from'], unit='ms'):%m-%d}→"
                    f"{pd.to_datetime(oos['test_to'], unit='ms'):%m-%d}")
        txt = (f"test on unseen days{span}: {oos['events']} simulated signals"
               + (f" ({oos.get('long', 0)} long / {oos.get('short', 0)} short)" if "long" in oos else "")
               + (f" · direction right {oos['dir_right'] * 100:.0f}%" if "dir_right" in oos else "")
               + f" · profitable after costs {oos['win_rate'] * 100:.0f}% · avg result after costs "
               f"{oos['avg_net']:+.2f}% · positive weeks {oos['good_weeks']}/{oos['weeks']}")
    if live and live["n"]:
        txt += (f" | live: {live['n']} finished · direction right {live['right'] * 100:.0f}% · profitable "
                f"{live['profitable'] * 100:.0f}% · avg after costs {live['avg']:+.2f}%")
    return txt


def live_record(con, hz, cost):
    rows = con.execute("SELECT forecast, outcome FROM signals WHERE horizon=? AND outcome IS NOT NULL", (hz,)).fetchall()
    if not rows:
        return {"n": 0}
    net = [np.sign(f) * o - cost for f, o in rows]
    return {"n": len(rows), "right": float(np.mean([np.sign(f) == np.sign(o) for f, o in rows])),
            "profitable": float(np.mean([x > 0 for x in net])), "avg": float(np.mean(net))}


# ============================================================ signals

_CRYPTO = {"t": 0, "set": set()}


def crypto_set():
    """Coins that are crypto (not stocks/ETFs/pre-IPO): Binance standard perpetuals + CoinGecko top 1000."""
    if time.time() - _CRYPTO["t"] < 12 * 3600 and _CRYPTO["set"]:
        return _CRYPTO["set"]
    out = set()
    try:
        for x in requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=20).json()["symbols"]:
            if x.get("contractType") == "PERPETUAL" and x.get("quoteAsset") == "USDT":
                out.add(re.sub(r"^(1000000|100000|10000|1000|1M)(?=[A-Z])", "", x["baseAsset"]))
    except Exception as e:
        log.warning(f"crypto list Binance: {e}")
    for page in (1, 2, 3, 4):
        try:
            out |= {c["symbol"].upper() for c in requests.get(
                "https://api.coingecko.com/api/v3/coins/markets", timeout=20,
                params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": page}).json()}
            time.sleep(6)
        except Exception:
            pass
    if out:
        _CRYPTO.update(t=time.time(), set=out)
    return _CRYPTO["set"]


def make_signals(cfg):
    """Forecast every coin of the last finished hour; keep only clear, non-conflicting crypto signals."""
    df, _ = build_dataset(cfg)
    if df.empty:
        return []
    con = db()
    now_ms = int(time.time() * 1000)
    finished = df[df.t <= now_ms]                  # only hours that are over
    if finished.empty or now_ms - int(finished.t.max()) > 2 * H:
        con.close()
        return []
    now = finished[finished.t == int(finished.t.max())]
    crypto = crypto_set() if cfg["crypto_only"] else None
    models = {hz: latest_model(con, hz) for hz in cfg["horizons_hours"]}
    cands = {}                                     # coin -> list of candidate signals (one per horizon)
    for row in now.itertuples():
        if crypto is not None and row.coin not in crypto:
            continue                               # stock / ETF / pre-IPO
        if row.sources < cfg["min_sources"] and math.expm1(row.authors) < cfg["min_authors"] - 0.5:
            continue                               # one post / one author is not a discussion
        entry = float(row.p0)
        if not entry == entry:
            continue
        x = np.array([getattr(row, f) for f in FNAMES], dtype=float)
        for hz, mdl in models.items():
            if not mdl:
                continue
            contrib = contributions(x, mdl["w"], mdl["mu"], mdl["sd"])
            fc = mdl["b0"] + contrib.sum()
            if abs(fc) >= cfg["signal_threshold_pct"] + cfg["cost_pct"]:
                cands.setdefault(row.coin, []).append(dict(hz=hz, fc=fc, contrib=contrib, x=x, entry=entry,
                                                           t=int(row.t), mdl=mdl))
    picked = []
    for coin, cs in cands.items():
        if len({np.sign(c["fc"]) for c in cs}) > 1:
            continue                               # 4h and 24h disagree: no signal
        picked.append((coin, max(cs, key=lambda c: abs(c["fc"]))))
    picked = sorted(picked, key=lambda p: -abs(p[1]["fc"]))[:cfg["max_signals_per_hour"]]
    sent = []
    for coin, c in picked:
        mdl, hz, fc = c["mdl"], c["hz"], c["fc"]
        tests_ok = (mdl["oos"].get("avg_net") or -1) > 0
        send = tests_ok or not cfg["send_only_if_tests_positive"]
        side = "LONG" if fc > 0 else "SHORT"
        inputs = dict(zip(FNAMES, c["x"].round(4).tolist()))
        inputs["_entry"] = c["entry"]
        try:
            with con:
                con.execute("INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,?,NULL)", (
                    c["t"], coin, hz, mdl["version"], float(fc), side, "sent" if send else "logged",
                    json.dumps(inputs), json.dumps(dict(zip(FNAMES, c["contrib"].round(4).tolist())))))
        except sqlite3.IntegrityError:
            continue
        if send:
            top = sorted(zip(FNAMES, c["x"], c["contrib"]), key=lambda z: -abs(z[2]))[:2]
            sent.append({"side": side, "coin": coin, "hz": hz, "fc": fc, "entry": c["entry"],
                         "target": c["entry"] * (1 + fc / 100), "t_in": c["t"], "t_out": c["t"] + hz * H,
                         "why": " · ".join(reason(f, v) for f, v, _ in top), "record": short_record(mdl["oos"])})
    con.close()
    if sent:
        telegram(cfg, signals_message(sent))
    log.info(f"signals: {len(cands)} candidates, {len(picked)} picked, {len(sent)} sent")
    return sent


def reason(f, v):
    """One input in plain words."""
    return {
        "mentions": lambda: f"{math.expm1(v):.0f} mentions this hour",
        "spike": lambda: f"mentions {math.exp(v):.1f}× normal",
        "accel": lambda: f"mentions {math.exp(v):.1f}× the previous hour",
        "authors": lambda: f"{math.expm1(v):.0f} different authors",
        "sources": lambda: f"talked about on {v:.0f} sites",
        "tone": lambda: "posts mostly positive" if v > 0.2 else "posts mostly negative" if v < -0.2 else "neutral tone",
        "pos_share": lambda: f"{v * 100:.0f}% very positive posts",
        "neg_share": lambda: f"{v * 100:.0f}% very negative posts",
        "engagement": lambda: f"big discussion ({math.expm1(v):,.0f} votes/comments)",
        "ret_1h": lambda: f"coin {'up' if v > 0 else 'down'} {abs(v):.1f}% vs BTC in the last hour",
        "ret_6h": lambda: f"coin {'up' if v > 0 else 'down'} {abs(v):.1f}% vs BTC in 6h",
        "ret_24h": lambda: f"coin already {'up' if v > 0 else 'down'} {abs(v):.1f}% vs BTC in 24h",
        "btc_6h": lambda: f"BTC {'up' if v > 0 else 'down'} {abs(v):.1f}% in 6h",
    }[f]()


def short_record(oos):
    if not oos or not oos.get("events"):
        return "no past tests yet"
    right = oos.get("dir_right", oos.get("win_rate"))
    return f"{oos['events']} tests · {right * 100:.0f}% right · avg {oos['avg_net']:+.2f}% after costs"


def signals_message(sent):
    sent = sorted(sent, key=lambda x: -abs(x["fc"]))
    cards = []
    for x in sent:
        icon = "🟢" if x["fc"] > 0 else "🔴"
        cards.append(
            f"{icon} <b>{x['side']} {x['coin']}/USDT</b>\n"
            f"Now {fmt_price(x['entry'])} → expected <b>{fmt_price(x['target'])}</b> ({x['fc']:+.1f}%) "
            f"by {pd.to_datetime(x['t_out'], unit='ms'):%m-%d %H:%M} UTC\n"
            f"Main reasons: {x['why']}\n"
            f"<i>This model's past tests: {x['record']}</i>")
    return "🔔 <b>Social signal</b> (alert only, no trading)\n\n" + "\n\n".join(cards)


def daily_summary(cfg):
    """Plain-language summary of the last 24 hours."""
    con = db()
    since = int(time.time() * 1000) - 24 * H
    rows = con.execute("SELECT coin, horizon, forecast, outcome, status FROM signals WHERE ts >= ?", (since,)).fetchall()
    sent = [r for r in rows if r[4] == "sent"]
    logged = [r for r in rows if r[4] == "logged"]
    done = [r for r in sent if r[3] is not None]
    lines = ["📊 <b>Social signals · last 24 hours</b>",
             f"Sent: {len(sent)} · held back (model's past tests not profitable): {len(logged)}"]
    if done:
        right = sum(np.sign(r[2]) == np.sign(r[3]) for r in done)
        avg = np.mean([np.sign(r[2]) * r[3] - cfg["cost_pct"] for r in done])
        lines.append(f"Finished: {len(done)} · right direction {right}/{len(done)} · avg result after costs {avg:+.2f}%")
        for coin, hz, fc, out, _ in done[:10]:
            ok = "✅" if np.sign(fc) == np.sign(out) else "❌"
            lines.append(f"{ok} {'LONG' if fc > 0 else 'SHORT'} {coin} {hz}h: expected {fc:+.1f}%, real {out:+.1f}%")
    for hz in cfg["horizons_hours"]:
        m = latest_model(con, hz)
        if m:
            lines.append(f"{hz}h model (v{m['version']}) past tests: {short_record(m['oos'])}")
    con.close()
    telegram(cfg, "\n".join(lines))


def fmt_price(p):
    return f"{p:,.2f}" if p >= 100 else f"{p:.4f}" if p >= 1 else f"{p:.4g}"


def fill_outcomes(cfg):
    """Write the real result into past signals once their horizon has passed."""
    con = db()
    rows = con.execute("SELECT ts, coin, horizon FROM signals WHERE outcome IS NULL").fetchall()
    if rows:
        now_ms = int(time.time() * 1000)
        coins = sorted({r[1] for r in rows} | {"BTC"})
        start = min(r[0] for r in rows) - 2 * H
        px = hourly_prices(coins, start, now_ms)
        with con:
            for ts, coin, hz in rows:
                p, b = px.get(coin), px.get("BTC")
                if p is None or b is None:
                    continue
                c0, c1 = p.get(ts - H), p.get(ts + hz * H - H)
                if c0 and c1:
                    con.execute("UPDATE signals SET outcome=? WHERE ts=? AND coin=? AND horizon=?",
                                ((c1 / c0 - 1) * 100, ts, coin, hz))
    con.close()


def telegram(cfg, text):
    if not cfg.get("telegram") or not text:
        return
    try:
        c = json.load(open("config.json"))
        tok, chat = c.get("telegram_bot_token"), c.get("telegram_chat_id")
        if tok and chat:
            for i in range(0, len(text), 3900):
                requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", timeout=10,
                              data={"chat_id": chat, "text": text[i:i + 3900], "parse_mode": "HTML"})
    except Exception as e:
        log.warning(f"telegram: {e}")


# ============================================================ reports

def report_model(cfg):
    con = db()
    for hz in cfg["horizons_hours"]:
        m = latest_model(con, hz)
        print(f"\n===== {hz}h model =====")
        if not m:
            print("no model yet (needs more data; run --retrain to check)")
            continue
        print(f"version v{m['version']} · trained on {m['n_train']} coin-hours "
              f"({pd.to_datetime(m['from'], unit='ms'):%Y-%m-%d} → {pd.to_datetime(m['to'], unit='ms'):%Y-%m-%d})")
        print("weights (how much 1 typical step of each input moves the forecast of the price move, in %):")
        for f, w in sorted(zip(FNAMES, m["w"]), key=lambda z: -abs(z[1])):
            print(f"  {w:+.3f}  {LABEL[f]}")
        o = m["oos"]
        print(record_text(o))
        print("  how it is made: for every past day, a model trained ONLY on earlier days forecasts each coin-hour"
              " of that day; a forecast beyond ±(threshold + costs) is one simulated LONG/SHORT signal, entered at"
              " that hour's close and closed after the horizon. 'direction right' = price moved the forecast way;"
              " 'profitable' = it also moved more than the costs. List: --unseen " + str(hz))
        for wk, txt in (o.get("by_week") or {}).items():
            print(f"    {wk}: {txt}")
        lv = live_record(con, hz, cfg["cost_pct"])
        print(f"live signals: {lv['n']} finished" + (f" · right {lv['right'] * 100:.0f}% · avg after costs "
                                                      f"{lv['avg']:+.2f}%" if lv["n"] else ""))
    con.close()


def report_unseen(hz):
    """Every simulated signal of the unseen-days test of the latest model for this horizon."""
    con = db()
    m = latest_model(con, hz)
    con.close()
    if not m or not m["oos"].get("list"):
        print("no simulated signals stored for this horizon yet (run --retrain after this update)")
        return
    print(record_text(m["oos"]))
    print(f"\n{'signal time':16} {'side':5} {'coin':10} {'entry':>12} {'forecast':>9} {'exit (+' + str(hz) + 'h)':>12}"
          f" {'real':>8} {'after costs':>11}  direction")
    for r in m["oos"]["list"]:
        print(f"{pd.to_datetime(r['t'], unit='ms'):%Y-%m-%d %H:%M} {r['side']:5} {r['coin']:10} "
              f"{fmt_price(r['entry']):>12} {r['forecast']:+8.2f}% {fmt_price(r['exit']):>12} {r['actual']:+7.2f}% "
              f"{r['net']:+10.2f}%  {'right' if np.sign(r['forecast']) == np.sign(r['actual']) else 'wrong'}")


def report_changes():
    con = db()
    for v, hz, text in con.execute("SELECT version, horizon, text FROM changes ORDER BY version DESC LIMIT 30"):
        print(f"\nv{v} ({hz}h):\n  " + text.replace("\n", "\n  "))
    con.close()


def report_signals(hours):
    con = db()
    since = int(time.time() * 1000) - hours * H
    rows = con.execute("SELECT ts, coin, horizon, version, forecast, status, contributions, outcome FROM signals "
                       "WHERE ts >= ? ORDER BY ts DESC", (since,)).fetchall()
    if not rows:
        print("no signals in this period")
    for ts, coin, hz, v, fc, st, contrib, out in rows:
        c = json.loads(contrib)
        entry = (json.loads(con.execute("SELECT inputs FROM signals WHERE ts=? AND coin=? AND horizon=?",
                                        (ts, coin, hz)).fetchone()[0]) or {}).get("_entry")
        top = sorted(c.items(), key=lambda kv: -abs(kv[1]))[:3]
        side = "LONG " if fc > 0 else "SHORT"
        target = f" → target {fmt_price(entry * (1 + fc / 100))}" if entry else ""
        if out is None:
            res = f"pending until {pd.to_datetime(ts + hz * H, unit='ms'):%m-%d %H:%M}"
        else:
            ex = f" exit {fmt_price(entry * (1 + out / 100))}" if entry else ""
            res = f"real {out:+.2f}%{ex} → {'✅ direction right' if np.sign(out) == np.sign(fc) else '❌ wrong'}"
        print(f"{pd.to_datetime(ts, unit='ms'):%m-%d %H:%M} {side} {coin:10} {hz:2d}h entry "
              f"{fmt_price(entry) if entry else '?'}{target} ({fc:+.2f}%) v{v} · {res} · why: "
              + ", ".join(f"{LABEL[f]} {x:+.2f}" for f, x in top))
    done = [r for r in rows if r[7] is not None]
    if done:
        cfg = load_cfg()
        net = [np.sign(r[4]) * r[7] - cfg["cost_pct"] for r in done]
        print(f"\nlive track record: {len(done)} finished · direction right "
              f"{np.mean([np.sign(r[4]) == np.sign(r[7]) for r in done]):.0%} · profitable after costs "
              f"{np.mean([n > 0 for n in net]):.0%} · avg after costs {np.mean(net):+.2f}%")
    con.close()


# ============================================================ loop

def run(cfg):
    db().close()
    last_hour = None
    log.info("signal engine started")
    while True:
        now = datetime.now(timezone.utc)
        try:
            if last_hour != (now.date(), now.hour) and now.minute >= 5:
                last_hour = (now.date(), now.hour)
                if now.hour % max(int(cfg["retrain_every_hours"]), 1) == 0:
                    retrain(cfg, notify=False)           # new version (see --changes); no Telegram spam
                if now.hour == cfg["daily_summary_hour_utc"]:
                    daily_summary(cfg)
                fill_outcomes(cfg)
                n = make_signals(cfg)
                log.info(f"hourly run: {len(n)} new signal(s)")
        except Exception as e:
            log.exception(f"cycle failed: {e}")
        time.sleep(30)


if __name__ == "__main__":
    import signal as _signal
    _signal.signal(_signal.SIGPIPE, _signal.SIG_DFL)     # quiet exit when piped into `head`
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--changes", action="store_true")
    ap.add_argument("--signals", type=int, metavar="HOURS")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--unseen", type=int, metavar="HORIZON", help="list the simulated test signals (4 or 24)")
    a = ap.parse_args()
    c = load_cfg()
    if a.model:
        report_model(c)
    elif a.changes:
        report_changes()
    elif a.signals:
        report_signals(a.signals)
    elif a.unseen:
        report_unseen(a.unseen)
    elif a.retrain:
        print(retrain(c, notify=False))
    else:
        run(c)
