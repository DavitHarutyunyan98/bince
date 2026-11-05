#!/usr/bin/env python3
"""Social / news collector for crypto (free sources only).

Collects posts with their ORIGINAL creation time, finds which coins they mention
and scores their tone, into social.db. Later an event study measures how price
moved in the hours after mention spikes.

Sources (each runs on its own timer, failures are logged and skipped):
  reddit        subreddits (needs a free Reddit "script" app: client id + secret)
  rss           CoinDesk, Cointelegraph, Decrypt, ... news feeds
  4chan         /biz/ board threads
  telegram      public channels via their t.me/s/<channel> web page
  binance       Binance announcements (new listings / delistings / news)
  cryptopanic   news aggregator (needs a free API token)
  trending      CoinGecko trending coins (stored separately)

Run:   .venv/bin/python social_collector.py              (collects forever)
Stats: .venv/bin/python social_collector.py --stats 24   (top coins of the last 24 h)
Settings: social_config.json (created on first start).
"""
import argparse
import html
import json
import logging
import os
import re
import sqlite3
import threading
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("social_collector.log"), logging.StreamHandler()])
log = logging.getLogger("social")

DB = "social.db"
CONFIG = "social_config.json"
UA = "bince-social-research/1.0 (personal research bot)"
DEFAULTS = {
    "reddit_client_id": "",
    "reddit_client_secret": "",
    "cryptopanic_token": "",
    "subreddits": ["CryptoCurrency", "CryptoMoonShots", "SatoshiStreetBets", "CryptoMarkets", "altcoin",
                   "Bitcoin", "ethereum", "solana", "binance", "wallstreetbets", "memecoins", "defi"],
    "rss_feeds": ["https://www.coindesk.com/arc/outboundfeeds/rss/", "https://cointelegraph.com/rss",
                  "https://decrypt.co/feed", "https://bitcoinmagazine.com/.rss/full/",
                  "https://www.theblock.co/rss.xml", "https://cryptoslate.com/feed/"],
    "telegram_channels": ["binance_announcements", "WatcherGuru", "whale_alert_io", "cointelegraph",
                          "bybit_announcements", "Bitget_Announcements", "crypto"],
    "intervals_seconds": {"reddit": 120, "rss": 300, "4chan": 300, "telegram": 300, "binance": 120,
                          "cryptopanic": 300, "trending": 600, "coins": 21600},
}

# Uppercase words that are also tickers but mostly mean something else: matched only as "$WORD".
AMBIGUOUS = set("""ALL AND THE FOR NOT ONE CEO USA SEC ETF ATH DCA API NFT DAO ICO IPO USD EUR GDP CPI FED
FOMC NEW TOP BIG HOT CAT DOG SUN GAS ACE ALT ANT ARK BAT FUN KEY JOE COW PEOPLE REAL SAFE MOVE NEAR SAND
EDGE GAME HIGH HOOK BOND SIGN MASK LOOK POWER STEP TRUE PUMP MEME WIN BANK CITY GOOD LIFE LOVE MAGIC
NEXT OPEN PRIME RARE RED SAGA SOLO STORY SUPER TIME TOKEN TREE WAVE ZERO DEEP DOGS HOME HYPE JUST KING
LAND LIVE LUCK MOON NOW ORDER PEPE PLAY PRO PUSH QUICK RAY ROSE SCR SPEC TAO TRUMP VANRY WAL
TAKE SOON USELESS HORIZON CORE FORM ARC CYBER APEX APP AUCTION PORTAL FOUR BREAKING JUST NEWS ALERT UPDATE WEEK MAJOR SPOT
FUTURES MARKET PRICE CASH GOLD SILVER OIL DATA FUND BANKS TRADE CHAIN SWAP LAYER BRIDGE AGENT VOTE
SHELL SKY SAFE SAHARA SONIC STORY TOWNS UNITED VIRTUAL WALLET WORLD YIELD ZONE HOLO HOME LIGHT""".split())
# Words that must not be matched as coin NAMES (lower case).
COMMON_NAMES = set("""bitcoin ethereum the one sun people near gas move real safe magic prime story super
time token tree wave zero home live luck moon play quick rose game edge high hook bond sign mask look
power step true pump meme open order dogs bank city good life love next rare land king just hype deep
status common stable form core cyber soon app four portal auction render graph flow ocean origin frontier harmony waves aurora
sonic stacks theta oasis mantle linea scroll starknet zeta blast mode celo kava band dash nano ankr
storj audio api3 radiant gains spell cream alpha beta gamma delta omega pixel voxel quant trust swap bridge
avalanche polygon cosmos arbitrum optimism immutable worldcoin bittensor chainlink internet computer
""".split()) - {"avalanche", "polygon", "cosmos", "arbitrum", "optimism", "chainlink", "worldcoin", "bittensor",
                 "immutable", "starknet"}       # real project names that are safe to match

# Extra tone words for crypto slang (VADER scale -4..+4).
CRYPTO_LEXICON = {"moon": 2.5, "mooning": 3.0, "pump": 1.5, "pumping": 2.0, "bullish": 2.5, "bull": 1.5,
                  "breakout": 2.0, "ath": 2.0, "gem": 2.0, "undervalued": 1.8, "lambo": 2.0, "hodl": 1.0,
                  "buy": 1.0, "long": 0.8, "listing": 1.2, "listed": 1.0, "rally": 2.0, "surge": 2.0,
                  "dump": -2.0, "dumping": -2.5, "bearish": -2.5, "bear": -1.5, "rug": -3.5, "rugpull": -3.5,
                  "scam": -3.0, "hack": -3.0, "hacked": -3.2, "exploit": -3.0, "rekt": -2.5, "short": -0.8,
                  "sell": -1.0, "crash": -3.0, "delist": -2.5, "delisting": -2.5, "liquidated": -2.0,
                  "fud": -1.5, "ponzi": -3.0, "plunge": -2.5, "drain": -2.5}


def load_config():
    cfg = {}
    if os.path.exists(CONFIG):
        try:
            cfg = json.load(open(CONFIG))
        except Exception as e:
            raise SystemExit(f"{CONFIG} is not valid JSON: {e}")
    else:
        json.dump(DEFAULTS, open(CONFIG, "w"), indent=2)
    merged = {**DEFAULTS, **cfg}
    merged["intervals_seconds"] = {**DEFAULTS["intervals_seconds"], **(cfg.get("intervals_seconds") or {})}
    return merged


def db():
    con = sqlite3.connect(DB, timeout=30)
    con.execute("""CREATE TABLE IF NOT EXISTS posts (
        uid TEXT PRIMARY KEY, source TEXT, channel TEXT, created INTEGER, collected INTEGER,
        author TEXT, title TEXT, body TEXT, url TEXT, score INTEGER, comments INTEGER, sentiment REAL)""")
    con.execute("CREATE TABLE IF NOT EXISTS mentions (uid TEXT, coin TEXT, created INTEGER, source TEXT, "
                "PRIMARY KEY (uid, coin))")
    con.execute("CREATE INDEX IF NOT EXISTS ix_mentions ON mentions(coin, created)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_posts ON posts(created)")
    con.execute("CREATE TABLE IF NOT EXISTS trending (ts INTEGER, coin TEXT, rank INTEGER, source TEXT)")
    return con


def strip_html(s):
    return html.unescape(re.sub(r"<[^>]+>", " ", s or "")).strip()


# ============================================================ coin detection & tone

class Analyzer:
    def __init__(self):
        self.lock = threading.Lock()
        self.tickers, self.names = set(), {}
        self.vader = None
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
            self.vader.lexicon.update(CRYPTO_LEXICON)
        except ImportError:
            log.warning("vaderSentiment not installed: tone scores disabled "
                        "(pip install vaderSentiment)")

    def refresh_coins(self):
        """Every coin traded as a USDT perpetual on Binance/Bybit/Bitget (incl. pre-IPO / TradFi
        perpetuals), plus names of the top CoinGecko coins for matching full names."""
        tick = set()
        try:
            for s in requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=20).json()["symbols"]:
                if s.get("quoteAsset") == "USDT" and "PERPETUAL" in str(s.get("contractType", "")):
                    tick.add(s["baseAsset"])
        except Exception as e:
            log.warning(f"coin list Binance: {e}")
        try:
            j = requests.get("https://api.bitget.com/api/v2/mix/market/tickers",
                             params={"productType": "USDT-FUTURES"}, timeout=20).json()
            tick |= {t["symbol"][:-4] for t in j.get("data") or [] if t["symbol"].endswith("USDT")}
        except Exception as e:
            log.warning(f"coin list Bitget: {e}")
        try:
            j = requests.get("https://api.bybit.com/v5/market/tickers", params={"category": "linear"},
                             timeout=20).json()
            tick |= {t["symbol"][:-4] for t in j.get("result", {}).get("list") or [] if t["symbol"].endswith("USDT")}
        except Exception as e:
            log.warning(f"coin list Bybit: {e}")
        tick = {re.sub(r"^(1000000|100000|10000|1000|1M)(?=[A-Z])", "", t) for t in tick}
        names = {}
        for page in (1, 2, 3, 4):
            try:
                rows = requests.get("https://api.coingecko.com/api/v3/coins/markets", timeout=20, params={
                    "vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": page}).json()
                for c in rows:
                    sym, name = c["symbol"].upper(), c["name"].lower()
                    # A name equal to its ticker ("Soon"/SOON, "Common"/COMMON) is matched by the ticker
                    # rules only; as a plain word it would count every normal sentence.
                    if sym in tick and len(name) >= 4 and name not in COMMON_NAMES and " " not in name[:3] \
                            and name.replace(" ", "").upper() != sym:
                        names[name] = sym
                time.sleep(8)      # free CoinGecko limit
            except Exception as e:
                log.warning(f"CoinGecko names page {page} (rate limit, retried next refresh): {str(e)[:80]}")
        if tick:
            with self.lock:
                self.tickers, self.names = tick, names
                # Pre-built patterns: names (any case), long tickers that are real words (any case).
                self.name_re = re.compile(r"\b(" + "|".join(re.escape(n) for n in
                                                            sorted(names, key=len, reverse=True)) + r")\b", re.I) \
                    if names else None
            log.info(f"coin list: {len(tick)} tickers, {len(names)} names")

    def coins_in(self, text):
        found = set()
        with self.lock:
            tick, name_re, names = self.tickers, getattr(self, "name_re", None), self.names
        for m in re.finditer(r"\$([A-Za-z][A-Za-z0-9]{1,14})\b", text):        # $XYZ — any ticker
            if m.group(1).upper() in tick:
                found.add(m.group(1).upper())
        letters = [ch for ch in text if ch.isalpha()]
        shouting = letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.5
        if not shouting:   # in ALL-CAPS headlines every word is capitalised: only $XYZ and names count
            for m in re.finditer(r"\b([A-Z][A-Z0-9]{2,14})\b", text):           # XYZ in capitals
                w = m.group(1)
                if w in tick and w not in AMBIGUOUS:
                    found.add(w)
        for m in re.finditer(r"\b([A-Z][a-z]+[A-Z]?[A-Za-z]*)\b", text):   # OpenAI, Anthropic, SpaceX…
            w = m.group(1).upper()
            if len(w) >= 6 and w in tick and w not in AMBIGUOUS and w.lower() not in COMMON_NAMES:
                found.add(w)
        if name_re:
            for m in name_re.finditer(text):                                      # "solana", "chainlink"…
                found.add(names[m.group(1).lower()])
        return found

    def tone(self, text):
        if not self.vader or not text:
            return None
        return round(self.vader.polarity_scores(text[:2000])["compound"], 3)


# ============================================================ sources

class Collector:
    def __init__(self):
        self.cfg = load_config()
        self.an = Analyzer()
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self.reddit_token = (None, 0)
        self.warned = set()

    def save(self, posts):
        """posts: list of dicts(uid, source, channel, created, author, title, body, url, score, comments)."""
        if not posts:
            return 0
        con = db()
        new = 0
        with con:
            for p in posts:
                text = f"{p.get('title') or ''}\n{p.get('body') or ''}"
                cur = con.execute("INSERT OR IGNORE INTO posts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                    p["uid"], p["source"], p.get("channel", ""), int(p["created"]), int(time.time()),
                    (p.get("author") or "")[:80], (p.get("title") or "")[:500], (p.get("body") or "")[:3000],
                    p.get("url", ""), int(p.get("score") or 0), int(p.get("comments") or 0), self.an.tone(text)))
                if cur.rowcount:
                    new += 1
                    for coin in self.an.coins_in(text):
                        con.execute("INSERT OR IGNORE INTO mentions VALUES (?,?,?,?)",
                                    (p["uid"], coin, int(p["created"]), p["source"]))
                else:   # seen before: keep votes/comments current
                    con.execute("UPDATE posts SET score=?, comments=? WHERE uid=?",
                                (int(p.get("score") or 0), int(p.get("comments") or 0), p["uid"]))
        con.close()
        return new

    def warn_once(self, key, msg):
        if key not in self.warned:
            self.warned.add(key)
            log.warning(msg)

    # ---------------- reddit (OAuth app-only, free)
    def reddit(self):
        cid, sec = self.cfg["reddit_client_id"], self.cfg["reddit_client_secret"]
        if not cid or not sec:
            self.warn_once("reddit", "Reddit skipped: add reddit_client_id / reddit_client_secret to "
                                     f"{CONFIG} (free 'script' app at https://www.reddit.com/prefs/apps)")
            return []
        tok, exp = self.reddit_token
        if not tok or time.time() > exp - 60:
            r = self.s.post("https://www.reddit.com/api/v1/access_token", auth=(cid, sec),
                            data={"grant_type": "client_credentials"}, timeout=15).json()
            tok, exp = r["access_token"], time.time() + int(r.get("expires_in", 3600))
            self.reddit_token = (tok, exp)
        h = {"Authorization": f"bearer {tok}", "User-Agent": UA}
        out = []
        for sub in self.cfg["subreddits"]:
            for kind in ("new", "comments"):
                try:
                    j = self.s.get(f"https://oauth.reddit.com/r/{sub}/{kind}", headers=h,
                                   params={"limit": 100}, timeout=15).json()
                except Exception as e:
                    log.warning(f"reddit r/{sub}/{kind}: {e}")
                    continue
                for c in (j.get("data") or {}).get("children") or []:
                    d = c["data"]
                    out.append({"uid": "reddit:" + d["name"], "source": "reddit", "channel": sub,
                                "created": d["created_utc"], "author": d.get("author"),
                                "title": d.get("title") or "", "body": d.get("selftext") or d.get("body") or "",
                                "url": "https://reddit.com" + (d.get("permalink") or ""),
                                "score": d.get("score"), "comments": d.get("num_comments")})
                time.sleep(1)
        return out

    # ---------------- RSS news
    def rss(self):
        out = []
        for url in self.cfg["rss_feeds"]:
            try:
                root = ET.fromstring(self.s.get(url, timeout=20).content)
            except Exception as e:
                self.warn_once("rss" + url, f"RSS {url}: {str(e)[:100]}")
                continue
            for it in root.iter("item"):
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                try:
                    created = parsedate_to_datetime(it.findtext("pubDate")).timestamp()
                except Exception:
                    created = time.time()
                out.append({"uid": "rss:" + (link or title)[:300], "source": "news", "channel": url.split("/")[2],
                            "created": created, "title": title,
                            "body": strip_html(it.findtext("description") or "")[:2000], "url": link})
        return out

    # ---------------- 4chan /biz/
    def fourchan(self):
        out = []
        try:
            pages = self.s.get("https://a.4cdn.org/biz/catalog.json", timeout=20).json()
        except Exception as e:
            log.warning(f"4chan: {e}")
            return out
        for pg in pages:
            for t in pg.get("threads") or []:
                out.append({"uid": f"4chan:{t['no']}", "source": "4chan", "channel": "biz", "created": t["time"],
                            "title": strip_html(t.get("sub", "")), "body": strip_html(t.get("com", ""))[:2000],
                            "url": f"https://boards.4chan.org/biz/thread/{t['no']}",
                            "comments": t.get("replies", 0)})
        return out

    # ---------------- Telegram public channels (web preview)
    def telegram(self):
        out = []
        for ch in self.cfg["telegram_channels"]:
            try:
                page = self.s.get(f"https://t.me/s/{ch}", timeout=20).text
            except Exception as e:
                self.warn_once("tg" + ch, f"Telegram {ch}: {e}")
                continue
            blocks = page.split('class="tgme_widget_message_wrap')
            if len(blocks) < 2:
                self.warn_once("tg" + ch, f"Telegram {ch}: no public messages (private or wrong name?)")
            for b in blocks[1:]:
                post = re.search(r'data-post="([^"]+)"', b)
                when = re.search(r'<time datetime="([^"]+)"', b)
                text = re.search(r'class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', b, re.S)
                if not post or not when:
                    continue
                created = datetime.fromisoformat(when.group(1).replace("Z", "+00:00")).timestamp()
                out.append({"uid": "tg:" + post.group(1), "source": "telegram", "channel": ch, "created": created,
                            "body": strip_html(text.group(1) if text else "")[:3000],
                            "url": "https://t.me/" + post.group(1)})
            time.sleep(2)
        return out

    # ---------------- Binance announcements
    def binance(self):
        out = []
        for cat, label in ((48, "new listings"), (161, "delistings"), (49, "latest news")):
            try:
                j = self.s.get("https://www.binance.com/bapi/composite/v1/public/cms/article/list/query",
                               params={"type": 1, "catalogId": cat, "pageNo": 1, "pageSize": 20}, timeout=20).json()
            except Exception as e:
                log.warning(f"Binance announcements {label}: {e}")
                continue
            for c in (j.get("data") or {}).get("catalogs") or []:
                for a in c.get("articles") or []:
                    out.append({"uid": f"binance:{a['code']}", "source": "binance", "channel": label,
                                "created": a["releaseDate"] / 1000, "title": a.get("title", ""),
                                "url": f"https://www.binance.com/en/support/announcement/{a['code']}"})
        return out

    # ---------------- CryptoPanic (free token)
    def cryptopanic(self):
        tok = self.cfg["cryptopanic_token"]
        if not tok:
            self.warn_once("cp", f"CryptoPanic skipped: add cryptopanic_token to {CONFIG} "
                                 "(free at https://cryptopanic.com/developers/api/)")
            return []
        out = []
        j = self.s.get("https://cryptopanic.com/api/developer/v2/posts/",
                       params={"auth_token": tok, "public": "true"}, timeout=20).json()
        for p in j.get("results") or []:
            created = datetime.fromisoformat(p["published_at"].replace("Z", "+00:00")).timestamp()
            coins = " ".join("$" + c["code"] for c in p.get("instruments") or p.get("currencies") or [])
            votes = p.get("votes") or {}
            out.append({"uid": f"cp:{p.get('id')}", "source": "cryptopanic", "channel": (p.get("source") or {}).get(
                "domain", ""), "created": created, "title": p.get("title", ""), "body": coins,
                "url": p.get("url", ""), "score": (votes.get("positive", 0) or 0) - (votes.get("negative", 0) or 0)})
        return out

    # ---------------- CoinGecko trending
    def trending(self):
        j = self.s.get("https://api.coingecko.com/api/v3/search/trending", timeout=20).json()
        now = int(time.time())
        rows = [(now, c["item"]["symbol"].upper(), i + 1, "coingecko") for i, c in enumerate(j.get("coins") or [])]
        con = db()
        with con:
            con.executemany("INSERT INTO trending VALUES (?,?,?,?)", rows)
        con.close()
        return []

    # ---------------- loop
    def run(self):
        db().close()
        self.an.refresh_coins()
        jobs = {"reddit": self.reddit, "rss": self.rss, "4chan": self.fourchan, "telegram": self.telegram,
                "binance": self.binance, "cryptopanic": self.cryptopanic, "trending": self.trending,
                "coins": self.an.refresh_coins}
        last = {k: 0 for k in jobs}
        while True:
            for name, fn in jobs.items():
                if time.time() - last[name] < self.cfg["intervals_seconds"][name]:
                    continue
                last[name] = time.time()
                try:
                    posts = fn() or []
                    new = self.save(posts)
                    if posts:
                        log.info(f"{name}: {len(posts)} items, {new} new")
                except Exception as e:
                    log.warning(f"{name} failed: {str(e)[:200]}")
            time.sleep(5)


# ============================================================ quick stats

def stats(hours):
    con = db()
    now = int(time.time())
    since, base_since = now - hours * 3600, now - 7 * 86400
    n_posts = con.execute("SELECT COUNT(*) FROM posts WHERE created >= ?", (since,)).fetchone()[0]
    by_src = con.execute("SELECT source, COUNT(*) FROM posts WHERE created >= ? GROUP BY source", (since,)).fetchall()
    print(f"Posts in the last {hours} h: {n_posts}  ({', '.join(f'{s}: {n}' for s, n in by_src)})")
    rows = con.execute("""
        SELECT m.coin, COUNT(*) AS n, AVG(p.sentiment), COUNT(DISTINCT m.source)
        FROM mentions m JOIN posts p ON p.uid = m.uid
        WHERE m.created >= ? GROUP BY m.coin ORDER BY n DESC LIMIT 40""", (since,)).fetchall()
    print(f"\n{'coin':12} {'mentions':>8} {'normal/'+str(hours)+'h':>11} {'spike x':>8} {'tone':>6} {'sources':>7}")
    for coin, n, tone, srcs in rows:
        base = con.execute("SELECT COUNT(*) FROM mentions WHERE coin=? AND created >= ? AND created < ?",
                           (coin, base_since, since)).fetchone()[0]
        span_h = max((since - max(base_since, con.execute("SELECT MIN(created) FROM posts").fetchone()[0] or since))
                     / 3600, 1)
        normal = base / span_h * hours
        spike = n / normal if normal > 0 else float("inf")
        print(f"{coin:12} {n:8d} {normal:11.1f} {spike:8.1f} {tone if tone is not None else 0:6.2f} {srcs:7d}")
    tr = con.execute("SELECT coin, MIN(rank) FROM trending WHERE ts >= ? GROUP BY coin ORDER BY MIN(rank) LIMIT 15",
                     (since,)).fetchall()
    if tr:
        print("\nCoinGecko trending:", ", ".join(f"{c}(#{r})" for c, r in tr))
    con.close()


def rescan():
    """Recompute coin mentions of all stored posts with the current matching rules."""
    an = Analyzer()
    an.refresh_coins()
    con = db()
    rows = con.execute("SELECT uid, title, body, created, source FROM posts").fetchall()
    with con:
        con.execute("DELETE FROM mentions")
        for uid, title, body, created, source in rows:
            for coin in an.coins_in(f"{title or ''}\n{body or ''}"):
                con.execute("INSERT OR IGNORE INTO mentions VALUES (?,?,?,?)", (uid, coin, created, source))
    n = con.execute("SELECT COUNT(*) FROM mentions").fetchone()[0]
    con.close()
    print(f"rescanned {len(rows)} posts -> {n} mentions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=int, metavar="HOURS", help="print top mentioned coins and exit")
    ap.add_argument("--rescan", action="store_true", help="recompute mentions of stored posts and exit")
    a = ap.parse_args()
    if a.rescan:
        rescan()
    elif a.stats:
        stats(a.stats)
    else:
        Collector().run()
