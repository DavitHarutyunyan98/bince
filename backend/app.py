"""
FastAPI backend for the trading dashboard.
Endpoints:
  POST   /optimization/start          start a background optimization job
  POST   /optimization/stop/{job_id}  stop a running job
  POST   /optimization/pause/{job_id} pause / resume a running job
  GET    /optimization/status/{job_id}
  GET    /optimization/logs/{job_id}?offset=0
  GET    /optimization/results/{job_id}
  GET    /config/trade                 read trade_config.json
  PUT    /config/trade                 save trade_config.json
  GET    /config/app                   read config.json (credentials masked)
  GET    /pairs/usdt-futures           list all USDT futures pairs from Binance
"""
import os
import sys
import json
import uuid
import logging

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import redis
from typing import Union
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rq import Queue
from rq.job import Job

from db import init_db, create_job, get_job, get_logs, get_results, update_job_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TRADE_CONFIG_PATH = os.path.join(ROOT, "trade_config.json")
APP_CONFIG_PATH = os.path.join(ROOT, "config.json")

_r = redis.from_url(REDIS_URL)
_q = Queue("optimizer", connection=_r)

app = FastAPI(title="Trading Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class OptimizationSettings(BaseModel):
    pairs: list[str]
    selected_params: list[str] = [
        "buy_signal_window", "buy_pattern_lookback",
        "sell_signal_window", "sell_pattern_lookback",
    ]
    bw_str: str = "5,20,1"
    bl_str: str = "1,10,1"
    sw_str: str = "5,20,1"
    slr_str: str = "1,10,1"
    atr_period_str: str = ""
    atr_mult_str: str = ""
    is_start: str = "2024-01-01"
    is_end: str = "2024-09-30"
    oos1_start: str = "2024-10-01"
    oos1_end: str = "2024-12-31"
    oos2_start: str = "2025-01-01"
    oos2_end: str = "2025-03-31"
    timeframe: str = "15m"
    min_trades: int = 10
    n_trials: int = 50
    min_candles: int = 100
    weight_return: float = 1.0
    weight_winrate: float = 0.5
    weight_trades: float = 0.2
    optimization_mode: str = "efficient"
    strategy_name: str = "Candlestick Patterns"
    use_stability: bool = False
    stability_weight: float = 0.5
    use_trade_balance_filter: bool = False
    max_trade_ratio: float = 3.0
    executor_type: str = "process"  # "process" (all cores) or "thread" (fallback)


# ---------------------------------------------------------------------------
# Optimization endpoints
# ---------------------------------------------------------------------------

@app.post("/optimization/start")
def start_optimization(settings: OptimizationSettings):
    job_id = str(uuid.uuid4())
    settings_dict = settings.model_dump()
    create_job(job_id, settings_dict)

    from worker_tasks import run_optimization
    rq_job = _q.enqueue(
        run_optimization,
        args=(job_id, settings_dict),
        job_id=job_id,
        job_timeout=7200,  # 2 hours max
    )
    logger.info("Enqueued optimization job %s", job_id)
    return {"job_id": job_id, "status": "pending"}


@app.post("/optimization/stop/{job_id}")
def stop_optimization(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    _r.set(f"opt:stop:{job_id}", "1", ex=3600)
    update_job_status(job_id, "stopping")
    return {"job_id": job_id, "status": "stopping"}


@app.post("/optimization/pause/{job_id}")
def toggle_pause(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    pause_key = f"opt:pause:{job_id}"
    if _r.exists(pause_key):
        _r.delete(pause_key)
        return {"job_id": job_id, "paused": False}
    else:
        _r.set(pause_key, "1", ex=3600)
        return {"job_id": job_id, "paused": True}


@app.get("/optimization/status/{job_id}")
def get_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    paused = bool(_r.exists(f"opt:pause:{job_id}"))
    return {
        "job_id": job_id,
        "status": job["status"],
        "paused": paused,
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


@app.get("/optimization/logs/{job_id}")
def get_job_logs(job_id: str, offset: int = 0):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    logs = get_logs(job_id, offset=offset)
    return {"job_id": job_id, "logs": logs, "count": len(logs)}


@app.get("/optimization/results/{job_id}")
def get_job_results(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    records = get_results(job_id)
    return {
        "job_id": job_id,
        "status": job["status"],
        "count": len(records),
        "results": records,
    }


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------

@app.get("/config/trade")
def get_trade_config():
    try:
        with open(TRADE_CONFIG_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


@app.put("/config/trade")
def save_trade_config(body: Union[list, dict] = Body(...)):
    with open(TRADE_CONFIG_PATH, "w") as f:
        json.dump(body, f, indent=2)
    return {"saved": True}


@app.get("/config/app")
def get_app_config():
    try:
        with open(APP_CONFIG_PATH) as f:
            cfg = json.load(f)
        # Mask sensitive fields
        masked = {k: ("***" if k in ("api_key", "secret_key", "telegram_bot_token") else v)
                  for k, v in cfg.items()}
        return masked
    except FileNotFoundError:
        raise HTTPException(404, "config.json not found")


# ---------------------------------------------------------------------------
# Pairs endpoint
# ---------------------------------------------------------------------------

@app.get("/pairs/usdt-futures")
def list_usdt_futures():
    try:
        from binance.client import Client
        with open(APP_CONFIG_PATH) as f:
            cfg = json.load(f)
        client = Client(cfg["api_key"], cfg["secret_key"])
        tickers = client.futures_ticker()
        pairs = sorted(
            [{"symbol": t["symbol"], "volume": float(t["quoteVolume"]),
              "change_pct": float(t["priceChangePercent"])}
             for t in tickers if t["symbol"].endswith("USDT")],
            key=lambda x: x["volume"],
            reverse=True,
        )
        return {"count": len(pairs), "pairs": pairs}
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Bot control endpoints (Phase 3)
# ---------------------------------------------------------------------------

@app.get("/bot/status")
def bot_status():
    import bot_service
    return bot_service.status()


@app.post("/bot/start")
def bot_start():
    import bot_service
    return bot_service.start()


@app.post("/bot/stop")
def bot_stop():
    import bot_service
    return bot_service.stop()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
