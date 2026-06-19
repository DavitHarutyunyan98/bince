# Bince Backend

FastAPI control plane for the trading system. Wraps the existing optimizer
(`main.py`) and bot (`hybrid_trader.py`) — neither of which was rewritten — so a
React dashboard can drive everything over HTTP.

```
React (Vercel) ──HTTP──► FastAPI (app.py)
                            ├─► Redis + RQ worker (worker_tasks.py) ─► main.py optimizer
                            ├─► SQLite (db.py)  live logs + results
                            ├─► trade_config.json  (config endpoints)
                            └─► bot_service.py  start/stop/status hybrid_trader.py
```

## Run locally

```bash
# 1. Redis
redis-server --daemonize yes

# 2. Python deps
python3 -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt

# 3. API + worker together
bash start.sh
# API on http://localhost:8000  (docs at /docs)
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/optimization/start` | Enqueue a job, returns `job_id` instantly |
| POST | `/optimization/stop/{job_id}` | Signal stop (via Redis) |
| POST | `/optimization/pause/{job_id}` | Toggle pause/resume |
| GET | `/optimization/status/{job_id}` | Job status + paused flag |
| GET | `/optimization/logs/{job_id}?offset=N` | Paged live log lines |
| GET | `/optimization/results/{job_id}` | Result rows (JSON) |
| GET / PUT | `/config/trade` | Read / write `trade_config.json` |
| GET | `/config/app` | Read `config.json` (secrets masked) |
| GET | `/pairs/usdt-futures` | Live Binance USDT-futures pairs |
| GET | `/bot/status` | Hybrid bot running state + pid |
| POST | `/bot/start` / `/bot/stop` | Start / stop the 24/7 bot |

## How it works

- **No globals across requests.** The worker writes every log line and the final
  results to SQLite (`db.py`); the API reads from there. Optimization globals in
  `main.py` are only touched inside the worker process.
- **Stop/pause cross processes.** Because the worker runs in a separate process,
  the API can't share a `threading.Event`. Instead `worker_tasks._RedisEvent`
  mimics `threading.Event` but reads/writes a Redis key, and the API flips that
  key. `main.py`'s optimizer already polls `stop_event.is_set()` each trial.
- **The optimizer is unchanged.** `worker_tasks.run_optimization` builds the same
  arguments the old Dash callback (`run_optimization_task`) passed and calls
  `FuturesTrader.optimize_trading_pairs[...]`.

## Production (always-on)

See `deploy/`:
- `bince-api.service`, `bince-worker.service`, `bince-bot.service` — systemd units
- `windows-nssm.md` — equivalent NSSM setup for Windows

```bash
sudo cp deploy/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now bince-api bince-worker bince-bot
```

> Run the bot EITHER via the `bince-bot` systemd unit OR via the API's
> `/bot/start` — never both, or two instances will trade the same pairs.

## Notes / TODO

- The plan's `ThreadPoolExecutor → ProcessPoolExecutor` swap in
  `optimize_trading_pairs` is intentionally **not** done here — `main.py` warns
  that multiprocessing deadlocks in that path. Since each optimization job now
  runs in its own RQ worker process already, you get process isolation for free;
  the per-pair fan-out can be migrated to `ProcessPoolExecutor` as a follow-up
  once the deadlock cause is addressed.
- `stability_metrics.py` is currently a stub; drop in the real implementation
  when ready (the import contract is `StabilityAnalyzer` +
  `integrate_stability_into_optimization`).
