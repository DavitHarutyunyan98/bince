"""
RQ worker task: runs the optimization in a background process.
All log messages and partial results are written to SQLite so the API
can stream them back to the React frontend without touching Python globals.
"""
import sys
import os
import threading
import json

# Make sure the project root is importable
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import redis
from binance.client import Client

from db import append_log, update_job_status, save_results, get_job

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_r = redis.from_url(REDIS_URL)

STOP_KEY = "opt:stop:{job_id}"
PAUSE_KEY = "opt:pause:{job_id}"


def _stop_key(job_id):
    return f"opt:stop:{job_id}"


def _pause_key(job_id):
    return f"opt:pause:{job_id}"


class _RedisEvent:
    """
    Drop-in replacement for threading.Event that reads its state from Redis.
    This lets the stop/pause signal cross process boundaries.
    """

    def __init__(self, key: str, r: redis.Redis):
        self._key = key
        self._r = r

    def is_set(self):
        return self._r.exists(self._key) > 0

    def set(self):
        self._r.set(self._key, "1", ex=3600)

    def clear(self):
        self._r.delete(self._key)


class _SqliteLogger:
    """Writes log lines to SQLite so the API can page them."""

    def __init__(self, job_id: str):
        self._job_id = job_id

    def __call__(self, message: str):
        append_log(self._job_id, message)


def run_optimization(job_id: str, settings: dict):
    """
    Entry point called by the RQ worker.
    Runs the optimizer from main.py's FuturesTrader class, writing all progress
    and results to SQLite.
    """
    update_job_status(job_id, "running")
    log = _SqliteLogger(job_id)

    stop_event = _RedisEvent(_stop_key(job_id), _r)
    pause_event = _RedisEvent(_pause_key(job_id), _r)

    # Clean up any stale stop/pause signals from a previous run
    stop_event.clear()
    pause_event.clear()

    try:
        log("🔧 Loading configuration...")

        config_path = os.path.join(ROOT, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)

        client = Client(cfg["api_key"], cfg["secret_key"])

        # Import here so we don't pull in Dash at import time
        from main import FuturesTrader, TIMEFRAME_TO_ANNUALIZATION_FACTOR, add_optimization_log
        import main as main_module

        # Monkey-patch the global log function so it also writes to SQLite
        original_add_log = main_module.add_optimization_log

        def patched_log(message):
            original_add_log(message)
            log(message)

        main_module.add_optimization_log = patched_log

        trader = FuturesTrader(client, cfg)

        pairs = settings["pairs"]
        param_ranges = {
            "buy_signal_window": settings.get("bw_str", ""),
            "buy_pattern_lookback": settings.get("bl_str", ""),
            "sell_signal_window": settings.get("sw_str", ""),
            "sell_pattern_lookback": settings.get("slr_str", ""),
            "atr_period": settings.get("atr_period_str", ""),
            "atr_multiplier": settings.get("atr_mult_str", ""),
        }
        weights = {
            "total_return": settings.get("weight_return", 1.0),
            "win_rate": settings.get("weight_winrate", 0.5),
            "total_trades": settings.get("weight_trades", 0.2),
        }

        optimization_mode = settings.get("optimization_mode", "efficient")
        strategy_name = settings.get("strategy_name", "Candlestick Patterns")
        use_stability = settings.get("use_stability", False)

        log(f"🚀 Starting {'STABILITY-BASED' if use_stability else 'STANDARD'} optimization "
            f"({optimization_mode.upper()} mode) with strategy: {strategy_name}")
        log(f"   Pairs: {pairs}")

        common_kwargs = dict(
            trading_pairs=pairs,
            param_ranges=param_ranges,
            selected_params=settings.get("selected_params", list(param_ranges.keys())),
            is_start_date=settings["is_start"],
            is_end_date=settings["is_end"],
            oos1_start_date=settings["oos1_start"],
            oos1_end_date=settings["oos1_end"],
            oos2_start_date=settings["oos2_start"],
            oos2_end_date=settings["oos2_end"],
            timeframe=settings.get("timeframe", "15m"),
            min_trades=settings.get("min_trades", 10),
            n_trials=settings.get("n_trials", 50),
            weights=weights,
            min_candles=settings.get("min_candles", 100),
            stop_event=stop_event,
            pause_event=pause_event,
            optimization_mode=optimization_mode,
            strategy_name=strategy_name,
        )

        if use_stability:
            df_results = trader.optimize_trading_pairs_with_stability(
                **common_kwargs,
                stability_weight=settings.get("stability_weight", 0.5),
            )
        else:
            df_results = trader.optimize_trading_pairs(**common_kwargs)

        # Restore original log function
        main_module.add_optimization_log = original_add_log

        if df_results is None or df_results.empty:
            log("⚠️ Optimization finished with no valid results.")
            update_job_status(job_id, "finished")
            save_results(job_id, [])
            return

        # Post-process: add trade difference column
        if "Long_Trades" in df_results.columns and "Short_Trades" in df_results.columns:
            df_results["Trade_Difference"] = abs(
                df_results["Long_Trades"] - df_results["Short_Trades"]
            )

        sort_col = (
            "Stability_Performance_Score"
            if "Stability_Performance_Score" in df_results.columns
            else "Score"
        )
        df_results = df_results.sort_values(by=sort_col, ascending=False).round(4)

        records = df_results.to_dict("records")
        save_results(job_id, records)

        final_status = "stopped" if stop_event.is_set() else "finished"
        log(f"✅ Optimization {'stopped' if stop_event.is_set() else 'complete'}. "
            f"{len(records)} total results saved.")
        update_job_status(job_id, final_status)

    except Exception as exc:
        log(f"❌ FATAL ERROR in worker: {exc}")
        import traceback
        log(traceback.format_exc())
        update_job_status(job_id, "failed")
