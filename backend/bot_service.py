"""
Process manager for the 24/7 hybrid trading bot.

The bot (hybrid_trader.HybridTraderManager) blocks when started, so we run it as
a detached child process and track it with a PID file. This lets the API
start/stop/status it without blocking the web server, and survives API restarts
because the PID is persisted on disk.

For true always-on operation (survive machine reboots), run the API itself under
systemd/NSSM and optionally have the bot supervised separately — see the unit
files in backend/deploy/.
"""
import os
import sys
import json
import signal
import subprocess

ROOT = os.path.dirname(os.path.dirname(__file__))
PID_FILE = os.path.join(os.path.dirname(__file__), "bot.pid")
BOT_LOG = os.path.join(os.path.dirname(__file__), "bot.out.log")
TRADE_CONFIG = os.path.join(ROOT, "trade_config.json")


def _read_pid():
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _pid_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)  # signal 0 just checks existence
        return True
    except OSError:
        return False


def _count_enabled_pairs():
    try:
        with open(TRADE_CONFIG) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return sum(1 for c in data if c.get("enabled", True))
    except Exception:
        return None


def status():
    pid = _read_pid()
    alive = _pid_alive(pid)
    if not alive and pid is not None:
        # Stale pid file — clean it up.
        try:
            os.remove(PID_FILE)
        except OSError:
            pass
        pid = None
    return {
        "running": alive,
        "pid": pid if alive else None,
        "pairs": _count_enabled_pairs(),
        "message": "Bot is running." if alive else "Bot is stopped.",
    }


def start():
    if _pid_alive(_read_pid()):
        return {"running": True, "message": "Bot already running.", "pid": _read_pid()}

    pairs = _count_enabled_pairs()
    if not pairs:
        return {"running": False, "message": "No enabled pairs in trade_config.json."}

    log_fh = open(BOT_LOG, "a")
    # Launch the bot module's __main__ as a detached process group leader so we
    # can signal the whole tree on stop.
    proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "hybrid_trader.py")],
        cwd=ROOT,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach into its own process group
    )
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))
    return {"running": True, "pid": proc.pid, "message": f"Bot started with {pairs} pairs."}


def stop():
    pid = _read_pid()
    if not _pid_alive(pid):
        return {"running": False, "message": "Bot is not running."}
    try:
        # Kill the whole process group (negative pid).
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    try:
        os.remove(PID_FILE)
    except OSError:
        pass
    return {"running": False, "message": "Bot stopped."}
