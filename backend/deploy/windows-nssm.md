# Running on Windows with NSSM

[NSSM](https://nssm.cc/) wraps any executable as a Windows service.

Assuming the project is at `C:\bince` with a venv at `C:\bince\.venv`:

## 1. API service

```bat
nssm install BinceAPI "C:\bince\.venv\Scripts\uvicorn.exe" "app:app --host 0.0.0.0 --port 8000"
nssm set BinceAPI AppDirectory C:\bince\backend
nssm set BinceAPI AppEnvironmentExtra REDIS_URL=redis://localhost:6379/0 PYTHONPATH=C:\bince;C:\bince\backend
nssm start BinceAPI
```

## 2. Optimization worker

```bat
nssm install BinceWorker "C:\bince\.venv\Scripts\rq.exe" "worker optimizer --url redis://localhost:6379/0"
nssm set BinceWorker AppDirectory C:\bince\backend
nssm set BinceWorker AppEnvironmentExtra PYTHONPATH=C:\bince;C:\bince\backend
nssm start BinceWorker
```

## 3. Trading bot (always-on)

```bat
nssm install BinceBot "C:\bince\.venv\Scripts\python.exe" "C:\bince\hybrid_trader.py"
nssm set BinceBot AppDirectory C:\bince
nssm start BinceBot
```

You also need Redis on Windows (use Memurai or WSL). As with the systemd setup,
run the bot EITHER as this NSSM service OR via the API's /bot/start — not both.
