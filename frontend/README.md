# Bince Dashboard (Frontend)

React + Vite dashboard for the trading backend. Deploys to Vercel.

## Local development

```bash
npm install
npm run dev      # serves on http://localhost:5173
```

Vite proxies `/api/*` to the FastAPI backend at `http://localhost:8000`
(see `vite.config.js`). Make sure the backend is running:

```bash
cd ../backend && bash start.sh
```

## Production / Vercel

The frontend talks to the backend through `VITE_API_BASE`. In the Vercel
project settings, add an environment variable:

```
VITE_API_BASE = https://your-backend-host.com
```

Then deploy:

```bash
npm i -g vercel
vercel --prod
```

If `VITE_API_BASE` is unset, the client falls back to `/api` (dev proxy).

## Screens

- **Optimizer** — controls (pairs, strategy, mode, param ranges, date ranges),
  start/stop/pause, and a live log view that polls `/optimization/logs`.
- **Results** — sortable table of optimization results, best-per-pair toggle,
  CSV export. Polls `/optimization/results` while a job runs.
- **Trade Config** — JSON editor for `trade_config.json` (GET/PUT `/config/trade`).
- **Bot Control** — start/stop/status for the 24/7 hybrid trader (Phase 3).
