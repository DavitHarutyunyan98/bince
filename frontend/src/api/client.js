// Centralized API client. In production set VITE_API_BASE to the FastAPI URL
// (e.g. https://your-server.com). In dev it defaults to "/api" which Vite
// proxies to http://localhost:8000.
const BASE = import.meta.env.VITE_API_BASE || "/api";

async function req(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

export const api = {
  // Health
  getHealth: () => req("/health"),

  // Optimization
  startOptimization: (settings) =>
    req("/optimization/start", { method: "POST", body: JSON.stringify(settings) }),
  stopOptimization: (jobId) =>
    req(`/optimization/stop/${jobId}`, { method: "POST" }),
  togglePause: (jobId) =>
    req(`/optimization/pause/${jobId}`, { method: "POST" }),
  getStatus: (jobId) => req(`/optimization/status/${jobId}`),
  getLogs: (jobId, offset = 0) =>
    req(`/optimization/logs/${jobId}?offset=${offset}`),
  getResults: (jobId) => req(`/optimization/results/${jobId}`),

  // Config
  getTradeConfig: () => req("/config/trade"),
  saveTradeConfig: (body) =>
    req("/config/trade", { method: "PUT", body: JSON.stringify(body) }),
  getAppConfig: () => req("/config/app"),

  // Pairs
  getUsdtFutures: () => req("/pairs/usdt-futures"),

  // Bot (Phase 3)
  getBotStatus: () => req("/bot/status"),
  startBot: () => req("/bot/start", { method: "POST" }),
  stopBot: () => req("/bot/stop", { method: "POST" }),
};
