// Centralized API client. In production set VITE_API_BASE to the FastAPI URL
// (e.g. https://your-server.com). In dev it defaults to "/api" which Vite
// proxies to http://localhost:8000.
const BASE = import.meta.env.VITE_API_BASE || "/api";

async function req(path, opts = {}) {
  const url = `${BASE}${path}`;
  let res;
  try {
    res = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
  } catch (e) {
    // fetch() rejects with a TypeError ("Failed to fetch") when the backend is
    // unreachable, blocked by CORS, or the dev proxy target is down. Turn that
    // opaque message into something actionable.
    throw new Error(
      `Cannot reach the API at "${url}". Is the backend (FastAPI on ` +
        `http://localhost:8000) running? In a deployed frontend, set ` +
        `VITE_API_BASE to the backend URL. (${e.message})`
    );
  }
  if (!res.ok) {
    // FastAPI returns errors as {"detail": "..."}; surface that message
    // directly so the UI can explain *why* a request failed.
    const text = await res.text();
    let detail = text;
    try {
      const body = JSON.parse(text);
      detail = typeof body?.detail === "string"
        ? body.detail
        : JSON.stringify(body?.detail ?? body);
    } catch {
      // body wasn't JSON; fall back to raw text
    }
    throw new Error(detail || `Request failed with status ${res.status}`);
  }
  return res.json();
}

export const api = {
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

  // Historical data fetching
  fetchData: ({ symbol, timeframe, start, end, limit = 500 }) =>
    req(
      `/data/fetch?symbol=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(timeframe)}` +
        `&start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}&limit=${limit}`
    ),

  // Bot (Phase 3)
  getBotStatus: () => req("/bot/status"),
  startBot: () => req("/bot/start", { method: "POST" }),
  stopBot: () => req("/bot/stop", { method: "POST" }),
};
