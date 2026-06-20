// Centralized API client. In production set VITE_API_BASE to the FastAPI URL
// (e.g. https://your-server.com). In dev it defaults to "/api" which Vite
// proxies to http://localhost:8000.
const BASE = import.meta.env.VITE_API_BASE || "/api";

// ---------------------------------------------------------------------------
// Activity log: a tiny pub/sub so the UI can show a live "terminal" of every
// action (request, success, error) the user triggers.
// ---------------------------------------------------------------------------
const _logEntries = [];
const _logListeners = new Set();
let _logSeq = 0;

export function logEvent(level, message) {
  const entry = {
    id: ++_logSeq,
    ts: new Date().toISOString(),
    level, // "info" | "success" | "error"
    message,
  };
  _logEntries.push(entry);
  if (_logEntries.length > 500) _logEntries.shift();
  _logListeners.forEach((fn) => fn(entry));
  return entry;
}

export function subscribeLog(fn) {
  _logListeners.add(fn);
  return () => _logListeners.delete(fn);
}

export function getLogEntries() {
  return _logEntries.slice();
}

async function req(path, opts = {}) {
  const url = `${BASE}${path}`;
  const method = opts.method || "GET";
  logEvent("info", `→ ${method} ${path}`);
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
    const msg =
      `Cannot reach the API at "${url}". Is the backend (FastAPI on ` +
      `http://localhost:8000) running? In a deployed frontend, set ` +
      `VITE_API_BASE to the backend URL. (${e.message})`;
    logEvent("error", `✖ ${method} ${path} — ${msg}`);
    throw new Error(msg);
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
    const msg = detail || `Request failed with status ${res.status}`;
    logEvent("error", `✖ ${method} ${path} [${res.status}] — ${msg}`);
    throw new Error(msg);
  }
  logEvent("success", `✓ ${method} ${path} [${res.status}]`);
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
