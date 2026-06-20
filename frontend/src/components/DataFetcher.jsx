import React, { useState } from "react";
import { api } from "../api/client.js";

const TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"];

const DEFAULTS = {
  symbol: "BTCUSDT",
  timeframe: "15m",
  start: "2024-01-01",
  end: "2024-03-31",
  limit: 500,
};

export default function DataFetcher() {
  const [form, setForm] = useState(DEFAULTS);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const fetchData = async () => {
    setError(null);
    setLoading(true);
    setData(null);
    try {
      const r = await api.fetchData({
        symbol: form.symbol.trim().toUpperCase(),
        timeframe: form.timeframe,
        start: form.start,
        end: form.end,
        limit: Number(form.limit),
      });
      setData(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const candles = data?.candles ?? [];
  // Show a window around the start/end so big ranges stay readable.
  const preview = candles.length > 50
    ? [...candles.slice(0, 25), ...candles.slice(-25)]
    : candles;

  return (
    <div className="panel">
      <h2>Historical Data Fetcher</h2>
      <div className="grid">
        <div>
          <label>Symbol</label>
          <input value={form.symbol} onChange={(e) => set("symbol", e.target.value)} />
        </div>
        <div>
          <label>Timeframe</label>
          <select value={form.timeframe} onChange={(e) => set("timeframe", e.target.value)}>
            {TIMEFRAMES.map((t) => <option key={t}>{t}</option>)}
          </select>
        </div>
        <div>
          <label>Start</label>
          <input type="date" value={form.start} onChange={(e) => set("start", e.target.value)} />
        </div>
        <div>
          <label>End</label>
          <input type="date" value={form.end} onChange={(e) => set("end", e.target.value)} />
        </div>
        <div>
          <label>Max candles (preview)</label>
          <input type="number" value={form.limit} onChange={(e) => set("limit", e.target.value)} />
        </div>
      </div>

      <div className="spacer" />
      <div className="row">
        <button className="btn" onClick={fetchData} disabled={loading}>
          {loading ? "Fetching…" : "⤓ Fetch Data"}
        </button>
        {data && (
          <span className="muted">
            {data.total_candles} candles ({data.returned} shown after down-sampling)
          </span>
        )}
      </div>
      {error && (
        <p style={{ color: "var(--red)", whiteSpace: "pre-wrap", marginTop: 8 }}>
          ⚠ Data fetch failed: {error}
        </p>
      )}

      {candles.length > 0 && (
        <>
          <div className="spacer" />
          <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Time</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th>
              </tr>
            </thead>
            <tbody>
              {preview.map((c, i) => (
                <tr key={i}>
                  <td>{c.time.replace("T", " ").slice(0, 16)}</td>
                  <td>{c.open}</td>
                  <td>{c.high}</td>
                  <td>{c.low}</td>
                  <td>{c.close}</td>
                  <td>{c.volume}</td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        </>
      )}
    </div>
  );
}
