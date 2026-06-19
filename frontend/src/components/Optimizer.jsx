import React, { useState, useEffect, useRef } from "react";
import { api } from "../api/client.js";

const DEFAULTS = {
  pairs: "BTCUSDT,ETHUSDT",
  strategy_name: "Candlestick Patterns",
  optimization_mode: "efficient",
  timeframe: "15m",
  n_trials: 50,
  min_trades: 10,
  min_candles: 100,
  bw_str: "5,20,1",
  bl_str: "1,10,1",
  sw_str: "5,20,1",
  slr_str: "1,10,1",
  atr_period_str: "",
  atr_mult_str: "",
  is_start: "2024-01-01",
  is_end: "2024-09-30",
  oos1_start: "2024-10-01",
  oos1_end: "2024-12-31",
  oos2_start: "2025-01-01",
  oos2_end: "2025-03-31",
  weight_return: 1.0,
  weight_winrate: 0.5,
  weight_trades: 0.2,
  use_stability: false,
  stability_weight: 0.5,
  executor_type: "process",
};

const STRATEGIES = [
  "Candlestick Patterns",
  "ATR SuperTrend",
  "RSI",
  "MA Crossover",
];
const MODES = ["efficient", "smart", "comprehensive"];
const TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"];

export default function Optimizer({ jobId, setJobId }) {
  const [form, setForm] = useState(DEFAULTS);
  const [status, setStatus] = useState("idle");
  const [paused, setPaused] = useState(false);
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState(null);
  const logRef = useRef(null);
  const offsetRef = useRef(0);
  const pollRef = useRef(null);

  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  // Poll status + logs while a job is active.
  useEffect(() => {
    if (!jobId) return;
    offsetRef.current = 0;
    setLogs([]);

    const poll = async () => {
      try {
        const st = await api.getStatus(jobId);
        setStatus(st.status);
        setPaused(st.paused);
        const res = await api.getLogs(jobId, offsetRef.current);
        if (res.logs.length) {
          offsetRef.current += res.logs.length;
          setLogs((prev) => [...prev, ...res.logs]);
        }
        if (["finished", "failed", "stopped"].includes(st.status)) {
          clearInterval(pollRef.current);
        }
      } catch (e) {
        setError(e.message);
      }
    };
    poll();
    pollRef.current = setInterval(poll, 1500);
    return () => clearInterval(pollRef.current);
  }, [jobId]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const start = async () => {
    setError(null);
    try {
      const payload = {
        ...form,
        pairs: form.pairs.split(",").map((s) => s.trim().toUpperCase()).filter(Boolean),
        n_trials: Number(form.n_trials),
        min_trades: Number(form.min_trades),
        min_candles: Number(form.min_candles),
        weight_return: Number(form.weight_return),
        weight_winrate: Number(form.weight_winrate),
        weight_trades: Number(form.weight_trades),
        stability_weight: Number(form.stability_weight),
      };
      const r = await api.startOptimization(payload);
      setJobId(r.job_id);
      setStatus("pending");
    } catch (e) {
      setError(e.message);
    }
  };

  const stop = async () => {
    if (jobId) await api.stopOptimization(jobId);
  };
  const togglePause = async () => {
    if (jobId) {
      const r = await api.togglePause(jobId);
      setPaused(r.paused);
    }
  };

  const running = ["pending", "running", "stopping"].includes(status);

  return (
    <>
      <div className="panel">
        <h2>Optimizer Controls</h2>
        <div className="grid">
          <div>
            <label>Trading Pairs (comma-separated)</label>
            <input value={form.pairs} onChange={(e) => set("pairs", e.target.value)} />
          </div>
          <div>
            <label>Strategy</label>
            <select value={form.strategy_name} onChange={(e) => set("strategy_name", e.target.value)}>
              {STRATEGIES.map((s) => <option key={s}>{s}</option>)}
            </select>
          </div>
          <div>
            <label>Mode</label>
            <select value={form.optimization_mode} onChange={(e) => set("optimization_mode", e.target.value)}>
              {MODES.map((m) => <option key={m}>{m}</option>)}
            </select>
          </div>
          <div>
            <label>Timeframe</label>
            <select value={form.timeframe} onChange={(e) => set("timeframe", e.target.value)}>
              {TIMEFRAMES.map((t) => <option key={t}>{t}</option>)}
            </select>
          </div>
          <div>
            <label>Trials per pair</label>
            <input type="number" value={form.n_trials} onChange={(e) => set("n_trials", e.target.value)} />
          </div>
          <div>
            <label>Min trades</label>
            <input type="number" value={form.min_trades} onChange={(e) => set("min_trades", e.target.value)} />
          </div>
          <div>
            <label>Parallelism</label>
            <select value={form.executor_type} onChange={(e) => set("executor_type", e.target.value)}>
              <option value="process">Process (all cores)</option>
              <option value="thread">Thread (fallback)</option>
            </select>
          </div>
        </div>

        <div className="spacer" />
        <h2>Parameter Ranges (low,high,step)</h2>
        <div className="grid">
          <div><label>Buy Signal Window</label><input value={form.bw_str} onChange={(e) => set("bw_str", e.target.value)} /></div>
          <div><label>Buy Pattern Lookback</label><input value={form.bl_str} onChange={(e) => set("bl_str", e.target.value)} /></div>
          <div><label>Sell Signal Window</label><input value={form.sw_str} onChange={(e) => set("sw_str", e.target.value)} /></div>
          <div><label>Sell Pattern Lookback</label><input value={form.slr_str} onChange={(e) => set("slr_str", e.target.value)} /></div>
          <div><label>ATR Period</label><input value={form.atr_period_str} onChange={(e) => set("atr_period_str", e.target.value)} placeholder="e.g. 7,21,1" /></div>
          <div><label>ATR Multiplier</label><input value={form.atr_mult_str} onChange={(e) => set("atr_mult_str", e.target.value)} placeholder="e.g. 1,5,0.5" /></div>
        </div>

        <div className="spacer" />
        <h2>Date Ranges</h2>
        <div className="grid">
          <div><label>In-Sample Start</label><input type="date" value={form.is_start} onChange={(e) => set("is_start", e.target.value)} /></div>
          <div><label>In-Sample End</label><input type="date" value={form.is_end} onChange={(e) => set("is_end", e.target.value)} /></div>
          <div><label>OOS1 Start</label><input type="date" value={form.oos1_start} onChange={(e) => set("oos1_start", e.target.value)} /></div>
          <div><label>OOS1 End</label><input type="date" value={form.oos1_end} onChange={(e) => set("oos1_end", e.target.value)} /></div>
          <div><label>OOS2 Start</label><input type="date" value={form.oos2_start} onChange={(e) => set("oos2_start", e.target.value)} /></div>
          <div><label>OOS2 End</label><input type="date" value={form.oos2_end} onChange={(e) => set("oos2_end", e.target.value)} /></div>
        </div>

        <div className="spacer" />
        <div className="checkbox-row">
          <input type="checkbox" id="stab" checked={form.use_stability} onChange={(e) => set("use_stability", e.target.checked)} />
          <label htmlFor="stab">Use stability-based optimization</label>
        </div>

        <div className="spacer" />
        <div className="row">
          <button className="btn" onClick={start} disabled={running}>▶ Start Optimization</button>
          <button className="btn warn" onClick={togglePause} disabled={!running}>{paused ? "Resume" : "Pause"}</button>
          <button className="btn danger" onClick={stop} disabled={!running}>■ Stop</button>
          {status !== "idle" && <span className={`badge ${status}`}>{status}{paused ? " (paused)" : ""}</span>}
        </div>
        {error && <p style={{ color: "var(--red)" }}>{error}</p>}
      </div>

      <div className="panel">
        <h2>Live Log</h2>
        <div className="log-view" ref={logRef}>
          {logs.length === 0 ? <span className="muted">No logs yet. Start an optimization to see progress.</span>
            : logs.map((l, i) => <div key={i}>[{l.ts}] {l.message}</div>)}
        </div>
      </div>
    </>
  );
}
