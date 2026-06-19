import React, { useState, useEffect, useRef } from "react";
import { api } from "../api/client.js";
import { Field, Section, AsyncButton } from "./ui.jsx";
import { toast } from "./Toast.jsx";

const DEFAULTS = {
  pairs: "BTCUSDT,ETHUSDT",
  strategy_name: "Candlestick Patterns",
  optimization_mode: "efficient",
  timeframe: "15m",
  n_trials: 50,
  min_trades: 10,
  min_candles: 100,
  executor_type: "process",
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
};

const STRATEGIES = ["Candlestick Patterns", "ATR SuperTrend", "RSI", "MA Crossover"];
const MODES = [
  ["efficient", "Efficient — fast, smart sampling"],
  ["smart", "Smart — balanced exploration"],
  ["comprehensive", "Comprehensive — thorough, slower"],
];
const TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"];

export default function Optimizer({ jobId, setJobId }) {
  const [form, setForm] = useState(DEFAULTS);
  const [status, setStatus] = useState("idle");
  const [paused, setPaused] = useState(false);
  const [logs, setLogs] = useState([]);
  const logRef = useRef(null);
  const offsetRef = useRef(0);
  const pollRef = useRef(null);

  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

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
          if (st.status === "finished") toast("Optimization complete 🎉", "success");
          if (st.status === "failed") toast("Optimization failed — check the log", "error");
          if (st.status === "stopped") toast("Optimization stopped", "info");
        }
      } catch (e) {
        // backend likely went away mid-run; stop hammering
        clearInterval(pollRef.current);
      }
    };
    poll();
    pollRef.current = setInterval(poll, 1500);
    return () => clearInterval(pollRef.current);
  }, [jobId]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const validate = (pairs) => {
    if (!pairs.length) return "Add at least one trading pair.";
    if (Number(form.n_trials) < 1) return "Trials per pair must be at least 1.";
    if (form.is_start >= form.is_end) return "In-sample start must be before its end.";
    return null;
  };

  const start = async () => {
    const pairs = form.pairs.split(",").map((s) => s.trim().toUpperCase()).filter(Boolean);
    const err = validate(pairs);
    if (err) { toast(err, "error"); return; }
    try {
      const payload = {
        ...form, pairs,
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
      toast(`Started optimization for ${pairs.length} pair(s)`, "success");
    } catch (e) {
      toast(`Couldn't start: ${e.message}`, "error");
    }
  };

  const stop = async () => {
    if (jobId) { await api.stopOptimization(jobId); toast("Stop requested…", "info"); }
  };
  const togglePause = async () => {
    if (jobId) {
      const r = await api.togglePause(jobId);
      setPaused(r.paused);
      toast(r.paused ? "Paused" : "Resumed", "info");
    }
  };

  const running = ["pending", "running", "stopping"].includes(status);

  return (
    <>
      <div className="panel">
        <div className="panel-head">
          <h2>⚡ Optimizer</h2>
          <div className="row">
            {status !== "idle" && (
              <span className={`badge ${status}`}>{status}{paused ? " · paused" : ""}</span>
            )}
            <AsyncButton onClick={start} disabled={running}>▶ Start</AsyncButton>
            <button className="btn warn" onClick={togglePause} disabled={!running}>
              {paused ? "Resume" : "Pause"}
            </button>
            <button className="btn danger" onClick={stop} disabled={!running}>■ Stop</button>
          </div>
        </div>

        <Section title="Basics" subtitle="what to optimize">
          <div className="grid">
            <Field label="Trading Pairs" help="Comma-separated Binance futures symbols, e.g. BTCUSDT, ETHUSDT.">
              <input value={form.pairs} onChange={(e) => set("pairs", e.target.value)} placeholder="BTCUSDT, ETHUSDT" />
            </Field>
            <Field label="Strategy" help="The trading strategy whose parameters will be tuned.">
              <select value={form.strategy_name} onChange={(e) => set("strategy_name", e.target.value)}>
                {STRATEGIES.map((s) => <option key={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="Mode" help="Efficient is fastest. Comprehensive explores more but takes longer.">
              <select value={form.optimization_mode} onChange={(e) => set("optimization_mode", e.target.value)}>
                {MODES.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
              </select>
            </Field>
            <Field label="Timeframe" help="Candle size used for backtesting.">
              <select value={form.timeframe} onChange={(e) => set("timeframe", e.target.value)}>
                {TIMEFRAMES.map((t) => <option key={t}>{t}</option>)}
              </select>
            </Field>
            <Field label="Trials per pair" help="How many parameter combinations to try per pair. More = better results, slower.">
              <input type="number" value={form.n_trials} onChange={(e) => set("n_trials", e.target.value)} />
            </Field>
            <Field label="Min trades" help="Discard results with fewer trades than this (avoids overfit flukes).">
              <input type="number" value={form.min_trades} onChange={(e) => set("min_trades", e.target.value)} />
            </Field>
            <Field label="Parallelism" help="Process uses all CPU cores (fastest). Thread is a safe fallback.">
              <select value={form.executor_type} onChange={(e) => set("executor_type", e.target.value)}>
                <option value="process">Process — all cores</option>
                <option value="thread">Thread — fallback</option>
              </select>
            </Field>
          </div>
        </Section>

        <Section title="Parameter Ranges" subtitle="low, high, step" defaultOpen={false}>
          <p className="muted hint">Each field is a search range as <code>low,high,step</code>. Leave ATR blank unless using SuperTrend.</p>
          <div className="grid">
            <Field label="Buy Signal Window" help="Lookback window for buy-signal confirmation."><input value={form.bw_str} onChange={(e) => set("bw_str", e.target.value)} /></Field>
            <Field label="Buy Pattern Lookback" help="Candles back to scan for the buy pattern."><input value={form.bl_str} onChange={(e) => set("bl_str", e.target.value)} /></Field>
            <Field label="Sell Signal Window" help="Lookback window for sell-signal confirmation."><input value={form.sw_str} onChange={(e) => set("sw_str", e.target.value)} /></Field>
            <Field label="Sell Pattern Lookback" help="Candles back to scan for the sell pattern."><input value={form.slr_str} onChange={(e) => set("slr_str", e.target.value)} /></Field>
            <Field label="ATR Period" help="SuperTrend only. e.g. 7,21,1"><input value={form.atr_period_str} onChange={(e) => set("atr_period_str", e.target.value)} placeholder="e.g. 7,21,1" /></Field>
            <Field label="ATR Multiplier" help="SuperTrend only. e.g. 1,5,0.5"><input value={form.atr_mult_str} onChange={(e) => set("atr_mult_str", e.target.value)} placeholder="e.g. 1,5,0.5" /></Field>
          </div>
        </Section>

        <Section title="Date Ranges" subtitle="in-sample + 2 out-of-sample" defaultOpen={false}>
          <p className="muted hint">Tune on the in-sample window, then validate on two out-of-sample windows to catch overfitting.</p>
          <div className="grid">
            <Field label="In-Sample Start"><input type="date" value={form.is_start} onChange={(e) => set("is_start", e.target.value)} /></Field>
            <Field label="In-Sample End"><input type="date" value={form.is_end} onChange={(e) => set("is_end", e.target.value)} /></Field>
            <Field label="OOS1 Start"><input type="date" value={form.oos1_start} onChange={(e) => set("oos1_start", e.target.value)} /></Field>
            <Field label="OOS1 End"><input type="date" value={form.oos1_end} onChange={(e) => set("oos1_end", e.target.value)} /></Field>
            <Field label="OOS2 Start"><input type="date" value={form.oos2_start} onChange={(e) => set("oos2_start", e.target.value)} /></Field>
            <Field label="OOS2 End"><input type="date" value={form.oos2_end} onChange={(e) => set("oos2_end", e.target.value)} /></Field>
          </div>
        </Section>

        <Section title="Scoring & Stability" subtitle="how results are ranked" defaultOpen={false}>
          <div className="grid">
            <Field label="Weight: Return" help="How much total return contributes to the score."><input type="number" step="0.1" value={form.weight_return} onChange={(e) => set("weight_return", e.target.value)} /></Field>
            <Field label="Weight: Win Rate" help="How much win rate contributes to the score."><input type="number" step="0.1" value={form.weight_winrate} onChange={(e) => set("weight_winrate", e.target.value)} /></Field>
            <Field label="Weight: Trades" help="How much trade count contributes to the score."><input type="number" step="0.1" value={form.weight_trades} onChange={(e) => set("weight_trades", e.target.value)} /></Field>
          </div>
          <div className="spacer" />
          <div className="checkbox-row">
            <input type="checkbox" id="stab" checked={form.use_stability} onChange={(e) => set("use_stability", e.target.checked)} />
            <label htmlFor="stab">Use stability-based optimization (favours consistent results across periods)</label>
          </div>
        </Section>
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>📋 Live Log</h2>
          {running && <span className="badge running">● streaming</span>}
        </div>
        <div className="log-view" ref={logRef}>
          {logs.length === 0 ? (
            <span className="muted">No logs yet. Configure above and hit <b>Start</b> to watch progress here.</span>
          ) : (
            logs.map((l, i) => <div key={i}><span className="log-ts">{l.ts}</span> {l.message}</div>)
          )}
        </div>
      </div>
    </>
  );
}
