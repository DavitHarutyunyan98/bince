import React, { useState, useEffect } from "react";
import { api } from "../api/client.js";

// Phase 3: start/stop/status for the 24/7 hybrid trader.
export default function BotControl() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    try {
      const s = await api.getBotStatus();
      setStatus(s);
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  };

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 4000);
    return () => clearInterval(iv);
  }, []);

  const doStart = async () => {
    setBusy(true);
    try { await api.startBot(); await refresh(); }
    catch (e) { setError(e.message); }
    setBusy(false);
  };
  const doStop = async () => {
    setBusy(true);
    try { await api.stopBot(); await refresh(); }
    catch (e) { setError(e.message); }
    setBusy(false);
  };

  const running = status?.running;
  const state = running ? "running" : "stopped";

  return (
    <div className="panel">
      <h2>Bot Control (24/7 Hybrid Trader)</h2>
      <div className="row">
        <span className={`badge ${state}`}>{running ? "RUNNING" : "STOPPED"}</span>
        {status?.pid && <span className="muted">PID {status.pid}</span>}
        {status?.pairs != null && <span className="muted">{status.pairs} pairs</span>}
      </div>
      <div className="spacer" />
      <div className="row">
        <button className="btn" onClick={doStart} disabled={busy || running}>▶ Start Bot</button>
        <button className="btn danger" onClick={doStop} disabled={busy || !running}>■ Stop Bot</button>
        <button className="btn ghost" onClick={refresh} disabled={busy}>Refresh</button>
      </div>
      {status?.message && <p className="muted">{status.message}</p>}
      {error && <p style={{ color: "var(--red)" }}>{error}</p>}
      <p className="muted">
        The bot runs the enabled pairs from <code>trade_config.json</code>. Edit those in the
        Trade Config tab, then start the bot here. For production, run it under systemd/NSSM
        so it survives restarts (see backend/README).
      </p>
    </div>
  );
}
