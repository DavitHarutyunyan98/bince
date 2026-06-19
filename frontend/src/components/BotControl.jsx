import React, { useState, useEffect } from "react";
import { api } from "../api/client.js";
import { AsyncButton } from "./ui.jsx";
import { toast } from "./Toast.jsx";

// Phase 3: start/stop/status for the 24/7 hybrid trader.
export default function BotControl() {
  const [status, setStatus] = useState(null);
  const [offline, setOffline] = useState(false);

  const refresh = async () => {
    try {
      setStatus(await api.getBotStatus());
      setOffline(false);
    } catch {
      setOffline(true);
    }
  };

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 4000);
    return () => clearInterval(iv);
  }, []);

  const doStart = async () => {
    try {
      const r = await api.startBot();
      await refresh();
      toast(r.running ? r.message : `Bot did not start: ${r.message}`, r.running ? "success" : "error");
    } catch (e) { toast(`Start failed: ${e.message}`, "error"); }
  };
  const doStop = async () => {
    try {
      const r = await api.stopBot();
      await refresh();
      toast(r.message || "Bot stopped", "info");
    } catch (e) { toast(`Stop failed: ${e.message}`, "error"); }
  };

  const running = status?.running;

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>🤖 Bot Control</h2>
        <span className={`badge ${running ? "running" : "stopped"}`}>
          {running ? "● RUNNING" : "○ STOPPED"}
        </span>
      </div>

      <div className="stat-row">
        <div className="stat">
          <div className="stat-label">State</div>
          <div className="stat-value">{offline ? "—" : running ? "Running" : "Stopped"}</div>
        </div>
        <div className="stat">
          <div className="stat-label">PID</div>
          <div className="stat-value">{status?.pid ?? "—"}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Enabled pairs</div>
          <div className="stat-value">{status?.pairs ?? "—"}</div>
        </div>
      </div>

      <div className="spacer" />
      <div className="row">
        <AsyncButton onClick={doStart} disabled={running || offline}>▶ Start Bot</AsyncButton>
        <AsyncButton className="btn danger" onClick={doStop} disabled={!running || offline}>■ Stop Bot</AsyncButton>
        <button className="btn ghost" onClick={refresh}>Refresh</button>
      </div>

      {status?.message && <p className="muted">{status.message}</p>}
      <p className="muted hint">
        The bot trades the enabled pairs from <b>Trade Config</b>. Edit those first, then start here.
        For production, run it under systemd/NSSM so it survives reboots (see backend/deploy).
      </p>
    </div>
  );
}
