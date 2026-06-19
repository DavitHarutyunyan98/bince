import React, { useState, useEffect } from "react";
import { api } from "../api/client.js";

// Editor for trade_config.json. The config can be a single object or a list of
// pair configs; we always edit it as JSON text for maximum flexibility, with a
// validate-on-save guard.
export default function ConfigEditor() {
  const [text, setText] = useState("");
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const cfg = await api.getTradeConfig();
      setText(JSON.stringify(cfg, null, 2));
      setError(null);
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  };

  useEffect(() => { load(); }, []);

  const save = async () => {
    setStatus(null);
    setError(null);
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (e) {
      setError("Invalid JSON: " + e.message);
      return;
    }
    try {
      await api.saveTradeConfig(parsed);
      setStatus("Saved ✓");
      setTimeout(() => setStatus(null), 2500);
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <div className="panel">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>trade_config.json</h2>
        <div className="row">
          <button className="btn ghost" onClick={load} disabled={loading}>Reload</button>
          <button className="btn" onClick={save}>Save</button>
          {status && <span className="badge finished">{status}</span>}
        </div>
      </div>
      {error && <p style={{ color: "var(--red)" }}>{error}</p>}
      <textarea
        rows={26}
        value={text}
        onChange={(e) => setText(e.target.value)}
        spellCheck={false}
        style={{ fontFamily: "SF Mono, Menlo, monospace" }}
      />
      <p className="muted">
        Each pair config needs: <code>strategy_name, symbol, bar_length, units_usdt,
        leverage</code> plus strategy params. Set <code>enabled: false</code> to skip a pair.
      </p>
    </div>
  );
}
