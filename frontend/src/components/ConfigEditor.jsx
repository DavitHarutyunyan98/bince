import React, { useState, useEffect } from "react";
import { api } from "../api/client.js";
import { AsyncButton } from "./ui.jsx";
import { toast } from "./Toast.jsx";

// Editor for trade_config.json. Supports a single object or a list of pair
// configs; edited as JSON text with validate-on-save and a live validity hint.
export default function ConfigEditor() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(true);
  const [dirty, setDirty] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const cfg = await api.getTradeConfig();
      setText(JSON.stringify(cfg, null, 2));
      setDirty(false);
    } catch (e) {
      toast(`Couldn't load config: ${e.message}`, "error");
    }
    setLoading(false);
  };

  useEffect(() => { load(); }, []);

  // Live JSON validity for inline feedback.
  let valid = true, parseErr = null;
  try { if (text.trim()) JSON.parse(text); } catch (e) { valid = false; parseErr = e.message; }

  const pairCount = (() => {
    try {
      const d = JSON.parse(text);
      const arr = Array.isArray(d) ? d : [d];
      return arr.filter((c) => c.enabled !== false).length;
    } catch { return null; }
  })();

  const save = async () => {
    if (!valid) { toast("Fix the JSON before saving", "error"); return; }
    try {
      await api.saveTradeConfig(JSON.parse(text));
      setDirty(false);
      toast("Trade config saved ✓", "success");
    } catch (e) {
      toast(`Save failed: ${e.message}`, "error");
    }
  };

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>⚙️ trade_config.json</h2>
        <div className="row">
          {pairCount != null && <span className="badge pending">{pairCount} enabled pair(s)</span>}
          <span className={`badge ${valid ? "finished" : "failed"}`}>{valid ? "valid JSON" : "invalid JSON"}</span>
          {dirty && <span className="badge stopped">unsaved</span>}
          <button className="btn ghost" onClick={load} disabled={loading}>Reload</button>
          <AsyncButton onClick={save} disabled={!valid}>Save</AsyncButton>
        </div>
      </div>
      {!valid && <p className="inline-err">⚠️ {parseErr}</p>}
      <textarea
        rows={24}
        value={text}
        onChange={(e) => { setText(e.target.value); setDirty(true); }}
        spellCheck={false}
        className="code"
      />
      <p className="muted hint">
        Each pair needs <code>strategy_name, symbol, bar_length, units_usdt, leverage</code> plus
        strategy params. Set <code>"enabled": false</code> to skip a pair without deleting it.
      </p>
    </div>
  );
}
