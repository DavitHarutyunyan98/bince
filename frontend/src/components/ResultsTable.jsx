import React, { useState, useEffect } from "react";
import { api } from "../api/client.js";

// Columns shown first; the rest follow dynamically.
const PRIORITY = [
  "Trading_Pair", "Timeframe", "Score", "Total_Return", "Win_Rate",
  "Total_Trades", "Max_Drawdown", "Sharpe_Ratio", "OOS1_Return", "OOS2_Return",
];

export default function ResultsTable({ jobId }) {
  const [rows, setRows] = useState([]);
  const [status, setStatus] = useState("");
  const [sortKey, setSortKey] = useState("Score");
  const [sortDir, setSortDir] = useState(-1);
  const [bestPerPair, setBestPerPair] = useState(true);

  useEffect(() => {
    if (!jobId) return;
    let active = true;
    const load = async () => {
      try {
        const r = await api.getResults(jobId);
        if (!active) return;
        setRows(r.results || []);
        setStatus(r.status);
        if (["finished", "failed", "stopped"].includes(r.status)) {
          clearInterval(iv);
        }
      } catch (e) { /* ignore until job exists */ }
    };
    load();
    const iv = setInterval(load, 2500);
    return () => { active = false; clearInterval(iv); };
  }, [jobId]);

  if (!rows.length) {
    return (
      <div className="panel">
        <h2>Results</h2>
        <p className="muted">No results yet {status && `(job ${status})`}.</p>
      </div>
    );
  }

  const allKeys = Object.keys(rows[0]);
  const cols = [...PRIORITY.filter((k) => allKeys.includes(k)),
    ...allKeys.filter((k) => !PRIORITY.includes(k))];

  let display = [...rows];
  if (bestPerPair) {
    const best = {};
    for (const r of rows) {
      const p = r.Trading_Pair;
      if (!best[p] || (r.Score ?? -1e9) > (best[p].Score ?? -1e9)) best[p] = r;
    }
    display = Object.values(best);
  }
  display.sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey];
    if (av == null) return 1;
    if (bv == null) return -1;
    return (av > bv ? 1 : av < bv ? -1 : 0) * sortDir;
  });

  const setSort = (k) => {
    if (k === sortKey) setSortDir((d) => -d);
    else { setSortKey(k); setSortDir(-1); }
  };

  const fmt = (v) => (typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(2)) : String(v));

  const downloadCsv = () => {
    const header = cols.join(",");
    const lines = display.map((r) => cols.map((c) => JSON.stringify(r[c] ?? "")).join(","));
    const blob = new Blob([[header, ...lines].join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `results_${jobId.slice(0, 8)}.csv`; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="panel">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>Results ({display.length} rows)</h2>
        <div className="row">
          <div className="checkbox-row">
            <input type="checkbox" id="bpp" checked={bestPerPair} onChange={(e) => setBestPerPair(e.target.checked)} />
            <label htmlFor="bpp">Best per pair</label>
          </div>
          <button className="btn ghost" onClick={downloadCsv}>Export CSV</button>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>{cols.map((c) => (
              <th key={c} onClick={() => setSort(c)}>
                {c.replace(/_/g, " ")}{sortKey === c ? (sortDir === -1 ? " ▼" : " ▲") : ""}
              </th>
            ))}</tr>
          </thead>
          <tbody>
            {display.map((r, i) => (
              <tr key={i}>{cols.map((c) => <td key={c}>{fmt(r[c])}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
