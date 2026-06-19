import React, { useState } from "react";
import Optimizer from "./components/Optimizer.jsx";
import ResultsTable from "./components/ResultsTable.jsx";
import ConfigEditor from "./components/ConfigEditor.jsx";
import BotControl from "./components/BotControl.jsx";

const TABS = [
  { id: "optimizer", label: "Optimizer" },
  { id: "config", label: "Trade Config" },
  { id: "bot", label: "Bot Control" },
];

export default function App() {
  const [tab, setTab] = useState("optimizer");
  // Lift the last completed job's results so the Optimizer and Results views share them.
  const [jobId, setJobId] = useState(null);

  return (
    <div className="app">
      <header className="topbar">
        <h1>📈 Bince Trading Dashboard</h1>
        <span className="muted">All controls in one place</span>
      </header>

      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={tab === t.id ? "active" : ""}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {tab === "optimizer" && (
        <>
          <Optimizer jobId={jobId} setJobId={setJobId} />
          {jobId && <ResultsTable jobId={jobId} />}
        </>
      )}
      {tab === "config" && <ConfigEditor />}
      {tab === "bot" && <BotControl />}
    </div>
  );
}
