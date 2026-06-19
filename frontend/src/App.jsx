import React, { useState, useEffect } from "react";
import Optimizer from "./components/Optimizer.jsx";
import ResultsTable from "./components/ResultsTable.jsx";
import ConfigEditor from "./components/ConfigEditor.jsx";
import BotControl from "./components/BotControl.jsx";
import { Toaster } from "./components/Toast.jsx";
import { api } from "./api/client.js";

const TABS = [
  { id: "optimizer", label: "Optimizer", icon: "⚡" },
  { id: "config", label: "Trade Config", icon: "⚙️" },
  { id: "bot", label: "Bot Control", icon: "🤖" },
];

// Polls the backend /health so the user always knows if the API is reachable —
// otherwise failing buttons look like bugs rather than "backend is offline".
function useBackendHealth() {
  const [state, setState] = useState({ status: "checking", redis: false });
  useEffect(() => {
    let active = true;
    const check = async () => {
      try {
        const h = await api.getHealth();
        if (active) setState({ status: "online", redis: h.redis });
      } catch {
        if (active) setState({ status: "offline", redis: false });
      }
    };
    check();
    const iv = setInterval(check, 8000);
    return () => { active = false; clearInterval(iv); };
  }, []);
  return state;
}

function ConnBadge({ health }) {
  const map = {
    checking: ["pending", "Checking…"],
    online: ["finished", "Backend online"],
    offline: ["failed", "Backend offline"],
  };
  const [cls, text] = map[health.status] || map.checking;
  return (
    <div className="conn">
      <span className={`badge ${cls}`}>● {text}</span>
      {health.status === "online" && (
        <span className={`badge ${health.redis ? "finished" : "stopped"}`}>
          Redis {health.redis ? "up" : "down"}
        </span>
      )}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("optimizer");
  const [jobId, setJobId] = useState(null);
  const health = useBackendHealth();

  return (
    <div className="app">
      <Toaster />
      <header className="topbar">
        <div className="brand">
          <h1>📈 Bince Trading Dashboard</h1>
          <span className="muted">One place to optimize, configure, and run your bot</span>
        </div>
        <ConnBadge health={health} />
      </header>

      {health.status === "offline" && (
        <div className="banner">
          The backend API isn't reachable. The dashboard is read-only until it's online —
          start the FastAPI server, or set <code>VITE_API_BASE</code> to your backend URL.
        </div>
      )}

      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={tab === t.id ? "active" : ""}
            onClick={() => setTab(t.id)}
          >
            <span className="tab-icon">{t.icon}</span> {t.label}
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

      <footer className="footer muted">
        Bince Dashboard · API: <code>{import.meta.env.VITE_API_BASE || "/api"}</code>
      </footer>
    </div>
  );
}
