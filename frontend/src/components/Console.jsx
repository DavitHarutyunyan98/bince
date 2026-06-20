import React, { useEffect, useRef, useState } from "react";
import { subscribeLog, getLogEntries, logEvent } from "../api/client.js";

const COLORS = {
  info: "var(--muted, #8aa)",
  success: "var(--green, #3fb950)",
  error: "var(--red, #f85149)",
};

// A persistent, app-wide "terminal" that shows every action and its result.
export default function Console() {
  const [entries, setEntries] = useState(getLogEntries());
  const [open, setOpen] = useState(true);
  const viewRef = useRef(null);

  useEffect(() => {
    // Seed with anything logged before mount, then stream new events.
    setEntries(getLogEntries());
    const unsub = subscribeLog(() => setEntries(getLogEntries()));
    return unsub;
  }, []);

  useEffect(() => {
    if (open && viewRef.current) {
      viewRef.current.scrollTop = viewRef.current.scrollHeight;
    }
  }, [entries, open]);

  return (
    <div className="panel">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2 style={{ margin: 0 }}>🖥 Activity Console</h2>
        <div className="row">
          <span className="muted">{entries.length} events</span>
          <button className="btn ghost" onClick={() => setOpen((o) => !o)}>
            {open ? "Hide" : "Show"}
          </button>
          <button
            className="btn ghost"
            onClick={() => {
              getLogEntries().length = 0; // clear shared buffer
              setEntries([]);
              logEvent("info", "Console cleared");
            }}
          >
            Clear
          </button>
        </div>
      </div>

      {open && (
        <div className="log-view" ref={viewRef} style={{ marginTop: 8 }}>
          {entries.length === 0 ? (
            <span className="muted">
              No activity yet. Every action you take will be logged here.
            </span>
          ) : (
            entries.map((e) => (
              <div key={e.id} style={{ color: COLORS[e.level] || "inherit" }}>
                [{e.ts.slice(11, 19)}] {e.message}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
