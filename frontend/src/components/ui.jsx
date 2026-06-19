import React, { useState } from "react";

// A labelled field with an optional help tooltip (the "?" hint).
export function Field({ label, help, children }) {
  return (
    <div className="field">
      <label>
        {label}
        {help && (
          <span className="help" tabIndex={0}>
            ?<span className="help-bubble">{help}</span>
          </span>
        )}
      </label>
      {children}
    </div>
  );
}

// A collapsible section so the long optimizer form isn't overwhelming.
export function Section({ title, children, defaultOpen = true, subtitle }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="section">
      <button className="section-head" onClick={() => setOpen((o) => !o)}>
        <span className={`chevron ${open ? "open" : ""}`}>▸</span>
        <span>{title}</span>
        {subtitle && <span className="section-subtitle">{subtitle}</span>}
      </button>
      {open && <div className="section-body">{children}</div>}
    </div>
  );
}

// A button that shows a spinner + disables itself while an async action runs.
export function AsyncButton({ onClick, children, className = "btn", disabled, ...rest }) {
  const [busy, setBusy] = useState(false);
  const handle = async () => {
    if (busy) return;
    setBusy(true);
    try {
      await onClick();
    } finally {
      setBusy(false);
    }
  };
  return (
    <button className={className} onClick={handle} disabled={busy || disabled} {...rest}>
      {busy && <span className="spinner" />} {children}
    </button>
  );
}
