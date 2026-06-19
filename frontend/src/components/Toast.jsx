import React, { useState, useEffect } from "react";

// Minimal event-based toast system: any module can call toast("msg", "error")
// without context/prop drilling. <Toaster /> mounts once at the app root.
const listeners = new Set();
let idSeq = 0;

export function toast(message, type = "info", ttl = 4000) {
  const item = { id: ++idSeq, message, type, ttl };
  listeners.forEach((fn) => fn(item));
}

export function Toaster() {
  const [items, setItems] = useState([]);

  useEffect(() => {
    const add = (item) => {
      setItems((prev) => [...prev, item]);
      if (item.ttl) {
        setTimeout(() => {
          setItems((prev) => prev.filter((i) => i.id !== item.id));
        }, item.ttl);
      }
    };
    listeners.add(add);
    return () => listeners.delete(add);
  }, []);

  const dismiss = (id) => setItems((prev) => prev.filter((i) => i.id !== id));

  return (
    <div className="toaster">
      {items.map((i) => (
        <div key={i.id} className={`toast toast-${i.type}`} onClick={() => dismiss(i.id)}>
          <span className="toast-icon">
            {i.type === "error" ? "⚠️" : i.type === "success" ? "✓" : "ℹ️"}
          </span>
          <span>{i.message}</span>
          <span className="toast-close">×</span>
        </div>
      ))}
    </div>
  );
}
