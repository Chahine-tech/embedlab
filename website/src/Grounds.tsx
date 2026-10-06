import { useEffect, useState } from "react";

const GROUNDS = [
  { id: "auto", label: "Auto, follow the system", swatch: "linear-gradient(135deg,#f3f1ec 0 50%,#0b0f14 50% 100%)" },
  { id: "slate", label: "Slate", swatch: "#0b0f14" },
  { id: "ink", label: "Ink", swatch: "#100d16" },
  { id: "graphite", label: "Graphite", swatch: "#1b1e22" },
  { id: "paper", label: "Paper", swatch: "#f3f1ec" },
] as const;

const KEY = "embedlab-ground";

/**
 * Auto is the default and means no override, so the control never claims a
 * ground the page is not actually using.
 */
export function Grounds() {
  const [ground, setGround] = useState<string>(() => {
    try {
      return localStorage.getItem(KEY) ?? "auto";
    } catch {
      return "auto";
    }
  });

  useEffect(() => {
    const root = document.documentElement;
    if (ground === "auto") root.removeAttribute("data-ground");
    else root.setAttribute("data-ground", ground);
    try {
      localStorage.setItem(KEY, ground);
    } catch {
      // Private window: the choice simply does not persist.
    }
  }, [ground]);

  return (
    <div className="grounds" role="group" aria-label="Background">
      <b>ground</b>
      {GROUNDS.map((option) => (
        <button
          key={option.id}
          type="button"
          className="swatch"
          style={{ background: option.swatch }}
          title={option.label}
          aria-label={option.label}
          aria-pressed={ground === option.id}
          onClick={() => setGround(option.id)}
        />
      ))}
    </div>
  );
}
