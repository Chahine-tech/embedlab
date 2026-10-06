import { useEffect, useState } from "react";

/**
 * A ground is a complete token set, so the swatch shows more than its
 * background: three of the five differ by a few points of near-black and are
 * one flat square each indistinguishable, while their direction hues are not.
 * The bar carries better, doubt and worse in that order.
 *
 * Auto has no fixed hues, so it stays the split square and shows no bar.
 */
const GROUNDS = [
  { id: "auto", label: "Auto, follow the system", ground: "linear-gradient(135deg,#f3f1ec 0 50%,#0b0f14 50% 100%)" },
  { id: "slate", label: "Slate", ground: "#0b0f14", hues: ["#5a9fe0", "#d9a441", "#d9636d"] },
  { id: "ink", label: "Ink", ground: "#100d16", hues: ["#8098f0", "#d9a441", "#e0707e"] },
  { id: "graphite", label: "Graphite", ground: "#1b1e22", hues: ["#5fa3dd", "#d2a04a", "#d9696f"] },
  { id: "paper", label: "Paper", ground: "#f3f1ec", hues: ["#2c6aad", "#8a6410", "#b04a4a"] },
] as const;

const KEY = "embedlab-ground";

function bar(hues: readonly string[]): string {
  const [better, doubt, worse] = hues;
  return `linear-gradient(90deg, ${better} 0 33.34%, ${doubt} 33.34% 66.67%, ${worse} 66.67% 100%)`;
}

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
          style={{ background: option.ground }}
          title={option.label}
          aria-label={option.label}
          aria-pressed={ground === option.id}
          onClick={() => setGround(option.id)}
        >
          {"hues" in option && <i style={{ background: bar(option.hues) }} />}
        </button>
      ))}
    </div>
  );
}
