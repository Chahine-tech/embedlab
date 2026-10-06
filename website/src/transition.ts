import { flushSync } from "react-dom";

/**
 * A view transition, when one would help and not otherwise.
 *
 * Three conditions, and the third is the one that matters. Held down, an arrow
 * key walks the grid faster than any animation can finish; twenty queued
 * cross-fades read worse than none at all. So a transition is skipped when the
 * last one was recent, which silently turns the effect off exactly while the
 * user is moving fast and keeps it for a deliberate click or single step.
 */
const QUIET_MS = 260;
let last = 0;

type StartViewTransition = (callback: () => void) => { finished: Promise<void> };

function supported(): boolean {
  return typeof (document as { startViewTransition?: unknown }).startViewTransition === "function";
}

function wanted(): boolean {
  return !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Apply a state change, animating it only when that is an improvement. */
export function transition(apply: () => void): void {
  const now = performance.now();
  const recent = now - last < QUIET_MS;
  last = now;

  if (recent || !supported() || !wanted()) {
    apply();
    return;
  }

  const start = (document as unknown as { startViewTransition: StartViewTransition })
    .startViewTransition;
  start.call(document, () => {
    // React batches by default; the transition needs the DOM updated inside
    // this callback, which is what flushSync forces.
    flushSync(apply);
  });
}
