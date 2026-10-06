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

interface ViewTransition {
  finished: Promise<void>;
  ready: Promise<void>;
  updateCallbackDone: Promise<void>;
}

type StartViewTransition = (callback: () => void) => ViewTransition;

function supported(): boolean {
  return typeof (document as { startViewTransition?: unknown }).startViewTransition === "function";
}

function wanted(): boolean {
  return !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function visible(): boolean {
  // A hidden document rejects the transition outright. Asking for one anyway
  // threw an unhandled InvalidStateError every time a background tab caught up
  // on a selection change.
  return document.visibilityState === "visible";
}

/** Apply a state change, animating it only when that is an improvement. */
export function transition(apply: () => void): void {
  const now = performance.now();
  const recent = now - last < QUIET_MS;
  last = now;

  if (recent || !supported() || !wanted() || !visible()) {
    apply();
    return;
  }

  const start = (document as unknown as { startViewTransition: StartViewTransition })
    .startViewTransition;
  const transitioning = start.call(document, () => {
    // React batches by default; the transition needs the DOM updated inside
    // this callback, which is what flushSync forces.
    flushSync(apply);
  });

  // A transition can still be refused after it starts, by a tab hidden between
  // the check above and the first frame. The state change has already been
  // applied by then, so the rejection is noise rather than a failure, and
  // leaving it unhandled turns it into a console error.
  const ignore = () => {};
  transitioning.finished.catch(ignore);
  transitioning.ready.catch(ignore);
  transitioning.updateCallbackDone.catch(ignore);
}
