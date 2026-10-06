import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Comparison, Direction, QueryRow, Significance } from "./data";
import { loadComparison, loadIndex } from "./data";
import { Forest } from "./Forest";
import { Grounds } from "./Grounds";
import { Rail } from "./Rail";
import { Sets } from "./Sets";
import { transition } from "./transition";

type Filter = "all" | Direction | "bothfail";

const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "all" },
  { id: "improved", label: "improved" },
  { id: "regressed", label: "regressed" },
  { id: "bothfail", label: "fails both" },
];

function matches(row: QueryRow, filter: Filter) {
  if (filter === "all") return true;
  if (filter === "bothfail") return row.left.symptom !== "ok" && row.right.symptom !== "ok";
  return row.direction === filter;
}

export function App() {
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [available, setAvailable] = useState<string[]>([]);
  const [wanted, setWanted] = useState<string>(() => location.hash.slice(1));
  const [failure, setFailure] = useState<string | null>(null);
  const [selected, setSelected] = useState(0);
  const [filter, setFilter] = useState<Filter>("all");
  const cells = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const index = await loadIndex();
        if (!live) return;
        setAvailable(index.comparisons);

        const id = wanted || index.comparisons[0];
        if (!id) throw new Error("no comparisons published yet");
        setComparison(null);
        const loaded = await loadComparison(id);
        if (!live) return;
        setComparison(loaded);
        setFailure(null);

        // Open on something worth seeing rather than on query one.
        const opener = loaded.queries.findIndex(
          (row) => row.direction === "regressed" && row.right.symptom !== "ok",
        );
        const fallback = loaded.queries.findIndex((row) => row.direction !== "unchanged");
        setSelected(opener >= 0 ? opener : Math.max(0, fallback));
      } catch (error) {
        if (live) setFailure(error instanceof Error ? error.message : String(error));
      }
    })();
    return () => {
      live = false;
    };
  }, [wanted]);

  // The hash is the address of a comparison, so a reload or a shared link lands
  // on the same one.
  useEffect(() => {
    function onHash() {
      setWanted(location.hash.slice(1));
    }
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const rows = comparison?.queries ?? [];

  const step = useCallback(
    (by: number) => {
      transition(() =>
        setSelected((current) => {
          let next = current + by;
          while (next >= 0 && next < rows.length && !matches(rows[next]!, filter)) {
            next += by > 0 ? 1 : -1;
          }
          return next >= 0 && next < rows.length ? next : current;
        }),
      );
    },
    [rows, filter],
  );

  /**
   * Narrowing the grid moves the selection into what is left.
   *
   * Without this the rail went on describing a query the filter had just
   * excluded, greyed out in the grid behind it: the detail panel contradicting
   * the filter above it.
   */
  const chooseFilter = useCallback(
    (next: Filter) => {
      setFilter(next);
      setSelected((current) => {
        if (rows[current] && matches(rows[current], next)) return current;
        const forward = rows.findIndex((row, index) => index >= current && matches(row, next));
        if (forward >= 0) return forward;
        const any = rows.findIndex((row) => matches(row, next));
        return any >= 0 ? any : current;
      });
    },
    [rows],
  );

  const columns = useCallback(() => {
    const host = cells.current;
    if (!host) return 1;
    const style = getComputedStyle(document.documentElement);
    const cell = parseFloat(style.getPropertyValue("--cell")) || 19;
    const gap = parseFloat(style.getPropertyValue("--gap")) || 4;
    return Math.max(1, Math.floor((host.clientWidth + gap) / (cell + gap)));
  }, []);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const moves: Record<string, number> = {
        ArrowRight: 1, l: 1, ArrowLeft: -1, h: -1,
        ArrowDown: columns(), j: columns(), ArrowUp: -columns(), k: -columns(),
      };
      const by = moves[event.key];
      if (by !== undefined) {
        event.preventDefault();
        step(by);
      } else if (event.key === "Escape") {
        chooseFilter("all");
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, columns, chooseFilter]);

function list(names: string[]): string {
  if (names.length <= 1) return names[0] ?? "";
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

/**
 * The headline sentence, written from the numbers rather than around them.
 *
 * The first version asserted "all measures moved up" and put a singular verb
 * after a list, because it was written while looking at a comparison where one
 * measure survived and every delta was positive. On a comparison where nothing
 * survives and a delta is negative it stated the opposite of the data.
 */
function verdictOf(significance: Record<string, Significance>, judged: number) {
  const entries = Object.entries(significance);
  const real = entries.filter(([, s]) => s.is_real).map(([m]) => m);
  const dead = entries.filter(([, s]) => !s.is_real).map(([m]) => m);
  const up = entries.filter(([, s]) => s.delta > 0).length;
  const down = entries.filter(([, s]) => s.delta < 0).length;

  return { real, dead, total: entries.length, up, down, judged };
}

  const verdict = useMemo(
    () =>
      comparison
        ? verdictOf(comparison.manifest.significance, comparison.right.calibration.n)
        : null,
    [comparison],
  );

  if (failure) {
    return (
      <div className="shell">
        <p className="state failed">
          Could not load a comparison: {failure}
        </p>
        <p className="state">
          Publish one with <code>uv run python -m embedlab.cli.demo</code>, then reload.
        </p>
      </div>
    );
  }
  if (!comparison || !verdict) {
    return <div className="shell"><p className="state">Reading artifacts…</p></div>;
  }

  const { manifest } = comparison;
  const counts = manifest.counts;
  const row = rows[selected];

  return (
    <div className="shell">
      <header>
        <h1 className="wordmark">
          Retrieval Diff <span>/ embedlab</span>
        </h1>
        <div className="runs">
          <span className="run">{manifest.left.name}</span>
          <span className="arrow">to</span>
          <span className="run">{manifest.right.name}</span>
        </div>
        <span className={manifest.trust.level === "suspect" ? "chip doubt" : "chip"}>
          trust <b>{manifest.trust.level}</b>
        </span>
        {available.length > 1 && (
          <select
            className="picker"
            aria-label="Comparison"
            value={comparison.id}
            onChange={(event) => {
              location.hash = event.target.value;
            }}
          >
            {available.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        )}
        <span className="meta">{comparison.right.calibration.n} judged queries</span>
        <Grounds />
      </header>

      <main>
        <div className="left">
          <section className="claim">
            <p className="eyebrow">What changed</p>
            <div className="claim-head">
              <p className="tally">
                <span className="up">+{counts.improved}</span> <small>improved</small>{" "}
                <span className="down">-{counts.regressed}</span> <small>regressed</small>{" "}
                <span className="same">={counts.unchanged}</span> <small>unchanged</small>
              </p>
              <p className="verdict">
                {verdict.real.length === 0 ? (
                  <>
                    Not one of the {verdict.total} measures survives the paired test.{" "}
                    <span className="noise">
                      Every interval crosses zero over {verdict.judged} queries
                    </span>
                    , so the two runs disagree about individual queries without differing
                    in aggregate.
                  </>
                ) : verdict.dead.length === 0 ? (
                  <>
                    <b>{list(verdict.real)}</b>{" "}
                    {verdict.real.length === 1 ? "survives" : "all survive"} the paired test.
                  </>
                ) : (
                  <>
                    <b>{list(verdict.real)}</b>{" "}
                    {verdict.real.length === 1 ? "survives" : "survive"} the paired test;{" "}
                    <span className="noise">{list(verdict.dead)}</span>{" "}
                    {verdict.dead.length === 1 ? "does" : "do"} not, and reporting{" "}
                    {verdict.dead.length === 1 ? "it" : "them"} as a difference would be
                    reporting noise.
                  </>
                )}
              </p>
            </div>
            <Forest significance={manifest.significance} />
            <p className="caption">
              Mean paired difference with its 95% bootstrap interval. An interval touching the zero
              rule is a difference we cannot claim.
            </p>
          </section>

          <section className="band">
            <div className="band-head">
              <p className="eyebrow" style={{ margin: 0 }}>
                Every judged query
              </p>
              <div className="filters" role="group" aria-label="Filter queries">
                {FILTERS.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    className="filter"
                    aria-pressed={filter === option.id}
                    onClick={() => chooseFilter(option.id)}
                  >
                    {option.id === "all" ? `all ${rows.length}` : option.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="cells" ref={cells} role="listbox" aria-label="Queries">
              {rows.map((item, index) => (
                <button
                  key={item.query_id}
                  type="button"
                  className="cell"
                  role="option"
                  data-dir={item.direction}
                  data-arb={item.arbitrary ? "1" : "0"}
                  data-muted={matches(item, filter) ? "0" : "1"}
                  aria-current={index === selected}
                  aria-label={`${item.query_id} ${item.direction}`}
                  title={`${item.query_id}: ${item.left_rank ?? "miss"} to ${item.right_rank ?? "miss"}`}
                  onClick={() => transition(() => setSelected(index))}
                />
              ))}
            </div>
            <p className="caption" style={{ margin: "10px 0 0" }}>
              Ordered by how far the gold moved, most improved first. Query order would
              leave the shape unreadable.
            </p>
            <div className="legend">
              <span><i style={{ background: "var(--better)" }} />gold moved up</span>
              <span><i style={{ background: "var(--worse)" }} />gold moved down</span>
              <span><i style={{ background: "var(--flat)" }} />unchanged</span>
              <span>
                <i style={{ background: "var(--doubt)", backgroundImage: "repeating-linear-gradient(45deg,transparent 0 2px,var(--ground) 2px 4px)" }} />
                decided by a score tie
              </span>
            </div>
          </section>

          <Sets
            left={comparison.sets.left}
            right={comparison.sets.right}
            leftName={manifest.left.name}
            rightName={manifest.right.name}
          />

          {/* A property of the comparison, not of the selected query: it never
              changed while arrowing through the grid, which is what gave it
              away as being in the wrong column. */}
          <section className="trust">
            <p className="eyebrow" style={{ margin: 0 }}>
              Why this diff is{" "}
              <span style={{ color: "var(--doubt-ink)" }}>{manifest.trust.level}</span>
            </p>
            <ul>
              {manifest.trust.reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          </section>
        </div>

        <Rail comparison={comparison} row={row} />
      </main>

      <footer>
        <span><kbd>arrows</kbd>move</span>
        <span><kbd>hjkl</kbd>move</span>
        <span><kbd>esc</kbd>clear filter</span>
        <span className="src">{comparison.id}</span>
      </footer>
    </div>
  );
}
