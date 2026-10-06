import { useEffect, useState } from "react";
import type { Cause, Comparison, Hit, QueryRow, Side } from "./data";
import { loadCauses, loadHits } from "./data";

function rank(value: number | null) {
  return value === null ? "not retrieved" : `#${value}`;
}

function Hits({ title, hits, goldRank }: { title: string; hits: Hit[]; goldRank: number | null }) {
  return (
    <div className="side">
      <h4>{title}</h4>
      {hits.map((hit) => {
        const isGold = goldRank !== null && hit.rank === goldRank;
        return (
          <div className={isGold ? "hit gold" : "hit"} key={hit.doc_id}>
            <span className="n">{isGold ? "✓" : hit.rank}</span>
            <span className="d" title={hit.doc_id}>
              {hit.doc_id}
            </span>
            <span className="s">{hit.score.toFixed(2)}</span>
          </div>
        );
      })}
    </div>
  );
}

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;

/**
 * Evidence, with one column where both runs agree.
 *
 * Showing every number as "left to right" implied a change that is usually not
 * there: the query-to-gold overlap is computed from the query and the gold
 * document, so it only differs when the two runs retrieved different relevant
 * documents. Rendering the same value twice reads as a transition. A second
 * column now appears only when there is a second value.
 */
function Fact({ label, left, right }: { label: string; left: number; right: number }) {
  const same = left === right;
  return (
    <>
      <dt>{label}</dt>
      <dd>
        {same ? pct(left) : `${pct(left)} to ${pct(right)}`}
        {same && <span className="same-note"> both</span>}
      </dd>
    </>
  );
}

function Facts({ left, right, calibration }: { left: Side; right: Side; calibration: { low_query_gold_overlap: number; high_competitor_gold_overlap: number } }) {
  return (
    <div>
      <p className="eyebrow">Evidence, deterministic</p>
      <dl className="facts">
        <Fact
          label="query to gold vocabulary"
          left={left.query_gold_overlap}
          right={right.query_gold_overlap}
        />
        <Fact
          label="rank-1 to gold vocabulary"
          left={left.competitor_gold_overlap}
          right={right.competitor_gold_overlap}
        />
        <dt>dataset low / high threshold</dt>
        <dd>{`${pct(calibration.low_query_gold_overlap)} / ${pct(calibration.high_competitor_gold_overlap)}`}</dd>
      </dl>
    </div>
  );
}

export function Rail({ comparison, row }: { comparison: Comparison; row: QueryRow | undefined }) {
  const [hits, setHits] = useState<{ left: Hit[]; right: Hit[] }>({ left: [], right: [] });
  const [causes, setCauses] = useState<Cause[]>([]);

  const queryId = row?.query_id;
  const leftId = comparison.manifest.left.run_id;
  const rightId = comparison.manifest.right.run_id;

  useEffect(() => {
    if (!queryId) return;
    let live = true;
    void Promise.all([
      loadHits(leftId, queryId),
      loadHits(rightId, queryId),
      loadCauses(rightId, queryId),
    ]).then(([l, r, c]) => {
      if (!live) return;
      setHits({ left: l, right: r });
      setCauses(c);
    });
    return () => {
      live = false;
    };
  }, [queryId, leftId, rightId]);

  if (!row) return <aside className="rail">Nothing selected.</aside>;

  const colour =
    row.direction === "improved"
      ? "var(--better)"
      : row.direction === "regressed"
        ? "var(--worse)"
        : "var(--faint)";

  return (
    <aside className="rail" aria-live="polite">
      <div>
        <div className="qid">
          <code>{row.query_id}</code>
          {row.arbitrary && <span className="chip doubt">score tie</span>}
          <span className="chip" style={{ color: colour, borderColor: colour }}>
            {row.direction}
          </span>
        </div>
      </div>

      <div>
        <p className="eyebrow">Where the gold landed</p>
        <div className="move" style={{ color: colour }}>
          <span className={row.left_rank === null ? "rank-miss" : undefined}>{rank(row.left_rank)}</span>
          <span className="to">to</span>
          <span className={row.right_rank === null ? "rank-miss" : undefined}>{rank(row.right_rank)}</span>
        </div>
        {row.direction === "unchanged" && <p className="caption">Both runs place it identically.</p>}
      </div>

      <div className="sides">
        <Hits title={`${comparison.manifest.left.name}, top 5`} hits={hits.left} goldRank={row.left_rank} />
        <Hits title={`${comparison.manifest.right.name}, top 5`} hits={hits.right} goldRank={row.right_rank} />
      </div>
      <p className="caption">
        Scores rank within a run. The two columns are not comparable to each other.
      </p>

      <Facts left={row.left} right={row.right} calibration={comparison.right.calibration} />

      <div>
        <p className="eyebrow">Hypotheses, {comparison.manifest.right.name} only</p>
        {causes.length === 0 ? (
          <p className="caption">No failure: a relevant document is first.</p>
        ) : (
          causes.map((cause) => (
            <div className="cause" data-kind={cause.kind} key={`${cause.kind}-${cause.because}`}>
              <b>{cause.kind.replace(/_/g, " ")}</b>{" "}
              <span style={{ color: "var(--faint)", fontSize: 10 }}>
                conf {cause.confidence.toFixed(1)} by {cause.producer}
              </span>
              <p>{cause.because}</p>
            </div>
          ))
        )}
      </div>

    </aside>
  );
}
