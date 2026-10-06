import type { Sets as SetCounts, Symptom } from "./data";

const SYMPTOMS: { id: Symptom; label: string }[] = [
  { id: "ok", label: "relevant first" },
  { id: "rank", label: "retrieved, ranked low" },
  { id: "miss", label: "never retrieved" },
];

function Bars({
  rows,
  total,
  leftName,
  rightName,
}: {
  rows: { label: string; left: number; right: number }[];
  total: number;
  leftName: string;
  rightName: string;
}) {
  const widest = Math.max(1, ...rows.flatMap((row) => [row.left, row.right]));
  return (
    <div className="bars">
      {rows.map((row) => (
        <div className="bar" key={row.label}>
          <span className="bar-label">{row.label}</span>
          <span className="bar-track">
            <i
              className="bar-fill left"
              style={{ width: `${(row.left / widest) * 100}%` }}
              title={`${leftName}: ${row.left} of ${total}`}
            />
            <i
              className="bar-fill right"
              style={{ width: `${(row.right / widest) * 100}%` }}
              title={`${rightName}: ${row.right} of ${total}`}
            />
          </span>
          <span className="bar-count">
            {row.left}
            <span className="to"> to </span>
            {row.right}
          </span>
        </div>
      ))}
    </div>
  );
}

/**
 * Two panels, not one.
 *
 * A symptom is what happened and every judged query has exactly one, so those
 * counts sum to the total. A cause is why, and a query may carry several, so
 * those counts sum to more than the number of failures. Putting them in one
 * list would double-count, which is why the engine keeps them on separate
 * axes and why this says "queries in set" rather than a percentage.
 */
export function Sets({
  left,
  right,
  leftName,
  rightName,
}: {
  left: SetCounts;
  right: SetCounts;
  leftName: string;
  rightName: string;
}) {
  const kinds = [...new Set([...Object.keys(left.causes), ...Object.keys(right.causes)])].sort(
    (a, b) => (right.causes[b] ?? 0) - (right.causes[a] ?? 0) || a.localeCompare(b),
  );

  return (
    <section className="sets">
      <div>
        <p className="eyebrow">Symptom, one per query</p>
        <Bars
          total={right.judged}
          leftName={leftName}
          rightName={rightName}
          rows={SYMPTOMS.map((symptom) => ({
            label: symptom.label,
            left: left.symptoms[symptom.id],
            right: right.symptoms[symptom.id],
          }))}
        />
        <p className="caption">
          Exclusive and exhaustive: these sum to the {right.judged} judged queries.
        </p>
      </div>

      <div>
        <p className="eyebrow">Hypothesised cause</p>
        <Bars
          total={right.judged}
          leftName={leftName}
          rightName={rightName}
          rows={kinds.map((kind) => ({
            label: kind.replace(/_/g, " "),
            left: left.causes[kind] ?? 0,
            right: right.causes[kind] ?? 0,
          }))}
        />
        <p className="caption">
          Queries in set, not a share of failures: a query can carry two causes, so these
          overlap on purpose.
        </p>
      </div>
    </section>
  );
}
