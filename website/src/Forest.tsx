import type { Significance } from "./data";

const WIDTH = 880;
const LEFT = 108;
const RIGHT = 196;
const ROW = 34;
const TOP = 22;

function signed(value: number, digits = 4) {
  return `${value >= 0 ? "+" : "-"}${Math.abs(value).toFixed(digits)}`;
}

/**
 * A delta with its interval, against a zero rule.
 *
 * Hue marks the exception only: a difference we can claim is drawn in plain
 * ink, a doubtful one is amber, dashed, hollow and labelled. The reading never
 * depends on telling two colours apart.
 */
export function Forest({ significance }: { significance: Record<string, Significance> }) {
  const rows = Object.entries(significance)
    .map(([measure, result]) => ({ measure, ...result }))
    .sort((a, b) => a.measure.localeCompare(b.measure));
  if (rows.length === 0) return null;

  let low = Math.min(0, ...rows.map((row) => row.ci_low));
  let high = Math.max(0, ...rows.map((row) => row.ci_high));
  const pad = (high - low) * 0.18 || 0.01;
  low -= pad;
  high += pad;

  const height = TOP + rows.length * ROW + 26;
  const x = (value: number) => LEFT + ((value - low) / (high - low)) * (WIDTH - LEFT - RIGHT);

  const step = 0.02;
  const ticks: number[] = [];
  for (let t = Math.ceil(low / step) * step; t <= high + 1e-9; t += step) ticks.push(t);

  return (
    <svg
      className="forest"
      viewBox={`0 0 ${WIDTH} ${height}`}
      width="100%"
      style={{ maxHeight: height + 10 }}
      role="img"
      aria-label="Per-measure difference with its confidence interval"
    >
      {ticks.map((tick) => (
        <g key={tick}>
          <line className="tickline" x1={x(tick)} y1={TOP} x2={x(tick)} y2={TOP + rows.length * ROW - 10} />
          <text x={x(tick)} y={TOP + rows.length * ROW + 8} textAnchor="middle">
            {Math.abs(tick) < 1e-9 ? "0" : signed(tick, 2)}
          </text>
        </g>
      ))}
      <line className="zero" x1={x(0)} y1={TOP - 8} x2={x(0)} y2={TOP + rows.length * ROW - 4} />

      {rows.map((row, index) => {
        const cy = TOP + index * ROW + ROW / 2 - 6;
        const colour = row.is_real ? "var(--text)" : "var(--doubt)";
        return (
          <g key={row.measure}>
            <text className={row.is_real ? "fname" : "fname fdead"} x={0} y={cy + 4}>
              {row.measure}
            </text>
            <line
              className="ci"
              x1={x(row.ci_low)}
              y1={cy}
              x2={x(row.ci_high)}
              y2={cy}
              stroke={colour}
              strokeDasharray={row.is_real ? undefined : "4 3"}
            />
            {[row.ci_low, row.ci_high].map((bound) => (
              <line key={bound} className="cap" x1={x(bound)} y1={cy - 4} x2={x(bound)} y2={cy + 4} stroke={colour} />
            ))}
            <circle
              cx={x(row.delta)}
              cy={cy}
              r={4}
              fill={row.is_real ? colour : "var(--ground)"}
              stroke={colour}
              strokeWidth={2}
            />
            <text x={WIDTH - RIGHT + 14} y={cy + 4} fill={colour}>
              {`${signed(row.delta)}  p=${row.p_value.toFixed(3)}`}
            </text>
            <text className="fdead" x={WIDTH - RIGHT + 14} y={cy + 16}>
              {row.is_real ? "we can claim this" : "cannot be claimed"}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
