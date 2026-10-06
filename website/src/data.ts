import { by, rows } from "./parquet";

/** Mirrors the engine's published layout. Bumped when a file moves. */
export const LAYOUT_VERSION = 1;

export type Trust = "identical" | "comparable" | "suspect" | "incomparable";
export type Direction = "improved" | "regressed" | "unchanged";
export type Symptom = "ok" | "rank" | "miss";

export interface Significance {
  delta: number;
  ci_low: number;
  ci_high: number;
  p_value: number;
  n: number;
  seed: number;
  resamples: number;
  /** The engine's judgement, stored rather than re-derived here. */
  is_real: boolean;
  underpowered: boolean;
}

export interface ComparisonManifest {
  layout_version: number;
  left: { run_id: string; name: string };
  right: { run_id: string; name: string };
  trust: { level: Trust; reasons: string[] };
  significance: Record<string, Significance>;
  counts: Record<Direction | "arbitrary", number>;
}

export interface RunManifest {
  layout_version: number;
  measures: string[];
  calibration: {
    n: number;
    sufficient: boolean;
    low_query_gold_overlap: number;
    high_competitor_gold_overlap: number;
  };
  manifest: { run_id: string; config: string; corpus: string; labels: string };
}

export interface Side {
  symptom: Symptom;
  gold_rank: number | null;
  top1: string;
  query_gold_overlap: number;
  competitor_gold_overlap: number;
  tied_with_top1: boolean;
  retrieved_with_zero_score: number;
}

export interface QueryRow {
  query_id: string;
  direction: Direction;
  arbitrary: boolean;
  left_rank: number | null;
  right_rank: number | null;
  left: Side;
  right: Side;
}

export interface Cause {
  kind: string;
  confidence: number;
  producer: string;
  because: string;
}

export interface Hit {
  doc_id: string;
  rank: number;
  score: number;
}

export interface Sets {
  /** Exclusive and exhaustive: one per judged query, so these sum to the total. */
  symptoms: Record<Symptom, number>;
  /** Overlapping: a query may carry two causes, so these do not sum to anything. */
  causes: Record<string, number>;
  judged: number;
}

export interface Comparison {
  id: string;
  manifest: ComparisonManifest;
  left: RunManifest;
  right: RunManifest;
  queries: QueryRow[];
  sets: { left: Sets; right: Sets };
}

const ROOT = "/runs";

const MISSING_RANK = 1 << 30;

/**
 * Sort weight for one query's movement.
 *
 * Ordering by query id leaves the grid a scatter of colour that says nothing.
 * Ordering by how far the gold moved groups the improvements at one end and
 * the regressions at the other, so the shape of the change is readable before
 * a single cell is clicked. A document that was never retrieved counts as
 * worse than any real rank, without pretending to be one.
 */
function movement(row: { left_rank: number | null; right_rank: number | null }): number {
  const before = row.left_rank ?? MISSING_RANK;
  const after = row.right_rank ?? MISSING_RANK;
  return after - before;
}

interface DeltaRow {
  query_id: string;
  direction: Direction;
  arbitrary: boolean;
  left_rank: number | null;
  right_rank: number | null;
}

interface EvidenceRow {
  query_id: string;
  symptom: Symptom;
  gold_rank: number | null;
  top1: string;
  query_gold_overlap: number;
  competitor_gold_overlap: number;
  tied_with_top1: boolean;
  retrieved_with_zero_score: number;
}

function sets(evidence: EvidenceRow[], causes: { kind: string }[]): Sets {
  const symptoms: Record<Symptom, number> = { ok: 0, rank: 0, miss: 0 };
  for (const row of evidence) symptoms[row.symptom] += 1;

  const counted: Record<string, number> = {};
  for (const cause of causes) counted[cause.kind] = (counted[cause.kind] ?? 0) + 1;

  return { symptoms, causes: counted, judged: evidence.length };
}

function side(row: EvidenceRow): Side {
  return {
    symptom: row.symptom,
    gold_rank: row.gold_rank,
    top1: row.top1,
    query_gold_overlap: row.query_gold_overlap,
    competitor_gold_overlap: row.competitor_gold_overlap,
    tied_with_top1: Boolean(row.tied_with_top1),
    retrieved_with_zero_score: row.retrieved_with_zero_score,
  };
}

async function json<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
  return (await response.json()) as T;
}

function runDir(id: string) {
  return `${ROOT}/run/${id}`;
}

export interface Index {
  layout_version: number;
  runs: string[];
  comparisons: string[];
}

/**
 * What the workspace holds, from the index the engine publishes.
 *
 * Not scraped from a directory listing: whether a server renders one at all is
 * its own business, and its markup is not a contract.
 */
export async function loadIndex(): Promise<Index> {
  return json<Index>(`${ROOT}/index.json`);
}

export async function loadComparison(id: string): Promise<Comparison> {
  const base = `${ROOT}/comparison/${id}`;
  const manifest = await json<ComparisonManifest>(`${base}/manifest.json`);

  if (manifest.layout_version !== LAYOUT_VERSION) {
    throw new Error(
      `${id} was written with layout version ${manifest.layout_version}; this reader knows ${LAYOUT_VERSION}`,
    );
  }

  const [left, right] = await Promise.all([
    json<RunManifest>(`${runDir(manifest.left.run_id)}/manifest.json`),
    json<RunManifest>(`${runDir(manifest.right.run_id)}/manifest.json`),
  ]);

  // Three small files and a Map, rather than a query engine. The engine
  // normalised these artifacts so a reader could join them; at this size that
  // join is a lookup, and SQL would have cost 34 MB of wasm to perform it.
  const [deltas, leftEvidence, rightEvidence] = await Promise.all([
    rows<DeltaRow>(`${base}/deltas.parquet`),
    rows<EvidenceRow>(`${runDir(manifest.left.run_id)}/evidence.parquet`),
    rows<EvidenceRow>(`${runDir(manifest.right.run_id)}/evidence.parquet`),
  ]);

  const leftById = by(leftEvidence, "query_id");
  const rightById = by(rightEvidence, "query_id");

  const queries: QueryRow[] = deltas
    .map((delta) => {
      const l = leftById.get(delta.query_id);
      const r = rightById.get(delta.query_id);
      if (!l || !r) return null;
      return {
        query_id: delta.query_id,
        direction: delta.direction,
        arbitrary: Boolean(delta.arbitrary),
        left_rank: delta.left_rank,
        right_rank: delta.right_rank,
        left: side(l),
        right: side(r),
      };
    })
    .filter((row): row is QueryRow => row !== null)
    .sort((a, b) => movement(a) - movement(b) || a.query_id.localeCompare(b.query_id));

  const [leftCauses, rightCauses] = await Promise.all([
    rows<{ query_id: string; kind: string }>(
      `${runDir(manifest.left.run_id)}/hypotheses.parquet`,
    ),
    rows<{ query_id: string; kind: string }>(
      `${runDir(manifest.right.run_id)}/hypotheses.parquet`,
    ),
  ]);

  return {
    id,
    manifest,
    left,
    right,
    queries,
    sets: {
      left: sets(leftEvidence, leftCauses),
      right: sets(rightEvidence, rightCauses),
    },
  };
}

/** Causes for one query, from whichever run bundle is asked for. */
export async function loadCauses(runId: string, queryId: string): Promise<Cause[]> {
  const all = await rows<Cause & { query_id: string }>(
    `${runDir(runId)}/hypotheses.parquet`,
  );
  return all
    .filter((row) => row.query_id === queryId)
    .sort((a, b) => b.confidence - a.confidence || a.kind.localeCompare(b.kind));
}

/** Top results for one query in one run. */
export async function loadHits(runId: string, queryId: string, limit = 5): Promise<Hit[]> {
  const all = await rows<Hit & { query_id: string }>(`${runDir(runId)}/run.parquet`);
  return all
    .filter((row) => row.query_id === queryId)
    .sort((a, b) => a.rank - b.rank)
    .slice(0, limit);
}
