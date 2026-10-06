import { parquetReadObjects } from "hyparquet";

/**
 * Reading the engine's Parquet, and nothing else.
 *
 * This replaced DuckDB-WASM, which shipped 34 MB of wasm to perform joins over
 * files of twelve kilobytes. SQL across several Parquet files is the right tool
 * once a run holds millions of rows and a query should not download all of
 * them; at three hundred queries it bought nothing a Map does not. The seam
 * stays here so the decision is one file wide if that scale ever arrives.
 *
 * The engine writes snappy for the same reason this reader can be small: a
 * neutral format is only neutral if it opens without an extra codec.
 */
const cache = new Map<string, Promise<unknown[]>>();

async function fetchRows(url: string): Promise<unknown[]> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
  const file = await response.arrayBuffer();
  return parquetReadObjects({ file });
}

/** Rows of one Parquet file, fetched once per URL. */
export function rows<T>(url: string): Promise<T[]> {
  let pending = cache.get(url);
  if (!pending) {
    pending = fetchRows(url);
    cache.set(url, pending);
  }
  return pending as Promise<T[]>;
}

/** Index rows by a key, for joining two files without a query engine. */
export function by<T, K extends keyof T>(items: T[], key: K): Map<T[K], T> {
  return new Map(items.map((item) => [item[key], item]));
}
