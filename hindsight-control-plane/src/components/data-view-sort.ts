/**
 * Sort configuration for the memories table in `data-view.tsx`. Kept React-free so
 * the column getters and the observation-only "sources" fallback can be unit-tested
 * without rendering the whole view.
 */
import { sortRows, type SortKind, type SortState } from "@/lib/table-sort";

/**
 * Entities and tags are deliberately absent: they render as chip lists, and neither
 * a chip count nor the first chip's name is an order anyone actually wants.
 */
export type MemoryTableSortKey = "memory" | "sources" | "occurred" | "mentioned";

/** Subset of a table row (graph `table_rows` or invalidated list items) used for sorting. */
export type MemoryTableRow = {
  text?: string | null;
  proof_count?: number | null;
  occurred_start?: string | null;
  mentioned_at?: string | null;
};

/** Matches the API's own order (mentioned_at DESC NULLS LAST), so the default is a no-op. */
export const MEMORY_TABLE_DEFAULT_SORT: SortState<MemoryTableSortKey> = {
  key: "mentioned",
  dir: "desc",
};

export const MEMORY_TABLE_COLUMNS: Record<
  MemoryTableSortKey,
  { kind: SortKind; get: (row: MemoryTableRow) => unknown }
> = {
  memory: { kind: "text", get: (row) => row.text },
  sources: { kind: "number", get: (row) => row.proof_count ?? 1 },
  occurred: { kind: "date", get: (row) => row.occurred_start },
  mentioned: { kind: "date", get: (row) => row.mentioned_at },
};

function kindsOf(keys: readonly MemoryTableSortKey[]): Record<MemoryTableSortKey, SortKind> {
  return Object.fromEntries(keys.map((k) => [k, MEMORY_TABLE_COLUMNS[k].kind])) as Record<
    MemoryTableSortKey,
    SortKind
  >;
}

const ALL_KEYS = Object.keys(MEMORY_TABLE_COLUMNS) as MemoryTableSortKey[];
const OBSERVATION_KINDS = kindsOf(ALL_KEYS);
const FACT_KINDS = kindsOf(ALL_KEYS.filter((k) => k !== "sources"));

/**
 * Sortable columns for a fact type. "sources" only exists on the observation table,
 * so world/experience omit it and a stale `?sort=sources` in the URL is rejected by
 * the hook. Returns module-level constants, so the reference is stable per fact type.
 */
export function memoryTableSortKinds(factType: string): Record<MemoryTableSortKey, SortKind> {
  return factType === "observation" ? OBSERVATION_KINDS : FACT_KINDS;
}

/** The sort actually applied: a key the current fact type doesn't show falls back to the default. */
export function effectiveMemoryTableSort(
  sort: SortState<MemoryTableSortKey> | null,
  factType: string
): SortState<MemoryTableSortKey> {
  if (!sort || !Object.prototype.hasOwnProperty.call(memoryTableSortKinds(factType), sort.key)) {
    return MEMORY_TABLE_DEFAULT_SORT;
  }
  return sort;
}

/** Sorts every loaded row (not just one page); returns a new array. */
export function sortMemoryTableRows<T extends MemoryTableRow>(
  rows: readonly T[],
  sort: SortState<MemoryTableSortKey> | null,
  factType: string,
  locale?: string
): T[] {
  return sortRows(rows, effectiveMemoryTableSort(sort, factType), MEMORY_TABLE_COLUMNS, locale);
}
