/**
 * Sort configuration for the failed consolidations dialog table in
 * `bank-stats-view.tsx`. Kept React-free so it can be unit-tested directly.
 */
import { sortRows, type SortKind, type SortState } from "@/lib/table-sort";

export type FailedConsolidationSortKey = "failedAt" | "type" | "text";

/** Subset of a failed memory row used for sorting. */
export type FailedConsolidationRow = {
  text: string;
  fact_type: string;
  consolidation_failed_at: string | null;
};

/**
 * Newest failure first. The list API orders by `mentioned_at DESC NULLS LAST`, which
 * is not a column in this table, so this is the closest visible order rather than a
 * no-op: a failure is what the dialog is about, and the latest one is the likeliest
 * to matter.
 */
export const FAILED_CONSOLIDATION_DEFAULT_SORT: SortState<FailedConsolidationSortKey> = {
  key: "failedAt",
  dir: "desc",
};

export const FAILED_CONSOLIDATION_COLUMNS: Record<
  FailedConsolidationSortKey,
  { kind: SortKind; get: (row: FailedConsolidationRow) => unknown }
> = {
  failedAt: { kind: "date", get: (row) => row.consolidation_failed_at },
  type: { kind: "text", get: (row) => row.fact_type },
  text: { kind: "text", get: (row) => row.text },
};

export const FAILED_CONSOLIDATION_SORT_KINDS = Object.fromEntries(
  Object.entries(FAILED_CONSOLIDATION_COLUMNS).map(([k, c]) => [k, c.kind])
) as Record<FailedConsolidationSortKey, SortKind>;

/** Returns a sorted copy; the input (dialog state) is never mutated. */
export function sortFailedConsolidations<T extends FailedConsolidationRow>(
  rows: readonly T[],
  sort: SortState<FailedConsolidationSortKey> | null,
  locale?: string
): T[] {
  return sortRows(rows, sort, FAILED_CONSOLIDATION_COLUMNS, locale);
}
