/**
 * Sort configuration for the sub-batches table in the operation detail dialog
 * (`bank-operations-view.tsx`). Kept React-free so it can be unit-tested directly.
 */
import { sortRows, type SortKind, type SortState } from "@/lib/table-sort";

export type SubBatchSortKey = "index" | "id" | "items" | "status";

/** Subset of a child operation used for sorting. */
export type SubBatchRow = {
  operation_id: string;
  status: string;
  sub_batch_index: number | null;
  items_count: number | null;
};

/**
 * Async-operation statuses, lowest rank first. The values are the API's
 * `async_operations.status` CHECK constraint (pending, processing, completed, failed,
 * cancelled). Ranked by how much attention a row needs, so the first (descending)
 * click leads with failed, then in-flight work, then pending, cancelled and finally
 * completed rows.
 */
export const OPERATION_STATUS_RANK = [
  "completed",
  "cancelled",
  "pending",
  "processing",
  "failed",
] as const;

/** The API returns children ORDER BY sub_batch_index, so the default is a no-op. */
export const SUB_BATCH_DEFAULT_SORT: SortState<SubBatchSortKey> = { key: "index", dir: "asc" };

export const SUB_BATCH_COLUMNS: Record<
  SubBatchSortKey,
  { kind: SortKind; get: (row: SubBatchRow) => unknown }
> = {
  index: { kind: "number", get: (row) => row.sub_batch_index },
  id: { kind: "text", get: (row) => row.operation_id },
  items: { kind: "number", get: (row) => row.items_count },
  status: { kind: { rank: OPERATION_STATUS_RANK }, get: (row) => row.status },
};

export const SUB_BATCH_SORT_KINDS = Object.fromEntries(
  Object.entries(SUB_BATCH_COLUMNS).map(([k, c]) => [k, c.kind])
) as Record<SubBatchSortKey, SortKind>;

/** Returns a sorted copy; the input (dialog state) is never mutated. */
export function sortSubBatches<T extends SubBatchRow>(
  rows: readonly T[],
  sort: SortState<SubBatchSortKey> | null,
  locale?: string
): T[] {
  return sortRows(rows, sort, SUB_BATCH_COLUMNS, locale);
}
