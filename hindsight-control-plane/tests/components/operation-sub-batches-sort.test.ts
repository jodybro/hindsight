/**
 * Sub-batches table in the operation detail dialog: the default must reproduce the
 * API's order (children are returned ORDER BY sub_batch_index), every column sorts
 * both ways with missing values last, and status follows the attention rank.
 */
import { describe, expect, it } from "vitest";

import {
  OPERATION_STATUS_RANK,
  SUB_BATCH_DEFAULT_SORT,
  SUB_BATCH_SORT_KINDS,
  sortSubBatches,
  type SubBatchRow,
} from "@/components/operation-sub-batches-sort";
import { defaultDirFor } from "@/lib/table-sort";

function child(
  operation_id: string,
  sub_batch_index: number | null,
  items_count: number | null,
  status: string
): SubBatchRow {
  return { operation_id, sub_batch_index, items_count, status };
}

// API order: by sub_batch_index ascending.
const rows: SubBatchRow[] = [
  child("c-op", 0, 40, "completed"),
  child("a-op", 1, null, "failed"),
  child("d-op", 2, 10, "processing"),
  child("b-op", 3, 25, "pending"),
];

const ids = (r: SubBatchRow[]) => r.map((x) => x.operation_id);

describe("sub-batches sort", () => {
  it("defaults to index asc, which is the API's order", () => {
    expect(SUB_BATCH_DEFAULT_SORT).toEqual({ key: "index", dir: "asc" });
    expect(ids(sortSubBatches(rows, SUB_BATCH_DEFAULT_SORT))).toEqual(ids(rows));
  });

  it("sorts index both ways", () => {
    expect(ids(sortSubBatches(rows, { key: "index", dir: "desc" }))).toEqual([
      "b-op",
      "d-op",
      "a-op",
      "c-op",
    ]);
  });

  it("sorts id A-Z and Z-A", () => {
    expect(ids(sortSubBatches(rows, { key: "id", dir: "asc" }))).toEqual([
      "a-op",
      "b-op",
      "c-op",
      "d-op",
    ]);
    expect(ids(sortSubBatches(rows, { key: "id", dir: "desc" }))).toEqual([
      "d-op",
      "c-op",
      "b-op",
      "a-op",
    ]);
  });

  it("sorts items both ways with a missing count last", () => {
    expect(ids(sortSubBatches(rows, { key: "items", dir: "desc" }))).toEqual([
      "c-op",
      "b-op",
      "d-op",
      "a-op",
    ]);
    expect(ids(sortSubBatches(rows, { key: "items", dir: "asc" }))).toEqual([
      "d-op",
      "b-op",
      "c-op",
      "a-op",
    ]);
  });

  it("keeps a missing sub_batch_index last in both directions", () => {
    const r = [
      child("x", null, 1, "pending"),
      child("y", 1, 1, "pending"),
      child("z", 0, 1, "pending"),
    ];
    expect(ids(sortSubBatches(r, { key: "index", dir: "asc" }))).toEqual(["z", "y", "x"]);
    expect(ids(sortSubBatches(r, { key: "index", dir: "desc" }))).toEqual(["y", "z", "x"]);
  });

  it("ranks status by attention: failed, processing, pending, cancelled, completed", () => {
    expect(OPERATION_STATUS_RANK).toEqual([
      "completed",
      "cancelled",
      "pending",
      "processing",
      "failed",
    ]);
    // First click on a rank column is descending.
    expect(defaultDirFor(SUB_BATCH_SORT_KINDS.status)).toBe("desc");
    const r = [...rows, child("e-op", 4, 1, "cancelled"), child("f-op", 5, 1, "mystery")];
    expect(ids(sortSubBatches(r, { key: "status", dir: "desc" }))).toEqual([
      "a-op",
      "d-op",
      "b-op",
      "e-op",
      "c-op",
      "f-op",
    ]);
    // Unknown statuses are treated as missing: last in both directions.
    expect(ids(sortSubBatches(r, { key: "status", dir: "asc" }))).toEqual([
      "c-op",
      "e-op",
      "b-op",
      "d-op",
      "a-op",
      "f-op",
    ]);
  });

  it("first clicks follow natural directions and the input is not mutated", () => {
    expect(defaultDirFor(SUB_BATCH_SORT_KINDS.id)).toBe("asc");
    expect(defaultDirFor(SUB_BATCH_SORT_KINDS.items)).toBe("desc");
    const copy = [...rows];
    sortSubBatches(rows, { key: "id", dir: "asc" });
    expect(rows).toEqual(copy);
  });
});
