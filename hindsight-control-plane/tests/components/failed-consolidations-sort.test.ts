/**
 * Failed consolidations dialog table. The API orders the list by mentioned_at, which
 * the table doesn't show, so the default is the closest visible order (newest
 * failure first); each column sorts both ways with missing values last.
 */
import { describe, expect, it } from "vitest";

import {
  FAILED_CONSOLIDATION_DEFAULT_SORT,
  FAILED_CONSOLIDATION_SORT_KINDS,
  sortFailedConsolidations,
  type FailedConsolidationRow,
} from "@/components/failed-consolidations-sort";
import { defaultDirFor } from "@/lib/table-sort";

type Row = FailedConsolidationRow & { id: string };

function row(id: string, failedAt: string | null, factType: string, text: string): Row {
  return { id, consolidation_failed_at: failedAt, fact_type: factType, text };
}

const rows: Row[] = [
  row("r1", "2026-10-01T10:00:00Z", "world", "banana"),
  row("r2", null, "experience", "Apple"),
  row("r3", "2026-10-03T10:00:00Z", "observation", "cherry"),
  row("r4", "2026-10-02T10:00:00Z", "world", ""),
];

const ids = (r: Row[]) => r.map((x) => x.id);

describe("failed consolidations sort", () => {
  it("defaults to newest failure first, missing timestamps last", () => {
    expect(FAILED_CONSOLIDATION_DEFAULT_SORT).toEqual({ key: "failedAt", dir: "desc" });
    expect(ids(sortFailedConsolidations(rows, FAILED_CONSOLIDATION_DEFAULT_SORT))).toEqual([
      "r3",
      "r4",
      "r1",
      "r2",
    ]);
  });

  it("matches the API order when failures were recorded in mention order", () => {
    // The usual case: a later-mentioned memory also failed later, so the default
    // keeps what the dialog showed before sorting existed.
    const apiOrder = [rows[2], rows[3], rows[0]];
    expect(ids(sortFailedConsolidations(apiOrder, FAILED_CONSOLIDATION_DEFAULT_SORT))).toEqual(
      ids(apiOrder)
    );
  });

  it("sorts failed-at oldest first with missing last", () => {
    expect(ids(sortFailedConsolidations(rows, { key: "failedAt", dir: "asc" }))).toEqual([
      "r1",
      "r4",
      "r3",
      "r2",
    ]);
  });

  it("sorts type both ways, ties kept in input order", () => {
    expect(ids(sortFailedConsolidations(rows, { key: "type", dir: "asc" }))).toEqual([
      "r2",
      "r3",
      "r1",
      "r4",
    ]);
    expect(ids(sortFailedConsolidations(rows, { key: "type", dir: "desc" }))).toEqual([
      "r1",
      "r4",
      "r3",
      "r2",
    ]);
  });

  it("sorts text case-insensitively with blank text last in both directions", () => {
    expect(ids(sortFailedConsolidations(rows, { key: "text", dir: "asc" }))).toEqual([
      "r2",
      "r1",
      "r3",
      "r4",
    ]);
    expect(ids(sortFailedConsolidations(rows, { key: "text", dir: "desc" }))).toEqual([
      "r3",
      "r1",
      "r2",
      "r4",
    ]);
  });

  it("first clicks follow natural directions and the input is not mutated", () => {
    expect(defaultDirFor(FAILED_CONSOLIDATION_SORT_KINDS.failedAt)).toBe("desc");
    expect(defaultDirFor(FAILED_CONSOLIDATION_SORT_KINDS.type)).toBe("asc");
    expect(defaultDirFor(FAILED_CONSOLIDATION_SORT_KINDS.text)).toBe("asc");
    const copy = [...rows];
    sortFailedConsolidations(rows, { key: "text", dir: "asc" });
    expect(rows).toEqual(copy);
  });
});
