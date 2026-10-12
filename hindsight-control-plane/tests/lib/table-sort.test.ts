import { describe, expect, it } from "vitest";
import {
  compareValues,
  defaultDirFor,
  isSortDir,
  nextSort,
  parseSortState,
  sortRows,
  type SortKind,
} from "@/lib/table-sort";

const sorted = (values: unknown[], kind: SortKind, dir: "asc" | "desc") =>
  [...values].sort(compareValues(kind, dir, "en"));

describe("compareValues — text", () => {
  it("sorts naturally and case-insensitively", () => {
    expect(sorted(["bank10", "Bank2", "apple"], "text", "asc")).toEqual([
      "apple",
      "Bank2",
      "bank10",
    ]);
    expect(sorted(["bank10", "Bank2", "apple"], "text", "desc")).toEqual([
      "bank10",
      "Bank2",
      "apple",
    ]);
  });

  it("puts null, undefined, empty and whitespace-only strings last in both directions", () => {
    const values = [null, "b", "", undefined, "a", "   "];
    expect(sorted(values, "text", "asc").slice(0, 2)).toEqual(["a", "b"]);
    expect(sorted(values, "text", "desc").slice(0, 2)).toEqual(["b", "a"]);
    expect(sorted(values, "text", "asc").slice(2)).toHaveLength(4);
  });
});

describe("compareValues — number", () => {
  it("sorts ascending and descending", () => {
    expect(sorted([3, 10, 1], "number", "asc")).toEqual([1, 3, 10]);
    expect(sorted([3, 10, 1], "number", "desc")).toEqual([10, 3, 1]);
  });

  it("accepts numeric strings", () => {
    expect(sorted(["10", 2, "1"], "number", "asc")).toEqual(["1", 2, "10"]);
  });

  it("puts null, undefined, NaN and non-numeric values last in both directions", () => {
    const values = [null, 5, NaN, undefined, 1, "abc", ""];
    expect(sorted(values, "number", "asc").slice(0, 2)).toEqual([1, 5]);
    expect(sorted(values, "number", "desc").slice(0, 2)).toEqual([5, 1]);
  });
});

describe("compareValues — date", () => {
  const early = "2024-01-01T00:00:00Z";
  const mid = new Date("2025-06-15T12:00:00Z");
  const late = "2026-03-01T00:00:00Z";

  it("sorts mixed ISO strings and Date objects", () => {
    expect(sorted([late, early, mid], "date", "asc")).toEqual([early, mid, late]);
    expect(sorted([late, early, mid], "date", "desc")).toEqual([late, mid, early]);
  });

  it("accepts epoch milliseconds", () => {
    const epoch = Date.parse("2025-01-01T00:00:00Z");
    expect(sorted([late, epoch, early], "date", "asc")).toEqual([early, epoch, late]);
  });

  it("puts null, undefined, empty and invalid dates last in both directions", () => {
    const values = [null, late, "not a date", undefined, early, "", new Date("nope")];
    expect(sorted(values, "date", "asc").slice(0, 2)).toEqual([early, late]);
    expect(sorted(values, "date", "desc").slice(0, 2)).toEqual([late, early]);
    expect(sorted(values, "date", "desc").slice(2)).toHaveLength(5);
  });
});

describe("compareValues — rank", () => {
  const kind: SortKind = { rank: ["low", "medium", "high"] };

  it("orders by position in the rank list", () => {
    expect(sorted(["high", "low", "medium"], kind, "asc")).toEqual(["low", "medium", "high"]);
    expect(sorted(["high", "low", "medium"], kind, "desc")).toEqual(["high", "medium", "low"]);
  });

  it("puts values outside the rank list last in both directions", () => {
    expect(sorted(["bogus", "high", null, "low"], kind, "asc").slice(0, 2)).toEqual([
      "low",
      "high",
    ]);
    expect(sorted(["bogus", "high", null, "low"], kind, "desc").slice(0, 2)).toEqual([
      "high",
      "low",
    ]);
  });
});

describe("compareValues — missing values", () => {
  it("treats two missing values as equal", () => {
    expect(compareValues("text", "asc")(null, undefined)).toBe(0);
    expect(compareValues("number", "desc")(NaN, null)).toBe(0);
    expect(compareValues("date", "asc")("garbage", "")).toBe(0);
  });
});

describe("defaultDirFor / nextSort", () => {
  it("picks the natural first-click direction per kind", () => {
    expect(defaultDirFor("text")).toBe("asc");
    expect(defaultDirFor("number")).toBe("desc");
    expect(defaultDirFor("date")).toBe("desc");
    expect(defaultDirFor({ rank: ["a", "b"] })).toBe("desc");
  });

  it("gives a new key its natural default direction", () => {
    expect(nextSort(null, "name", "text")).toEqual({ key: "name", dir: "asc" });
    expect(nextSort(null, "created", "date")).toEqual({ key: "created", dir: "desc" });
    expect(nextSort({ key: "name", dir: "asc" }, "count", "number")).toEqual({
      key: "count",
      dir: "desc",
    });
  });

  it("toggles the direction when the same key is clicked again", () => {
    const first = nextSort(null, "created", "date");
    const second = nextSort(first, "created", "date");
    const third = nextSort(second, "created", "date");
    expect([first.dir, second.dir, third.dir]).toEqual(["desc", "asc", "desc"]);
  });
});

type Row = { id: string; name: string | null; count: number };

const columns = {
  name: { kind: "text" as const, get: (r: Row) => r.name },
  count: { kind: "number" as const, get: (r: Row) => r.count },
};

describe("sortRows", () => {
  const rows: Row[] = [
    { id: "1", name: "beta", count: 2 },
    { id: "2", name: null, count: 1 },
    { id: "3", name: "Alpha", count: 2 },
    { id: "4", name: "gamma", count: 2 },
  ];
  const ids = (rs: Row[]) => rs.map((r) => r.id);

  it("sorts by the selected column without mutating the input", () => {
    const snapshot = [...rows];
    const out = sortRows(rows, { key: "name", dir: "asc" }, columns);
    expect(ids(out)).toEqual(["3", "1", "4", "2"]);
    expect(rows).toEqual(snapshot);
    expect(out).not.toBe(rows);
  });

  it("is stable for ties", () => {
    expect(ids(sortRows(rows, { key: "count", dir: "desc" }, columns))).toEqual([
      "1",
      "3",
      "4",
      "2",
    ]);
    expect(ids(sortRows(rows, { key: "count", dir: "asc" }, columns))).toEqual([
      "2",
      "1",
      "3",
      "4",
    ]);
  });

  it("returns a copy in original order when there is no sort", () => {
    const out = sortRows(rows, null, columns);
    expect(ids(out)).toEqual(["1", "2", "3", "4"]);
    expect(out).not.toBe(rows);
  });

  it("returns a copy in original order for an unknown key", () => {
    const out = sortRows(rows, { key: "bogus" as "name", dir: "asc" }, columns);
    expect(ids(out)).toEqual(["1", "2", "3", "4"]);
    expect(out).not.toBe(rows);
  });
});

describe("isSortDir / parseSortState", () => {
  const keys = ["name", "created"] as const;

  it("recognises only asc and desc", () => {
    expect(isSortDir("asc")).toBe(true);
    expect(isSortDir("desc")).toBe(true);
    expect(isSortDir("ASC")).toBe(false);
    expect(isSortDir(null)).toBe(false);
  });

  it("accepts a valid key and direction", () => {
    expect(parseSortState("created", "asc", keys)).toEqual({ key: "created", dir: "asc" });
  });

  it("rejects an unknown key", () => {
    expect(parseSortState("bogus", "asc", keys)).toBeNull();
  });

  it("rejects an invalid direction", () => {
    expect(parseSortState("name", "sideways", keys)).toBeNull();
  });

  it("rejects missing params", () => {
    expect(parseSortState(null, "asc", keys)).toBeNull();
    expect(parseSortState("name", undefined, keys)).toBeNull();
    expect(parseSortState(undefined, null, keys)).toBeNull();
  });
});
