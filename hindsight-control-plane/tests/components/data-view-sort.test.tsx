// @vitest-environment jsdom
/**
 * Sortable memories table. The pure config (column getters, the observation-only
 * "sources" fallback) is pinned directly; the render tests prove the sort reads the
 * URL, runs over every loaded row before the 100-row page slice, writes back with
 * router.replace and resets to page 1.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";

import {
  MEMORY_TABLE_DEFAULT_SORT,
  effectiveMemoryTableSort,
  memoryTableSortKinds,
  sortMemoryTableRows,
  type MemoryTableRow,
} from "@/components/data-view-sort";

let currentParams = new URLSearchParams();
const replace = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace, push: vi.fn() }),
  usePathname: () => "/banks/b1",
  useSearchParams: () => currentParams,
}));

vi.mock("next-intl", () => ({
  useTranslations: (ns?: string) => (key: string, values?: Record<string, unknown>) =>
    values ? `${ns}.${key}:${JSON.stringify(values)}` : key,
  useLocale: () => "en",
}));

vi.mock("@/lib/bank-context", () => ({
  useBank: () => ({ currentBank: "b1" }),
}));

const getGraph = vi.fn();
vi.mock("@/lib/api", () => ({
  client: {
    getGraph: (...args: unknown[]) => getGraph(...args),
    getBankStats: vi.fn().mockResolvedValue({ pending_consolidation: 0 }),
    listMemories: vi.fn().mockResolvedValue({ items: [] }),
    listObservationScopes: vi.fn().mockResolvedValue({ scopes: [] }),
  },
}));

// Canvas-heavy / network-backed children the table doesn't need.
vi.mock("@/components/constellation", () => ({ Constellation: () => null }));
vi.mock("@/components/memory-detail-modal", () => ({ MemoryDetailModal: () => null }));
vi.mock("@/components/tag-filter-input", () => ({ TagFilterInput: () => null }));

const { DataView } = await import("@/components/data-view");

type Row = MemoryTableRow & { id: string };

function row(id: string, extra: Partial<Row> = {}): Row {
  return { id, text: id, entities: "", tags: [], ...extra };
}

const texts = (rows: Row[]) => rows.map((r) => r.id);

describe("memory table sort config", () => {
  const rows: Row[] = [
    row("a", { mentioned_at: "2024-01-02T00:00:00Z", occurred_start: null }),
    row("b", { mentioned_at: "2024-03-01T00:00:00Z", occurred_start: "2023-05-01T00:00:00Z" }),
    row("c", { mentioned_at: null, occurred_start: "2023-01-01T00:00:00Z" }),
    row("d", { mentioned_at: "2024-02-01T00:00:00Z", occurred_start: "2023-09-01T00:00:00Z" }),
  ];

  it("defaults to mentioned desc with missing dates last", () => {
    expect(MEMORY_TABLE_DEFAULT_SORT).toEqual({ key: "mentioned", dir: "desc" });
    expect(texts(sortMemoryTableRows(rows, MEMORY_TABLE_DEFAULT_SORT, "world"))).toEqual([
      "b",
      "d",
      "a",
      "c",
    ]);
  });

  it("keeps rows without occurred_start last in both directions", () => {
    expect(texts(sortMemoryTableRows(rows, { key: "occurred", dir: "desc" }, "world"))).toEqual([
      "d",
      "b",
      "c",
      "a",
    ]);
    expect(texts(sortMemoryTableRows(rows, { key: "occurred", dir: "asc" }, "world"))).toEqual([
      "c",
      "b",
      "d",
      "a",
    ]);
  });

  it("sorts memory text A-Z case-insensitively", () => {
    const r = [
      row("1", { text: "banana" }),
      row("2", { text: "Apple" }),
      row("3", { text: "cherry" }),
    ];
    expect(
      sortMemoryTableRows(r, { key: "memory", dir: "asc" }, "world").map((x) => x.text)
    ).toEqual(["Apple", "banana", "cherry"]);
  });

  it("counts entities and tags", () => {
    const r = [
      row("one", { entities: "Alice", tags: ["x", "y", "z"] }),
      row("none", { entities: "", tags: undefined }),
      row("three", { entities: "Alice, Bob, Carol", tags: ["x"] }),
    ];
    expect(texts(sortMemoryTableRows(r, { key: "entities", dir: "desc" }, "world"))).toEqual([
      "three",
      "one",
      "none",
    ]);
    expect(texts(sortMemoryTableRows(r, { key: "tags", dir: "asc" }, "world"))).toEqual([
      "none",
      "three",
      "one",
    ]);
  });

  it("sorts observations by proof_count, treating missing as 1", () => {
    const r = [
      row("p3", { proof_count: 3 }),
      row("pnull", { proof_count: null }),
      row("p7", { proof_count: 7 }),
    ];
    expect(texts(sortMemoryTableRows(r, { key: "sources", dir: "desc" }, "observation"))).toEqual([
      "p7",
      "p3",
      "pnull",
    ]);
  });

  it("falls back to the default when sources is requested outside observations", () => {
    expect(effectiveMemoryTableSort({ key: "sources", dir: "asc" }, "world")).toEqual(
      MEMORY_TABLE_DEFAULT_SORT
    );
    expect(effectiveMemoryTableSort({ key: "sources", dir: "asc" }, "observation")).toEqual({
      key: "sources",
      dir: "asc",
    });
    expect(effectiveMemoryTableSort(null, "world")).toEqual(MEMORY_TABLE_DEFAULT_SORT);
    expect(Object.keys(memoryTableSortKinds("experience"))).not.toContain("sources");
    expect(Object.keys(memoryTableSortKinds("observation"))).toContain("sources");
    const r = [
      row("p3", { proof_count: 3, mentioned_at: "2024-01-01T00:00:00Z" }),
      row("p9", { proof_count: 9, mentioned_at: "2024-06-01T00:00:00Z" }),
    ];
    expect(texts(sortMemoryTableRows(r, { key: "sources", dir: "asc" }, "world"))).toEqual([
      "p9",
      "p3",
    ]);
  });

  it("does not mutate its input", () => {
    const copy = [...rows];
    sortMemoryTableRows(rows, { key: "memory", dir: "desc" }, "world");
    expect(rows).toEqual(copy);
  });
});

describe("DataView table sorting", () => {
  beforeEach(() => {
    currentParams = new URLSearchParams();
    replace.mockClear();
    getGraph.mockReset();
  });
  afterEach(() => cleanup());

  // Ids are zero-padded so text order == numeric order. The API's order (by
  // mentioned_at desc) is the reverse: m149 first.
  function graphOf(n: number) {
    const rows: Row[] = [];
    for (let i = n - 1; i >= 0; i--) {
      const id = `m${String(i).padStart(3, "0")}`;
      rows.push(
        row(id, {
          mentioned_at: new Date(Date.UTC(2024, 0, 1) + i * 86_400_000).toISOString(),
          occurred_start: null,
        })
      );
    }
    return { nodes: [], edges: [], table_rows: rows, total_units: n };
  }

  async function openTable(factType: "world" | "observation" = "world") {
    render(<DataView factType={factType} />);
    fireEvent.click(await screen.findByRole("button", { name: "table" }));
    return screen.getByRole("table");
  }

  function bodyTexts(table: HTMLElement): string[] {
    const body = table.querySelector("tbody")!;
    return within(body)
      .getAllByRole("row")
      .map((tr) => tr.querySelector("td")!.textContent!.trim());
  }

  it("shows the API order (mentioned desc) by default", async () => {
    getGraph.mockResolvedValue(graphOf(150));
    const table = await openTable();
    const rows = bodyTexts(table);
    expect(rows).toHaveLength(100);
    expect(rows[0]).toBe("m149");
    expect(rows[99]).toBe("m050");
    const mentioned = screen.getByRole("columnheader", { name: /columnMentioned/ });
    expect(mentioned.getAttribute("aria-sort")).toBe("descending");
  });

  it("sorts every loaded row before paginating", async () => {
    currentParams = new URLSearchParams("sort=memory&dir=asc");
    getGraph.mockResolvedValue(graphOf(150));
    const table = await openTable();
    const rows = bodyTexts(table);
    // Page 1 holds the globally smallest 100, not the first page re-ordered.
    expect(rows[0]).toBe("m000");
    expect(rows[99]).toBe("m099");
    expect(
      screen.getByRole("columnheader", { name: /columnMemory/ }).getAttribute("aria-sort")
    ).toBe("ascending");
  });

  it("writes the sort to the URL and resets to page 1 on header click", async () => {
    currentParams = new URLSearchParams("view=data&subTab=world");
    getGraph.mockResolvedValue(graphOf(150));
    await openTable();
    // Move to page 2 first.
    fireEvent.click(screen.getByText("1 / 2").nextElementSibling as HTMLElement);
    expect(screen.getByText("2 / 2")).toBeTruthy();

    const memoryLabel = 'common.sortBy:{"column":"columnMemory"}';
    fireEvent.click(screen.getByRole("button", { name: memoryLabel }));
    expect(replace).toHaveBeenCalledWith("/banks/b1?view=data&subTab=world&sort=memory&dir=asc", {
      scroll: false,
    });
    expect(screen.getByText("1 / 2")).toBeTruthy();
  });

  it("ignores a stale sources sort outside observations and offers no sources header", async () => {
    currentParams = new URLSearchParams("sort=sources&dir=asc");
    getGraph.mockResolvedValue(graphOf(5));
    const table = await openTable("world");
    expect(bodyTexts(table)[0]).toBe("m004");
    expect(screen.queryByRole("columnheader", { name: /columnSources/ })).toBeNull();
  });

  it("sorts observations by sources", async () => {
    currentParams = new URLSearchParams("sort=sources&dir=desc");
    getGraph.mockResolvedValue({
      nodes: [],
      edges: [],
      total_units: 3,
      table_rows: [
        row("o1", { proof_count: 2 }),
        row("o2", { proof_count: 9 }),
        row("o3", { proof_count: 5 }),
      ],
    });
    const table = await openTable("observation");
    expect(bodyTexts(table)).toEqual(["o2", "o3", "o1"]);
    expect(
      screen.getByRole("columnheader", { name: /columnSources/ }).getAttribute("aria-sort")
    ).toBe("descending");
  });
});
