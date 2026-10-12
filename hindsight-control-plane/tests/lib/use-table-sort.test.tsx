// @vitest-environment jsdom
/**
 * Header-click state for sortable tables. Local mode is plain state; URL mode reads
 * `?sort=&dir=` and writes back with router.replace, keeping every other param
 * (view, subTab, ...) so re-sorting never knocks the user off their tab.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, renderHook } from "@testing-library/react";

let currentParams = new URLSearchParams();
const replace = vi.fn();
const push = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace, push }),
  usePathname: () => "/banks/b1",
  useSearchParams: () => currentParams,
}));

import { useTableSort } from "@/lib/use-table-sort";
import type { SortKind, SortState } from "@/lib/table-sort";

type Key = "name" | "count" | "created";
const kinds: Record<Key, SortKind> = { name: "text", count: "number", created: "date" };
const initial: SortState<Key> = { key: "created", dir: "desc" };

beforeEach(() => {
  currentParams = new URLSearchParams();
  replace.mockClear();
  push.mockClear();
});

afterEach(() => cleanup());

describe("useTableSort (local state)", () => {
  it("starts at the initial sort and reports it as default", () => {
    const { result } = renderHook(() => useTableSort(kinds, initial));
    expect(result.current.sort).toEqual(initial);
    expect(result.current.isDefault).toBe(true);
  });

  it("first click uses the column's natural direction, second click flips it", () => {
    const { result } = renderHook(() => useTableSort(kinds, initial));
    act(() => result.current.toggle("name"));
    expect(result.current.sort).toEqual({ key: "name", dir: "asc" });
    expect(result.current.isDefault).toBe(false);
    act(() => result.current.toggle("name"));
    expect(result.current.sort).toEqual({ key: "name", dir: "desc" });
  });

  it("switching column resets to that column's default direction", () => {
    const { result } = renderHook(() => useTableSort(kinds, initial));
    act(() => result.current.toggle("name"));
    act(() => result.current.toggle("name")); // name desc
    act(() => result.current.toggle("count"));
    expect(result.current.sort).toEqual({ key: "count", dir: "desc" });
    act(() => result.current.toggle("name"));
    expect(result.current.sort).toEqual({ key: "name", dir: "asc" });
  });

  it("flipping the initial column toggles its direction", () => {
    const { result } = renderHook(() => useTableSort(kinds, initial));
    act(() => result.current.toggle("created"));
    expect(result.current.sort).toEqual({ key: "created", dir: "asc" });
  });

  it("never touches the router", () => {
    const { result } = renderHook(() => useTableSort(kinds, initial));
    act(() => result.current.toggle("name"));
    expect(replace).not.toHaveBeenCalled();
    expect(push).not.toHaveBeenCalled();
  });

  it("calls onChange with the next state", () => {
    const onChange = vi.fn();
    const { result } = renderHook(() => useTableSort(kinds, initial, { onChange }));
    act(() => result.current.toggle("count"));
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith({ key: "count", dir: "desc" });
  });

  it("tolerates inline kinds/initial objects without looping", () => {
    let renders = 0;
    const { result } = renderHook(() => {
      renders++;
      return useTableSort<Key>(
        { name: "text", count: "number", created: "date" },
        { key: "created", dir: "desc" }
      );
    });
    act(() => result.current.toggle("name"));
    expect(result.current.sort).toEqual({ key: "name", dir: "asc" });
    expect(renders).toBeLessThan(5);
  });
});

describe("useTableSort (URL-synced)", () => {
  it("reads the sort from the URL", () => {
    currentParams = new URLSearchParams("sort=name&dir=asc");
    const { result } = renderHook(() => useTableSort(kinds, initial, { syncUrl: true }));
    expect(result.current.sort).toEqual({ key: "name", dir: "asc" });
    expect(result.current.isDefault).toBe(false);
  });

  it("falls back to initial for an unknown key or direction", () => {
    currentParams = new URLSearchParams("sort=bogus&dir=up");
    const { result } = renderHook(() => useTableSort(kinds, initial, { syncUrl: true }));
    expect(result.current.sort).toEqual(initial);
    expect(result.current.isDefault).toBe(true);
  });

  it("falls back to initial for a valid key with an invalid direction", () => {
    currentParams = new URLSearchParams("sort=name&dir=sideways");
    const { result } = renderHook(() => useTableSort(kinds, initial, { syncUrl: true }));
    expect(result.current.sort).toEqual(initial);
  });

  it("writes with replace, preserving the other params, without scrolling", () => {
    currentParams = new URLSearchParams("view=data&subTab=world");
    const { result } = renderHook(() => useTableSort(kinds, initial, { syncUrl: true }));
    act(() => result.current.toggle("name"));
    expect(push).not.toHaveBeenCalled();
    expect(replace).toHaveBeenCalledTimes(1);
    const [href, options] = replace.mock.calls[0];
    expect(options).toEqual({ scroll: false });
    const [path, query] = (href as string).split("?");
    expect(path).toBe("/banks/b1");
    const params = new URLSearchParams(query);
    expect(params.get("view")).toBe("data");
    expect(params.get("subTab")).toBe("world");
    expect(params.get("sort")).toBe("name");
    expect(params.get("dir")).toBe("asc");
  });

  it("flips the direction of the column already in the URL", () => {
    currentParams = new URLSearchParams("view=data&sort=name&dir=asc");
    const { result } = renderHook(() => useTableSort(kinds, initial, { syncUrl: true }));
    act(() => result.current.toggle("name"));
    const params = new URLSearchParams((replace.mock.calls[0][0] as string).split("?")[1]);
    expect(params.get("sort")).toBe("name");
    expect(params.get("dir")).toBe("desc");
    expect(params.get("view")).toBe("data");
  });

  it("removes sort/dir when toggling back to the initial sort", () => {
    currentParams = new URLSearchParams("view=data&subTab=world&sort=created&dir=asc");
    const { result } = renderHook(() => useTableSort(kinds, initial, { syncUrl: true }));
    act(() => result.current.toggle("created")); // asc -> desc == initial
    expect(replace).toHaveBeenCalledWith("/banks/b1?view=data&subTab=world", { scroll: false });
  });

  it("uses the bare pathname when no params remain", () => {
    currentParams = new URLSearchParams("sort=created&dir=asc");
    const { result } = renderHook(() => useTableSort(kinds, initial, { syncUrl: true }));
    act(() => result.current.toggle("created"));
    expect(replace).toHaveBeenCalledWith("/banks/b1", { scroll: false });
  });

  it("does not mutate the current search params", () => {
    currentParams = new URLSearchParams("view=data");
    const { result } = renderHook(() => useTableSort(kinds, initial, { syncUrl: true }));
    act(() => result.current.toggle("name"));
    expect(currentParams.toString()).toBe("view=data");
  });

  it("calls onChange with the next state", () => {
    const onChange = vi.fn();
    const { result } = renderHook(() => useTableSort(kinds, initial, { syncUrl: true, onChange }));
    act(() => result.current.toggle("count"));
    expect(onChange).toHaveBeenCalledWith({ key: "count", dir: "desc" });
  });
});
