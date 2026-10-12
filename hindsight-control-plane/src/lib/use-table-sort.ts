"use client";

/**
 * React state for a sortable table's header clicks, built on the pure helpers in
 * `table-sort.ts`. Top-level tables pass `syncUrl` so the sort lives in the address
 * bar as `?sort=<key>&dir=asc|desc` and survives reloads and shared links; nested or
 * modal tables keep it in local state.
 */
import { useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

import { nextSort, parseSortState, type SortKind, type SortState } from "@/lib/table-sort";

export type UseTableSortOptions<K extends string> = {
  /** Persist the sort in the URL (`sort` / `dir` params) instead of local state. */
  syncUrl?: boolean;
  /** Called with the new sort on every header click, e.g. to reset paging. */
  onChange?: (next: SortState<K>) => void;
};

export type UseTableSortResult<K extends string> = {
  sort: SortState<K>;
  toggle: (key: K) => void;
  /** True when the table is showing its initial (default) order. */
  isDefault: boolean;
};

function sameSort<K extends string>(a: SortState<K>, b: SortState<K>): boolean {
  return a.key === b.key && a.dir === b.dir;
}

export function useTableSort<K extends string>(
  kinds: Record<K, SortKind>,
  initial: SortState<K>,
  opts?: UseTableSortOptions<K>
): UseTableSortResult<K> {
  const syncUrl = opts?.syncUrl ?? false;
  const onChange = opts?.onChange;

  // Navigation hooks are called unconditionally (rules of hooks); in local mode
  // their values are simply ignored.
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [localSort, setLocalSort] = useState<SortState<K>>(initial);

  // In URL mode the URL is the source of truth: derive on every render rather than
  // mirroring into state, so back/forward and hand edits are reflected without
  // effects. Unknown keys or directions fall back to `initial` silently.
  const sort: SortState<K> = syncUrl
    ? (parseSortState(
        searchParams?.get("sort"),
        searchParams?.get("dir"),
        Object.keys(kinds) as K[]
      ) ?? initial)
    : localSort;

  // Deliberately not memoized and no effects: callers often pass `kinds` / `initial`
  // inline, so they are only read at click time and can never trigger a render loop.
  const toggle = (key: K) => {
    const next = nextSort(sort, key, kinds[key]);
    if (syncUrl) {
      // Copy the current params so sibling state (view, subTab, ...) survives.
      const merged = new URLSearchParams(searchParams?.toString() ?? "");
      if (sameSort(next, initial)) {
        // The default order needs no params; keeps URLs clean.
        merged.delete("sort");
        merged.delete("dir");
      } else {
        merged.set("sort", next.key);
        merged.set("dir", next.dir);
      }
      const query = merged.toString();
      // replace, not push: re-sorting shouldn't fill the history stack.
      router.replace(query ? `${pathname}?${query}` : pathname, { scroll: false });
    } else {
      setLocalSort(next);
    }
    onChange?.(next);
  };

  return { sort, toggle, isDefault: sameSort(sort, initial) };
}
