/**
 * Pure, React-free sorting primitives shared by every sortable data table in the
 * control plane. Tables declare each sortable column as a `kind` plus a getter; this
 * module owns the comparison rules so all tables order values the same way.
 */

export type SortDir = "asc" | "desc";

export type SortState<K extends string = string> = { key: K; dir: SortDir };

/**
 * How a column's values compare. `rank` orders categorical values by their position
 * in the list, which is given lowest first (index 0 = lowest rank).
 */
export type SortKind = "text" | "number" | "date" | { rank: readonly string[] };

/**
 * The direction a column sorts in on its first click. Text reads naturally A-Z, while
 * numbers, dates and ranks are almost always wanted largest / newest / highest first.
 */
export function defaultDirFor(kind: SortKind): SortDir {
  return kind === "text" ? "asc" : "desc";
}

/**
 * Header-click transition. Clicking the active column flips its direction; clicking a
 * different column starts it at its natural direction. There is deliberately no third
 * "unsorted" state — a table always has a well-defined order once a header is clicked.
 */
export function nextSort<K extends string>(
  current: SortState<K> | null,
  key: K,
  kind: SortKind
): SortState<K> {
  if (current && current.key === key) {
    return { key, dir: current.dir === "asc" ? "desc" : "asc" };
  }
  return { key, dir: defaultDirFor(kind) };
}

// Normalizes a raw cell value to something comparable for its kind, or `null` when
// the value is missing (null, undefined, blank, unparseable or outside the rank list).
function normalize(kind: SortKind, value: unknown): string | number | null {
  if (value === null || value === undefined) return null;
  if (typeof kind === "object") {
    const idx = kind.rank.indexOf(String(value));
    return idx === -1 ? null : idx;
  }
  switch (kind) {
    case "text": {
      const s = String(value);
      return s.trim() === "" ? null : s;
    }
    case "number": {
      // Number("") and Number("  ") are 0, which would silently rank blanks as zero.
      if (typeof value === "string" && value.trim() === "") return null;
      const n = typeof value === "number" ? value : Number(value);
      return Number.isFinite(n) ? n : null;
    }
    case "date": {
      const t =
        value instanceof Date
          ? value.getTime()
          : typeof value === "number"
            ? value
            : Date.parse(String(value));
      return Number.isFinite(t) ? t : null;
    }
  }
}

/**
 * Builds a comparator for one column. Missing values always sort last regardless of
 * direction: a "newest first" or "oldest first" view should both lead with real data,
 * not a block of blanks. Two missing values compare equal so a stable sort keeps their
 * input order. Text uses a numeric, case-insensitive collator so "bank2" < "bank10".
 */
export function compareValues(
  kind: SortKind,
  dir: SortDir,
  locale?: string
): (a: unknown, b: unknown) => number {
  const sign = dir === "asc" ? 1 : -1;
  const collator =
    kind === "text" ? new Intl.Collator(locale, { numeric: true, sensitivity: "base" }) : null;
  return (a, b) => {
    const x = normalize(kind, a);
    const y = normalize(kind, b);
    if (x === null || y === null) {
      if (x === null && y === null) return 0;
      return x === null ? 1 : -1;
    }
    const cmp = collator ? collator.compare(String(x), String(y)) : (x as number) - (y as number);
    return cmp === 0 ? 0 : sign * Math.sign(cmp);
  };
}

/**
 * Returns a sorted copy of `rows`; the input is never mutated because it is usually
 * React state or query-cache data. With no sort (or a key the table doesn't define)
 * the copy keeps the original order. Ties keep input order — Array.prototype.sort is
 * stable as of ES2019, so the server's order acts as the secondary sort.
 */
export function sortRows<T, K extends string>(
  rows: readonly T[],
  sort: SortState<K> | null,
  columns: Record<K, { kind: SortKind; get: (row: T) => unknown }>,
  locale?: string
): T[] {
  const copy = [...rows];
  if (!sort || !Object.prototype.hasOwnProperty.call(columns, sort.key)) return copy;
  const { kind, get } = columns[sort.key];
  const cmp = compareValues(kind, sort.dir, locale);
  return copy.sort((a, b) => cmp(get(a), get(b)));
}

export function isSortDir(v: unknown): v is SortDir {
  return v === "asc" || v === "desc";
}

/**
 * Validates a sort read from untrusted input such as URL params. Hand-edited or stale
 * URLs must never yield a key the table doesn't know, so anything off-list is `null`.
 */
export function parseSortState<K extends string>(
  key: string | null | undefined,
  dir: string | null | undefined,
  validKeys: readonly K[]
): SortState<K> | null {
  if (!key || !isSortDir(dir)) return null;
  if (!(validKeys as readonly string[]).includes(key)) return null;
  return { key: key as K, dir };
}
