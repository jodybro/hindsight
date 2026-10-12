// @vitest-environment jsdom
/**
 * The sortable header is the click target and the screen-reader signal for every
 * sortable table, so pin its aria-sort, direction icon and click contract.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import { SortableTableHead } from "@/components/ui/table";
import type { SortState } from "@/lib/table-sort";

type Key = "name" | "count";

function renderHead(
  sort: SortState<Key>,
  extra: {
    sortKey?: Key;
    onSort?: (k: Key) => void;
    align?: "left" | "right";
    sortLabel?: string;
  } = {}
) {
  const onSort = extra.onSort ?? vi.fn();
  render(
    <table>
      <thead>
        <tr>
          <SortableTableHead
            sortKey={extra.sortKey ?? "name"}
            sort={sort}
            onSort={onSort}
            align={extra.align}
            sortLabel={extra.sortLabel}
          >
            Name
          </SortableTableHead>
        </tr>
      </thead>
    </table>
  );
  return {
    th: screen.getByRole("columnheader"),
    button: screen.getByRole("button"),
    icon: () => screen.getByRole("columnheader").querySelector("[data-sort-icon]"),
  };
}

afterEach(() => cleanup());

describe("SortableTableHead", () => {
  it("is aria-sort=none with the neutral icon when another column is sorted", () => {
    const { th, button, icon } = renderHead({ key: "count", dir: "desc" });
    expect(th.getAttribute("aria-sort")).toBe("none");
    expect(icon()?.getAttribute("data-sort-icon")).toBe("none");
    expect(button.className.split(" ")).not.toContain("text-foreground");
  });

  it("is aria-sort=ascending with the up icon when sorted ascending", () => {
    const { th, button, icon } = renderHead({ key: "name", dir: "asc" });
    expect(th.getAttribute("aria-sort")).toBe("ascending");
    expect(icon()?.getAttribute("data-sort-icon")).toBe("asc");
    expect(button.className.split(" ")).toContain("text-foreground");
  });

  it("is aria-sort=descending with the down icon when sorted descending", () => {
    const { th, icon } = renderHead({ key: "name", dir: "desc" });
    expect(th.getAttribute("aria-sort")).toBe("descending");
    expect(icon()?.getAttribute("data-sort-icon")).toBe("desc");
  });

  it("calls onSort with its key when clicked", () => {
    const onSort = vi.fn();
    const { button } = renderHead({ key: "name", dir: "asc" }, { sortKey: "count", onSort });
    expect(button.getAttribute("type")).toBe("button");
    fireEvent.click(button);
    expect(onSort).toHaveBeenCalledTimes(1);
    expect(onSort).toHaveBeenCalledWith("count");
  });

  it("uses sortLabel as the button's accessible name and title", () => {
    renderHead({ key: "name", dir: "asc" }, { sortLabel: "Sort by Name" });
    const button = screen.getByRole("button", { name: "Sort by Name" });
    expect(button.getAttribute("title")).toBe("Sort by Name");
  });

  it("puts the icon after the label by default and before it when right-aligned", () => {
    const left = renderHead({ key: "name", dir: "asc" });
    expect(left.button.lastElementChild?.hasAttribute("data-sort-icon")).toBe(true);
    cleanup();
    const right = renderHead({ key: "name", dir: "asc" }, { align: "right" });
    expect(right.button.firstElementChild?.hasAttribute("data-sort-icon")).toBe(true);
    expect(right.button.className).toContain("justify-end");
    expect(right.th.className).toContain("text-right");
  });
});
