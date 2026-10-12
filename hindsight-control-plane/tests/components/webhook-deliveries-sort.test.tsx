// @vitest-environment jsdom
/**
 * Webhook deliveries table. The pure config is pinned directly (API order default,
 * each column both ways, missing values last, status rank). The render tests prove
 * the sort re-applies after "load more", the loaded-only hint shows only for a
 * non-default sort with more pages, and an expanded detail row moves with its
 * delivery.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

import {
  WEBHOOK_DELIVERY_DEFAULT_SORT,
  WEBHOOK_DELIVERY_SORT_KINDS,
  WEBHOOK_DELIVERY_STATUS_RANK,
  sortWebhookDeliveries,
} from "@/components/webhook-deliveries-sort";
import { defaultDirFor } from "@/lib/table-sort";
import type { WebhookDelivery } from "@/lib/api";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  usePathname: () => "/banks/b1",
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("next-intl", () => ({
  useTranslations: (ns?: string) => (key: string, values?: Record<string, unknown>) =>
    values ? `${ns}.${key}:${JSON.stringify(values)}` : key,
  useLocale: () => "en",
}));

vi.mock("@/lib/bank-context", () => ({
  useBank: () => ({ currentBank: "b1" }),
}));

const listWebhooks = vi.fn();
const listWebhookDeliveries = vi.fn();
vi.mock("@/lib/api", () => ({
  client: {
    listWebhooks: (...args: unknown[]) => listWebhooks(...args),
    listWebhookDeliveries: (...args: unknown[]) => listWebhookDeliveries(...args),
  },
}));

const { WebhooksView } = await import("@/components/webhooks-view");

function delivery(id: string, extra: Partial<WebhookDelivery> = {}): WebhookDelivery {
  return {
    id,
    webhook_id: "w1",
    url: "https://example.test/hook",
    event_type: "retain.completed",
    status: "completed",
    attempts: 1,
    next_retry_at: null,
    last_error: null,
    last_response_status: 200,
    last_response_body: null,
    last_attempt_at: null,
    created_at: null,
    updated_at: null,
    ...extra,
  };
}

const ids = (r: WebhookDelivery[]) => r.map((x) => x.id);

describe("webhook deliveries sort config", () => {
  // API order: created_at DESC.
  const rows = [
    delivery("d1", {
      created_at: "2026-10-04T00:00:00Z",
      status: "completed",
      last_response_status: 200,
      attempts: 1,
      event_type: "retain.completed",
    }),
    delivery("d2", {
      created_at: "2026-10-03T00:00:00Z",
      status: "failed",
      last_response_status: 500,
      attempts: 5,
      event_type: "consolidation.completed",
    }),
    delivery("d3", {
      created_at: "2026-10-02T00:00:00Z",
      status: "pending",
      last_response_status: null,
      attempts: 2,
      event_type: "memory_defense.triggered",
    }),
    delivery("d4", {
      created_at: null,
      status: "processing",
      last_response_status: 404,
      attempts: 3,
      event_type: "retain.completed",
    }),
  ];

  it("defaults to created_at desc, which is the API's order", () => {
    expect(WEBHOOK_DELIVERY_DEFAULT_SORT).toEqual({ key: "createdAt", dir: "desc" });
    expect(ids(sortWebhookDeliveries(rows, WEBHOOK_DELIVERY_DEFAULT_SORT))).toEqual(ids(rows));
  });

  it("sorts created_at oldest first with missing last", () => {
    expect(ids(sortWebhookDeliveries(rows, { key: "createdAt", dir: "asc" }))).toEqual([
      "d3",
      "d2",
      "d1",
      "d4",
    ]);
  });

  it("sorts HTTP status both ways with no response last", () => {
    expect(ids(sortWebhookDeliveries(rows, { key: "http", dir: "desc" }))).toEqual([
      "d2",
      "d4",
      "d1",
      "d3",
    ]);
    expect(ids(sortWebhookDeliveries(rows, { key: "http", dir: "asc" }))).toEqual([
      "d1",
      "d4",
      "d2",
      "d3",
    ]);
  });

  it("sorts attempts both ways", () => {
    expect(ids(sortWebhookDeliveries(rows, { key: "attempts", dir: "desc" }))).toEqual([
      "d2",
      "d4",
      "d3",
      "d1",
    ]);
    expect(ids(sortWebhookDeliveries(rows, { key: "attempts", dir: "asc" }))).toEqual([
      "d1",
      "d3",
      "d4",
      "d2",
    ]);
  });

  it("sorts event type both ways, ties kept in input order", () => {
    expect(ids(sortWebhookDeliveries(rows, { key: "event", dir: "asc" }))).toEqual([
      "d2",
      "d3",
      "d1",
      "d4",
    ]);
    expect(ids(sortWebhookDeliveries(rows, { key: "event", dir: "desc" }))).toEqual([
      "d1",
      "d4",
      "d3",
      "d2",
    ]);
  });

  it("ranks status failed, processing, pending, cancelled, completed; unknown last", () => {
    expect(WEBHOOK_DELIVERY_STATUS_RANK).toEqual([
      "completed",
      "cancelled",
      "pending",
      "processing",
      "failed",
    ]);
    expect(defaultDirFor(WEBHOOK_DELIVERY_SORT_KINDS.status)).toBe("desc");
    const r = [...rows, delivery("d5", { status: "cancelled" }), delivery("d6", { status: "?" })];
    expect(ids(sortWebhookDeliveries(r, { key: "status", dir: "desc" }))).toEqual([
      "d2",
      "d4",
      "d3",
      "d5",
      "d1",
      "d6",
    ]);
    expect(ids(sortWebhookDeliveries(r, { key: "status", dir: "asc" }))).toEqual([
      "d1",
      "d5",
      "d3",
      "d4",
      "d2",
      "d6",
    ]);
  });

  it("first clicks follow natural directions and the input is not mutated", () => {
    expect(defaultDirFor(WEBHOOK_DELIVERY_SORT_KINDS.event)).toBe("asc");
    expect(defaultDirFor(WEBHOOK_DELIVERY_SORT_KINDS.http)).toBe("desc");
    expect(defaultDirFor(WEBHOOK_DELIVERY_SORT_KINDS.attempts)).toBe("desc");
    const copy = [...rows];
    sortWebhookDeliveries(rows, { key: "event", dir: "asc" });
    expect(rows).toEqual(copy);
  });
});

describe("WebhooksView deliveries dialog sorting", () => {
  const webhook = {
    id: "w1",
    bank_id: "b1",
    url: "https://example.test/hook",
    secret: null,
    event_types: ["retain.completed"],
    enabled: true,
    http_config: null,
    created_at: "2026-10-01T00:00:00Z",
    updated_at: "2026-10-01T00:00:00Z",
  };

  // Page 1 newest-first, then an older page 2 that holds the most attempts.
  const page1 = [
    delivery("p1-a", { created_at: "2026-10-10T00:00:00Z", attempts: 2 }),
    delivery("p1-b", {
      created_at: "2026-10-09T00:00:00Z",
      attempts: 3,
      status: "failed",
      last_error: "boom-p1-b",
    }),
  ];
  const page2 = [delivery("p2-a", { created_at: "2026-10-08T00:00:00Z", attempts: 9 })];

  beforeEach(() => {
    listWebhooks.mockReset().mockResolvedValue({ items: [webhook], total: 1 });
    listWebhookDeliveries
      .mockReset()
      .mockImplementation((_bank: string, _id: string, _limit: number, cursor?: string) =>
        Promise.resolve(
          cursor ? { items: page2, next_cursor: null } : { items: page1, next_cursor: "c1" }
        )
      );
  });
  afterEach(() => cleanup());

  const label = (column: string) => `common.sortBy:${JSON.stringify({ column })}`;

  async function openDeliveries() {
    render(<WebhooksView />);
    fireEvent.click(await screen.findByRole("button", { name: /deliveriesButton/ }));
    const dialog = await screen.findByRole("dialog");
    await within(dialog).findByRole("table");
    return dialog;
  }

  // Event-type cells are identical, so identify rows by created_at / attempts text.
  function attemptsColumn(dialog: HTMLElement): string[] {
    const body = dialog.querySelector("tbody")!;
    return Array.from(body.querySelectorAll("tr"))
      .filter((tr) => tr.children.length === 6)
      .map((tr) => tr.children[3].textContent!.trim());
  }

  it("opens in API order, then sorts loaded rows and re-sorts after load more", async () => {
    const dialog = await openDeliveries();
    expect(attemptsColumn(dialog)).toEqual(["2", "3"]);
    expect(within(dialog).queryByText("deliveriesSortLoadedOnly")).toBeNull();

    fireEvent.click(
      within(dialog).getByRole("button", { name: label("deliveriesTableHeaderAttempts") })
    );
    expect(attemptsColumn(dialog)).toEqual(["3", "2"]);
    // Non-default sort with more pages: the hint explains the partial scope.
    expect(within(dialog).getByText("deliveriesSortLoadedOnly")).toBeTruthy();

    fireEvent.click(within(dialog).getByRole("button", { name: "loadMore" }));
    await waitFor(() => expect(attemptsColumn(dialog)).toEqual(["9", "3", "2"]));
    // No more pages: neither the button nor the hint remain.
    expect(within(dialog).queryByRole("button", { name: "loadMore" })).toBeNull();
    expect(within(dialog).queryByText("deliveriesSortLoadedOnly")).toBeNull();
  });

  it("keeps an expanded detail row directly under its delivery after sorting", async () => {
    const dialog = await openDeliveries();
    // Expand p1-b (the failed one with an error).
    fireEvent.click(within(dialog).getByText("3").closest("tr")!);
    expect(within(dialog).getByText("boom-p1-b")).toBeTruthy();

    // Status desc puts failed first.
    fireEvent.click(
      within(dialog).getByRole("button", { name: label("deliveriesTableHeaderStatus") })
    );
    const rows = Array.from(dialog.querySelector("tbody")!.querySelectorAll("tr"));
    expect(rows[0].children[3].textContent).toBe("3");
    expect(rows[1].textContent).toContain("boom-p1-b");
    expect(rows[2].children[3].textContent).toBe("2");
  });
});
