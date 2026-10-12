/**
 * Sort configuration for the webhook deliveries table in `webhooks-view.tsx`. Kept
 * React-free so it can be unit-tested directly. Deliveries are cursor-paged, so the
 * sort only ever covers the pages loaded so far.
 */
import { sortRows, type SortKind, type SortState } from "@/lib/table-sort";
import { OPERATION_STATUS_RANK } from "./operation-sub-batches-sort";

export type WebhookDeliverySortKey = "status" | "http" | "attempts" | "event" | "createdAt";

/** Subset of a `WebhookDelivery` used for sorting. */
export type WebhookDeliveryRow = {
  status: string;
  last_response_status: number | null;
  attempts: number;
  event_type: string;
  created_at: string | null;
};

/**
 * A delivery is an `async_operations` row (operation_type 'webhook_delivery') and its
 * status is that row's status, so it shares the operations rank: failed, processing,
 * pending, cancelled, completed on the first (descending) click.
 */
export const WEBHOOK_DELIVERY_STATUS_RANK = OPERATION_STATUS_RANK;

/** The API pages deliveries ORDER BY created_at DESC, so the default is a no-op. */
export const WEBHOOK_DELIVERY_DEFAULT_SORT: SortState<WebhookDeliverySortKey> = {
  key: "createdAt",
  dir: "desc",
};

export const WEBHOOK_DELIVERY_COLUMNS: Record<
  WebhookDeliverySortKey,
  { kind: SortKind; get: (row: WebhookDeliveryRow) => unknown }
> = {
  status: { kind: { rank: WEBHOOK_DELIVERY_STATUS_RANK }, get: (row) => row.status },
  http: { kind: "number", get: (row) => row.last_response_status },
  attempts: { kind: "number", get: (row) => row.attempts },
  event: { kind: "text", get: (row) => row.event_type },
  createdAt: { kind: "date", get: (row) => row.created_at },
};

export const WEBHOOK_DELIVERY_SORT_KINDS = Object.fromEntries(
  Object.entries(WEBHOOK_DELIVERY_COLUMNS).map(([k, c]) => [k, c.kind])
) as Record<WebhookDeliverySortKey, SortKind>;

/** Returns a sorted copy of the loaded deliveries; the input is never mutated. */
export function sortWebhookDeliveries<T extends WebhookDeliveryRow>(
  rows: readonly T[],
  sort: SortState<WebhookDeliverySortKey> | null,
  locale?: string
): T[] {
  return sortRows(rows, sort, WEBHOOK_DELIVERY_COLUMNS, locale);
}
