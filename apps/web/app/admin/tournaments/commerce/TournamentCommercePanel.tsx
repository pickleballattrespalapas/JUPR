"use client";

import Link from "next/link";
import { useMemo, useRef, useState } from "react";

import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, actionUncertain, type ActionCompletion } from "@/components/interaction";
import {
  adminTournamentCommerceExportUrl,
  AdminTournamentCommerceDetail,
  formatCommerceMoney,
  getAdminTournamentCommerceDetail,
  getAdminTournamentCommerceOperation,
  getAdminTournamentCommerceStatus,
  mutateAdminTournamentCommerce,
  TournamentCommerceBundle,
  TournamentCommerceBundleComponent,
  TournamentCommerceCatalogStatus,
  TournamentCommerceItem,
  TournamentCommercePromotion,
  TournamentCommerceVariant
} from "@/lib/tournamentCommerceApi";
import { useAuthenticatedAutoLoad } from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";

type Props = {
  clubId: string;
  tournamentId: string;
  tournamentName: string;
};

type EditableCatalog = {
  currency: "USD";
  catalog_revision: number;
  catalog_fingerprint: string;
  items: TournamentCommerceItem[];
  variants: TournamentCommerceVariant[];
  bundles: TournamentCommerceBundle[];
  bundle_components: TournamentCommerceBundleComponent[];
  promotions: TournamentCommercePromotion[];
  event_options: Array<Record<string, unknown>>;
};

type Tab = "catalog" | "orders" | "fulfillment" | "recovery";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const inputStyle = {
  width: "100%",
  minWidth: 0,
  boxSizing: "border-box" as const,
  padding: "0.55rem",
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  font: "inherit"
};

const buttonStyle = {
  padding: "0.6rem 0.9rem",
  borderRadius: "999px",
  border: "1px solid #0f172a",
  background: "#0f172a",
  color: "white",
  fontWeight: 800,
  cursor: "pointer"
};

const ghostButtonStyle = {
  ...buttonStyle,
  background: "white",
  color: "#0f172a"
};

const dangerButtonStyle = {
  ...ghostButtonStyle,
  borderColor: "#fecaca",
  color: "#b91c1c"
};

const statusOptions: TournamentCommerceCatalogStatus[] = [
  "DRAFT",
  "ACTIVE",
  "ARCHIVED"
];

function cloneCatalog(catalog: EditableCatalog): EditableCatalog {
  return JSON.parse(JSON.stringify(catalog)) as EditableCatalog;
}

function stringValue(
  row: Record<string, unknown>,
  key: string,
  fallback = ""
): string {
  const value = row[key];
  return value == null ? fallback : String(value);
}

function numberValue(
  row: Record<string, unknown>,
  key: string,
  fallback = 0
): number {
  const parsed = Number(row[key]);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function recordValue(
  row: Record<string, unknown>,
  key: string
): Record<string, unknown> {
  const value = row[key];
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function recordListValue(
  row: Record<string, unknown>,
  key: string
): Array<Record<string, unknown>> {
  const value = row[key];
  return Array.isArray(value)
    ? value.filter(
        (entry): entry is Record<string, unknown> =>
          Boolean(entry) && typeof entry === "object"
      )
    : [];
}

function inputDateTime(value?: string | null): string {
  return value ? String(value).slice(0, 16) : "";
}

function outputDateTime(value: string): string | null {
  if (!value) return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed.toISOString();
}

function nullableWholeNumber(value: string): number | null {
  if (!value.trim()) return null;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? Math.max(0, parsed) : null;
}

function moneyMinor(value: string): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.round(parsed * 100) : 0;
}

function targetValue(promotion: TournamentCommercePromotion): string {
  if (promotion.target_type === "ITEM") return `ITEM:${promotion.item_id || ""}`;
  if (promotion.target_type === "ITEM_VARIANT") {
    return `ITEM_VARIANT:${promotion.variant_id || ""}`;
  }
  return `BUNDLE:${promotion.bundle_id || ""}`;
}

function operationStatusStyle(status: string) {
  const normalized = status.toUpperCase();
  if (normalized === "COMPLETED") {
    return { color: "#166534", background: "#f0fdf4" };
  }
  if (normalized === "RECOVERY_REQUIRED" || normalized === "FAILED") {
    return { color: "#991b1b", background: "#fef2f2" };
  }
  return { color: "#92400e", background: "#fffbeb" };
}

export default function TournamentCommercePanel({
  clubId,
  tournamentId,
  tournamentName
}: Props) {
  const {
    accessToken,
    loading: sessionLoading,
    message: sessionMessage
  } = useAdminSession();
  const [status, setStatus] = useState<Record<string, unknown> | null>(null);
  const [selectedTournamentId, setSelectedTournamentId] = useState(tournamentId);
  const [detail, setDetail] =
    useState<AdminTournamentCommerceDetail | null>(null);
  const [draft, setDraft] = useState<EditableCatalog | null>(null);
  const [tab, setTab] = useState<Tab>("catalog");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [catalogReviewed, setCatalogReviewed] = useState(false);
  const [paymentChoices, setPaymentChoices] = useState<Record<string, string>>(
    {}
  );
  const [cancelReasons, setCancelReasons] = useState<Record<string, string>>({});
  const [fulfillmentStatuses, setFulfillmentStatuses] = useState<
    Record<string, string>
  >({});
  const [fulfillmentNotes, setFulfillmentNotes] = useState<
    Record<string, string>
  >({});
  const [operationEvidence, setOperationEvidence] =
    useState<Record<string, unknown> | null>(null);
  const operationKeysRef = useRef(new Map<string, { fingerprint: string; key: string }>());

  function stableOperationKey(scope: string, request: Record<string, unknown>): string {
    const fingerprint = JSON.stringify(request);
    const pending = operationKeysRef.current.get(scope);
    if (pending?.fingerprint === fingerprint) return pending.key;
    const key = `commerce:${crypto.randomUUID()}`;
    operationKeysRef.current.set(scope, { fingerprint, key });
    return key;
  }

  function clearOperationKey(scope: string, key: string) {
    if (operationKeysRef.current.get(scope)?.key === key) operationKeysRef.current.delete(scope);
  }

  const runtime = recordValue(status || {}, "runtime");
  const environment = stringValue(runtime, "environment", "local");
  const adminWriteReady =
    environment === "production"
      ? false
      : environment === "staging"
        ? Boolean(runtime.admin_write_ready)
        : true;

  const variantsByItem = useMemo(() => {
    const grouped = new Map<string, TournamentCommerceVariant[]>();
    for (const variant of draft?.variants || []) {
      const rows = grouped.get(variant.item_id) || [];
      rows.push(variant);
      grouped.set(variant.item_id, rows);
    }
    return grouped;
  }, [draft?.variants]);

  const componentsByBundle = useMemo(() => {
    const grouped = new Map<string, TournamentCommerceBundleComponent[]>();
    for (const component of draft?.bundle_components || []) {
      const rows = grouped.get(String(component.bundle_id || "")) || [];
      rows.push(component);
      grouped.set(String(component.bundle_id || ""), rows);
    }
    return grouped;
  }, [draft?.bundle_components]);

  const savedItemIds = useMemo(
    () => new Set((detail?.catalog.items || []).map((item) => item.id)),
    [detail?.catalog.items]
  );
  const savedBundleIds = useMemo(
    () => new Set((detail?.catalog.bundles || []).map((bundle) => bundle.id)),
    [detail?.catalog.bundles]
  );

  const eventLabels = useMemo(
    () =>
      new Map(
        (draft?.event_options || []).map((event) => [
          stringValue(event, "id"),
          stringValue(
            event,
            "label",
            stringValue(event, "division_name", "Tournament event")
          )
        ])
      ),
    [draft?.event_options]
  );

  const variantLabels = useMemo(() => {
    const itemById = new Map(
      (draft?.items || []).map((item) => [item.id, item.name])
    );
    return new Map(
      (draft?.variants || []).map((variant) => [
        variant.id,
        `${itemById.get(variant.item_id) || "Extra"} — ${variant.name}`
      ])
    );
  }, [draft?.items, draft?.variants]);

  async function loadWorkspace() {
    if (!accessToken) return;
    setBusy(true);
    setMessage(null);
    const statusResponse = await getAdminTournamentCommerceStatus(
      clubId,
      accessToken
    );
    if (statusResponse.error) {
      setBusy(false);
      setMessage(statusResponse.error);
      setStatus(null);
      return;
    }
    setStatus(statusResponse.data);
    await loadDetail(tournamentId);
  }

  async function loadDetail(tournamentId = selectedTournamentId) {
    setSelectedTournamentId(tournamentId);
    setDetail(null);
    setDraft(null);
    setCatalogReviewed(false);
    setOperationEvidence(null);
    if (!tournamentId || !accessToken) return;
    setBusy(true);
    setMessage(null);
    const response = await getAdminTournamentCommerceDetail(
      clubId,
      tournamentId,
      accessToken
    );
    setBusy(false);
    if (response.error || !response.data) {
      setMessage(response.error || "Unable to load tournament extras.");
      return;
    }
    setDetail(response.data);
    setDraft(cloneCatalog(response.data.catalog as EditableCatalog));
    setPaymentChoices(
      Object.fromEntries(
        response.data.orders.map((order) => [
          stringValue(order, "registration_id"),
          stringValue(order, "payment_status", "UNPAID")
        ])
      )
    );
    setFulfillmentStatuses(
      Object.fromEntries(
        response.data.fulfillment.map((row) => [
          stringValue(row, "id"),
          stringValue(row, "status", "PENDING")
        ])
      )
    );
    setFulfillmentNotes(
      Object.fromEntries(
        response.data.fulfillment.map((row) => [
          stringValue(row, "id"),
          stringValue(row, "notes")
        ])
      )
    );
    setMessage(null);
  }

  function updateDraft(change: (next: EditableCatalog) => void) {
    setDraft((current) => {
      if (!current) return current;
      const next = cloneCatalog(current);
      change(next);
      return next;
    });
    setCatalogReviewed(false);
  }

  function addItem() {
    const itemId = crypto.randomUUID();
    const variantId = crypto.randomUUID();
    updateDraft((next) => {
      next.items.push({
        id: itemId,
        name: "New extra",
        description: "",
        kind: "OTHER",
        status: "DRAFT",
        base_price_minor: 0,
        inventory_limit: null,
        max_per_registration: 1,
        available_from: null,
        available_until: null,
        requires_fulfillment: true,
        fulfillment_instructions: "",
        sort_order: next.items.length
      });
      next.variants.push({
        id: variantId,
        item_id: itemId,
        name: "Standard",
        sku: "",
        status: "ACTIVE",
        price_delta_minor: 0,
        price_minor: 0,
        inventory_limit: null,
        sort_order: 0
      });
    });
  }

  function removeItem(itemId: string) {
    updateDraft((next) => {
      const variantIds = new Set(
        next.variants
          .filter((variant) => variant.item_id === itemId)
          .map((variant) => variant.id)
      );
      next.items = next.items.filter((item) => item.id !== itemId);
      next.variants = next.variants.filter(
        (variant) => variant.item_id !== itemId
      );
      next.bundle_components = next.bundle_components.filter(
        (component) =>
          component.item_id !== itemId &&
          !variantIds.has(String(component.variant_id || ""))
      );
      next.promotions = next.promotions.filter(
        (promotion) =>
          promotion.item_id !== itemId &&
          !variantIds.has(String(promotion.variant_id || ""))
      );
    });
  }

  function addVariant(itemId: string) {
    const item = draft?.items.find((row) => row.id === itemId);
    updateDraft((next) => {
      next.variants.push({
        id: crypto.randomUUID(),
        item_id: itemId,
        name: "New option",
        sku: "",
        status: "ACTIVE",
        price_delta_minor: 0,
        price_minor: item?.base_price_minor || 0,
        inventory_limit: null,
        sort_order: next.variants.filter((row) => row.item_id === itemId).length
      });
    });
  }

  function addVariantPreset(itemId: string, names: string[]) {
    const item = draft?.items.find((row) => row.id === itemId);
    updateDraft((next) => {
      const existing = new Set(
        next.variants
          .filter((variant) => variant.item_id === itemId)
          .map((variant) => variant.name.trim().toLowerCase())
      );
      let sortOrder = next.variants.filter((row) => row.item_id === itemId).length;
      for (const name of names) {
        if (existing.has(name.toLowerCase())) continue;
        next.variants.push({
          id: crypto.randomUUID(),
          item_id: itemId,
          name,
          sku: "",
          status: "ACTIVE",
          price_delta_minor: 0,
          price_minor: item?.base_price_minor || 0,
          inventory_limit: null,
          sort_order: sortOrder
        });
        sortOrder += 1;
      }
    });
  }

  function removeVariant(variantId: string) {
    updateDraft((next) => {
      next.variants = next.variants.filter(
        (variant) => variant.id !== variantId
      );
      next.bundle_components = next.bundle_components.filter(
        (component) => component.variant_id !== variantId
      );
      next.promotions = next.promotions.filter(
        (promotion) => promotion.variant_id !== variantId
      );
    });
  }

  function addBundle() {
    updateDraft((next) => {
      next.bundles.push({
        id: crypto.randomUUID(),
        name: "New bundle",
        description: "",
        status: "DRAFT",
        price_minor: 0,
        max_per_registration: 1,
        available_from: null,
        available_until: null,
        sort_order: next.bundles.length
      });
    });
  }

  function removeBundle(bundleId: string) {
    updateDraft((next) => {
      next.bundles = next.bundles.filter((bundle) => bundle.id !== bundleId);
      next.bundle_components = next.bundle_components.filter(
        (component) => component.bundle_id !== bundleId
      );
      next.promotions = next.promotions.filter(
        (promotion) => promotion.bundle_id !== bundleId
      );
    });
  }

  function addBundleComponent(bundleId: string, selection: string) {
    if (!selection) return;
    const [componentType, targetId] = selection.split(":");
    updateDraft((next) => {
      if (
        next.bundle_components.some(
          (row) =>
            row.bundle_id === bundleId &&
            (row.event_option_id === targetId || row.variant_id === targetId)
        )
      ) {
        return;
      }
      if (componentType === "EVENT_OPTION") {
        next.bundle_components.push({
          id: crypto.randomUUID(),
          bundle_id: bundleId,
          component_type: "EVENT_OPTION",
          event_option_id: targetId,
          item_id: null,
          variant_id: null,
          quantity: 1
        });
      } else {
        const variant = next.variants.find((row) => row.id === targetId);
        if (!variant) return;
        next.bundle_components.push({
          id: crypto.randomUUID(),
          bundle_id: bundleId,
          component_type: "ITEM_VARIANT",
          event_option_id: null,
          item_id: variant.item_id,
          variant_id: variant.id,
          quantity: 1
        });
      }
    });
  }

  function setBundleEventChoice(
    bundleId: string,
    eventOptionId: string,
    enabled: boolean
  ) {
    updateDraft((next) => {
      const current = next.bundle_components.filter(
        (row) =>
          row.bundle_id === bundleId && row.component_type === "EVENT_CHOICE"
      );
      const requiredCount = Math.max(1, Number(current[0]?.quantity || 1));
      next.bundle_components = next.bundle_components.filter(
        (row) =>
          !(
            row.bundle_id === bundleId &&
            row.component_type === "EVENT_CHOICE" &&
            row.event_option_id === eventOptionId
          )
      );
      if (enabled) {
        next.bundle_components.push({
          id: crypto.randomUUID(),
          bundle_id: bundleId,
          component_type: "EVENT_CHOICE",
          event_option_id: eventOptionId,
          item_id: null,
          variant_id: null,
          quantity: requiredCount
        });
      }
      const eligibleCount = next.bundle_components.filter(
        (row) =>
          row.bundle_id === bundleId && row.component_type === "EVENT_CHOICE"
      ).length;
      const clamped = Math.max(1, Math.min(requiredCount, eligibleCount || 1));
      for (const row of next.bundle_components) {
        if (
          row.bundle_id === bundleId &&
          row.component_type === "EVENT_CHOICE"
        ) {
          row.quantity = clamped;
        }
      }
      const bundle = next.bundles.find((row) => row.id === bundleId);
      if (bundle && eligibleCount) bundle.max_per_registration = 1;
    });
  }

  function setBundleEventChoiceCount(bundleId: string, value: number) {
    updateDraft((next) => {
      const eligible = next.bundle_components.filter(
        (row) =>
          row.bundle_id === bundleId && row.component_type === "EVENT_CHOICE"
      );
      const count = Math.max(1, Math.min(Math.floor(value || 1), eligible.length || 1));
      for (const row of eligible) row.quantity = count;
      const bundle = next.bundles.find((row) => row.id === bundleId);
      if (bundle && eligible.length) bundle.max_per_registration = 1;
    });
  }

  function addPromotion() {
    const item = draft?.items[0];
    const variant = draft?.variants[0];
    const bundle = draft?.bundles[0];
    if (!item && !bundle) {
      setMessage("Add an extra or bundle before creating a giveaway.");
      return;
    }
    updateDraft((next) => {
      next.promotions.push({
        id: crypto.randomUUID(),
        name: "New giveaway",
        promotion_type: "DATE_WINDOW_FREE",
        target_type: variant ? "ITEM_VARIANT" : bundle ? "BUNDLE" : "ITEM",
        item_id: variant?.item_id || item?.id || null,
        variant_id: variant?.id || null,
        bundle_id: variant || item ? null : bundle?.id || null,
        starts_at: null,
        ends_at: null,
        giveaway_limit: null,
        per_registration_limit: 1,
        priority: 100,
        status: "DRAFT"
      });
    });
  }

  function setPromotionTarget(promotionId: string, selection: string) {
    const [targetType, targetId] = selection.split(":");
    updateDraft((next) => {
      const promotion = next.promotions.find((row) => row.id === promotionId);
      if (!promotion) return;
      promotion.target_type = targetType as TournamentCommercePromotion["target_type"];
      promotion.item_id = null;
      promotion.variant_id = null;
      promotion.bundle_id = null;
      if (targetType === "ITEM") promotion.item_id = targetId;
      if (targetType === "ITEM_VARIANT") {
        const variant = next.variants.find((row) => row.id === targetId);
        promotion.item_id = variant?.item_id || null;
        promotion.variant_id = targetId;
      }
      if (targetType === "BUNDLE") promotion.bundle_id = targetId;
    });
  }

  function localCatalogError(): string | null {
    if (!draft) return "Load a tournament first.";
    for (const item of draft.items) {
      if (!item.name.trim()) return "Every extra needs a name.";
      if (
        item.status === "ACTIVE" &&
        !(variantsByItem.get(item.id) || []).some(
          (variant) => variant.status === "ACTIVE"
        )
      ) {
        return "An active item needs at least one active option";
      }
    }
    for (const bundle of draft.bundles) {
      const components = componentsByBundle.get(bundle.id) || [];
      if (bundle.status === "ACTIVE" && !components.length) {
        return "An active bundle needs at least one event or extra.";
      }
      const choices = components.filter(
        (component) => component.component_type === "EVENT_CHOICE"
      );
      if (choices.length) {
        const requiredCount = Number(choices[0]?.quantity || 1);
        if (choices.some((choice) => Number(choice.quantity || 1) !== requiredCount)) {
          return `${bundle.name}: every flexible event must use the same required-event count.`;
        }
        if (requiredCount < 1 || requiredCount > choices.length) {
          return `${bundle.name}: choose-any count must be between 1 and the number of eligible events.`;
        }
        if (bundle.max_per_registration !== 1) {
          return `${bundle.name}: flexible event bundles are limited to one per registration.`;
        }
      }
    }
    return null;
  }

  async function previewCatalog() {
    const localError = localCatalogError();
    if (localError || !draft || !selectedTournamentId) {
      setMessage(localError || "Load a tournament first.");
      return;
    }
    setBusy(true);
    setMessage(null);
    const path = `/admin/clubs/${encodeURIComponent(
      clubId
    )}/tournaments/commerce/tournaments/${encodeURIComponent(
      selectedTournamentId
    )}/catalog/preview`;
    const response = await mutateAdminTournamentCommerce<{
      ok: boolean;
      catalog: Record<string, unknown>;
    }>(path, "POST", { catalog: draft }, accessToken);
    setBusy(false);
    if (response.error) {
      setCatalogReviewed(false);
      setMessage(response.error);
      return;
    }
    setCatalogReviewed(true);
    setMessage(
      "Catalog review passed. Confirm the save only when the staging write wave is open."
    );
  }

  async function saveCatalog(confirmationText: string): Promise<ActionCompletion> {
    if (!draft || !detail || !selectedTournamentId) throw new Error("Load and review a tournament catalog before saving.");
    const path = `/admin/clubs/${encodeURIComponent(
      clubId
    )}/tournaments/commerce/tournaments/${encodeURIComponent(
      selectedTournamentId
    )}/catalog`;
    const request = {
      expected_catalog_fingerprint: detail.catalog.catalog_fingerprint || "",
      catalog: draft,
      confirmation_text: confirmationText,
      source: "next_tournament_commerce_admin"
    };
    const operationScope = `catalog:${selectedTournamentId}`;
    const idempotencyKey = stableOperationKey(operationScope, request);
    setBusy(true);
    setMessage(null);
    const response = await mutateAdminTournamentCommerce<Record<string, unknown>>(
      path,
      "PUT",
      { ...request, idempotency_key: idempotencyKey },
      accessToken
    );
    setBusy(false);
    if (response.error) {
      setMessage(response.error);
      if (response.status == null || response.status >= 500) {
        return actionUncertain("Catalog save needs verification", response.error, idempotencyKey, "Retry exact catalog save", () => saveCatalog(confirmationText));
      }
      throw new Error(response.error);
    }
    clearOperationKey(operationScope, idempotencyKey);
    await loadDetail(selectedTournamentId);
    setMessage("Tournament extras catalog saved.");
    return actionSuccess("Tournament catalog saved", "The reviewed extras, bundles, inventory, and promotions catalog was saved.");
  }

  async function updatePayment(
    registrationId: string,
    orderUpdatedAt: string,
    confirmationText: string
  ): Promise<ActionCompletion> {
    const paymentStatus = paymentChoices[registrationId] || "UNPAID";
    const path = `/admin/clubs/${encodeURIComponent(
      clubId
    )}/tournaments/commerce/tournaments/${encodeURIComponent(
      selectedTournamentId
    )}/orders/${encodeURIComponent(registrationId)}/payment`;
    const request = {
      payment_status: paymentStatus,
      expected_order_updated_at: orderUpdatedAt,
      confirmation_text: confirmationText,
      source: "next_tournament_commerce_admin"
    };
    const operationScope = `payment:${selectedTournamentId}:${registrationId}`;
    const idempotencyKey = stableOperationKey(operationScope, request);
    setBusy(true);
    const response = await mutateAdminTournamentCommerce<Record<string, unknown>>(
      path,
      "PATCH",
      { ...request, idempotency_key: idempotencyKey },
      accessToken
    );
    setBusy(false);
    if (response.error) {
      setMessage(response.error);
      if (response.status == null || response.status >= 500) {
        return actionUncertain("Payment update needs verification", response.error, idempotencyKey, "Retry exact payment update", () => updatePayment(registrationId, orderUpdatedAt, confirmationText));
      }
      throw new Error(response.error);
    }
    clearOperationKey(operationScope, idempotencyKey);
    await loadDetail(selectedTournamentId);
    setMessage(`Payment marked ${paymentStatus.toLowerCase()}.`);
    return actionSuccess("Payment status saved", `The payment is now ${paymentStatus.toLowerCase()}.`);
  }

  async function cancelOrder(
    registrationId: string,
    orderUpdatedAt: string,
    confirmationText: string
  ): Promise<ActionCompletion> {
    const reason = (cancelReasons[registrationId] || "").trim();
    if (!reason) {
      setMessage("Enter a cancellation reason first.");
      throw new Error("Enter a cancellation reason first.");
    }
    const path = `/admin/clubs/${encodeURIComponent(
      clubId
    )}/tournaments/commerce/tournaments/${encodeURIComponent(
      selectedTournamentId
    )}/orders/${encodeURIComponent(registrationId)}/cancel`;
    const request = {
      expected_order_updated_at: orderUpdatedAt,
      reason,
      confirmation_text: confirmationText,
      source: "next_tournament_commerce_admin"
    };
    const operationScope = `cancel:${selectedTournamentId}:${registrationId}`;
    const idempotencyKey = stableOperationKey(operationScope, request);
    setBusy(true);
    const response = await mutateAdminTournamentCommerce<Record<string, unknown>>(
      path,
      "POST",
      { ...request, idempotency_key: idempotencyKey },
      accessToken
    );
    setBusy(false);
    if (response.error) {
      setMessage(response.error);
      if (response.status == null || response.status >= 500) {
        return actionUncertain("Order cancellation needs verification", response.error, idempotencyKey, "Retry exact cancellation", () => cancelOrder(registrationId, orderUpdatedAt, confirmationText));
      }
      throw new Error(response.error);
    }
    clearOperationKey(operationScope, idempotencyKey);
    await loadDetail(selectedTournamentId);
    setMessage("Extras order cancelled and inventory released.");
    return actionSuccess("Extras order cancelled", "The order was cancelled, active inventory was released, and its audit history was retained.");
  }

  async function updateFulfillment(row: Record<string, unknown>, confirmationText: string): Promise<ActionCompletion> {
    const id = stringValue(row, "id");
    const currentStatus = stringValue(row, "status", "PENDING").toUpperCase();
    const nextStatus = (fulfillmentStatuses[id] || currentStatus).toUpperCase();
    const notes = (fulfillmentNotes[id] || "").trim();
    if (
      currentStatus === "FULFILLED" &&
      nextStatus !== "FULFILLED" &&
      notes.length < 8
    ) {
      setMessage(
        "Add a correction note of at least 8 characters before changing a fulfilled item."
      );
      throw new Error("Add a correction note of at least 8 characters before changing a fulfilled item.");
    }
    const path = `/admin/clubs/${encodeURIComponent(
      clubId
    )}/tournaments/commerce/tournaments/${encodeURIComponent(
      selectedTournamentId
    )}/fulfillment/${encodeURIComponent(id)}`;
    const request = {
      status: nextStatus,
      notes,
      expected_updated_at: stringValue(row, "updated_at"),
      confirmation_text: confirmationText,
      source: "next_tournament_commerce_admin"
    };
    const operationScope = `fulfillment:${selectedTournamentId}:${id}`;
    const idempotencyKey = stableOperationKey(operationScope, request);
    setBusy(true);
    const response = await mutateAdminTournamentCommerce<Record<string, unknown>>(
      path,
      "PATCH",
      { ...request, idempotency_key: idempotencyKey },
      accessToken
    );
    setBusy(false);
    if (response.error) {
      setMessage(response.error);
      if (response.status == null || response.status >= 500) {
        return actionUncertain("Fulfillment update needs verification", response.error, idempotencyKey, "Retry exact fulfillment update", () => updateFulfillment(row, confirmationText));
      }
      throw new Error(response.error);
    }
    clearOperationKey(operationScope, idempotencyKey);
    await loadDetail(selectedTournamentId);
    setMessage("Fulfillment status saved.");
    return actionSuccess("Fulfillment status saved", `The fulfillment item is now ${nextStatus.toLowerCase()}.`);
  }

  async function downloadFulfillment() {
    const url = adminTournamentCommerceExportUrl(
      clubId,
      selectedTournamentId
    );
    if (!url || !accessToken) {
      setMessage("The authenticated export is unavailable.");
      return;
    }
    setBusy(true);
    try {
      const response = await fetch(url, {
        headers: { Authorization: `Bearer ${accessToken}` }
      });
      if (!response.ok) {
        throw new Error(`Export failed (${response.status}).`);
      }
      const blob = await response.blob();
      const href = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = `tournament-fulfillment-${selectedTournamentId}.csv`;
      anchor.click();
      URL.revokeObjectURL(href);
      setMessage("Fulfillment CSV downloaded.");
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "Unable to download export."
      );
    } finally {
      setBusy(false);
    }
  }

  async function inspectOperation(operationId: string) {
    setBusy(true);
    const response = await getAdminTournamentCommerceOperation(
      clubId,
      selectedTournamentId,
      operationId,
      accessToken
    );
    setBusy(false);
    if (response.error || !response.data) {
      setMessage(response.error || "Unable to inspect operation.");
      return;
    }
    setOperationEvidence(response.data);
    setMessage("Loaded authoritative recovery evidence.");
  }

  useAuthenticatedAutoLoad(`${accessToken}\u0000${tournamentId}`, loadWorkspace);

  if (!accessToken) {
    return (
      <article style={{ ...cardStyle, background: "#fffbeb" }}>
        <h2 style={{ marginTop: 0 }}>Admin sign-in required</h2>
        <p style={{ color: "#92400e" }}>
          {sessionLoading
            ? "Checking the saved admin session…"
            : sessionMessage || "Sign in to manage tournament extras."}
        </p>
        {!sessionLoading ? <Link href="/admin/login">Open admin login</Link> : null}
      </article>
    );
  }

  return (
    <section data-commerce-form aria-label={`${tournamentName} payments, extras, and fulfillment`} style={{ display: "grid", gap: "1rem" }}>
      <style>{`[data-commerce-form] label { min-width: 0; } [data-commerce-form] input, [data-commerce-form] select, [data-commerce-form] textarea, [data-commerce-form] button { box-sizing: border-box; max-width: 100%; } [data-commerce-form] summary { cursor: pointer; }`}</style>


      {detail && draft ? (
        <>
          <nav
            aria-label="Tournament extras sections"
            style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}
          >
            {(
              [
                ["catalog", "Catalog & offers"],
                ["orders", "Offline payments"],
                ["fulfillment", "Pickup & fulfillment"],
                ["recovery", "Recovery & audit"]
              ] as Array<[Tab, string]>
            ).map(([value, label]) => (
              <button
                key={value}
                type="button"
                onClick={() => setTab(value)}
                aria-pressed={tab === value}
                style={tab === value ? buttonStyle : ghostButtonStyle}
              >
                {label}
              </button>
            ))}
          </nav>

          {tab === "catalog" ? (
            <section style={{ display: "grid", gap: "1rem" }}>
              <article style={cardStyle}>
                <h2 style={{ marginTop: 0 }}>Extras</h2>
                <p style={{ color: "#475569" }}>
                  Use options for sizes, room types, meal choices, or pack
                  variations. Inventory can be limited at the extra or option
                  level.
                </p>
                <button type="button" onClick={addItem} style={buttonStyle}>
                  Add extra
                </button>
              </article>

              {draft.items.map((item) => (
                <details key={item.id} open={!savedItemIds.has(item.id)} style={cardStyle}>
                  <summary style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
                    <span><strong>{item.name || "Untitled extra"}</strong><br /><small>{formatCommerceMoney(item.base_price_minor)} · {item.status} · {item.kind.toLowerCase()}</small></span>
                    <span style={{ fontWeight: 800 }}>{savedItemIds.has(item.id) ? "Edit" : "New extra"}</span>
                  </summary>
                  <div style={{ marginTop: "1rem" }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      gap: "0.75rem",
                      alignItems: "start",
                      flexWrap: "wrap"
                    }}
                  >
                    <h3 style={{ marginTop: 0 }}>{item.name || "Untitled extra"}</h3>
                    <button
                      type="button"
                      onClick={() => removeItem(item.id)}
                      style={dangerButtonStyle}
                    >
                      Remove extra
                    </button>
                  </div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns:
                        "repeat(auto-fit, minmax(min(100%, 220px), 1fr))",
                      gap: "0.75rem"
                    }}
                  >
                    <label>
                      Name
                      <br />
                      <input
                        value={item.name}
                        onChange={(event) =>
                          updateDraft((next) => {
                            next.items.find((row) => row.id === item.id)!.name =
                              event.target.value;
                          })
                        }
                        style={inputStyle}
                      />
                    </label>
                    <label>
                      Type
                      <br />
                      <select
                        value={item.kind}
                        onChange={(event) =>
                          updateDraft((next) => {
                            next.items.find((row) => row.id === item.id)!.kind =
                              event.target.value as TournamentCommerceItem["kind"];
                          })
                        }
                        style={inputStyle}
                      >
                        <option value="MERCHANDISE">Merchandise</option>
                        <option value="LODGING">Accommodation</option>
                        <option value="MEAL">Meal</option>
                        <option value="DRINK_PACK">Drink pack</option>
                        <option value="OTHER">Other</option>
                      </select>
                    </label>
                    <label>
                      Base price (USD)
                      <br />
                      <input
                        key={`${item.id}-${detail?.catalog.catalog_revision || 0}-base-price`}
                        type="text"
                        inputMode="decimal"
                        defaultValue={(item.base_price_minor / 100).toFixed(2)}
                        onBlur={(event) =>
                          updateDraft((next) => {
                            next.items.find(
                              (row) => row.id === item.id
                            )!.base_price_minor = moneyMinor(event.target.value);
                          })
                        }
                        style={inputStyle}
                      />
                    </label>
                    <label>
                      Total inventory
                      <br />
                      <input
                        type="number"
                        min="0"
                        placeholder="Unlimited"
                        value={item.inventory_limit ?? ""}
                        onChange={(event) =>
                          updateDraft((next) => {
                            next.items.find(
                              (row) => row.id === item.id
                            )!.inventory_limit = nullableWholeNumber(
                              event.target.value
                            );
                          })
                        }
                        style={inputStyle}
                      />
                    </label>
                    <label>
                      Maximum per registration
                      <br />
                      <input
                        type="number"
                        min="1"
                        max="20"
                        value={item.max_per_registration}
                        onChange={(event) =>
                          updateDraft((next) => {
                            next.items.find(
                              (row) => row.id === item.id
                            )!.max_per_registration = Math.max(
                              1,
                              Number(event.target.value) || 1
                            );
                          })
                        }
                        style={inputStyle}
                      />
                    </label>
                    <label>
                      Display order
                      <br />
                      <input
                        type="number"
                        min="0"
                        value={item.sort_order}
                        onChange={(event) =>
                          updateDraft((next) => {
                            next.items.find(
                              (row) => row.id === item.id
                            )!.sort_order = Math.max(
                              0,
                              Number(event.target.value) || 0
                            );
                          })
                        }
                        style={inputStyle}
                      />
                    </label>
                    <label>
                      Status
                      <br />
                      <select
                        value={item.status}
                        onChange={(event) =>
                          updateDraft((next) => {
                            next.items.find(
                              (row) => row.id === item.id
                            )!.status = event.target
                              .value as TournamentCommerceCatalogStatus;
                          })
                        }
                        style={inputStyle}
                      >
                        {statusOptions.map((value) => (
                          <option key={value}>{value}</option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Available from
                      <br />
                      <input
                        type="datetime-local"
                        value={inputDateTime(item.available_from)}
                        onChange={(event) =>
                          updateDraft((next) => {
                            next.items.find(
                              (row) => row.id === item.id
                            )!.available_from = outputDateTime(
                              event.target.value
                            );
                          })
                        }
                        style={inputStyle}
                      />
                    </label>
                    <label>
                      Available through
                      <br />
                      <input
                        type="datetime-local"
                        value={inputDateTime(item.available_until)}
                        onChange={(event) =>
                          updateDraft((next) => {
                            next.items.find(
                              (row) => row.id === item.id
                            )!.available_until = outputDateTime(
                              event.target.value
                            );
                          })
                        }
                        style={inputStyle}
                      />
                    </label>
                  </div>
                  <label style={{ display: "block", marginTop: "0.75rem" }}>
                    Description
                    <br />
                    <textarea
                      rows={2}
                      value={item.description}
                      onChange={(event) =>
                        updateDraft((next) => {
                          next.items.find(
                            (row) => row.id === item.id
                          )!.description = event.target.value;
                        })
                      }
                      style={inputStyle}
                    />
                  </label>
                  <label
                    style={{
                      display: "flex",
                      gap: "0.5rem",
                      marginTop: "0.75rem"
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={item.requires_fulfillment}
                      onChange={(event) =>
                        updateDraft((next) => {
                          next.items.find(
                            (row) => row.id === item.id
                          )!.requires_fulfillment = event.target.checked;
                        })
                      }
                    />
                    Track pickup or fulfillment for this extra
                  </label>
                  {item.requires_fulfillment ? (
                    <label style={{ display: "block", marginTop: "0.75rem" }}>
                      Pickup instructions
                      <br />
                      <input
                        value={item.fulfillment_instructions || ""}
                        onChange={(event) =>
                          updateDraft((next) => {
                            next.items.find(
                              (row) => row.id === item.id
                            )!.fulfillment_instructions = event.target.value;
                          })
                        }
                        style={inputStyle}
                      />
                    </label>
                  ) : null}
                  <h4 style={{ marginBottom: "0.25rem" }}>Options (variants)</h4>
                  <p style={{ color: "#64748b", margin: "0 0 0.65rem" }}>
                    Use options when one extra has selectable variants, such as
                    T-shirt sizes, lodging room types, meal choices, or package
                    levels. Each option can have its own SKU, price change,
                    inventory, and status. Leave this section empty when the
                    extra has no choices.
                  </p>
                  <div style={{ display: "grid", gap: "0.65rem" }}>
                    {(variantsByItem.get(item.id) || []).map((variant) => (
                      <div
                        key={variant.id}
                        style={{
                          display: "grid",
                          gridTemplateColumns:
                            "repeat(auto-fit, minmax(min(100%, 150px), 1fr))",
                          gap: "0.5rem",
                          alignItems: "end",
                          border: "1px solid #e2e8f0",
                          borderRadius: "10px",
                          padding: "0.65rem"
                        }}
                      >
                        <label>
                          Option name
                          <br />
                          <input
                            value={variant.name}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.variants.find(
                                  (row) => row.id === variant.id
                                )!.name = event.target.value;
                              })
                            }
                            style={inputStyle}
                          />
                        </label>
                        <label>
                          SKU
                          <br />
                          <input
                            value={variant.sku || ""}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.variants.find(
                                  (row) => row.id === variant.id
                                )!.sku = event.target.value;
                              })
                            }
                            style={inputStyle}
                          />
                        </label>
                        <label>
                          Price change (USD)
                          <br />
                          <input
                            key={`${variant.id}-${detail?.catalog.catalog_revision || 0}-price-delta`}
                            type="text"
                            inputMode="decimal"
                            defaultValue={(variant.price_delta_minor / 100).toFixed(2)}
                            onBlur={(event) =>
                              updateDraft((next) => {
                                next.variants.find(
                                  (row) => row.id === variant.id
                                )!.price_delta_minor = moneyMinor(
                                  event.target.value
                                );
                              })
                            }
                            style={inputStyle}
                          />
                        </label>
                        <label>
                          Option inventory
                          <br />
                          <input
                            type="number"
                            min="0"
                            placeholder="Use total inventory"
                            value={variant.inventory_limit ?? ""}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.variants.find(
                                  (row) => row.id === variant.id
                                )!.inventory_limit = nullableWholeNumber(
                                  event.target.value
                                );
                              })
                            }
                            style={inputStyle}
                          />
                        </label>
                        <label>
                          Status
                          <br />
                          <select
                            value={variant.status}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.variants.find(
                                  (row) => row.id === variant.id
                                )!.status = event.target
                                  .value as TournamentCommerceCatalogStatus;
                              })
                            }
                            style={inputStyle}
                          >
                            {statusOptions.map((value) => (
                              <option key={value}>{value}</option>
                            ))}
                          </select>
                        </label>
                        <button
                          type="button"
                          onClick={() => removeVariant(variant.id)}
                          style={dangerButtonStyle}
                        >
                          Remove option
                        </button>
                      </div>
                    ))}
                  </div>
                  <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginTop: "0.75rem" }}>
                    <button
                      type="button"
                      onClick={() => addVariant(item.id)}
                      style={ghostButtonStyle}
                    >
                      Add option
                    </button>
                    <button
                      type="button"
                      onClick={() => addVariantPreset(item.id, ["XS", "S", "M", "L", "XL", "2XL"])}
                      style={ghostButtonStyle}
                    >
                      Add T-shirt sizes
                    </button>
                    <button
                      type="button"
                      onClick={() => addVariantPreset(item.id, ["Chicken", "Vegetarian", "Vegan"])}
                      style={ghostButtonStyle}
                    >
                      Add meal choices
                    </button>
                    <button
                      type="button"
                      onClick={() => addVariantPreset(item.id, ["Standard", "Poolside", "Premium"])}
                      style={ghostButtonStyle}
                    >
                      Add room choices
                    </button>
                  </div>
                  <small style={{ color: "#64748b" }}>Presets add only missing options and never overwrite existing rows.</small>
                </div>
                </details>
              ))}

              <article style={cardStyle}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    gap: "0.75rem",
                    flexWrap: "wrap"
                  }}
                >
                  <div>
                    <h2 style={{ marginTop: 0 }}>Bundles</h2>
                    <p style={{ color: "#475569" }}>
                      Combine required events and extras at one lower price.
                      Matching bundles apply automatically.
                    </p>
                  </div>
                  <button type="button" onClick={addBundle} style={buttonStyle}>
                    Add bundle
                  </button>
                </div>
                <div style={{ display: "grid", gap: "0.75rem" }}>
                  {draft.bundles.map((bundle) => (
                    <details key={bundle.id} open={!savedBundleIds.has(bundle.id)} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem" }}>
                      <summary style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
                        <span><strong>{bundle.name || "Untitled bundle"}</strong><br /><small>{formatCommerceMoney(bundle.price_minor)} · {bundle.status} · {(componentsByBundle.get(bundle.id) || []).filter((component) => component.component_type !== "EVENT_CHOICE").length} fixed parts · {(componentsByBundle.get(bundle.id) || []).filter((component) => component.component_type === "EVENT_CHOICE").length} eligible registration divisions</small></span>
                        <span style={{ fontWeight: 800 }}>{savedBundleIds.has(bundle.id) ? "Edit" : "New bundle"}</span>
                      </summary>
                      <section style={{ marginTop: "1rem" }}>
                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns:
                            "repeat(auto-fit, minmax(min(100%, 180px), 1fr))",
                          gap: "0.5rem"
                        }}
                      >
                        <label>
                          Name
                          <br />
                          <input
                            value={bundle.name}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.bundles.find(
                                  (row) => row.id === bundle.id
                                )!.name = event.target.value;
                              })
                            }
                            style={inputStyle}
                          />
                        </label>
                        <label>
                          Bundle price (USD)
                          <br />
                          <input
                            key={`${bundle.id}-${detail?.catalog.catalog_revision || 0}-bundle-price`}
                            type="text"
                            inputMode="decimal"
                            defaultValue={(bundle.price_minor / 100).toFixed(2)}
                            onBlur={(event) =>
                              updateDraft((next) => {
                                next.bundles.find(
                                  (row) => row.id === bundle.id
                                )!.price_minor = moneyMinor(event.target.value);
                              })
                            }
                            style={inputStyle}
                          />
                        </label>
                        <label>
                          Maximum per registration
                          <br />
                          <input
                            type="number"
                            min="1"
                            max="20"
                            value={bundle.max_per_registration}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.bundles.find(
                                  (row) => row.id === bundle.id
                                )!.max_per_registration = Math.max(
                                  1,
                                  Number(event.target.value) || 1
                                );
                              })
                            }
                            style={inputStyle}
                          />
                        </label>
                        <label>
                          Status
                          <br />
                          <select
                            value={bundle.status}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.bundles.find(
                                  (row) => row.id === bundle.id
                                )!.status = event.target
                                  .value as TournamentCommerceCatalogStatus;
                              })
                            }
                            style={inputStyle}
                          >
                            {statusOptions.map((value) => (
                              <option key={value}>{value}</option>
                            ))}
                          </select>
                        </label>
                        <label>
                          Available from
                          <br />
                          <input
                            type="datetime-local"
                            value={inputDateTime(bundle.available_from)}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.bundles.find(
                                  (row) => row.id === bundle.id
                                )!.available_from = outputDateTime(
                                  event.target.value
                                );
                              })
                            }
                            style={inputStyle}
                          />
                        </label>
                        <label>
                          Available through
                          <br />
                          <input
                            type="datetime-local"
                            value={inputDateTime(bundle.available_until)}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.bundles.find(
                                  (row) => row.id === bundle.id
                                )!.available_until = outputDateTime(
                                  event.target.value
                                );
                              })
                            }
                            style={inputStyle}
                          />
                        </label>
                        <label>
                          Display order
                          <br />
                          <input
                            type="number"
                            min="0"
                            value={bundle.sort_order}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.bundles.find(
                                  (row) => row.id === bundle.id
                                )!.sort_order = Math.max(
                                  0,
                                  Number(event.target.value) || 0
                                );
                              })
                            }
                            style={inputStyle}
                          />
                        </label>
                      </div>
                      <label style={{ display: "block", marginTop: "0.5rem" }}>
                        Description
                        <br />
                        <input
                          value={bundle.description}
                          onChange={(event) =>
                            updateDraft((next) => {
                              next.bundles.find(
                                (row) => row.id === bundle.id
                              )!.description = event.target.value;
                            })
                          }
                          style={inputStyle}
                        />
                      </label>
                      <h4>Required parts</h4>
                      <ul>
                        {(componentsByBundle.get(bundle.id) || [])
                          .filter((component) => component.component_type !== "EVENT_CHOICE")
                          .map((component) => (
                            <li key={component.id}>
                              <label>
                                Quantity{" "}
                                <input
                                  aria-label={`${bundle.name} component quantity`}
                                  type="number"
                                  min="1"
                                  max="20"
                                  value={component.quantity || 1}
                                  onChange={(event) =>
                                    updateDraft((next) => {
                                      next.bundle_components.find(
                                        (row) => row.id === component.id
                                      )!.quantity = Math.max(
                                        1,
                                        Number(event.target.value) || 1
                                      );
                                    })
                                  }
                                  style={{ width: "4rem" }}
                                />
                              </label>{" "}
                              ×{" "}
                              {component.component_type === "EVENT_OPTION"
                                ? eventLabels.get(
                                    String(component.event_option_id || "")
                                  )
                                : variantLabels.get(
                                    String(component.variant_id || "")
                                  )}{" "}
                              <button
                                type="button"
                                onClick={() =>
                                  updateDraft((next) => {
                                    next.bundle_components =
                                      next.bundle_components.filter(
                                        (row) => row.id !== component.id
                                      );
                                  })
                                }
                                style={dangerButtonStyle}
                              >
                                Remove
                              </button>
                            </li>
                          )
                        )}
                      </ul>
                      <section
                        style={{
                          marginTop: "1rem",
                          padding: "0.85rem",
                          border: "1px solid #bfdbfe",
                          borderRadius: "12px",
                          background: "#eff6ff"
                        }}
                      >
                        <h4 style={{ margin: 0 }}>Flexible event choice</h4>
                        <p style={{ color: "#475569", margin: "0.35rem 0 0.75rem" }}>
                          Select the eligible registration divisions—the event/division entries a player can choose during registration—then choose how many of them are required. For example, “any 1 of these divisions” or “any 2 of these divisions.” Fixed extras can still be added below.
                        </p>
                        {(() => {
                          const choiceComponents = (componentsByBundle.get(bundle.id) || []).filter(
                            (component) => component.component_type === "EVENT_CHOICE"
                          );
                          const selectedIds = new Set(
                            choiceComponents.map((component) => String(component.event_option_id || ""))
                          );
                          const requiredCount = Math.max(
                            1,
                            Math.min(
                              Number(choiceComponents[0]?.quantity || 1),
                              choiceComponents.length || 1
                            )
                          );
                          return (
                            <>
                              <label style={{ display: "block", maxWidth: "18rem", marginBottom: "0.75rem" }}>
                                Divisions required from eligible list
                                <br />
                                <input
                                  type="number"
                                  min="1"
                                  max={Math.max(1, choiceComponents.length)}
                                  value={requiredCount}
                                  disabled={!choiceComponents.length}
                                  onChange={(event) =>
                                    setBundleEventChoiceCount(bundle.id, Number(event.target.value))
                                  }
                                  style={inputStyle}
                                />
                              </label>
                              <div style={{ display: "grid", gap: "0.45rem" }}>
                                {draft.event_options.map((event) => {
                                  const eventId = stringValue(event, "id");
                                  return (
                                    <label key={`${bundle.id}-choice-${eventId}`} style={{ display: "flex", gap: "0.55rem", alignItems: "flex-start" }}>
                                      <input
                                        type="checkbox"
                                        checked={selectedIds.has(eventId)}
                                        onChange={(change) =>
                                          setBundleEventChoice(bundle.id, eventId, change.target.checked)
                                        }
                                      />
                                      <span>{eventLabels.get(eventId) || "Tournament event"}</span>
                                    </label>
                                  );
                                })}
                              </div>
                              {choiceComponents.length ? (
                                <p style={{ color: "#1d4ed8", marginBottom: 0, fontWeight: 700 }}>
                                  Bundle applies when the registration includes any {requiredCount} of {choiceComponents.length} eligible registration division{choiceComponents.length === 1 ? "" : "s"}.
                                </p>
                              ) : (
                                <p style={{ color: "#64748b", marginBottom: 0 }}>
                                  No flexible registration-division pool configured. The bundle may still use exact required parts below.
                                </p>
                              )}
                            </>
                          );
                        })()}
                      </section>
                      <label style={{ display: "block", marginTop: "1rem" }}>
                        Add an exact required event or extra
                        <br />
                        <select
                          key={`${bundle.id}-${draft.variants.length}-${draft.event_options.length}`}
                          defaultValue=""
                          onChange={(event) => {
                            addBundleComponent(bundle.id, event.target.value);
                            event.target.value = "";
                          }}
                          style={inputStyle}
                        >
                          <option value="">Choose…</option>
                          <optgroup label="Events">
                            {draft.event_options.map((event) => (
                              <option
                                key={stringValue(event, "id")}
                                value={`EVENT_OPTION:${stringValue(event, "id")}`}
                              >
                                {stringValue(
                                  event,
                                  "label",
                                  stringValue(event, "division_name", "Event")
                                )}
                              </option>
                            ))}
                          </optgroup>
                          <optgroup label="Extras">
                            {draft.variants.map((variant) => (
                              <option
                                key={variant.id}
                                value={`ITEM_VARIANT:${variant.id}`}
                              >
                                {variantLabels.get(variant.id)}
                              </option>
                            ))}
                          </optgroup>
                        </select>
                      </label>
                      <p>
                        <button
                          type="button"
                          onClick={() => removeBundle(bundle.id)}
                          style={dangerButtonStyle}
                        >
                          Remove bundle
                        </button>
                      </p>
                    </section>
                    </details>
                  ))}
                </div>
              </article>

              <article style={cardStyle}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    gap: "0.75rem",
                    flexWrap: "wrap"
                  }}
                >
                  <div>
                    <h2 style={{ marginTop: 0 }}>Free offers and giveaways</h2>
                    <p style={{ color: "#475569" }}>
                      Make an extra free during a date window, for the first
                      registrants, or for the first claims.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={addPromotion}
                    style={buttonStyle}
                  >
                    Add giveaway
                  </button>
                </div>
                <div style={{ display: "grid", gap: "0.75rem" }}>
                  {draft.promotions.map((promotion) => (
                    <section
                      key={promotion.id}
                      style={{
                        border: "1px solid #e2e8f0",
                        borderRadius: "12px",
                        padding: "0.75rem"
                      }}
                    >
                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns:
                            "repeat(auto-fit, minmax(min(100%, 180px), 1fr))",
                          gap: "0.5rem"
                        }}
                      >
                        <label>
                          Name
                          <br />
                          <input
                            value={promotion.name}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.promotions.find(
                                  (row) => row.id === promotion.id
                                )!.name = event.target.value;
                              })
                            }
                            style={inputStyle}
                          />
                        </label>
                        <label>
                          Offer rule
                          <br />
                          <select
                            value={promotion.promotion_type}
                            onChange={(event) =>
                              updateDraft((next) => {
                                const row = next.promotions.find(
                                  (item) => item.id === promotion.id
                                )!;
                                row.promotion_type = event.target
                                  .value as TournamentCommercePromotion["promotion_type"];
                                if (row.promotion_type === "DATE_WINDOW_FREE") {
                                  row.giveaway_limit = null;
                                } else if (!row.giveaway_limit) {
                                  row.giveaway_limit = 10;
                                }
                              })
                            }
                            style={inputStyle}
                          >
                            <option value="DATE_WINDOW_FREE">
                              Free within dates
                            </option>
                            <option value="FIRST_N_REGISTRANTS">
                              First registrants
                            </option>
                            <option value="FIRST_N_CLAIMS">First claims</option>
                          </select>
                        </label>
                        <label>
                          Applies to
                          <br />
                          <select
                            value={targetValue(promotion)}
                            onChange={(event) =>
                              setPromotionTarget(
                                promotion.id,
                                event.target.value
                              )
                            }
                            style={inputStyle}
                          >
                            <optgroup label="Extras">
                              {draft.items.map((item) => (
                                <option
                                  key={`item-${item.id}`}
                                  value={`ITEM:${item.id}`}
                                >
                                  All {item.name} options
                                </option>
                              ))}
                              {draft.variants.map((variant) => (
                                <option
                                  key={`variant-${variant.id}`}
                                  value={`ITEM_VARIANT:${variant.id}`}
                                >
                                  {variantLabels.get(variant.id)}
                                </option>
                              ))}
                            </optgroup>
                            <optgroup label="Bundles">
                              {draft.bundles.map((bundle) => (
                                <option
                                  key={`bundle-${bundle.id}`}
                                  value={`BUNDLE:${bundle.id}`}
                                >
                                  {bundle.name}
                                </option>
                              ))}
                            </optgroup>
                          </select>
                        </label>
                        <label>
                          Free per registration
                          <br />
                          <input
                            type="number"
                            min="1"
                            max="20"
                            value={promotion.per_registration_limit}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.promotions.find(
                                  (row) => row.id === promotion.id
                                )!.per_registration_limit = Math.max(
                                  1,
                                  Number(event.target.value) || 1
                                );
                              })
                            }
                            style={inputStyle}
                          />
                        </label>
                        {promotion.promotion_type !== "DATE_WINDOW_FREE" ? (
                          <label>
                            Giveaway limit
                            <br />
                            <input
                              type="number"
                              min="1"
                              value={promotion.giveaway_limit || 1}
                              onChange={(event) =>
                                updateDraft((next) => {
                                  next.promotions.find(
                                    (row) => row.id === promotion.id
                                  )!.giveaway_limit = Math.max(
                                    1,
                                    Number(event.target.value) || 1
                                  );
                                })
                              }
                              style={inputStyle}
                            />
                          </label>
                        ) : null}
                        <label>
                          Starts
                          <br />
                          <input
                            type="datetime-local"
                            value={inputDateTime(promotion.starts_at)}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.promotions.find(
                                  (row) => row.id === promotion.id
                                )!.starts_at = outputDateTime(
                                  event.target.value
                                );
                              })
                            }
                            style={inputStyle}
                          />
                        </label>
                        <label>
                          Ends
                          <br />
                          <input
                            type="datetime-local"
                            value={inputDateTime(promotion.ends_at)}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.promotions.find(
                                  (row) => row.id === promotion.id
                                )!.ends_at = outputDateTime(event.target.value);
                              })
                            }
                            style={inputStyle}
                          />
                        </label>
                        <label>
                          Status
                          <br />
                          <select
                            value={promotion.status}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.promotions.find(
                                  (row) => row.id === promotion.id
                                )!.status = event.target
                                  .value as TournamentCommerceCatalogStatus;
                              })
                            }
                            style={inputStyle}
                          >
                            {statusOptions.map((value) => (
                              <option key={value}>{value}</option>
                            ))}
                          </select>
                        </label>
                        <label>
                          Offer priority
                          <br />
                          <input
                            type="number"
                            min="0"
                            value={promotion.priority ?? 100}
                            onChange={(event) =>
                              updateDraft((next) => {
                                next.promotions.find(
                                  (row) => row.id === promotion.id
                                )!.priority = Math.max(
                                  0,
                                  Number(event.target.value) || 0
                                );
                              })
                            }
                            style={inputStyle}
                          />
                        </label>
                      </div>
                      <p>
                        <button
                          type="button"
                          onClick={() =>
                            updateDraft((next) => {
                              next.promotions = next.promotions.filter(
                                (row) => row.id !== promotion.id
                              );
                            })
                          }
                          style={dangerButtonStyle}
                        >
                          Remove giveaway
                        </button>
                      </p>
                    </section>
                  ))}
                </div>
              </article>

              <article style={cardStyle}>
                <h2 style={{ marginTop: 0 }}>Review and save</h2>
                <p style={{ color: "#475569" }}>
                  Review validates prices, dates, inventory, bundle parts, and
                  giveaway targets without changing the catalog.
                </p>
                <div
                  style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}
                >
                  <button
                    type="button"
                    onClick={previewCatalog}
                    disabled={busy || !adminWriteReady}
                    style={ghostButtonStyle}
                  >
                    {busy ? "Reviewing…" : "Review catalog"}
                  </button>
                  <ConfirmAction
                    triggerLabel="Save reviewed catalog"
                    title="Save this tournament extras catalog?"
                    description="This creates a new forward-only catalog revision. Existing order revisions remain immutable."
                    confirmLabel="Yes, save catalog"
                    confirmationText="SAVE"
                    disabled={
                      busy || !catalogReviewed || !adminWriteReady || !accessToken
                    }
                    busy={busy}
                    onConfirm={saveCatalog}
                  />
                </div>
              </article>
            </section>
          ) : null}

          {tab === "orders" ? (
            <article style={cardStyle}>
              <h2 style={{ marginTop: 0 }}>Offline payment tracking</h2>
              <p style={{ color: "#475569" }}>
                No online card payment is taken. Staff record payment status
                here against the immutable order total.
              </p>
              <div style={{ overflowX: "auto" }}>
                <table
                  style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    minWidth: "1080px"
                  }}
                >
                  <thead>
                    <tr>
                      <th align="left">Registrant</th>
                      <th align="left">Order</th>
                      <th align="left">Registration and extras</th>
                      <th align="left">Total</th>
                      <th align="left">Payment</th>
                      <th align="left">Cancellation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.orders.map((order) => {
                      const registration = recordValue(order, "registration");
                      const registrationId = stringValue(
                        order,
                        "registration_id"
                      );
                      const updatedAt = stringValue(order, "updated_at");
                      const statusValue = stringValue(order, "status");
                      const lines = recordListValue(order, "lines");
                      return (
                        <tr key={stringValue(order, "id")}>
                          <td>
                            <strong>
                              {stringValue(
                                registration,
                                "display_name",
                                "Registrant"
                              )}
                            </strong>
                            <br />
                            {stringValue(registration, "email")}
                          </td>
                          <td>{statusValue}</td>
                          <td>
                            {lines.length ? (
                              <ul
                                style={{
                                  margin: 0,
                                  paddingLeft: "1.1rem",
                                  minWidth: "220px"
                                }}
                              >
                                {lines.map((line) => {
                                  const option = stringValue(
                                    line,
                                    "option_snapshot"
                                  );
                                  const quantity = numberValue(
                                    line,
                                    "quantity",
                                    1
                                  );
                                  return (
                                    <li key={stringValue(line, "id")}>
                                      {stringValue(
                                        line,
                                        "label_snapshot",
                                        "Selection"
                                      )}
                                      {option ? ` — ${option}` : ""}
                                      {quantity > 1 ? ` × ${quantity}` : ""}
                                    </li>
                                  );
                                })}
                              </ul>
                            ) : (
                              "No active selections"
                            )}
                          </td>
                          <td>
                            {formatCommerceMoney(
                              numberValue(order, "total_minor")
                            )}
                          </td>
                          <td>
                            <select
                              aria-label={`${stringValue(
                                registration,
                                "display_name",
                                "Registrant"
                              )} payment status`}
                              value={
                                paymentChoices[registrationId] ||
                                stringValue(order, "payment_status", "UNPAID")
                              }
                              onChange={(event) =>
                                setPaymentChoices((current) => ({
                                  ...current,
                                  [registrationId]: event.target.value
                                }))
                              }
                              style={inputStyle}
                            >
                              <option value="UNPAID">Unpaid</option>
                              <option value="PAID">Paid</option>
                              <option value="WAIVED">Waived</option>
                              <option value="REFUNDED">Refunded</option>
                            </select>
                            <ConfirmAction
                              triggerLabel="Save payment"
                              title="Save this payment status?"
                              description="The payment status change is version-checked and added to the tournament commerce audit trail."
                              confirmLabel="Yes, save payment status"
                              confirmationText="SAVE PAYMENT STATUS"
                              disabled={
                                busy || !adminWriteReady || statusValue === "CANCELLED"
                              }
                              busy={busy}
                              onConfirm={(confirmationText) => updatePayment(registrationId, updatedAt, confirmationText)}
                            />
                          </td>
                          <td>
                            <input
                              aria-label={`${stringValue(
                                registration,
                                "display_name",
                                "Registrant"
                              )} cancellation reason`}
                              placeholder="Cancellation reason"
                              value={cancelReasons[registrationId] || ""}
                              onChange={(event) =>
                                setCancelReasons((current) => ({
                                  ...current,
                                  [registrationId]: event.target.value
                                }))
                              }
                              disabled={statusValue === "CANCELLED"}
                              style={inputStyle}
                            />
                            <div style={{ marginTop: "0.35rem" }}>
                              <ConfirmAction
                                triggerLabel="Cancel extras order"
                                title="Cancel this extras order?"
                                description="Cancellation releases active inventory and retains all prior order revisions and audit evidence."
                                confirmLabel="Yes, cancel order"
                                confirmationText="CANCEL"
                                tone="danger"
                                disabled={
                                  busy ||
                                  !adminWriteReady ||
                                  statusValue === "CANCELLED" ||
                                  !(cancelReasons[registrationId] || "").trim()
                                }
                                busy={busy}
                                onConfirm={(confirmationText) =>
                                  cancelOrder(
                                    registrationId,
                                    updatedAt,
                                    confirmationText
                                  )
                                }
                              />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {!detail.orders.length ? <p>No extras orders yet.</p> : null}
            </article>
          ) : null}

          {tab === "fulfillment" ? (
            <article style={cardStyle}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  gap: "0.75rem",
                  flexWrap: "wrap"
                }}
              >
                <div>
                  <h2 style={{ marginTop: 0 }}>Pickup and fulfillment</h2>
                  <p style={{ color: "#475569" }}>
                    Track each shirt, room, meal, pack, or bundled fulfillment
                    component separately.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={downloadFulfillment}
                  disabled={busy}
                  style={ghostButtonStyle}
                >
                  Download CSV
                </button>
              </div>
              <div style={{ overflowX: "auto" }}>
                <table
                  style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    minWidth: "1080px"
                  }}
                >
                  <thead>
                    <tr>
                      <th align="left">Registrant</th>
                      <th align="left">Item</th>
                      <th align="left">Quantity</th>
                      <th align="left">Status</th>
                      <th align="left">Notes</th>
                      <th align="left">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.fulfillment.map((row) => {
                      const id = stringValue(row, "id");
                      const registration = recordValue(row, "registration");
                      const currentStatus = stringValue(
                        row,
                        "status",
                        "PENDING"
                      );
                      return (
                        <tr key={id}>
                          <td>
                            {stringValue(
                              registration,
                              "display_name",
                              "Registrant"
                            )}
                            <br />
                            {stringValue(registration, "email")}
                          </td>
                          <td>
                            <strong>
                              {stringValue(row, "label_snapshot", "Extra")}
                            </strong>
                            <br />
                            {stringValue(row, "option_snapshot")}
                          </td>
                          <td>{numberValue(row, "quantity", 1)}</td>
                          <td>
                            <select
                              value={
                                fulfillmentStatuses[id] || currentStatus
                              }
                              onChange={(event) =>
                                setFulfillmentStatuses((current) => ({
                                  ...current,
                                  [id]: event.target.value
                                }))
                              }
                              style={inputStyle}
                            >
                              <option value="PENDING">Pending</option>
                              <option value="READY">Ready</option>
                              <option value="FULFILLED">Fulfilled</option>
                              <option value="CANCELLED">Cancelled</option>
                            </select>
                          </td>
                          <td>
                            <textarea
                              aria-label={`${stringValue(
                                row,
                                "label_snapshot",
                                "Extra"
                              )} fulfillment notes`}
                              rows={2}
                              value={fulfillmentNotes[id] || ""}
                              onChange={(event) =>
                                setFulfillmentNotes((current) => ({
                                  ...current,
                                  [id]: event.target.value
                                }))
                              }
                              placeholder={
                                currentStatus === "FULFILLED"
                                  ? "Correction note required to move backward"
                                  : "Optional pickup note"
                              }
                              style={inputStyle}
                            />
                          </td>
                          <td>
                            <ConfirmAction
                              triggerLabel="Save fulfillment"
                              title="Save this fulfillment status?"
                              description="The status change is version-checked and added to the tournament commerce audit trail."
                              confirmLabel="Yes, save status"
                              confirmationText="SAVE FULFILLMENT STATUS"
                              disabled={busy || !adminWriteReady}
                              busy={busy}
                              onConfirm={(confirmationText) => updateFulfillment(row, confirmationText)}
                            />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {!detail.fulfillment.length ? (
                <p>No fulfillment items are currently active.</p>
              ) : null}
            </article>
          ) : null}

          {tab === "recovery" ? (
            <section style={{ display: "grid", gap: "1rem" }}>
              <article style={cardStyle}>
                <h2 style={{ marginTop: 0 }}>Operation recovery</h2>
                <p style={{ color: "#475569" }}>
                  Inspect authoritative operation, order, and audit evidence
                  before retrying any interrupted action.
                </p>
                <div style={{ overflowX: "auto" }}>
                  <table
                    style={{
                      width: "100%",
                      borderCollapse: "collapse",
                      minWidth: "760px"
                    }}
                  >
                    <thead>
                      <tr>
                        <th align="left">Created</th>
                        <th align="left">Action</th>
                        <th align="left">Status</th>
                        <th align="left">Evidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detail.operations.map((operation) => {
                        const operationId = stringValue(operation, "id");
                        const operationStatus = stringValue(
                          operation,
                          "status",
                          "UNKNOWN"
                        );
                        return (
                          <tr key={operationId}>
                            <td>{stringValue(operation, "created_at")}</td>
                            <td>{stringValue(operation, "action")}</td>
                            <td>
                              <span
                                style={{
                                  ...operationStatusStyle(operationStatus),
                                  borderRadius: "999px",
                                  padding: "0.15rem 0.5rem"
                                }}
                              >
                                {operationStatus}
                              </span>
                            </td>
                            <td>
                              <button
                                type="button"
                                onClick={() =>
                                  void inspectOperation(operationId)
                                }
                                disabled={busy}
                                style={ghostButtonStyle}
                              >
                                Inspect
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                {!detail.operations.length ? (
                  <p>No commerce operations are recorded.</p>
                ) : null}
              </article>

              {operationEvidence ? (
                <article style={cardStyle}>
                  <h2 style={{ marginTop: 0 }}>Authoritative evidence</h2>
                  <dl
                    style={{
                      display: "grid",
                      gridTemplateColumns:
                        "repeat(auto-fit, minmax(min(100%, 180px), 1fr))",
                      gap: "0.75rem"
                    }}
                  >
                    <div>
                      <dt>Recovery state</dt>
                      <dd>
                        {stringValue(operationEvidence, "recovery_state", "—")}
                      </dd>
                    </div>
                    <div>
                      <dt>Mutation complete</dt>
                      <dd>
                        {Boolean(
                          operationEvidence.authoritative_mutation_complete
                        )
                          ? "Yes"
                          : "No"}
                      </dd>
                    </div>
                    <div>
                      <dt>Safe retry</dt>
                      <dd>
                        {Boolean(operationEvidence.safe_retry) ? "Yes" : "No"}
                      </dd>
                    </div>
                    <div>
                      <dt>Retry mode</dt>
                      <dd>
                        {stringValue(operationEvidence, "retry_mode", "—")}
                      </dd>
                    </div>
                  </dl>
                </article>
              ) : null}

              <article style={cardStyle}>
                <h2 style={{ marginTop: 0 }}>Recent audit trail</h2>
                <div style={{ overflowX: "auto" }}>
                  <table
                    style={{
                      width: "100%",
                      borderCollapse: "collapse",
                      minWidth: "760px"
                    }}
                  >
                    <thead>
                      <tr>
                        <th align="left">Created</th>
                        <th align="left">Action</th>
                        <th align="left">Actor</th>
                        <th align="left">Source</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detail.audit.map((row) => (
                        <tr key={stringValue(row, "id")}>
                          <td>{stringValue(row, "created_at")}</td>
                          <td>{stringValue(row, "action")}</td>
                          <td>{stringValue(row, "actor_label")}</td>
                          <td>{stringValue(row, "source")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </article>
            </section>
          ) : null}
        </>
      ) : null}

      <p
        aria-live="polite"
        role={message?.toLowerCase().includes("unable") ? "alert" : "status"}
        style={{
          color: message?.toLowerCase().includes("unable")
            ? "#b91c1c"
            : "#475569",
          margin: 0
        }}
      >
        {message}
      </p>
    </section>
  );
}
