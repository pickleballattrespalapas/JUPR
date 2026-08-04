"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type {
  AdminTournamentDetailResponse,
  AdminTournamentRegistration,
  AdminTournamentSelection,
  AdminTournamentStatusResponse,
  AdminTournamentWriteResponse
} from "@/lib/adminTournamentApi";
import {
  formatCommerceMoney,
  getAdminTournamentCommerceDetail,
  type AdminTournamentCommerceDetail,
  type TournamentCommerceQuote
} from "@/lib/tournamentCommerceApi";
import {
  useAuthenticatedAutoLoad,
  useLatestRequestGuard
} from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminTournamentStatusResponse;
  tournamentId: string;
  tournamentName: string;
  registrationId: string;
};

type RegistrationEdit = {
  registrationStatus: string;
  paymentStatus: string;
  notes: string;
};

type SelectionEdit = {
  eventOptionId: string;
  partnerMode: string;
  partnerNote: string;
  partnerSelectionId: string;
};

type ExtrasQuoteResponse = {
  ok: boolean;
  quote: TournamentCommerceQuote;
  current_order?: {
    status?: string | null;
    payment_status?: string | null;
    updated_at?: string | null;
    quote_fingerprint?: string | null;
  } | null;
};

const REGISTRATION_STATUS_OPTIONS = ["confirmed", "waitlist", "cancelled"];
const PAYMENT_STATUS_OPTIONS = ["unpaid", "paid", "waived", "refunded"];
const PARTNER_MODE_OPTIONS = ["NONE", "NEEDS_PARTNER"];
const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white",
  minWidth: 0
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
  display: "inline-block",
  padding: "0.6rem 0.9rem",
  borderRadius: "999px",
  border: "1px solid #0f172a",
  background: "#0f172a",
  color: "white",
  fontWeight: 800,
  cursor: "pointer"
};

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}
function registrationEdit(
  row: AdminTournamentRegistration | null
): RegistrationEdit {
  return {
    registrationStatus: row?.registration_status || "confirmed",
    paymentStatus: row?.payment_status || "unpaid",
    notes: row?.notes || ""
  };
}
function selectionEdit(row: AdminTournamentSelection | null): SelectionEdit {
  return {
    eventOptionId: row?.event_option_id || "",
    partnerMode:
      row?.partner_mode === "HAS_PARTNER"
        ? "NONE"
        : row?.partner_mode || "NONE",
    partnerNote: row?.partner_note || "",
    partnerSelectionId: row?.partner_selection_id || ""
  };
}
function selectedHref(
  path: string,
  tournamentId: string,
  tournamentName: string
): string {
  const params = new URLSearchParams({ tournament: tournamentId });
  if (tournamentName) params.set("name", tournamentName);
  return `${path}?${params.toString()}`;
}
function eventLabel(row: Record<string, unknown>): string {
  const family = String(row.event_family_label || "").trim();
  const division = String(row.division_name || row.label || "").trim();
  return family && division && family !== division
    ? `${family} / ${division}`
    : division || family || String(row.id || "Event");
}
function recordValue(
  row: Record<string, unknown> | null | undefined,
  key: string
): Record<string, unknown> {
  const value = row?.[key];
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}
function recordList(
  row: Record<string, unknown> | null | undefined,
  key: string
): Array<Record<string, unknown>> {
  const value = row?.[key];
  return Array.isArray(value)
    ? value.filter(
        (item): item is Record<string, unknown> =>
          Boolean(item) && typeof item === "object"
      )
    : [];
}
function stringValue(
  row: Record<string, unknown> | null | undefined,
  key: string
): string {
  const value = row?.[key];
  return value == null ? "" : String(value);
}
function numberValue(
  row: Record<string, unknown> | null | undefined,
  key: string
): number | null {
  const value = Number(row?.[key]);
  return Number.isFinite(value) ? value : null;
}

export default function TournamentRegistrantEditPanel({
  apiBase,
  clubId,
  status,
  tournamentId,
  tournamentName,
  registrationId
}: Props) {
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [detail, setDetail] =
    useState<AdminTournamentDetailResponse | null>(null);
  const [commerce, setCommerce] =
    useState<AdminTournamentCommerceDetail | null>(null);
  const [commerceWarning, setCommerceWarning] = useState<string | null>(null);
  const [registrationDraft, setRegistrationDraft] =
    useState<RegistrationEdit>(() => registrationEdit(null));
  const [selectedSelectionId, setSelectedSelectionId] = useState("");
  const [selectionDraft, setSelectionDraft] = useState<SelectionEdit>(() =>
    selectionEdit(null)
  );
  const [extraQuantities, setExtraQuantities] = useState<
    Record<string, number>
  >({});
  const [extrasQuote, setExtrasQuote] =
    useState<TournamentCommerceQuote | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const detailRequest = useLatestRequestGuard(
    `${accessToken}\u0000${tournamentId}\u0000${registrationId}`,
    clearProtectedState
  );
  const actionRequest = useLatestRequestGuard(accessToken);

  const registration =
    detail?.registrations.find((row) => row.id === registrationId) || null;
  const selections = (detail?.selections || []).filter(
    (row) => row.registration_id === registrationId
  );
  const selectedSelection =
    selections.find((row) => row.id === selectedSelectionId) || null;
  const currentOrder =
    commerce?.orders.find(
      (row) => stringValue(row, "registration_id") === registrationId
    ) || null;

  function clearProtectedState() {
    actionRequest.invalidate();
    setDetail(null);
    setCommerce(null);
    setCommerceWarning(null);
    setRegistrationDraft(registrationEdit(null));
    setSelectedSelectionId("");
    setSelectionDraft(selectionEdit(null));
    setExtraQuantities({});
    setExtrasQuote(null);
    setBusy(false);
    setMessage(null);
  }

  async function requestJson<T>(
    path: string,
    options?: RequestInit
  ): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken)
      throw new Error("Sign in before editing tournament registrations.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), {
      ...options,
      headers
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok)
      throw new Error(
        String(payload?.detail || `API error (${response.status})`)
      );
    return payload as T;
  }

  function quantitiesFromCommerce(
    commerceDetail: AdminTournamentCommerceDetail | null
  ): Record<string, number> {
    const order =
      commerceDetail?.orders.find(
        (row) => stringValue(row, "registration_id") === registrationId
      ) || null;
    const quote = recordValue(order, "quote");
    const request = recordValue(quote, "request");
    const result: Record<string, number> = {};
    for (const selection of recordList(request, "item_selections")) {
      const variantId = stringValue(selection, "variant_id");
      const quantity = Math.max(0, numberValue(selection, "quantity") || 0);
      if (variantId && quantity) result[variantId] = quantity;
    }
    return result;
  }

  async function loadDetail(preferredSelectionId = selectedSelectionId) {
    const generation = detailRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentDetailResponse>(
        `/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`
      );
      if (!detailRequest.isCurrent(generation)) return;
      const nextRegistration =
        payload.registrations.find((row) => row.id === registrationId) || null;
      if (!nextRegistration) throw new Error("Registration was not found.");

      let nextCommerce: AdminTournamentCommerceDetail | null = null;
      let nextCommerceWarning: string | null = null;
      try {
        const commerceResponse = await getAdminTournamentCommerceDetail(
          clubId,
          tournamentId,
          accessToken
        );
        if (!detailRequest.isCurrent(generation)) return;
        nextCommerce = commerceResponse.data || null;
        nextCommerceWarning = commerceResponse.error
          ? `Extras are temporarily unavailable. ${commerceResponse.error}`
          : null;
      } catch (error) {
        nextCommerceWarning = `Extras are temporarily unavailable. ${
          error instanceof Error ? error.message : "Unable to load commerce."
        }`;
      }

      setDetail(payload);
      setCommerce(nextCommerce);
      setCommerceWarning(nextCommerceWarning);
      setRegistrationDraft(registrationEdit(nextRegistration));
      setExtraQuantities(quantitiesFromCommerce(nextCommerce));
      setExtrasQuote(null);
      const nextSelections = payload.selections.filter(
        (row) => row.registration_id === registrationId
      );
      const nextSelection =
        nextSelections.find((row) => row.id === preferredSelectionId) ||
        nextSelections[0] ||
        null;
      setSelectedSelectionId(nextSelection?.id || "");
      setSelectionDraft(selectionEdit(nextSelection));
    } catch (error) {
      if (detailRequest.isCurrent(generation))
        setMessage(
          error instanceof Error
            ? error.message
            : "Unable to load registration."
        );
    } finally {
      if (detailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function chooseSelection(selectionId: string) {
    const row =
      selections.find((selection) => selection.id === selectionId) || null;
    setSelectedSelectionId(row?.id || "");
    setSelectionDraft(selectionEdit(row));
    setMessage(null);
  }

  async function saveRegistration(confirmationText: string) {
    if (!registration?.updated_at) {
      setMessage("Reload this registration before saving.");
      return;
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/admin/tournaments/${encodeURIComponent(
          tournamentId
        )}/registrations/${encodeURIComponent(registration.id)}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            registration_status: registrationDraft.registrationStatus,
            payment_status: registrationDraft.paymentStatus,
            notes: registrationDraft.notes,
            expected_updated_at: registration.updated_at,
            confirmation_text: confirmationText,
            source: "next_tournament_registration_detail"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      await loadDetail(selectedSelectionId);
      if (actionRequest.isCurrent(generation))
        setMessage("Registration saved.");
    } catch (error) {
      if (actionRequest.isCurrent(generation))
        setMessage(
          error instanceof Error
            ? error.message
            : "Unable to save registration."
        );
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveSelection(confirmationText: string) {
    if (!selectedSelection?.updated_at) {
      setMessage("Reload this event entry before saving.");
      return;
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/admin/tournaments/${encodeURIComponent(
          tournamentId
        )}/selections/${encodeURIComponent(selectedSelection.id)}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            event_option_id: selectionDraft.eventOptionId,
            partner_mode: selectedSelection.partner_selection_id
              ? "HAS_PARTNER"
              : selectionDraft.partnerMode,
            partner_note: selectionDraft.partnerNote,
            expected_updated_at: selectedSelection.updated_at,
            confirmation_text: confirmationText,
            source: "next_tournament_registration_detail"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      await loadDetail(selectedSelection.id);
      if (actionRequest.isCurrent(generation))
        setMessage("Event entry saved.");
    } catch (error) {
      if (actionRequest.isCurrent(generation))
        setMessage(
          error instanceof Error
            ? error.message
            : "Unable to save event entry."
        );
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function savePartner(confirmationText: string) {
    if (!selectedSelection?.updated_at) {
      setMessage("Reload this event entry before changing its partner.");
      return;
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/admin/tournaments/${encodeURIComponent(
          tournamentId
        )}/selections/${encodeURIComponent(selectedSelection.id)}/partner`,
        {
          method: "PUT",
          body: JSON.stringify({
            partner_selection_id:
              selectionDraft.partnerSelectionId || null,
            unpaired_mode: selectionDraft.partnerMode,
            expected_updated_at: selectedSelection.updated_at,
            confirmation_text: confirmationText,
            source: "next_tournament_registration_detail"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      await loadDetail(selectedSelection.id);
      if (actionRequest.isCurrent(generation))
        setMessage(
          selectionDraft.partnerSelectionId
            ? "Event partner saved."
            : "Event partner removed."
        );
    } catch (error) {
      if (actionRequest.isCurrent(generation))
        setMessage(
          error instanceof Error
            ? error.message
            : "Unable to save event partner."
        );
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function itemSelections() {
    return Object.entries(extraQuantities)
      .filter(([, quantity]) => Number(quantity) > 0)
      .map(([variant_id, quantity]) => ({
        variant_id,
        quantity: Math.floor(Number(quantity))
      }));
  }

  async function reviewExtras() {
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const response = await requestJson<ExtrasQuoteResponse>(
        `/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/commerce/tournaments/${encodeURIComponent(
          tournamentId
        )}/orders/${encodeURIComponent(registrationId)}/quote`,
        {
          method: "POST",
          body: JSON.stringify({ item_selections: itemSelections() })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      setExtrasQuote(response.quote);
      setMessage(
        `Extras reviewed. New total: ${formatCommerceMoney(
          response.quote.total_minor
        )}.`
      );
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setExtrasQuote(null);
        setMessage(
          error instanceof Error ? error.message : "Unable to review extras."
        );
      }
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveExtras(confirmationText: string) {
    if (!extrasQuote) {
      setMessage("Review the updated extras total before saving.");
      return;
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<Record<string, unknown>>(
        `/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/commerce/tournaments/${encodeURIComponent(
          tournamentId
        )}/orders/${encodeURIComponent(registrationId)}`,
        {
          method: "PUT",
          body: JSON.stringify({
            item_selections: itemSelections(),
            expected_quote_fingerprint: extrasQuote.quote_fingerprint,
            expected_order_updated_at: currentOrder
              ? stringValue(currentOrder, "updated_at") || null
              : null,
            confirmation_text: confirmationText,
            idempotency_key: crypto.randomUUID(),
            source: "next_tournament_registration_detail"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      await loadDetail(selectedSelectionId);
      if (actionRequest.isCurrent(generation))
        setMessage("Registrant extras saved and totals recalculated.");
    } catch (error) {
      if (actionRequest.isCurrent(generation))
        setMessage(
          error instanceof Error ? error.message : "Unable to save extras."
        );
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(
    status.enabled
      ? `${accessToken}\u0000${tournamentId}\u0000${registrationId}`
      : "",
    loadDetail
  );

  const financial = useMemo(() => {
    const quote = recordValue(currentOrder, "quote");
    const totalMinor =
      numberValue(currentOrder, "total_minor") ??
      numberValue(quote, "total_minor");
    const extras = recordList(quote, "lines")
      .filter((line) =>
        ["ITEM", "BUNDLE"].includes(
          stringValue(line, "line_type").toUpperCase()
        )
      )
      .map((line) => stringValue(line, "label") || "Extra");
    return { totalMinor, extras };
  }, [currentOrder]);

  const partnerCandidates = useMemo(() => {
    if (!selectedSelection || !detail) return [];
    const registrationById = new Map(
      detail.registrations.map((row) => [row.id, row])
    );
    return detail.selections
      .filter(
        (row) =>
          row.id !== selectedSelection.id &&
          row.event_option_id === selectedSelection.event_option_id &&
          row.registration_id !== selectedSelection.registration_id &&
          (!row.partner_selection_id ||
            row.partner_selection_id === selectedSelection.id)
      )
      .filter((row) => {
        const candidate = registrationById.get(row.registration_id);
        return candidate?.registration_status !== "cancelled";
      })
      .map((row) => ({
        selection: row,
        registration: registrationById.get(row.registration_id)
      }))
      .sort((left, right) =>
        String(left.registration?.display_name || "").localeCompare(
          String(right.registration?.display_name || "")
        )
      );
  }, [detail, selectedSelection]);

  const activeVariants = useMemo(() => {
    if (!commerce) return [];
    const itemById = new Map(
      commerce.catalog.items.map((item) => [item.id, item])
    );
    return commerce.catalog.variants
      .filter((variant) => {
        const item = itemById.get(variant.item_id);
        return variant.status === "ACTIVE" && item?.status === "ACTIVE";
      })
      .map((variant) => ({ variant, item: itemById.get(variant.item_id)! }))
      .sort((left, right) =>
        `${left.item.name} ${left.variant.name}`.localeCompare(
          `${right.item.name} ${right.variant.name}`
        )
      );
  }, [commerce]);

  if (sessionLoading && !accessToken)
    return <p role="status">Loading registration…</p>;
  if (!accessToken)
    return (
      <article style={{ ...cardStyle, background: "#fffbeb" }}>
        <h2 style={{ marginTop: 0 }}>Admin sign-in required</h2>
        <p>
          <Link href="/admin/login">Open admin login</Link>
        </p>
      </article>
    );

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <p>
        <Link
          href={selectedHref(
            "/admin/tournaments/registration/registrants",
            tournamentId,
            tournamentName
          )}
        >
          ← Back to registrations
        </Link>
      </p>
      {message ? (
        <p
          role="status"
          style={{
            color: /unable|error|required|not found|changed/i.test(message)
              ? "#b91c1c"
              : "#166534"
          }}
        >
          {message}
        </p>
      ) : null}
      {busy && !registration ? (
        <p role="status">Loading registration…</p>
      ) : null}
      {registration ? (
        <>
          <article
            style={{
              ...cardStyle,
              background: "#eff6ff",
              borderColor: "#bfdbfe"
            }}
          >
            <h2 style={{ marginTop: 0 }}>{registration.display_name}</h2>
            <p style={{ color: "#475569", overflowWrap: "anywhere" }}>
              {registration.email || "No email"}
              {registration.phone ? ` · ${registration.phone}` : ""}
            </p>
            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "repeat(auto-fit, minmax(160px, 1fr))",
                gap: "0.75rem"
              }}
            >
              <div>
                <strong>Amount</strong>
                <br />
                {financial.totalMinor == null
                  ? "—"
                  : formatCommerceMoney(financial.totalMinor)}
              </div>
              <div>
                <strong>Events</strong>
                <br />
                {selections.length
                  ? selections
                      .map((row) => row.event_label || row.event_option_id)
                      .join(", ")
                  : "None"}
              </div>
              <div>
                <strong>Extras</strong>
                <br />
                {financial.extras.length
                  ? financial.extras.join(", ")
                  : "None"}
              </div>
            </div>
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Registration</h2>
            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "repeat(auto-fit, minmax(190px, 1fr))",
                gap: "0.75rem"
              }}
            >
              <label style={{ minWidth: 0 }}>
                <strong>Registration status</strong>
                <br />
                <select
                  value={registrationDraft.registrationStatus}
                  onChange={(event) =>
                    setRegistrationDraft((current) => ({
                      ...current,
                      registrationStatus: event.target.value
                    }))
                  }
                  style={inputStyle}
                >
                  {REGISTRATION_STATUS_OPTIONS.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
              <label style={{ minWidth: 0 }}>
                <strong>Payment status</strong>
                <br />
                <select
                  value={registrationDraft.paymentStatus}
                  onChange={(event) =>
                    setRegistrationDraft((current) => ({
                      ...current,
                      paymentStatus: event.target.value
                    }))
                  }
                  style={inputStyle}
                >
                  {PAYMENT_STATUS_OPTIONS.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <label style={{ display: "block", marginTop: "0.75rem" }}>
              <strong>Admin notes</strong>
              <br />
              <textarea
                value={registrationDraft.notes}
                onChange={(event) =>
                  setRegistrationDraft((current) => ({
                    ...current,
                    notes: event.target.value
                  }))
                }
                rows={4}
                style={inputStyle}
              />
            </label>
            <p>
              <ConfirmAction
                triggerLabel={busy ? "Saving…" : "Save registration"}
                title="Save this registration update?"
                description={`Update ${registration.display_name}'s status, offline payment state, and notes.`}
                confirmLabel="Yes, save registration"
                confirmationText="SAVE REGISTRATION"
                busy={busy}
                onConfirm={saveRegistration}
              />
            </p>
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Event entries and partners</h2>
            {selections.length ? (
              <>
                <label>
                  <strong>Select entry</strong>
                  <br />
                  <select
                    value={selectedSelectionId}
                    onChange={(event) => chooseSelection(event.target.value)}
                    style={inputStyle}
                  >
                    {selections.map((row) => (
                      <option key={row.id} value={row.id}>
                        {row.event_label || row.event_option_id || row.id}
                      </option>
                    ))}
                  </select>
                </label>
                {selectedSelection ? (
                  <div style={{ marginTop: "0.75rem" }}>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns:
                          "repeat(auto-fit, minmax(190px, 1fr))",
                        gap: "0.75rem"
                      }}
                    >
                      <label style={{ minWidth: 0 }}>
                        <strong>Division</strong>
                        <br />
                        <select
                          value={selectionDraft.eventOptionId}
                          onChange={(event) =>
                            setSelectionDraft((current) => ({
                              ...current,
                              eventOptionId: event.target.value
                            }))
                          }
                          disabled={Boolean(
                            selectedSelection.partner_selection_id
                          )}
                          style={inputStyle}
                        >
                          <option value="">Choose division</option>
                          {detail?.event_options.map((row) => (
                            <option key={String(row.id)} value={String(row.id)}>
                              {eventLabel(row)}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label style={{ minWidth: 0 }}>
                        <strong>Unpaired mode</strong>
                        <br />
                        <select
                          value={selectionDraft.partnerMode}
                          onChange={(event) =>
                            setSelectionDraft((current) => ({
                              ...current,
                              partnerMode: event.target.value
                            }))
                          }
                          disabled={Boolean(
                            selectionDraft.partnerSelectionId
                          )}
                          style={inputStyle}
                        >
                          {PARTNER_MODE_OPTIONS.map((value) => (
                            <option key={value} value={value}>
                              {value === "NONE"
                                ? "No partner request"
                                : "Needs partner"}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                    <label
                      style={{ display: "block", marginTop: "0.75rem" }}
                    >
                      <strong>Assigned partner</strong>
                      <br />
                      <select
                        value={selectionDraft.partnerSelectionId}
                        onChange={(event) =>
                          setSelectionDraft((current) => ({
                            ...current,
                            partnerSelectionId: event.target.value
                          }))
                        }
                        style={inputStyle}
                      >
                        <option value="">No assigned partner</option>
                        {selectedSelection.partner_selection_id &&
                        !partnerCandidates.some(
                          (candidate) =>
                            candidate.selection.id ===
                            selectedSelection.partner_selection_id
                        ) ? (
                          <option value={selectedSelection.partner_selection_id}>
                            {selectedSelection.partner_display_name ||
                              "Current partner"}
                          </option>
                        ) : null}
                        {partnerCandidates.map(({ selection, registration }) => (
                          <option key={selection.id} value={selection.id}>
                            {registration?.display_name || selection.id}
                          </option>
                        ))}
                      </select>
                    </label>
                    <p style={{ color: "#475569" }}>
                      Only compatible, non-cancelled registrations in this
                      division are shown. Existing teams and imported draws are
                      protected from conflicting assignments.
                    </p>
                    <label
                      style={{ display: "block", marginTop: "0.75rem" }}
                    >
                      <strong>Partner note</strong>
                      <br />
                      <textarea
                        value={selectionDraft.partnerNote}
                        onChange={(event) =>
                          setSelectionDraft((current) => ({
                            ...current,
                            partnerNote: event.target.value
                          }))
                        }
                        rows={3}
                        style={inputStyle}
                      />
                    </label>
                    <div
                      style={{
                        display: "flex",
                        gap: "0.75rem",
                        flexWrap: "wrap",
                        marginTop: "0.75rem"
                      }}
                    >
                      <ConfirmAction
                        triggerLabel={busy ? "Saving…" : "Save event entry"}
                        title="Save this event-entry update?"
                        description="Update the selected division and partner-board note."
                        confirmLabel="Yes, save event entry"
                        confirmationText="SAVE SELECTION"
                        busy={busy}
                        onConfirm={saveSelection}
                      />
                      <ConfirmAction
                        triggerLabel={
                          busy ? "Saving…" : "Save assigned partner"
                        }
                        title="Save this partner assignment?"
                        description={
                          selectionDraft.partnerSelectionId
                            ? "Assign or replace the confirmed partner for this event."
                            : "Remove the confirmed partner and return the entry to its selected unpaired mode."
                        }
                        confirmLabel="Yes, save partner"
                        confirmationText="SAVE PARTNER"
                        busy={busy}
                        onConfirm={savePartner}
                      />
                    </div>
                  </div>
                ) : null}
              </>
            ) : (
              <p style={{ color: "#64748b" }}>
                No event entries are attached to this registration.
              </p>
            )}
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Extras and quantities</h2>
            <p style={{ color: "#475569" }}>
              Add, remove, or change extras and option quantities. Review
              recalculates bundle discounts, inventory, fulfillment, and the
              offline amount due before saving.
            </p>
            {commerceWarning ? (
              <p
                role="status"
                style={{
                  padding: "0.75rem",
                  border: "1px solid #fde68a",
                  borderRadius: "10px",
                  background: "#fffbeb",
                  color: "#92400e"
                }}
              >
                {commerceWarning} Registration and partner editing remain
                available.
              </p>
            ) : null}
            {activeVariants.length ? (
              <div style={{ display: "grid", gap: "0.65rem" }}>
                {activeVariants.map(({ item, variant }) => (
                  <div
                    key={variant.id}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "minmax(0, 1fr) 8rem",
                      gap: "0.75rem",
                      alignItems: "end",
                      padding: "0.75rem",
                      border: "1px solid #e2e8f0",
                      borderRadius: "12px"
                    }}
                  >
                    <div>
                      <strong>{item.name}</strong>
                      <br />
                      <span>{variant.name}</span>
                      {variant.sku ? (
                        <small style={{ display: "block", color: "#64748b" }}>
                          SKU {variant.sku}
                        </small>
                      ) : null}
                      <small style={{ display: "block", color: "#475569" }}>
                        {formatCommerceMoney(variant.price_minor)}
                        {variant.remaining != null
                          ? ` · ${variant.remaining} remaining`
                          : ""}
                      </small>
                    </div>
                    <label>
                      Quantity
                      <br />
                      <input
                        type="number"
                        min="0"
                        max={Math.min(
                          item.max_per_registration || 20,
                          variant.remaining ?? 20
                        )}
                        value={extraQuantities[variant.id] || 0}
                        onChange={(event) => {
                          setExtraQuantities((current) => ({
                            ...current,
                            [variant.id]: Math.max(
                              0,
                              Math.floor(Number(event.target.value) || 0)
                            )
                          }));
                          setExtrasQuote(null);
                        }}
                        style={inputStyle}
                      />
                    </label>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: "#64748b" }}>
                No active extras are available for this tournament.
              </p>
            )}
            {extrasQuote ? (
              <div
                style={{
                  marginTop: "0.75rem",
                  padding: "0.85rem",
                  background: "#eff6ff",
                  border: "1px solid #bfdbfe",
                  borderRadius: "12px"
                }}
              >
                <strong>Reviewed total: {formatCommerceMoney(extrasQuote.total_minor)}</strong>
                <br />
                List subtotal {formatCommerceMoney(extrasQuote.list_subtotal_minor)} · Savings {formatCommerceMoney(extrasQuote.discount_minor)}
                <ul>
                  {extrasQuote.lines.map((line) => (
                    <li key={line.line_key}>
                      {line.quantity} × {line.label}
                      {line.option_label ? ` — ${line.option_label}` : ""}: {formatCommerceMoney(line.final_total_minor)}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
            <div
              style={{
                display: "flex",
                gap: "0.75rem",
                flexWrap: "wrap",
                marginTop: "0.75rem"
              }}
            >
              <button
                type="button"
                style={buttonStyle}
                disabled={busy}
                onClick={() => void reviewExtras()}
              >
                {busy ? "Reviewing…" : "Review updated extras"}
              </button>
              <ConfirmAction
                triggerLabel={busy ? "Saving…" : "Save reviewed extras"}
                title="Save this registrant's extras?"
                description="Apply the reviewed extras, option quantities, inventory, fulfillment, discounts, and offline amount due."
                confirmLabel="Yes, save extras"
                confirmationText="SAVE EXTRAS"
                disabled={!extrasQuote}
                busy={busy}
                onConfirm={saveExtras}
              />
            </div>
          </article>
        </>
      ) : null}
    </div>
  );
}
