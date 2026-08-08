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
  firstName: string;
  lastName: string;
  displayName: string;
  email: string;
  phone: string;
  playerId: string;
  gender: string;
  age: string;
  ageBracket: string;
  duprId: string;
  doublesSkill: string;
  singlesSkill: string;
  wantsPartnerBoardContact: boolean;
  registrationStatus: string;
  paymentStatus: string;
  notes: string;
};

type SelectionEdit = {
  eventOptionId: string;
  partnerMode: string;
  partnerName: string;
  partnerEmail: string;
  partnerPhone: string;
  partnerDuprId: string;
  partnerSkill: string;
  partnerAge: string;
  partnerGender: string;
  partnerNote: string;
  showOnPartnerBoard: boolean;
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
const PARTNER_MODE_OPTIONS = ["NONE", "HAS_PARTNER", "NEEDS_PARTNER"];
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
    firstName: row?.first_name || "",
    lastName: row?.last_name || "",
    displayName: row?.display_name || "",
    email: row?.email || "",
    phone: row?.phone || "",
    playerId: row?.player_id == null ? "" : String(row.player_id),
    gender: row?.gender || "",
    age: row?.age == null ? "" : String(row.age),
    ageBracket: row?.age_bracket || "",
    duprId: row?.dupr_id || "",
    doublesSkill: row?.doubles_skill == null ? "" : String(row.doubles_skill),
    singlesSkill: row?.singles_skill == null ? "" : String(row.singles_skill),
    wantsPartnerBoardContact: Boolean(row?.wants_partner_board_contact),
    registrationStatus: row?.registration_status || "confirmed",
    paymentStatus: row?.payment_status || "unpaid",
    notes: row?.notes || ""
  };
}
function selectionEdit(row: AdminTournamentSelection | null): SelectionEdit {
  return {
    eventOptionId: row?.event_option_id || "",
    partnerMode: row?.partner_selection_id ? "NONE" : row?.partner_mode || "NONE",
    partnerName: row?.partner_name || "",
    partnerEmail: row?.partner_email || "",
    partnerPhone: row?.partner_phone || "",
    partnerDuprId: row?.partner_dupr_id || "",
    partnerSkill: row?.partner_skill == null ? "" : String(row.partner_skill),
    partnerAge: row?.partner_age == null ? "" : String(row.partner_age),
    partnerGender: row?.partner_gender || "",
    partnerNote: row?.partner_note || "",
    showOnPartnerBoard: Boolean(row?.show_on_partner_board),
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
  const [newEventOptionId, setNewEventOptionId] = useState("");
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
    setNewEventOptionId("");
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
      const selectedEventIds = new Set(nextSelections.map((row) => row.event_option_id));
      setNewEventOptionId(
        String(payload.event_options.find((row) => !selectedEventIds.has(String(row.id)))?.id || "")
      );
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
            first_name: registrationDraft.firstName || null,
            last_name: registrationDraft.lastName || null,
            display_name: registrationDraft.displayName || null,
            email: registrationDraft.email || null,
            phone: registrationDraft.phone || null,
            player_id: registrationDraft.playerId ? Number(registrationDraft.playerId) : null,
            gender: registrationDraft.gender || null,
            age: registrationDraft.age ? Number(registrationDraft.age) : null,
            age_bracket: registrationDraft.ageBracket || null,
            dupr_id: registrationDraft.duprId || null,
            doubles_skill: registrationDraft.doublesSkill ? Number(registrationDraft.doublesSkill) : null,
            singles_skill: registrationDraft.singlesSkill ? Number(registrationDraft.singlesSkill) : null,
            wants_partner_board_contact: registrationDraft.wantsPartnerBoardContact,
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

  async function addSelection(confirmationText: string) {
    if (!newEventOptionId) {
      setMessage("Choose a Division before adding an event entry.");
      return;
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const selectedEvent = detail?.event_options.find(
        (row) => String(row.id) === newEventOptionId
      );
      const response = await requestJson<AdminTournamentWriteResponse & { selection?: AdminTournamentSelection }>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/registrations/${encodeURIComponent(registrationId)}/selections`,
        {
          method: "POST",
          body: JSON.stringify({
            event_option_id: newEventOptionId,
            partner_mode: Boolean(selectedEvent?.partner_required) ? "NEEDS_PARTNER" : "NONE",
            expected_state_fingerprint: detail?.state_fingerprint || null,
            confirmation_text: confirmationText,
            source: "next_tournament_registration_detail"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      await loadDetail(response.selection?.id || "");
      if (actionRequest.isCurrent(generation)) setMessage("Event entry added.");
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to add event entry.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function removeSelection(confirmationText: string) {
    if (!selectedSelection?.updated_at) {
      setMessage("Reload this event entry before removing it.");
      return;
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/selections/${encodeURIComponent(selectedSelection.id)}`,
        {
          method: "DELETE",
          body: JSON.stringify({
            expected_updated_at: selectedSelection.updated_at,
            confirmation_text: confirmationText,
            source: "next_tournament_registration_detail"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      await loadDetail("");
      if (actionRequest.isCurrent(generation)) setMessage("Event entry removed.");
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to remove event entry.");
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
      const selectionPatch = selectedSelection.partner_selection_id
        ? {
            partner_note: selectionDraft.partnerNote
          }
        : {
            event_option_id: selectionDraft.eventOptionId,
            partner_mode: selectionDraft.partnerMode,
            partner_name: selectionDraft.partnerName || null,
            partner_email: selectionDraft.partnerEmail || null,
            partner_phone: selectionDraft.partnerPhone || null,
            partner_dupr_id: selectionDraft.partnerDuprId || null,
            partner_skill: selectionDraft.partnerSkill
              ? Number(selectionDraft.partnerSkill)
              : null,
            partner_age: selectionDraft.partnerAge
              ? Number(selectionDraft.partnerAge)
              : null,
            partner_gender: selectionDraft.partnerGender || null,
            partner_note: selectionDraft.partnerNote,
            show_on_partner_board: selectionDraft.showOnPartnerBoard
          };
      await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/admin/tournaments/${encodeURIComponent(
          tournamentId
        )}/selections/${encodeURIComponent(selectedSelection.id)}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            ...selectionPatch,
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

  const availableNewEvents = useMemo(() => {
    const selectedIds = new Set(selections.map((row) => row.event_option_id));
    return (detail?.event_options || []).filter((row) => !selectedIds.has(String(row.id)));
  }, [detail, selections]);

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
            <h2 style={{ marginTop: 0 }}>Complete registration</h2>
            <p style={{ color: "#475569" }}>
              Edit the registrant&apos;s identity, contact information, player linkage, eligibility data, status, payment state, consent, and notes. Event entries, partners, extras, and fulfillment are managed below.
            </p>
            <h3>Registrant identity and eligibility</h3>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
                gap: "0.75rem",
                marginBottom: "0.75rem"
              }}
            >
              <label><strong>First name</strong><br /><input value={registrationDraft.firstName} onChange={(event) => setRegistrationDraft((current) => ({ ...current, firstName: event.target.value }))} style={inputStyle} /></label>
              <label><strong>Last name</strong><br /><input value={registrationDraft.lastName} onChange={(event) => setRegistrationDraft((current) => ({ ...current, lastName: event.target.value }))} style={inputStyle} /></label>
              <label><strong>Display name</strong><br /><input value={registrationDraft.displayName} onChange={(event) => setRegistrationDraft((current) => ({ ...current, displayName: event.target.value }))} style={inputStyle} /></label>
              <label><strong>Email</strong><br /><input type="email" value={registrationDraft.email} onChange={(event) => setRegistrationDraft((current) => ({ ...current, email: event.target.value }))} style={inputStyle} /></label>
              <label><strong>Phone</strong><br /><input type="tel" value={registrationDraft.phone} onChange={(event) => setRegistrationDraft((current) => ({ ...current, phone: event.target.value }))} style={inputStyle} /></label>
              <label><strong>Linked player ID</strong><br /><input type="number" min="1" value={registrationDraft.playerId} onChange={(event) => setRegistrationDraft((current) => ({ ...current, playerId: event.target.value }))} style={inputStyle} /><small>Clear to unlink this tournament registration from a player profile.</small></label>
              <label><strong>Gender</strong><br /><select value={registrationDraft.gender} onChange={(event) => setRegistrationDraft((current) => ({ ...current, gender: event.target.value }))} style={inputStyle}><option value="">Not specified</option><option value="Women">Women</option><option value="Men">Men</option><option value="Non-binary">Non-binary</option><option value="Other">Other</option><option value="Prefer not to say">Prefer not to say</option></select></label>
              <label><strong>Age</strong><br /><input type="number" min="5" max="120" value={registrationDraft.age} onChange={(event) => setRegistrationDraft((current) => ({ ...current, age: event.target.value }))} style={inputStyle} /></label>
              <label><strong>Age bracket note</strong><br /><input value={registrationDraft.ageBracket} onChange={(event) => setRegistrationDraft((current) => ({ ...current, ageBracket: event.target.value }))} style={inputStyle} /></label>
              <label><strong>DUPR ID</strong><br /><input value={registrationDraft.duprId} onChange={(event) => setRegistrationDraft((current) => ({ ...current, duprId: event.target.value }))} style={inputStyle} /></label>
              <label><strong>Doubles skill</strong><br /><input type="number" min="1" max="7" step="0.01" value={registrationDraft.doublesSkill} onChange={(event) => setRegistrationDraft((current) => ({ ...current, doublesSkill: event.target.value }))} style={inputStyle} /></label>
              <label><strong>Singles skill</strong><br /><input type="number" min="1" max="7" step="0.01" value={registrationDraft.singlesSkill} onChange={(event) => setRegistrationDraft((current) => ({ ...current, singlesSkill: event.target.value }))} style={inputStyle} /></label>
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.9rem" }}>
              <input type="checkbox" checked={registrationDraft.wantsPartnerBoardContact} onChange={(event) => setRegistrationDraft((current) => ({ ...current, wantsPartnerBoardContact: event.target.checked }))} />
              Registrant consents to contact information appearing on the Partner Board
            </label>
            <h3>Administrative status and payment</h3>
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
                description={`Update every editable registration detail for ${registration.display_name}. Eligibility-affecting changes are rechecked when event entries are saved and during Tournament Review.`}
                confirmLabel="Yes, save registration"
                confirmationText="SAVE REGISTRATION"
                busy={busy}
                onConfirm={saveRegistration}
              />
            </p>
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Event entries and partners</h2>
            <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) auto", gap: "0.75rem", alignItems: "end", marginBottom: "1rem" }}>
              <label>
                <strong>Add another event entry</strong><br />
                <select value={newEventOptionId} onChange={(event) => setNewEventOptionId(event.target.value)} style={inputStyle}>
                  <option value="">{availableNewEvents.length ? "Choose Division" : "All available Divisions are already selected"}</option>
                  {availableNewEvents.map((row) => <option key={String(row.id)} value={String(row.id)}>{eventLabel(row)}</option>)}
                </select>
              </label>
              <ConfirmAction
                triggerLabel={busy ? "Adding…" : "Add event entry"}
                title="Add this event entry?"
                description="Add the selected Division to this registration. Eligibility is checked before the entry is saved."
                confirmLabel="Yes, add event entry"
                confirmationText="SAVE SELECTION"
                disabled={!newEventOptionId || busy}
                busy={busy}
                onConfirm={addSelection}
              />
            </div>
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
                                : value === "HAS_PARTNER"
                                  ? "Has manual / unregistered partner"
                                  : "Needs partner"}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                    {selectionDraft.partnerMode === "HAS_PARTNER" && !selectionDraft.partnerSelectionId ? (
                      <div style={{ marginTop: "0.75rem", padding: "0.85rem", border: "1px solid #bfdbfe", borderRadius: "12px", background: "#eff6ff" }}>
                        <h3 style={{ marginTop: 0 }}>Manual partner details</h3>
                        <p style={{ color: "#475569" }}>Use this when the partner does not have a separate tournament registration. These fields are used for skill, age, and gender eligibility.</p>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem" }}>
                          <label><strong>Partner name</strong><br /><input value={selectionDraft.partnerName} onChange={(event) => setSelectionDraft((current) => ({ ...current, partnerName: event.target.value }))} style={inputStyle} /></label>
                          <label><strong>Partner email</strong><br /><input type="email" value={selectionDraft.partnerEmail} onChange={(event) => setSelectionDraft((current) => ({ ...current, partnerEmail: event.target.value }))} style={inputStyle} /></label>
                          <label><strong>Partner phone</strong><br /><input type="tel" value={selectionDraft.partnerPhone} onChange={(event) => setSelectionDraft((current) => ({ ...current, partnerPhone: event.target.value }))} style={inputStyle} /></label>
                          <label><strong>Partner DUPR ID</strong><br /><input value={selectionDraft.partnerDuprId} onChange={(event) => setSelectionDraft((current) => ({ ...current, partnerDuprId: event.target.value }))} style={inputStyle} /></label>
                          <label><strong>Partner doubles skill</strong><br /><input type="number" min="1" max="7" step="0.01" value={selectionDraft.partnerSkill} onChange={(event) => setSelectionDraft((current) => ({ ...current, partnerSkill: event.target.value }))} style={inputStyle} /></label>
                          <label><strong>Partner age</strong><br /><input type="number" min="5" max="120" value={selectionDraft.partnerAge} onChange={(event) => setSelectionDraft((current) => ({ ...current, partnerAge: event.target.value }))} style={inputStyle} /></label>
                          <label><strong>Partner gender</strong><br /><select value={selectionDraft.partnerGender} onChange={(event) => setSelectionDraft((current) => ({ ...current, partnerGender: event.target.value }))} style={inputStyle}><option value="">Not specified</option><option value="Women">Women</option><option value="Men">Men</option><option value="Non-binary">Non-binary</option><option value="Other">Other</option><option value="Prefer not to say">Prefer not to say</option></select></label>
                        </div>
                      </div>
                    ) : null}
                    {selectionDraft.partnerMode === "NEEDS_PARTNER" && !selectionDraft.partnerSelectionId ? (
                      <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "0.75rem" }}>
                        <input
                          type="checkbox"
                          checked={selectionDraft.showOnPartnerBoard}
                          disabled={!Boolean(registration.wants_partner_board_contact)}
                          onChange={(event) => setSelectionDraft((current) => ({ ...current, showOnPartnerBoard: event.target.checked }))}
                        />
                        Show this partner request on the Partner Board
                        {!registration.wants_partner_board_contact ? <small>Enable and save the registrant&apos;s Partner Board contact consent above first.</small> : null}
                      </label>
                    ) : null}
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
                        description="Update the Division, unpaired mode, complete manual-partner details, eligibility data, Partner Board visibility, and partner note."
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
                      <ConfirmAction
                        triggerLabel={busy ? "Removing…" : "Remove event entry"}
                        title="Remove this event entry?"
                        description="Remove the Division from this registration. Confirmed teams and imported draws remain protected and must be changed in their canonical workflow first."
                        confirmLabel="Yes, remove event entry"
                        confirmationText="REMOVE SELECTION"
                        tone="danger"
                        busy={busy}
                        onConfirm={removeSelection}
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
