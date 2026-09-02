"use client";

import Link from "next/link";
import { FormEvent, useCallback, useMemo, useState } from "react";
import {
  PublicRegistrationEditRegistration,
  PublicRegistrationEditSelection,
  PublicRegistrationEvent,
  PublicRegistrationPlayer,
  PublicRegistrationSelectionPayload,
  PublicRegistrationDay,
  submitClubTournamentRegistrationEdit
} from "@/lib/tournamentRegistrationApi";
import {
  TournamentCommerceCatalog,
  TournamentCommerceOrder,
  TournamentCommerceQuote,
  TournamentCommerceSelection
} from "@/lib/tournamentCommerceApi";
import { publicEventEligibilityReason, publicEventFamilyKey } from "@/lib/tournamentRegistrationEligibility";
import { InteractionDialog } from "@/components/interaction";
import TournamentCommerceChooser from "../TournamentCommerceChooser";

type EditTournamentRegistrationFormProps = {
  clubSlug: string;
  tournamentId: string;
  registrationSlug?: string | null;
  editToken: string;
  registration: PublicRegistrationEditRegistration;
  selections: PublicRegistrationEditSelection[];
  days: PublicRegistrationDay[];
  events: PublicRegistrationEvent[];
  players: PublicRegistrationPlayer[];
  commerce?: TournamentCommerceCatalog | null;
  commerceOrder?: TournamentCommerceOrder | null;
};

type EventSelectionDraft = Omit<
  PublicRegistrationSelectionPayload,
  "event_option_id" | "registration_day_id" | "partner_mode"
>;

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

function numberOrNull(value: FormDataEntryValue | null): number | null {
  const text = String(value ?? "").trim();
  if (!text) return null;
  const parsed = Number(text);
  return Number.isFinite(parsed) ? parsed : null;
}

function textValue(formData: FormData, key: string): string {
  return String(formData.get(key) ?? "").trim();
}

function eventMeta(event: PublicRegistrationEvent): string {
  const pieces = [event.skill_label, event.age_label, event.event_format, event.scoring]
    .filter((item) => item && String(item).trim())
    .map(String);
  if (event.price_usd != null) pieces.push(`$${Number(event.price_usd).toFixed(2)}`);
  if (event.capacity_teams != null) pieces.push(`Cap ${event.capacity_teams}`);
  return pieces.join(" • ");
}

function scheduledDaysLabel(
  event: PublicRegistrationEvent,
  daysById: Map<string, PublicRegistrationDay>
): string {
  const scheduledIds = event.scheduled_day_ids?.length
    ? event.scheduled_day_ids
    : [event.registration_day_id];
  return scheduledIds
    .map((dayId) => daysById.get(dayId))
    .filter(Boolean)
    .map((day) =>
      day?.event_date ? `${day.label} · ${day.event_date}` : day?.label
    )
    .join(" · ");
}

function numericState(value: string): number | null {
  if (!value.trim()) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export default function EditTournamentRegistrationForm({
  clubSlug,
  tournamentId,
  registrationSlug,
  editToken,
  registration,
  selections,
  days,
  events,
  players,
  commerce,
  commerceOrder
}: EditTournamentRegistrationFormProps) {
  const initialSelectionIds = selections.map((selection) => selection.event_option_id).filter(Boolean);
  const [selectedIds, setSelectedIds] = useState<string[]>(initialSelectionIds);
  const [partnerModes, setPartnerModes] = useState<Record<string, "NONE" | "HAS_PARTNER" | "NEEDS_PARTNER">>(() => {
    const modes: Record<string, "NONE" | "HAS_PARTNER" | "NEEDS_PARTNER"> = {};
    for (const selection of selections) {
      const mode = selection.partner_mode === "HAS_PARTNER" || selection.partner_mode === "NEEDS_PARTNER" ? selection.partner_mode : "NONE";
      modes[selection.event_option_id] = mode;
    }
    return modes;
  });
  const [selectionDrafts, setSelectionDrafts] = useState<
    Record<string, EventSelectionDraft>
  >(() =>
    Object.fromEntries(
      selections.map((selection) => [
        selection.event_option_id,
        {
          partner_name: selection.partner_name || "",
          partner_email: selection.partner_email || "",
          partner_phone: selection.partner_phone || "",
          partner_dupr_id: selection.partner_dupr_id || "",
          partner_skill: selection.partner_skill ?? null,
          partner_age: selection.partner_age ?? null,
          partner_gender: selection.partner_gender || "",
          partner_note: selection.partner_note || "",
          show_on_partner_board: Boolean(selection.show_on_partner_board)
        }
      ])
    )
  );
  const [editingEventId, setEditingEventId] = useState<string | null>(null);
  const [addEventOpen, setAddEventOpen] = useState(false);
  const [gender, setGender] = useState(String(registration.gender || ""));
  const [ageDraft, setAgeDraft] = useState(String(registration.age ?? ""));
  const [doublesSkill, setDoublesSkill] = useState(String(registration.doubles_skill ?? ""));
  const [singlesSkill, setSinglesSkill] = useState(String(registration.singles_skill ?? ""));
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<{ confirmationToken: string; deliveryStatus: string } | null>(null);
  const [commerceSelections, setCommerceSelections] = useState<
    TournamentCommerceSelection[]
  >(() => commerceOrder?.quote?.request.item_selections || []);
  const [commerceQuote, setCommerceQuote] =
    useState<TournamentCommerceQuote | null>(
      commerceOrder?.quote || null
    );
  const [commerceIdempotencyKey, setCommerceIdempotencyKey] = useState(() =>
    crypto.randomUUID()
  );

  const eventById = useMemo(() => new Map(events.map((event) => [event.id, event])), [events]);
  const linkedPlayer = useMemo(() => players.find((player) => player.id === String(registration.player_id ?? "")) ?? null, [players, registration.player_id]);
  const eligibilityProfile = useMemo(() => ({
    gender,
    age: numericState(ageDraft),
    doublesSkill: linkedPlayer?.doubles_skill ?? numericState(doublesSkill),
    singlesSkill: linkedPlayer?.singles_skill ?? numericState(singlesSkill)
  }), [ageDraft, gender, linkedPlayer, doublesSkill, singlesSkill]);
  const selectionByEventId = useMemo(() => new Map(selections.map((selection) => [selection.event_option_id, selection])), [selections]);
  const selectionByFamily = useMemo(() => {
    const lookup = new Map<string, PublicRegistrationEditSelection>();
    for (const selection of selections) {
      const priorEvent = eventById.get(selection.event_option_id);
      if (priorEvent) lookup.set(publicEventFamilyKey(priorEvent), selection);
    }
    return lookup;
  }, [selections, eventById]);
  const visibleEvents = useMemo(() => events.filter((event) => event.selectable || initialSelectionIds.includes(event.id)), [events, initialSelectionIds]);
  const groupedEvents = useMemo(() => {
    return days.map((day) => ({
      day,
      events: visibleEvents.filter((event) => event.registration_day_id === day.id)
    })).filter((row) => row.events.length > 0);
  }, [days, visibleEvents]);
  const dayById = useMemo(
    () => new Map(days.map((day) => [day.id, day])),
    [days]
  );

  const totalPrice = selectedIds.reduce((sum, id) => {
    const price = eventById.get(id)?.price_usd;
    return sum + (price == null ? 0 : Number(price));
  }, 0);

  const updateCommerceReview = useCallback(
    (
      nextSelections: TournamentCommerceSelection[],
      nextQuote: TournamentCommerceQuote | null
    ) => {
      const nextFingerprint = nextQuote?.quote_fingerprint || null;
      const currentFingerprint =
        commerceQuote?.quote_fingerprint || null;
      setCommerceSelections(nextSelections);
      setCommerceQuote(nextQuote);
      if (
        currentFingerprint &&
        nextFingerprint !== currentFingerprint
      ) {
        setCommerceIdempotencyKey(crypto.randomUUID());
      }
    },
    [commerceQuote?.quote_fingerprint]
  );

  function toggleEvent(eventId: string, checked: boolean) {
    setSelectedIds((current) => {
      if (!checked) return current.filter((id) => id !== eventId);
      const nextEvent = eventById.get(eventId);
      if (!nextEvent) return current;
      const sameGroup = (id: string) => eventById.has(id) && publicEventFamilyKey(eventById.get(id)!) === publicEventFamilyKey(nextEvent);
      return [...current.filter((id) => id !== eventId && !sameGroup(id)), eventId];
    });
    if (checked && !partnerModes[eventId]) {
      setPartnerModes((current) => ({ ...current, [eventId]: eventById.get(eventId)?.partner_required ? "NEEDS_PARTNER" : "NONE" }));
    }
    if (checked && !selectionDrafts[eventId]) {
      const nextEvent = eventById.get(eventId);
      const prior = nextEvent
        ? selectionByFamily.get(publicEventFamilyKey(nextEvent))
        : undefined;
      setSelectionDrafts((current) => ({
        ...current,
        [eventId]: {
          partner_name: prior?.partner_name || "",
          partner_email: prior?.partner_email || "",
          partner_phone: prior?.partner_phone || "",
          partner_dupr_id: prior?.partner_dupr_id || "",
          partner_skill: prior?.partner_skill ?? null,
          partner_age: prior?.partner_age ?? null,
          partner_gender: prior?.partner_gender || "",
          partner_note: prior?.partner_note || "",
          show_on_partner_board: Boolean(prior?.show_on_partner_board)
        }
      }));
    }
    if (commerce?.available && commerceQuote) {
      setCommerceQuote(null);
      setCommerceIdempotencyKey(crypto.randomUUID());
    }
  }

  function updateSelectionDraft(
    eventId: string,
    patch: Partial<EventSelectionDraft>
  ) {
    setSelectionDrafts((current) => ({
      ...current,
      [eventId]: { ...(current[eventId] || {}), ...patch }
    }));
    if (commerce?.available && commerceQuote) {
      setCommerceQuote(null);
      setCommerceIdempotencyKey(crypto.randomUUID());
    }
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    const formData = new FormData(event.currentTarget);
    const submittedAge = numberOrNull(formData.get("age"));
    if (submittedAge == null || submittedAge < 1 || submittedAge > 120) {
      setError("Enter an age between 1 and 120 before choosing or saving events.");
      return;
    }
    if (!selectedIds.length) {
      setError("Select at least one event.");
      return;
    }
    const ineligible = selectedIds
      .map((id) => eventById.get(id))
      .filter((row): row is PublicRegistrationEvent => Boolean(row))
      .map((row) => ({
        row,
        reason: publicEventEligibilityReason(row, {
          ...eligibilityProfile,
          age: submittedAge
        })
      }))
      .find((item) => item.reason);
    if (ineligible?.reason) {
      setError(`${ineligible.row.division_name}: ${ineligible.reason}`);
      return;
    }
    if (commerce?.available && !commerceQuote) {
      setError(
        "Review extras and the current total before saving. You can choose zero extras."
      );
      return;
    }
    const payloadSelections: Array<PublicRegistrationSelectionPayload & { id?: string }> = selectedIds.map((eventId) => {
      const eventOption = eventById.get(eventId);
      const prior = selectionByEventId.get(eventId) ?? (eventOption ? selectionByFamily.get(publicEventFamilyKey(eventOption)) : undefined);
      const mode = partnerModes[eventId] ?? (eventOption?.partner_required ? "NEEDS_PARTNER" : "NONE");
      const draft = selectionDrafts[eventId] || {};
      return {
        id: prior?.id,
        event_option_id: eventId,
        registration_day_id: eventOption?.registration_day_id,
        partner_mode: mode,
        partner_name: String(draft.partner_name || "").trim(),
        partner_email: String(draft.partner_email || "").trim(),
        partner_phone: String(draft.partner_phone || "").trim(),
        partner_dupr_id: String(draft.partner_dupr_id || "").trim(),
        partner_skill: draft.partner_skill ?? null,
        partner_age: draft.partner_age ?? null,
        partner_gender: String(draft.partner_gender || "").trim(),
        partner_note: String(draft.partner_note || "").trim(),
        show_on_partner_board:
          mode === "NEEDS_PARTNER" &&
          Boolean(draft.show_on_partner_board)
      };
    });

    setPending(true);
    const response = await submitClubTournamentRegistrationEdit(clubSlug, {
      edit_token: editToken,
      expected_updated_at: registration.updated_at,
      expected_selection_versions: selections.map((selection) => ({ id: selection.id, updated_at: selection.updated_at })),
      tournament_id: tournamentId,
      registration_slug: registrationSlug || null,
      first_name: textValue(formData, "first_name"),
      last_name: textValue(formData, "last_name"),
      display_name: textValue(formData, "display_name"),
      email: registration.email,
      phone: textValue(formData, "phone"),
      player_id: registration.player_id ?? null,
      dupr_id: textValue(formData, "dupr_id"),
      doubles_skill: numberOrNull(formData.get("doubles_skill")),
      singles_skill: numberOrNull(formData.get("singles_skill")),
      age: submittedAge,
      gender: textValue(formData, "gender"),
      notes: textValue(formData, "notes"),
      wants_partner_board_contact: formData.get("wants_partner_board_contact") === "on",
      terms_accepted: formData.get("terms_accepted") === "on",
      website: textValue(formData, "website"),
      selections: payloadSelections,
      commerce: commerce?.available
        ? {
            item_selections: commerceSelections,
            expected_quote_fingerprint:
              commerceQuote?.quote_fingerprint || "",
            idempotency_key: commerceIdempotencyKey,
            expected_order_updated_at: commerceOrder?.updated_at || null
          }
        : null
    });
    setPending(false);

    if (response.error || !response.data?.registration_id) {
      if (response.status === 409 && response.current_quote) {
        const nextQuote = response.current_quote;
        if (
          (nextQuote?.quote_fingerprint || null) !==
          (commerceQuote?.quote_fingerprint || null)
        ) {
          setCommerceIdempotencyKey(crypto.randomUUID());
        }
        setCommerceSelections(nextQuote.request.item_selections || []);
        setCommerceQuote(nextQuote);
        setError(
          response.error ||
            "The total changed. Review the updated price before saving."
        );
        return;
      }
      setError(response.error || "Unable to save registration changes.");
      return;
    }
    if (!response.data.confirmation_token) {
      setError(response.data.email_delivery?.message || "Your changes were saved, but secure confirmation access is unavailable. Please contact tournament staff before submitting again.");
      return;
    }

    setSuccess({
      confirmationToken: response.data.confirmation_token,
      deliveryStatus: response.data.email_delivery?.status || response.data.confirmation_delivery?.status || "unknown"
    });
  }

  if (success) {
    const query = new URLSearchParams({ confirmation_token: success.confirmationToken });
    if (success.deliveryStatus) query.set("email_status", success.deliveryStatus);
    return (
      <section style={{ ...cardStyle, background: "#f0fdf4", borderColor: "#bbf7d0" }}>
        <h2 style={{ marginTop: 0 }}>Registration changes saved</h2>
        <p style={{ color: "#166534" }}>
          {success.deliveryStatus === "sent" || success.deliveryStatus === "staging_redirect"
            ? "An updated confirmation was sent."
            : success.deliveryStatus === "dry_run"
              ? "Confirmation email delivery is in dry-run mode."
              : "The confirmation email could not be delivered. Contact tournament staff if you need a copy."}
        </p>
        <Link href={`/clubs/${clubSlug}/tournament-registration/confirmation?${query.toString()}`}>View updated registration</Link>
      </section>
    );
  }

  return (
    <form onSubmit={onSubmit} style={{ display: "grid", gap: "1rem" }}>
      <input type="text" name="website" autoComplete="off" tabIndex={-1} style={{ position: "absolute", left: "-10000px" }} aria-hidden="true" />

      <section style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Player information</h2>
        <p style={{ color: "#475569" }}>Email is locked for security. Request a new registration if the email address needs to change.</p>
        {registration.player_id != null ? (
          <p style={{ color: "#475569" }}>
            Linked JUPR profile: <strong>{linkedPlayer?.display_name || `Player ${registration.player_id}`}</strong>. The linked profile and its verified rating are locked by this edit link.
          </p>
        ) : null}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
          <label>First name<br /><input name="first_name" defaultValue={registration.first_name || ""} required style={{ width: "100%" }} /></label>
          <label>Last name<br /><input name="last_name" defaultValue={registration.last_name || ""} required style={{ width: "100%" }} /></label>
          <label>Display name<br /><input name="display_name" defaultValue={registration.display_name || ""} placeholder="Optional if first/last entered" style={{ width: "100%" }} /></label>
          <label>Email<br /><input name="email" type="email" value={registration.email} disabled style={{ width: "100%" }} /></label>
          <label>Phone<br /><input name="phone" defaultValue={registration.phone || ""} style={{ width: "100%" }} /></label>
          <label>DUPR ID<br /><input name="dupr_id" defaultValue={registration.dupr_id || ""} style={{ width: "100%" }} /></label>
          <label>Doubles skill<br /><input name="doubles_skill" value={linkedPlayer?.doubles_skill ?? doublesSkill} onChange={(event) => setDoublesSkill(event.target.value)} disabled={registration.player_id != null} type="number" min="1" max="7" step="0.01" style={{ width: "100%" }} /></label>
          <label>Singles skill<br /><input name="singles_skill" value={linkedPlayer?.singles_skill ?? singlesSkill} onChange={(event) => setSinglesSkill(event.target.value)} disabled={registration.player_id != null} type="number" min="1" max="7" step="0.01" style={{ width: "100%" }} /></label>
          <label>Age<br /><input name="age" value={ageDraft} onChange={(event) => setAgeDraft(event.target.value)} type="number" min="1" max="120" required style={{ width: "100%" }} /></label>
          <label>Gender<br /><select name="gender" value={gender} onChange={(event) => setGender(event.target.value)} style={{ width: "100%" }}><option value="">Select</option><option>Women</option><option>Men</option><option>Non-binary</option><option>Prefer not to say</option></select></label>
        </div>
      </section>

      <section style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Registered events</h2>
        <p style={{ color: "#475569" }}>
          Each event has its own edit action for division and partner details.
          Changes are staged until you save the full registration below.
        </p>
        <div style={{ display: "grid", gap: "0.65rem" }}>
          {selectedIds.map((eventId) => {
            const eventOption = eventById.get(eventId);
            if (!eventOption) return null;
            const schedule = scheduledDaysLabel(eventOption, dayById);
            const mode =
              partnerModes[eventId] ||
              (eventOption.partner_required ? "NEEDS_PARTNER" : "NONE");
            const eligibilityReason = publicEventEligibilityReason(eventOption, eligibilityProfile);
            return (
              <article
                key={eventId}
                style={{
                  border: "1px solid #cbd5e1",
                  borderRadius: "12px",
                  padding: "0.85rem",
                  background: "#f8fafc",
                  display: "flex",
                  justifyContent: "space-between",
                  gap: "0.75rem",
                  alignItems: "flex-start",
                  flexWrap: "wrap"
                }}
              >
                <div>
                  <strong>
                    {eventOption.event_family_label} — {eventOption.division_name}
                  </strong>
                  <p style={{ color: "#64748b", margin: "0.3rem 0" }}>
                    {schedule || "Schedule TBD"} · {eventMeta(eventOption)}
                  </p>
                  <span style={{ color: "#475569" }}>
                    Partner: {mode.replaceAll("_", " ").toLowerCase()}
                  </span>
                  {eligibilityReason ? <p role="alert" style={{ color: "#b91c1c", marginBottom: 0 }}>{eligibilityReason} Update the player details or choose another division before saving.</p> : null}
                </div>
                <button
                  type="button"
                  onClick={() => setEditingEventId(eventId)}
                  style={{
                    border: "1px solid #0f172a",
                    borderRadius: "9px",
                    padding: "0.5rem 0.75rem",
                    background: "white",
                    fontWeight: 800
                  }}
                >
                  Edit event
                </button>
              </article>
            );
          })}
        </div>
        {!selectedIds.length ? (
          <p style={{ color: "#92400e" }}>No events are currently selected.</p>
        ) : null}
        <button
          type="button"
          onClick={() => setAddEventOpen(true)}
          disabled={!visibleEvents.some((event) => (
            !selectedIds.includes(event.id)
              && event.selectable
              && !publicEventEligibilityReason(event, eligibilityProfile)
          ))}
          style={{
            marginTop: "0.8rem",
            border: "1px solid #2563eb",
            borderRadius: "9px",
            padding: "0.55rem 0.8rem",
            color: "#1d4ed8",
            background: "white",
            fontWeight: 800
          }}
        >
          + Add Event
        </button>
        <p><strong>Estimated total:</strong> ${totalPrice.toFixed(2)}</p>
      </section>

      {addEventOpen ? (
        <InteractionDialog
          open={addEventOpen}
          phase="ready"
          title="Add Event"
          description="Choose an available division. Choosing another division in the same event family replaces the current division."
          onRequestClose={() => setAddEventOpen(false)}
          actions={(
            <button type="button" onClick={() => setAddEventOpen(false)}>
              Close
            </button>
          )}
        >
          {groupedEvents.map(({ day, events: dayEvents }) => {
            const available = dayEvents.filter(
              (eventOption) =>
                !selectedIds.includes(eventOption.id) &&
                eventOption.selectable &&
                !publicEventEligibilityReason(eventOption, eligibilityProfile)
            );
            if (!available.length) return null;
            return (
              <section key={day.id}>
                <h3>{day.label}{day.event_date ? ` · ${day.event_date}` : ""}</h3>
                <div style={{ display: "grid", gap: "0.5rem" }}>
                  {available.map((eventOption) => (
                    <button
                      key={eventOption.id}
                      type="button"
                      onClick={() => {
                        toggleEvent(eventOption.id, true);
                        setAddEventOpen(false);
                        setEditingEventId(eventOption.id);
                      }}
                      style={{
                        textAlign: "left",
                        border: "1px solid #cbd5e1",
                        borderRadius: "10px",
                        padding: "0.7rem",
                        background: "white"
                      }}
                    >
                      <strong>{eventOption.event_family_label} — {eventOption.division_name}</strong>
                      <br />
                      <span style={{ color: "#64748b" }}>{scheduledDaysLabel(eventOption, dayById) || "Schedule TBD"}<br />{eventMeta(eventOption)}</span>
                    </button>
                  ))}
                </div>
              </section>
            );
          })}
        </InteractionDialog>
      ) : null}

      {editingEventId && eventById.get(editingEventId) ? (() => {
        const eventOption = eventById.get(editingEventId)!;
        const prior = selectionDrafts[editingEventId];
        const mode =
          partnerModes[editingEventId] ||
          (eventOption.partner_required ? "NEEDS_PARTNER" : "NONE");
        return (
          <InteractionDialog
            open={Boolean(editingEventId)}
            phase="ready"
            title="Edit event"
            description={`${eventOption.event_family_label} — ${eventOption.division_name}`}
            onRequestClose={() => setEditingEventId(null)}
            actions={(
              <>
                <button
                  type="button"
                  onClick={() => {
                    toggleEvent(editingEventId, false);
                    setEditingEventId(null);
                  }}
                  style={{ color: "#b91c1c" }}
                >
                  Remove event
                </button>
                <button type="button" onClick={() => setEditingEventId(null)} style={{ fontWeight: 800 }}>
                  Apply event changes
                </button>
              </>
            )}
          >
            <h3>{eventOption.event_family_label} — {eventOption.division_name}</h3>
            <p style={{ color: "#475569" }}>{scheduledDaysLabel(eventOption, dayById) || "Schedule TBD"}<br />{eventMeta(eventOption)}</p>
            <div style={{ display: "grid", gap: "0.6rem" }}>
              <label>Partner status<br />
                <select
                  value={mode}
                  onChange={(event) =>
                    setPartnerModes((current) => ({
                      ...current,
                      [editingEventId]: event.target.value as "NONE" | "HAS_PARTNER" | "NEEDS_PARTNER"
                    }))
                  }
                  style={{ width: "100%" }}
                >
                  {!eventOption.partner_required ? <option value="NONE">No partner needed</option> : null}
                  {eventOption.partner_required ? <option value="HAS_PARTNER">I have a partner</option> : null}
                  {eventOption.partner_required ? <option value="NEEDS_PARTNER">I need a partner</option> : null}
                </select>
              </label>
              {mode === "HAS_PARTNER" ? (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.5rem" }}>
                  <label>Partner name<br /><input defaultValue={prior?.partner_name || ""} onChange={(event) => updateSelectionDraft(editingEventId, { partner_name: event.target.value })} required style={{ width: "100%" }} /></label>
                  <label>Partner email<br /><input defaultValue={prior?.partner_email || ""} onChange={(event) => updateSelectionDraft(editingEventId, { partner_email: event.target.value })} type="email" required style={{ width: "100%" }} /></label>
                  <label>Partner phone<br /><input defaultValue={prior?.partner_phone || ""} onChange={(event) => updateSelectionDraft(editingEventId, { partner_phone: event.target.value })} style={{ width: "100%" }} /></label>
                  <label>Partner DUPR ID<br /><input defaultValue={prior?.partner_dupr_id || ""} onChange={(event) => updateSelectionDraft(editingEventId, { partner_dupr_id: event.target.value })} style={{ width: "100%" }} /></label>
                  <label>Partner skill<br /><input defaultValue={prior?.partner_skill ?? ""} onChange={(event) => updateSelectionDraft(editingEventId, { partner_skill: numberOrNull(event.target.value) })} type="number" min="1" max="7" step="0.01" style={{ width: "100%" }} /></label>
                  <label>Partner age<br /><input defaultValue={prior?.partner_age ?? ""} type="number" min="1" max="120" required onChange={(event) => updateSelectionDraft(editingEventId, { partner_age: numberOrNull(event.target.value) })} style={{ width: "100%" }} /></label>
                  <label>Partner gender<br /><select defaultValue={prior?.partner_gender || ""} required onChange={(event) => updateSelectionDraft(editingEventId, { partner_gender: event.target.value })} style={{ width: "100%" }}><option value="">Select</option><option value="Women">Women</option><option value="Men">Men</option><option value="Non-binary">Non-binary</option><option value="Other">Other</option><option value="Prefer not to say">Prefer not to say</option></select></label>
                </div>
              ) : null}
              {mode === "NEEDS_PARTNER" ? (
                <label style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                  <input
                    type="checkbox"
                    defaultChecked={Boolean(prior?.show_on_partner_board)}
                    disabled={!eventOption.partner_board_enabled}
                    onChange={(event) => updateSelectionDraft(editingEventId, { show_on_partner_board: event.target.checked })}
                  /> Show me on the public partner board for this event
                </label>
              ) : null}
              <label>Partner note<br /><textarea defaultValue={prior?.partner_note || ""} onChange={(event) => updateSelectionDraft(editingEventId, { partner_note: event.target.value })} rows={2} style={{ width: "100%" }} /></label>
            </div>
          </InteractionDialog>
        );
      })() : null}

      {commerce?.available ? (
        <TournamentCommerceChooser
          clubSlug={clubSlug}
          tournamentId={tournamentId}
          registrationId={registration.id}
          eventOptionIds={selectedIds}
          catalog={commerce}
          initialSelections={commerceSelections}
          disabled={pending}
          onReviewChange={updateCommerceReview}
        />
      ) : null}

      <section style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Notes and policies</h2>
        <label>Notes for organizers<br /><textarea name="notes" defaultValue={registration.notes || ""} rows={4} style={{ width: "100%" }} /></label>
        <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginTop: "0.75rem" }}>
          <input name="wants_partner_board_contact" type="checkbox" defaultChecked={Boolean(registration.wants_partner_board_contact)} /> Organizers may use my contact info for partner-board coordination.
        </label>
        <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginTop: "0.75rem" }}>
          <input name="terms_accepted" type="checkbox" required /> I confirm these registration changes are accurate and agree to tournament policies.
        </label>
      </section>

      {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
      <button type="submit" disabled={pending || !visibleEvents.length} style={{ padding: "0.75rem 1rem", borderRadius: "10px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 700 }}>
        {pending ? "Saving…" : "Save registration changes"}
      </button>
    </form>
  );
}
