"use client";

import Link from "next/link";
import { FormEvent, useMemo, useState } from "react";
import {
  PublicRegistrationEditRegistration,
  PublicRegistrationEditSelection,
  PublicRegistrationEvent,
  PublicRegistrationPlayer,
  PublicRegistrationSelectionPayload,
  PublicRegistrationDay,
  submitClubTournamentRegistrationEdit
} from "@/lib/tournamentRegistrationApi";
import { publicEventEligibilityReason, publicEventFamilyKey } from "@/lib/tournamentRegistrationEligibility";

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
};

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

function numericState(value: string): number | null {
  if (!value.trim()) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export default function EditTournamentRegistrationForm({ clubSlug, tournamentId, registrationSlug, editToken, registration, selections, days, events, players }: EditTournamentRegistrationFormProps) {
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
  const [gender, setGender] = useState(String(registration.gender || ""));
  const [doublesSkill, setDoublesSkill] = useState(String(registration.doubles_skill ?? ""));
  const [singlesSkill, setSinglesSkill] = useState(String(registration.singles_skill ?? ""));
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<{ registrationId: string; deliveryStatus: string } | null>(null);

  const eventById = useMemo(() => new Map(events.map((event) => [event.id, event])), [events]);
  const linkedPlayer = useMemo(() => players.find((player) => player.id === String(registration.player_id ?? "")) ?? null, [players, registration.player_id]);
  const eligibilityProfile = useMemo(() => ({
    gender,
    doublesSkill: linkedPlayer?.doubles_skill ?? numericState(doublesSkill),
    singlesSkill: linkedPlayer?.singles_skill ?? numericState(singlesSkill)
  }), [gender, linkedPlayer, doublesSkill, singlesSkill]);
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

  const totalPrice = selectedIds.reduce((sum, id) => {
    const price = eventById.get(id)?.price_usd;
    return sum + (price == null ? 0 : Number(price));
  }, 0);

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
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    if (!selectedIds.length) {
      setError("Select at least one event.");
      return;
    }
    const ineligible = selectedIds
      .map((id) => eventById.get(id))
      .filter((row): row is PublicRegistrationEvent => Boolean(row))
      .map((row) => ({ row, reason: publicEventEligibilityReason(row, eligibilityProfile) }))
      .find((item) => item.reason);
    if (ineligible?.reason) {
      setError(`${ineligible.row.division_name}: ${ineligible.reason}`);
      return;
    }
    const formData = new FormData(event.currentTarget);
    const payloadSelections: Array<PublicRegistrationSelectionPayload & { id?: string }> = selectedIds.map((eventId) => {
      const eventOption = eventById.get(eventId);
      const prior = selectionByEventId.get(eventId) ?? (eventOption ? selectionByFamily.get(publicEventFamilyKey(eventOption)) : undefined);
      const mode = partnerModes[eventId] ?? (eventOption?.partner_required ? "NEEDS_PARTNER" : "NONE");
      return {
        id: prior?.id,
        event_option_id: eventId,
        registration_day_id: eventOption?.registration_day_id,
        partner_mode: mode,
        partner_name: textValue(formData, `partner_name_${eventId}`),
        partner_email: textValue(formData, `partner_email_${eventId}`),
        partner_phone: textValue(formData, `partner_phone_${eventId}`),
        partner_dupr_id: textValue(formData, `partner_dupr_id_${eventId}`),
        partner_skill: numberOrNull(formData.get(`partner_skill_${eventId}`)),
        partner_age: numberOrNull(formData.get(`partner_age_${eventId}`)),
        partner_gender: textValue(formData, `partner_gender_${eventId}`),
        partner_note: textValue(formData, `partner_note_${eventId}`),
        show_on_partner_board: mode === "NEEDS_PARTNER" && formData.get(`show_on_partner_board_${eventId}`) === "on"
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
      age: numberOrNull(formData.get("age")),
      gender: textValue(formData, "gender"),
      notes: textValue(formData, "notes"),
      wants_partner_board_contact: formData.get("wants_partner_board_contact") === "on",
      terms_accepted: formData.get("terms_accepted") === "on",
      website: textValue(formData, "website"),
      selections: payloadSelections
    });
    setPending(false);

    if (response.error || !response.data?.registration_id) {
      setError(response.error || "Unable to save registration changes.");
      return;
    }

    setSuccess({
      registrationId: String(response.data.registration_id),
      deliveryStatus: response.data.confirmation_delivery?.status || "unknown"
    });
  }

  if (success) {
    const query = new URLSearchParams({ registration_id: success.registrationId });
    if (registrationSlug) query.set("tournament", registrationSlug);
    else query.set("tournament_id", tournamentId);
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
          <label>Doubles skill<br /><input name="doubles_skill" value={linkedPlayer?.doubles_skill ?? doublesSkill} onChange={(event) => setDoublesSkill(event.target.value)} disabled={registration.player_id != null} type="number" min="0" max="7" step="0.01" style={{ width: "100%" }} /></label>
          <label>Singles skill<br /><input name="singles_skill" value={linkedPlayer?.singles_skill ?? singlesSkill} onChange={(event) => setSinglesSkill(event.target.value)} disabled={registration.player_id != null} type="number" min="0" max="7" step="0.01" style={{ width: "100%" }} /></label>
          <label>Age<br /><input name="age" defaultValue={registration.age ?? ""} type="number" min="1" max="120" style={{ width: "100%" }} /></label>
          <label>Gender<br /><select name="gender" value={gender} onChange={(event) => setGender(event.target.value)} style={{ width: "100%" }}><option value="">Select</option><option>Women</option><option>Men</option><option>Non-binary</option><option>Prefer not to say</option></select></label>
        </div>
      </section>

      <section style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Choose events</h2>
        <p style={{ color: "#475569" }}>Choose one division per day and event family. Existing closed divisions may be preserved or removed, but cannot be newly added.</p>
        {groupedEvents.map(({ day, events: dayEvents }) => (
          <div key={day.id} style={{ marginBottom: "1rem" }}>
            <h3 style={{ marginBottom: "0.35rem" }}>{day.label}{day.event_date ? ` · ${day.event_date}` : ""}</h3>
            <div style={{ display: "grid", gap: "0.5rem" }}>
              {dayEvents.map((eventOption) => {
                const selected = selectedIds.includes(eventOption.id);
                const prior = selectionByEventId.get(eventOption.id);
                const mode = partnerModes[eventOption.id] ?? (eventOption.partner_required ? "NEEDS_PARTNER" : "NONE");
                const eligibilityReason = publicEventEligibilityReason(eventOption, eligibilityProfile);
                return (
                  <article key={eventOption.id} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: selected ? "#f8fafc" : "white" }}>
                    <label style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start" }}>
                      <input type="checkbox" checked={selected} disabled={(!eventOption.selectable && !prior) || (Boolean(eligibilityReason) && !prior)} onChange={(e) => toggleEvent(eventOption.id, e.target.checked)} />
                      <span><strong>{eventOption.event_family_label} — {eventOption.division_name}</strong><br /><span style={{ color: "#64748b" }}>{eventMeta(eventOption)}{!eventOption.selectable ? " • no longer selectable" : ""}</span></span>
                    </label>
                    {eligibilityReason ? <p style={{ color: "#b91c1c", margin: "0.4rem 0 0" }}>{eligibilityReason}</p> : null}
                    {selected ? (
                      <div style={{ marginTop: "0.75rem", display: "grid", gap: "0.5rem" }}>
                        <label>Partner status<br />
                          <select value={mode} onChange={(e) => setPartnerModes((current) => ({ ...current, [eventOption.id]: e.target.value as "NONE" | "HAS_PARTNER" | "NEEDS_PARTNER" }))} style={{ width: "100%" }}>
                            {!eventOption.partner_required ? <option value="NONE">No partner needed</option> : null}
                            {eventOption.partner_required ? <option value="HAS_PARTNER">I have a partner</option> : null}
                            {eventOption.partner_required ? <option value="NEEDS_PARTNER">I need a partner</option> : null}
                          </select>
                        </label>
                        {mode === "HAS_PARTNER" ? (
                          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.5rem" }}>
                            <label>Partner name<br /><input name={`partner_name_${eventOption.id}`} defaultValue={prior?.partner_name || ""} required style={{ width: "100%" }} /></label>
                            <label>Partner email<br /><input name={`partner_email_${eventOption.id}`} defaultValue={prior?.partner_email || ""} type="email" required style={{ width: "100%" }} /></label>
                            <label>Partner phone<br /><input name={`partner_phone_${eventOption.id}`} defaultValue={prior?.partner_phone || ""} style={{ width: "100%" }} /></label>
                            <label>Partner DUPR ID<br /><input name={`partner_dupr_id_${eventOption.id}`} defaultValue={prior?.partner_dupr_id || ""} style={{ width: "100%" }} /></label>
                            <label>Partner skill<br /><input name={`partner_skill_${eventOption.id}`} defaultValue={prior?.partner_skill ?? ""} type="number" min="1" max="7" step="0.01" style={{ width: "100%" }} /></label>
                            <label>Partner age<br /><input name={`partner_age_${eventOption.id}`} defaultValue={prior?.partner_age ?? ""} type="number" min="1" max="120" style={{ width: "100%" }} /></label>
                            <label>Partner gender<br /><select name={`partner_gender_${eventOption.id}`} required={String(eventOption.gender_restriction || "ANY").toUpperCase() !== "ANY"} style={{ width: "100%" }}><option value="">Select</option><option value="Women">Women</option><option value="Men">Men</option><option value="Other">Other</option><option value="Prefer not to say">Prefer not to say</option></select></label>
                          </div>
                        ) : null}
                        {mode === "NEEDS_PARTNER" ? (
                          <label style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                            <input name={`show_on_partner_board_${eventOption.id}`} type="checkbox" defaultChecked={Boolean(prior?.show_on_partner_board)} disabled={!eventOption.partner_board_enabled} /> Show me on the public partner board for this event
                          </label>
                        ) : null}
                        <label>Partner note<br /><textarea name={`partner_note_${eventOption.id}`} defaultValue={prior?.partner_note || ""} rows={2} style={{ width: "100%" }} /></label>
                      </div>
                    ) : null}
                  </article>
                );
              })}
            </div>
          </div>
        ))}
        {!visibleEvents.length ? <p style={{ color: "#64748b" }}>No events are currently available for editing.</p> : null}
        <p><strong>Estimated total:</strong> ${totalPrice.toFixed(2)}</p>
      </section>

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
