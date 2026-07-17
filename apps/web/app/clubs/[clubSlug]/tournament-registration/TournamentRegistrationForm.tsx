"use client";

import { FormEvent, useMemo, useState } from "react";
import {
  PublicRegistrationDay,
  PublicRegistrationEvent,
  PublicRegistrationSelectionPayload,
  submitClubTournamentRegistration
} from "@/lib/tournamentRegistrationApi";
import { publicEventEligibilityReason, publicEventFamilyKey } from "@/lib/tournamentRegistrationEligibility";

type TournamentRegistrationFormProps = {
  clubSlug: string;
  tournamentId: string;
  registrationSlug?: string | null;
  days: PublicRegistrationDay[];
  events: PublicRegistrationEvent[];
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

export default function TournamentRegistrationForm({ clubSlug, tournamentId, registrationSlug, days, events }: TournamentRegistrationFormProps) {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [partnerModes, setPartnerModes] = useState<Record<string, "NONE" | "HAS_PARTNER" | "NEEDS_PARTNER">>({});
  const [gender, setGender] = useState("");
  const [doublesSkill, setDoublesSkill] = useState("");
  const [singlesSkill, setSinglesSkill] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectableEvents = useMemo(() => events.filter((event) => event.selectable), [events]);
  const eventById = useMemo(() => new Map(events.map((event) => [event.id, event])), [events]);
  const eligibilityProfile = useMemo(() => ({
    gender,
    doublesSkill: numericState(doublesSkill),
    singlesSkill: numericState(singlesSkill)
  }), [gender, doublesSkill, singlesSkill]);
  const daysById = useMemo(() => new Map(days.map((day) => [day.id, day])), [days]);
  const groupedEvents = useMemo(() => {
    return days.map((day) => ({
      day,
      events: selectableEvents.filter((event) => event.registration_day_id === day.id)
    })).filter((row) => row.events.length > 0);
  }, [days, selectableEvents]);

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
    const selections: PublicRegistrationSelectionPayload[] = selectedIds.map((eventId) => {
      const eventOption = eventById.get(eventId);
      const mode = partnerModes[eventId] ?? (eventOption?.partner_required ? "NEEDS_PARTNER" : "NONE");
      return {
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
    const response = await submitClubTournamentRegistration(clubSlug, {
      tournament_id: tournamentId,
      registration_slug: registrationSlug || null,
      first_name: textValue(formData, "first_name"),
      last_name: textValue(formData, "last_name"),
      display_name: textValue(formData, "display_name"),
      email: textValue(formData, "email"),
      phone: textValue(formData, "phone"),
      // Initial public submissions remain unlinked until staff verify a profile.
      player_id: null,
      dupr_id: textValue(formData, "dupr_id"),
      doubles_skill: numberOrNull(formData.get("doubles_skill")),
      singles_skill: numberOrNull(formData.get("singles_skill")),
      age: numberOrNull(formData.get("age")),
      gender: textValue(formData, "gender"),
      notes: textValue(formData, "notes"),
      wants_partner_board_contact: formData.get("wants_partner_board_contact") === "on",
      terms_accepted: formData.get("terms_accepted") === "on",
      website: textValue(formData, "website"),
      selections
    });
    setPending(false);

    if (response.error || !response.data?.registration_id) {
      setError(response.error || "Unable to submit registration.");
      return;
    }

    const query = new URLSearchParams({ registration_id: String(response.data.registration_id) });
    if (registrationSlug) query.set("tournament", registrationSlug);
    else query.set("tournament_id", tournamentId);
    window.location.href = `/clubs/${clubSlug}/tournament-registration/confirmation?${query.toString()}`;
  }

  return (
    <form onSubmit={onSubmit} style={{ display: "grid", gap: "1rem" }}>
      <input type="text" name="website" autoComplete="off" tabIndex={-1} style={{ position: "absolute", left: "-10000px" }} aria-hidden="true" />

      <section style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Player information</h2>
        <p style={{ color: "#475569" }}>Public submissions remain unlinked until tournament staff verify the matching club profile.</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
          <label>First name<br /><input name="first_name" required style={{ width: "100%" }} /></label>
          <label>Last name<br /><input name="last_name" required style={{ width: "100%" }} /></label>
          <label>Display name<br /><input name="display_name" placeholder="Optional if first/last entered" style={{ width: "100%" }} /></label>
          <label>Email<br /><input name="email" type="email" required style={{ width: "100%" }} /></label>
          <label>Phone<br /><input name="phone" style={{ width: "100%" }} /></label>
          <label>DUPR ID<br /><input name="dupr_id" style={{ width: "100%" }} /></label>
          <label>Doubles skill<br /><input name="doubles_skill" type="number" min="0" max="7" step="0.01" value={doublesSkill} onChange={(event) => setDoublesSkill(event.target.value)} style={{ width: "100%" }} /></label>
          <label>Singles skill<br /><input name="singles_skill" type="number" min="0" max="7" step="0.01" value={singlesSkill} onChange={(event) => setSinglesSkill(event.target.value)} style={{ width: "100%" }} /></label>
          <label>Age<br /><input name="age" type="number" min="1" max="120" style={{ width: "100%" }} /></label>
          <label>Gender<br /><select name="gender" value={gender} onChange={(event) => setGender(event.target.value)} style={{ width: "100%" }}><option value="">Select</option><option>Women</option><option>Men</option><option>Non-binary</option><option>Prefer not to say</option></select></label>
        </div>
      </section>

      <section style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Choose events</h2>
        <p style={{ color: "#475569" }}>Choose one division per day and event family. Selecting another division in the same group replaces the previous choice.</p>
        {groupedEvents.map(({ day, events: dayEvents }) => (
          <div key={day.id} style={{ marginBottom: "1rem" }}>
            <h3 style={{ marginBottom: "0.35rem" }}>{day.label}{day.event_date ? ` · ${day.event_date}` : ""}</h3>
            <div style={{ display: "grid", gap: "0.5rem" }}>
              {dayEvents.map((eventOption) => {
                const selected = selectedIds.includes(eventOption.id);
                const mode = partnerModes[eventOption.id] ?? (eventOption.partner_required ? "NEEDS_PARTNER" : "NONE");
                const eligibilityReason = publicEventEligibilityReason(eventOption, eligibilityProfile);
                return (
                  <article key={eventOption.id} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: selected ? "#f8fafc" : "white" }}>
                    <label style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start" }}>
                      <input type="checkbox" checked={selected} disabled={Boolean(eligibilityReason)} onChange={(e) => toggleEvent(eventOption.id, e.target.checked)} />
                      <span><strong>{eventOption.event_family_label} — {eventOption.division_name}</strong><br /><span style={{ color: "#64748b" }}>{eventMeta(eventOption)}</span></span>
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
                            <label>Partner name<br /><input name={`partner_name_${eventOption.id}`} required style={{ width: "100%" }} /></label>
                            <label>Partner email<br /><input name={`partner_email_${eventOption.id}`} type="email" required style={{ width: "100%" }} /></label>
                            <label>Partner phone<br /><input name={`partner_phone_${eventOption.id}`} style={{ width: "100%" }} /></label>
                            <label>Partner DUPR ID<br /><input name={`partner_dupr_id_${eventOption.id}`} style={{ width: "100%" }} /></label>
                            <label>Partner skill<br /><input name={`partner_skill_${eventOption.id}`} type="number" min="1" max="7" step="0.01" style={{ width: "100%" }} /></label>
                            <label>Partner age<br /><input name={`partner_age_${eventOption.id}`} type="number" min="1" max="120" style={{ width: "100%" }} /></label>
                            <label>Partner gender<br /><select name={`partner_gender_${eventOption.id}`} required={String(eventOption.gender_restriction || "ANY").toUpperCase() !== "ANY"} style={{ width: "100%" }}><option value="">Select</option><option value="Women">Women</option><option value="Men">Men</option><option value="Other">Other</option><option value="Prefer not to say">Prefer not to say</option></select></label>
                          </div>
                        ) : null}
                        {mode === "NEEDS_PARTNER" ? (
                          <label style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                            <input name={`show_on_partner_board_${eventOption.id}`} type="checkbox" disabled={!eventOption.partner_board_enabled} /> Show me on the public partner board for this event
                          </label>
                        ) : null}
                        <label>Partner note<br /><textarea name={`partner_note_${eventOption.id}`} rows={2} style={{ width: "100%" }} /></label>
                      </div>
                    ) : null}
                  </article>
                );
              })}
            </div>
          </div>
        ))}
        {!selectableEvents.length ? <p style={{ color: "#64748b" }}>No selectable events are currently open.</p> : null}
        <p><strong>Estimated total:</strong> ${totalPrice.toFixed(2)}</p>
      </section>

      <section style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Notes and policies</h2>
        <label>Notes for organizers<br /><textarea name="notes" rows={4} style={{ width: "100%" }} /></label>
        <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginTop: "0.75rem" }}>
          <input name="wants_partner_board_contact" type="checkbox" /> Organizers may use my contact info for partner-board coordination.
        </label>
        <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginTop: "0.75rem" }}>
          <input name="terms_accepted" type="checkbox" required /> I confirm this registration is accurate and agree to tournament policies.
        </label>
      </section>

      {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
      <button type="submit" disabled={pending || !selectableEvents.length} style={{ padding: "0.75rem 1rem", borderRadius: "10px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 700 }}>
        {pending ? "Submitting…" : "Submit registration"}
      </button>
    </form>
  );
}
