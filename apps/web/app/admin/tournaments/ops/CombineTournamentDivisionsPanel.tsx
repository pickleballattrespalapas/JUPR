"use client";

import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type { ActionCompletion } from "@/components/interaction";
import type { AdminTournamentDraw } from "@/lib/adminTournamentApi";

type EventOption = Record<string, unknown>;
type Props = {
  draw: AdminTournamentDraw;
  draws: AdminTournamentDraw[];
  eventOptions: EventOption[];
  hasGames: boolean;
  disabled: boolean;
  busy: boolean;
  onCombine: (eventOptionIds: string[], name: string, confirmation: string) => Promise<ActionCompletion>;
};

const normalized = (value: unknown) => String(value || "").trim().replace(/\s+/g, " ").toLowerCase();
const truthy = (value: unknown) => ["true", "1", "yes", "on"].includes(normalized(value));
const inactiveStatuses = new Set(["disabled", "cancelled", "canceled", "inactive", "archived", "deleted", "void", "voided"]);

export function drawSourceDivisionIds(draw: AdminTournamentDraw): string[] {
  return [...new Set([String(draw.event_option_id || ""), ...(draw.source_event_option_ids || [])].filter(Boolean))];
}

export function sourceDivisionLabel(event: EventOption): string {
  const family = String(event.event_family_label || "").trim();
  const division = String(event.division_name || event.label || "").trim();
  return family && division && family !== division ? `${family} / ${division}` : division || family || "Division";
}

function familyKey(event: EventOption): string {
  return JSON.stringify([
    normalized(event.event_family_key || event.event_family_label),
    normalized(event.event_type),
    normalized(event.gender_restriction || "ANY"),
    truthy(event.partner_required),
    normalized(event.competition_format || "STANDARD"),
  ]);
}

export function compatibleDivision(event: EventOption, primary: EventOption, dayId: string): boolean {
  const scheduledDays = event.scheduled_day_ids;
  return event.enabled === true
    && !inactiveStatuses.has(normalized(event.status))
    && normalized(event.competition_format || "STANDARD") === "standard"
    && ![event.eligibility_mode, event.skill_mode].some(mode => normalized(mode) === "combined_rating_cap")
    && familyKey(event) === familyKey(primary)
    && String(event.registration_day_id || "") === dayId
    && (scheduledDays == null || (Array.isArray(scheduledDays) && (!scheduledDays.length || scheduledDays.includes(dayId))));
}

function suggestedName(events: EventOption[]): string {
  const family = String(events[0]?.event_family_label || "").trim();
  const divisions = events.map(event => String(event.division_name || event.label || "Division").trim());
  return `${family ? `${family} / ` : ""}${divisions.join(" / ")}`;
}

export default function CombineTournamentDivisionsPanel({ draw, draws, eventOptions, hasGames, disabled, busy, onCombine }: Props) {
  const savedIds = drawSourceDivisionIds(draw);
  const [addedIds, setAddedIds] = useState<string[]>([]);
  const [editedName, setEditedName] = useState<string | null>(null);
  const primary = eventOptions.find(event => String(event.id || "") === draw.event_option_id);
  const dayId = String(draw.registration_day_id || primary?.registration_day_id || "");
  const selectedIds = [...savedIds, ...addedIds];
  const selectedEvents = selectedIds.map(id => eventOptions.find(event => String(event.id || "") === id)).filter((event): event is EventOption => Boolean(event));
  const combinedName = editedName ?? suggestedName(selectedEvents);
  const candidates = primary && dayId ? eventOptions.filter(event => !savedIds.includes(String(event.id || "")) && compatibleDivision(event, primary, dayId)) : [];
  const lockedReason = hasGames ? "Divisions cannot be combined after games have been generated."
    : draw.pool_config ? "Remove and save this draw’s pool split before combining divisions."
    : !["DRAFT", "READY"].includes(String(draw.status || "DRAFT").toUpperCase()) ? "Select a draft or ready draw before combining divisions."
    : !primary || !dayId ? "Select a draw belonging to a registration division and tournament day."
    : !compatibleDivision(primary, primary, dayId) || String(draw.draw_kind || "STANDARD").toUpperCase() !== "STANDARD" ? "Combining requires an enabled standard division scheduled on this day. Combined-rating-cap events cannot be combined."
    : null;
  const controlsDisabled = disabled || busy || Boolean(lockedReason);
  const blockedByCombinedDraw = (id: string) => draws.some(other => other.id !== draw.id && !other.combined_into_draw_id && !other.hidden_from_primary_ops && drawSourceDivisionIds(other).length > 1 && drawSourceDivisionIds(other).includes(id));

  function toggleDivision(id: string, checked: boolean) {
    setAddedIds(current => checked ? [...current, id] : current.filter(value => value !== id));
  }

  return (
    <article data-testid="combine-tournament-divisions" style={{ border: "1px solid #cbd5e1", borderRadius: "14px", padding: "1rem", background: "#f8fafc" }}>
      <h2 style={{ marginTop: 0 }}>Combine divisions</h2>
      <p>Play multiple registration divisions in one draw. Players keep their original registration division. The combined draw has one schedule, optional pools, shared playoffs, and one podium.</p>
      <fieldset disabled={controlsDisabled} style={{ border: 0, padding: 0, margin: 0 }}>
        <legend style={{ fontWeight: 700, marginBottom: "0.5rem" }}>Registration divisions in this draw</legend>
        <div style={{ display: "grid", gap: "0.5rem" }}>
          {savedIds.map(id => {
            const event = eventOptions.find(row => String(row.id || "") === id);
            return <label key={id}><input type="checkbox" checked disabled /> {event ? sourceDivisionLabel(event) : "Saved division"} <span style={{ color: "#475569" }}>(already included)</span></label>;
          })}
          {candidates.map(event => {
            const id = String(event.id || "");
            const blocked = blockedByCombinedDraw(id);
            return <label key={id}><input type="checkbox" checked={addedIds.includes(id)} disabled={controlsDisabled || blocked || (selectedIds.length >= 16 && !addedIds.includes(id))} onChange={e => toggleDivision(id, e.target.checked)} /> {sourceDivisionLabel(event)}{blocked ? <span style={{ color: "#92400e" }}> — already belongs to another combined draw; select that draw to add divisions</span> : null}</label>;
          })}
        </div>
        {!candidates.length ? <p style={{ color: "#475569" }}>No additional compatible divisions are available on this day. Divisions must use the same event family and player format.</p> : null}
        <label style={{ display: "block", marginTop: "1rem", fontWeight: 700 }}>Combined draw name
          <input value={combinedName} maxLength={180} onChange={event => setEditedName(event.target.value)} style={{ display: "block", width: "100%", maxWidth: "36rem", marginTop: "0.25rem", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" }} />
        </label>
      </fieldset>
      {combinedName.trim().length > 180 ? <p role="alert">Shorten the combined draw name to 180 characters or fewer.</p> : null}
      {lockedReason ? <p role="status" style={{ color: "#92400e" }}>{lockedReason}</p> : <p style={{ color: "#475569" }}>Existing teams in the selected draft draws will move into this draw. Then import confirmed registrations to include any teams not yet imported. Combining is blocked if any selected draw has games, saved pools, or day-live activity.</p>}
      <ConfirmAction
        triggerLabel="Combine selected divisions"
        title="Combine these divisions into one draw?"
        description={`Combine ${selectedEvents.map(sourceDivisionLabel).join(" + ")} as “${combinedName.trim()}”. Existing draft rosters move into this draw, and the other draft draws are retired. Original registrations remain unchanged. All teams compete for one shared podium.`}
        confirmLabel="Yes, combine divisions"
        confirmationText="COMBINE DIVISIONS"
        disabled={controlsDisabled || !addedIds.length || !combinedName.trim() || combinedName.trim().length > 180 || addedIds.some(blockedByCombinedDraw)}
        busy={busy}
        onConfirm={confirmation => onCombine(selectedIds, combinedName.trim(), confirmation)}
      />
    </article>
  );
}
