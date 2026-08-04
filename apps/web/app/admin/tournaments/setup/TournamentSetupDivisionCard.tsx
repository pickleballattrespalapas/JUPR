"use client";

import { useId } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import {
  cleanString,
  dayLabel,
  dayReference,
  eventAgeMode,
  eventDayReferences,
  eventDivisionName,
  eventFamilyDefaults,
  eventFamilyName,
  numberInputValue,
  recordBoolean,
  type BuilderRow,
  type ValidationIssue
} from "../../tournament-setup/tournamentSetupBuilder";
import styles from "../../tournament-setup/TournamentSetupBuilder.module.css";

type Props = {
  row: BuilderRow;
  position: number;
  eventFamilies: BuilderRow[];
  days: BuilderRow[];
  disabled: boolean;
  issues: ValidationIssue[];
  onEdit: () => void;
  onRemove: () => void;
};

function optionLabel(value: string): string {
  return value
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function divisionFormat(value: Record<string, unknown>): string {
  if (cleanString(value.competition_format).toUpperCase() === "FOUR_PLAYER_TEAM") return "Four-player team";
  if (cleanString(value.eligibility_mode).toUpperCase() === "COMBINED_RATING_CAP") {
    return `Combined-rating doubles · cap ${numberInputValue(value.combined_rating_cap) || "not set"}`;
  }
  return "Standard singles or doubles";
}

export default function TournamentSetupDivisionCard({
  row,
  position,
  eventFamilies,
  days,
  disabled,
  issues,
  onEdit,
  onRemove
}: Props) {
  const issueId = useId();
  const value = row.value;
  const name = eventDivisionName(value);
  const family = eventFamilyName(value);
  const familyDefaults = eventFamilyDefaults(eventFamilies, family) || {};
  const dayNames = new Map(days.map((day) => [dayReference(day.value), dayLabel(day.value) || dayReference(day.value)]));
  const selectedDays = eventDayReferences(value).map((day) => dayNames.get(day) || day);
  const participantType = cleanString(value.event_type || value.participant_type || familyDefaults.participant_type) || "GENDER_DOUBLES";
  const gender = participantType === "MIXED_DOUBLES"
    ? "MIXED"
    : cleanString(value.gender_restriction || familyDefaults.gender_restriction) || "ANY";

  return (
    <article className={styles.rowCard} aria-describedby={issues.length ? issueId : undefined}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "flex-start", flexWrap: "wrap" }}>
        <div>
          <h3 style={{ margin: 0 }}>Division {position + 1}: {name || "Untitled division"}</h3>
          <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>
            Parent event: {family || "None"} · {selectedDays.join(", ") || "No tournament days"}
          </p>
        </div>
        <div className={styles.rowActions}>
          <button type="button" className={styles.smallButton} disabled={disabled} onClick={onEdit}>
            Edit
          </button>
          <ConfirmAction
            triggerLabel="Remove division"
            title={`Remove ${name || `division ${position + 1}`}?`}
            description="This removes the division from the unpublished setup draft. Published tournament data remains unchanged until final review and publication."
            confirmLabel="Yes, remove division"
            confirmationText=""
            tone="danger"
            disabled={disabled}
            onConfirm={onRemove}
          />
        </div>
      </div>
      <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.7rem", margin: "0.9rem 0 0" }}>
        <div><dt style={{ fontWeight: 800 }}>Skill</dt><dd style={{ margin: 0 }}>{cleanString(value.skill_label) || "Open"}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Age</dt><dd style={{ margin: 0 }}>{cleanString(value.age_label) || optionLabel(eventAgeMode(value))}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Participant type</dt><dd style={{ margin: 0 }}>{optionLabel(participantType)}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Gender</dt><dd style={{ margin: 0 }}>{optionLabel(gender)}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Division format</dt><dd style={{ margin: 0 }}>{divisionFormat(value)}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Capacity</dt><dd style={{ margin: 0 }}>{numberInputValue(value.capacity_teams) || "—"}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Entry fee</dt><dd style={{ margin: 0 }}>${Number(value.price_usd || 0).toFixed(2)}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Waitlist</dt><dd style={{ margin: 0 }}>{recordBoolean(value.waitlist_enabled, true) ? "Enabled" : "Disabled"}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Partner Board</dt><dd style={{ margin: 0 }}>{participantType !== "SINGLES" && recordBoolean(value.partner_board_enabled, true) ? "Enabled" : "Disabled"}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Draw override</dt><dd style={{ margin: 0 }}>{cleanString(value.event_format_override || value.division_format) || "Use event default"}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Scoring override</dt><dd style={{ margin: 0 }}>{cleanString(value.scoring_override || value.division_scoring) || "Use event default"}</dd></div>
      </dl>
      {cleanString(value.division_notes || value.notes) ? (
        <p style={{ marginBottom: 0 }}><strong>Notes:</strong> {cleanString(value.division_notes || value.notes)}</p>
      ) : null}
      {issues.length ? (
        <ul id={issueId} className={styles.issues}>
          {issues.map((issue) => <li key={`${issue.path}-${issue.message}`}>{issue.message}</li>)}
        </ul>
      ) : null}
    </article>
  );
}
