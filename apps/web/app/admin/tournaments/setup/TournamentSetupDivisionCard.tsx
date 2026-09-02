"use client";

import { useId } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess } from "@/components/interaction";
import {
  cleanString,
  dayLabel,
  dayReference,
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
import { skillEligibilityLabel, skillEligibilitySummary } from "@/lib/tournamentSkillEligibility";
import {
  DIVISION_AGE_POLICY_FIELDS,
  EVENT_AGE_POLICY_FIELDS,
  agePolicySummary,
  readAgePolicy
} from "./TournamentAgePolicyEditor";

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
  const competitionFormat = cleanString(familyDefaults.competition_format || value.competition_format).toUpperCase() || "STANDARD";
  const eventFormat = competitionFormat === "FOUR_PLAYER_TEAM" ? "Four-player team" : optionLabel(participantType);
  const gender = participantType === "MIXED_DOUBLES"
    ? "MIXED"
    : cleanString(value.gender_restriction || familyDefaults.gender_restriction) || "ANY";
  const agePolicySource = cleanString(value.age_policy_source).toUpperCase() === "OVERRIDE" ? "OVERRIDE" : "INHERIT_EVENT";
  const agePolicy = agePolicySource === "OVERRIDE"
    ? readAgePolicy(value, DIVISION_AGE_POLICY_FIELDS)
    : readAgePolicy(familyDefaults, EVENT_AGE_POLICY_FIELDS);
  const drawOverride = cleanString(value.event_format_override || value.division_format);
  const scoringOverride = cleanString(value.scoring_override || value.division_scoring);
  const resolvedDraw = drawOverride || cleanString(familyDefaults.default_format) || cleanString(value.event_format_default) || "ROUND_ROBIN_PLUS_PLAYOFF";
  const resolvedScoring = scoringOverride || cleanString(familyDefaults.default_scoring) || cleanString(value.scoring_default) || "GAME_TO_15";

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
            onConfirm={async () => {
              onRemove();
              return actionSuccess("Division removed", "The division was removed from the unpublished setup draft.");
            }}
          />
        </div>
      </div>
      <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.7rem", margin: "0.9rem 0 0" }}>
        <div><dt style={{ fontWeight: 800 }}>Skill</dt><dd style={{ margin: 0 }}>{skillEligibilityLabel(value)}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Age policy</dt><dd style={{ margin: 0 }}>{agePolicySummary(agePolicy)}{agePolicySource === "INHERIT_EVENT" ? " (from event)" : " (division override)"}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Event format</dt><dd style={{ margin: 0 }}>{eventFormat}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Gender</dt><dd style={{ margin: 0 }}>{optionLabel(gender)}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Division eligibility</dt><dd style={{ margin: 0 }}>{skillEligibilitySummary(value)}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Capacity</dt><dd style={{ margin: 0 }}>{numberInputValue(value.capacity_teams) || "—"}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Entry fee</dt><dd style={{ margin: 0 }}>${Number(value.price_usd || 0).toFixed(2)}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Waitlist</dt><dd style={{ margin: 0 }}>{recordBoolean(value.waitlist_enabled, true) ? "Enabled" : "Disabled"}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Partner Board</dt><dd style={{ margin: 0 }}>{participantType !== "SINGLES" && recordBoolean(value.partner_board_enabled, true) ? "Enabled" : "Disabled"}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Draw format</dt><dd style={{ margin: 0 }}>{optionLabel(resolvedDraw)}{drawOverride ? " (division override)" : " (from event)"}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Scoring</dt><dd style={{ margin: 0 }}>{optionLabel(resolvedScoring)}{scoringOverride ? " (division override)" : " (from event)"}</dd></div>
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
