"use client";

import { useId } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import {
  cleanString,
  dayLabel,
  dayReference,
  eventDayReferences,
  eventFamilyName,
  numberInputValue,
  recordBoolean,
  type BuilderRow,
  type ValidationIssue
} from "../../tournament-setup/tournamentSetupBuilder";
import styles from "../../tournament-setup/TournamentSetupBuilder.module.css";
import {
  EVENT_AGE_POLICY_FIELDS,
  agePolicySummary,
  readAgePolicy
} from "./TournamentAgePolicyEditor";

type Props = {
  row: BuilderRow;
  position: number;
  days: BuilderRow[];
  disabled: boolean;
  issues: ValidationIssue[];
  divisionCount: number;
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

export default function TournamentSetupEventFamilyCard({
  row,
  position,
  days,
  disabled,
  issues,
  divisionCount,
  onEdit,
  onRemove
}: Props) {
  const issueId = useId();
  const value = row.value;
  const name = eventFamilyName(value);
  const dayNames = new Map(days.map((day) => [dayReference(day.value), dayLabel(day.value) || dayReference(day.value)]));
  const selectedDays = eventDayReferences(value).map((day) => dayNames.get(day) || day);
  const participantType = cleanString(value.participant_type) || "GENDER_DOUBLES";
  const competitionFormat = cleanString(value.competition_format).toUpperCase() || "STANDARD";
  const eventFormat = competitionFormat === "FOUR_PLAYER_TEAM"
    ? "Four-player team"
    : optionLabel(participantType);
  const gender = participantType === "MIXED_DOUBLES" ? "MIXED" : cleanString(value.gender_restriction) || "ANY";
  const agePolicy = readAgePolicy(value, EVENT_AGE_POLICY_FIELDS);

  return (
    <article className={styles.rowCard} aria-describedby={issues.length ? issueId : undefined}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "flex-start", flexWrap: "wrap" }}>
        <div>
          <h3 style={{ margin: 0 }}>Event {position + 1}: {name || "Untitled event"}</h3>
          <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>
            {selectedDays.join(", ") || "No tournament days"}
          </p>
        </div>
        <div className={styles.rowActions}>
          <button type="button" className={styles.smallButton} disabled={disabled} onClick={onEdit}>
            Edit
          </button>
          <ConfirmAction
            triggerLabel="Remove event"
            title={`Remove ${name || `event ${position + 1}`}?`}
            description={
              divisionCount
                ? `This event still has ${divisionCount} division${divisionCount === 1 ? "" : "s"}. Reassign or remove those divisions first.`
                : "This removes the event from the unpublished setup draft. Published tournament data remains unchanged until final review and publication."
            }
            confirmLabel="Yes, remove event"
            cancelLabel="No, keep event"
            confirmationText=""
            tone="danger"
            disabled={disabled || divisionCount > 0}
            onConfirm={onRemove}
          />
        </div>
      </div>
      <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.7rem", margin: "0.9rem 0 0" }}>
        <div><dt style={{ fontWeight: 800 }}>Event format</dt><dd style={{ margin: 0 }}>{eventFormat}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Gender</dt><dd style={{ margin: 0 }}>{optionLabel(gender)}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Default draw</dt><dd style={{ margin: 0 }}>{optionLabel(cleanString(value.default_format) || "ROUND_ROBIN_PLUS_PLAYOFF")}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Default scoring</dt><dd style={{ margin: 0 }}>{optionLabel(cleanString(value.default_scoring) || "GAME_TO_15")}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Age policy</dt><dd style={{ margin: 0 }}>{agePolicySummary(agePolicy)}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Capacity</dt><dd style={{ margin: 0 }}>{numberInputValue(value.default_capacity_teams) || "—"}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Entry fee</dt><dd style={{ margin: 0 }}>${Number(value.default_price_usd || 0).toFixed(2)}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Waitlist</dt><dd style={{ margin: 0 }}>{recordBoolean(value.default_waitlist, true) ? "Enabled" : "Disabled"}</dd></div>
        <div><dt style={{ fontWeight: 800 }}>Partner Board</dt><dd style={{ margin: 0 }}>{participantType !== "SINGLES" && recordBoolean(value.default_partner_board, true) ? "Enabled" : "Disabled"}</dd></div>
        {competitionFormat === "FOUR_PLAYER_TEAM" ? (
          <div><dt style={{ fontWeight: 800 }}>Team rules</dt><dd style={{ margin: 0 }}>2 men + 2 women · {optionLabel(cleanString(value.team_tiebreak_mode) || "SINGLES")} tiebreak · {recordBoolean(value.team_allow_substitutes, false) ? "Substitutes allowed" : "No substitutes"}</dd></div>
        ) : null}
        <div><dt style={{ fontWeight: 800 }}>Divisions</dt><dd style={{ margin: 0 }}>{divisionCount}</dd></div>
      </dl>
      {issues.length ? (
        <ul id={issueId} className={styles.issues}>
          {issues.map((issue) => <li key={`${issue.path}-${issue.message}`}>{issue.message}</li>)}
        </ul>
      ) : null}
    </article>
  );
}
