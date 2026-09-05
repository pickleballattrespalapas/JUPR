"use client";

import { useId } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess } from "@/components/interaction";
import {
  COMPETITION_FORMATS,
  DIVISION_STATUSES,
  GENDER_RESTRICTIONS,
  PARTICIPANT_TYPES,
  SCORING_OPTIONS,
  cleanString,
  editableString,
  eventFamilyName,
  numberInputValue,
  recordBoolean,
  setRecordNumber,
  setRecordString,
  type BuilderRow,
  type SetupRecord,
  type ValidationIssue
} from "./tournamentSetupBuilder";
import styles from "./TournamentSetupBuilder.module.css";

type TournamentSetupFamilyCardProps = {
  row: BuilderRow;
  position: number;
  total: number;
  disabled: boolean;
  issues: ValidationIssue[];
  onChange: (value: SetupRecord) => void;
  onMove: (direction: -1 | 1) => void;
  onRemove: () => void;
};

function optionsWithCurrent(options: readonly string[], current: string): string[] {
  return current && !options.includes(current) ? [current, ...options] : [...options];
}

export function TournamentSetupFamilyCard({
  row,
  position,
  total,
  disabled,
  issues,
  onChange,
  onMove,
  onRemove
}: TournamentSetupFamilyCardProps) {
  const issueId = useId();
  const value = row.value;
  const name = eventFamilyName(value);
  const editableName = editableString(value.event_family ?? value.event_family_label);
  const participantType = cleanString(value.participant_type) || "GENDER_DOUBLES";
  const gender = cleanString(value.gender_restriction) || "ANY";
  const format = cleanString(value.default_format) || "ROUND_ROBIN_PLUS_PLAYOFF";
  const scoring = cleanString(value.default_scoring) || "GAME_TO_15";
  const status = cleanString(value.default_status) || "open";

  return (
    <fieldset className={styles.rowCard} aria-describedby={issues.length ? issueId : undefined}>
      <legend className={styles.legend}>Event {position + 1}: {name || "Untitled event"}</legend>
      <div className={styles.rowActions}>
        <button
          type="button"
          className={styles.smallButton}
          disabled={disabled || position === 0}
          onClick={() => onMove(-1)}
          aria-label={`Move ${name || `event ${position + 1}`} earlier`}
        >
          Move up
        </button>
        <button
          type="button"
          className={styles.smallButton}
          disabled={disabled || position === total - 1}
          onClick={() => onMove(1)}
          aria-label={`Move ${name || `event ${position + 1}`} later`}
        >
          Move down
        </button>
        <ConfirmAction
          triggerLabel="Remove event"
          title={`Remove ${name || `event ${position + 1}`}?`}
          description="This removes the event defaults from the local draft. Divisions using this event must be removed or reassigned first."
          confirmLabel="Yes, remove event"
          cancelLabel="No, keep event"
          confirmationText=""
          tone="danger"
          disabled={disabled}
          onConfirm={async () => {
            onRemove();
            return actionSuccess("Event removed", "The event defaults were removed from the local draft.");
          }}
        />
      </div>
      <div className={styles.grid}>
        <label className={`${styles.label} ${styles.wide}`}>
          Event name
          <input
            className={styles.input}
            value={editableName}
            disabled={disabled}
            aria-invalid={issues.some((issue) => issue.path.endsWith(".event_family")) || undefined}
            onChange={(event) => onChange(setRecordString(value, ["event_family", "event_family_label"], event.target.value))}
          />
        </label>
        <label className={styles.label}>
          Participant type
          <select
            className={styles.select}
            value={participantType}
            disabled={disabled}
            onChange={(event) => onChange(setRecordString(value, ["participant_type"], event.target.value))}
          >
            {optionsWithCurrent(PARTICIPANT_TYPES, participantType).map((option) => (
              <option key={option} value={option}>{option.replaceAll("_", " ")}</option>
            ))}
          </select>
        </label>
        <label className={styles.label}>
          Gender
          <select
            className={styles.select}
            value={gender}
            disabled={disabled}
            onChange={(event) => onChange(setRecordString(value, ["gender_restriction"], event.target.value))}
          >
            {optionsWithCurrent(GENDER_RESTRICTIONS, gender).map((option) => (
              <option key={option} value={option}>{option.replaceAll("_", " ")}</option>
            ))}
          </select>
        </label>
        <label className={styles.label}>
          Default format
          <select
            className={styles.select}
            value={format}
            disabled={disabled}
            onChange={(event) => onChange(setRecordString(value, ["default_format"], event.target.value))}
          >
            {optionsWithCurrent(COMPETITION_FORMATS, format).map((option) => (
              <option key={option} value={option}>{option.replaceAll("_", " ")}</option>
            ))}
          </select>
        </label>
        <label className={styles.label}>
          Default scoring
          <select
            className={styles.select}
            value={scoring}
            disabled={disabled}
            onChange={(event) => onChange(setRecordString(value, ["default_scoring"], event.target.value))}
          >
            {optionsWithCurrent(SCORING_OPTIONS, scoring).map((option) => (
              <option key={option} value={option}>{option.replaceAll("_", " ")}</option>
            ))}
          </select>
        </label>
        <label className={styles.label}>
          Default capacity
          <input
            className={styles.input}
            type="number"
            min="1"
            step="1"
            value={numberInputValue(value.default_capacity_teams)}
            disabled={disabled}
            aria-invalid={issues.some((issue) => issue.path.endsWith(".default_capacity_teams")) || undefined}
            onChange={(event) => onChange(setRecordNumber(value, "default_capacity_teams", event.target.value))}
          />
        </label>
        <label className={styles.label}>
          Default price (USD)
          <input
            className={styles.input}
            type="number"
            min="0"
            step="0.01"
            value={numberInputValue(value.default_price_usd)}
            disabled={disabled}
            aria-invalid={issues.some((issue) => issue.path.endsWith(".default_price_usd")) || undefined}
            onChange={(event) => onChange(setRecordNumber(value, "default_price_usd", event.target.value))}
          />
        </label>
        <label className={styles.label}>
          Default registration status
          <select
            className={styles.select}
            value={status}
            disabled={disabled}
            onChange={(event) => onChange(setRecordString(value, ["default_status"], event.target.value))}
          >
            {optionsWithCurrent(DIVISION_STATUSES, status).map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </label>
        <label className={styles.checkbox}>
          <input
            type="checkbox"
            checked={recordBoolean(value.default_waitlist, true)}
            disabled={disabled}
            onChange={(event) => onChange({ ...value, default_waitlist: event.target.checked })}
          />
          Waitlist by default
        </label>
        <label className={styles.checkbox}>
          <input
            type="checkbox"
            checked={recordBoolean(value.default_partner_board, true)}
            disabled={disabled}
            onChange={(event) => onChange({ ...value, default_partner_board: event.target.checked })}
          />
          Players Needing Partners by default
        </label>
      </div>
      {issues.length ? (
        <ul id={issueId} className={styles.issues}>
          {issues.map((issue) => <li key={`${issue.path}-${issue.message}`}>{issue.message}</li>)}
        </ul>
      ) : null}
    </fieldset>
  );
}
