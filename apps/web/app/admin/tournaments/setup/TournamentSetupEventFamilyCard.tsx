"use client";

import { useId } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import {
  COMPETITION_FORMATS,
  GENDER_RESTRICTIONS,
  PARTICIPANT_TYPES,
  SCORING_OPTIONS,
  cleanString,
  dayLabel,
  dayReference,
  eventDayReference,
  eventFamilyName,
  numberInputValue,
  recordBoolean,
  setRecordNumber,
  setRecordString,
  type BuilderRow,
  type SetupRecord,
  type ValidationIssue
} from "../../tournament-setup/tournamentSetupBuilder";
import styles from "../../tournament-setup/TournamentSetupBuilder.module.css";

type Props = {
  row: BuilderRow;
  position: number;
  total: number;
  days: BuilderRow[];
  disabled: boolean;
  issues: ValidationIssue[];
  divisionCount: number;
  onChange: (value: SetupRecord) => void;
  onMove: (direction: -1 | 1) => void;
  onRemove: () => void;
};

function optionLabel(value: string): string {
  return value
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function optionsWithCurrent(options: readonly string[], current: string): string[] {
  return current && !options.includes(current) ? [current, ...options] : [...options];
}

export default function TournamentSetupEventFamilyCard({
  row,
  position,
  total,
  days,
  disabled,
  issues,
  divisionCount,
  onChange,
  onMove,
  onRemove
}: Props) {
  const issueId = useId();
  const value = row.value;
  const name = eventFamilyName(value);
  const currentDay = eventDayReference(value);
  const participantType = cleanString(value.participant_type) || "GENDER_DOUBLES";
  const gender = cleanString(value.gender_restriction) || "ANY";
  const drawFormat = cleanString(value.default_format) || "ROUND_ROBIN_PLUS_PLAYOFF";
  const scoring = cleanString(value.default_scoring) || "GAME_TO_15";
  const dayOptions = days.map((day) => ({
    value: dayReference(day.value),
    label: dayLabel(day.value) || dayReference(day.value),
    enabled: recordBoolean(day.value.enabled, true)
  }));

  function updateParticipantType(nextType: string) {
    const next = setRecordString(value, ["participant_type"], nextType);
    if (nextType === "SINGLES") {
      next.default_partner_board = false;
      next.gender_restriction = gender === "MIXED" ? "ANY" : gender;
    }
    onChange(next);
  }

  return (
    <fieldset
      className={styles.rowCard}
      aria-describedby={issues.length ? issueId : undefined}
    >
      <legend className={styles.legend}>
        Event {position + 1}: {name || "Untitled event"}
      </legend>

      <div className={styles.rowActions}>
        <button
          type="button"
          className={styles.smallButton}
          disabled={disabled || position === 0}
          onClick={() => onMove(-1)}
        >
          Move up
        </button>
        <button
          type="button"
          className={styles.smallButton}
          disabled={disabled || position === total - 1}
          onClick={() => onMove(1)}
        >
          Move down
        </button>
        <ConfirmAction
          triggerLabel="Remove event"
          title={`Remove ${name || `event ${position + 1}`}?`}
          description={
            divisionCount
              ? `This event still has ${divisionCount} division${divisionCount === 1 ? "" : "s"}. Reassign or remove those divisions first.`
              : "This removes the event from the setup draft. Published tournament data does not change until final review."
          }
          confirmLabel="Yes, remove event"
          cancelLabel="No, keep event"
          confirmationText=""
          tone="danger"
          disabled={disabled || divisionCount > 0}
          onConfirm={onRemove}
        />
      </div>

      <div className={styles.grid}>
        <label className={`${styles.label} ${styles.wide}`}>
          Event name
          <input
            className={styles.input}
            value={name}
            disabled={disabled}
            placeholder="Gender Doubles"
            onChange={(event) =>
              onChange(
                setRecordString(
                  value,
                  ["event_family", "event_family_label"],
                  event.target.value
                )
              )
            }
          />
        </label>

        <label className={styles.label}>
          Tournament day
          <select
            className={styles.select}
            value={currentDay}
            disabled={disabled}
            onChange={(event) =>
              onChange(
                setRecordString(
                  value,
                  ["registration_day_id", "assigned_day"],
                  event.target.value
                )
              )
            }
          >
            <option value="">Choose a day</option>
            {dayOptions.map((option) => (
              <option
                key={option.value}
                value={option.value}
                disabled={!option.enabled}
              >
                {option.label}
                {option.enabled ? "" : " (disabled)"}
              </option>
            ))}
          </select>
        </label>

        <label className={styles.label}>
          Participant type
          <select
            className={styles.select}
            value={participantType}
            disabled={disabled}
            onChange={(event) => updateParticipantType(event.target.value)}
          >
            {optionsWithCurrent(PARTICIPANT_TYPES, participantType).map((option) => (
              <option key={option} value={option}>
                {optionLabel(option)}
              </option>
            ))}
          </select>
        </label>

        <label className={styles.label}>
          Gender
          <select
            className={styles.select}
            value={gender}
            disabled={disabled}
            onChange={(event) =>
              onChange(
                setRecordString(
                  value,
                  ["gender_restriction"],
                  event.target.value
                )
              )
            }
          >
            {optionsWithCurrent(GENDER_RESTRICTIONS, gender).map((option) => (
              <option key={option} value={option}>
                {optionLabel(option)}
              </option>
            ))}
          </select>
        </label>

        <label className={styles.label}>
          Default draw format
          <select
            className={styles.select}
            value={drawFormat}
            disabled={disabled}
            onChange={(event) =>
              onChange(
                setRecordString(value, ["default_format"], event.target.value)
              )
            }
          >
            {optionsWithCurrent(COMPETITION_FORMATS, drawFormat).map((option) => (
              <option key={option} value={option}>
                {optionLabel(option)}
              </option>
            ))}
          </select>
        </label>

        <label className={styles.label}>
          Default scoring
          <select
            className={styles.select}
            value={scoring}
            disabled={disabled}
            onChange={(event) =>
              onChange(
                setRecordString(value, ["default_scoring"], event.target.value)
              )
            }
          >
            {optionsWithCurrent(SCORING_OPTIONS, scoring).map((option) => (
              <option key={option} value={option}>
                {optionLabel(option)}
              </option>
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
            onChange={(event) =>
              onChange(
                setRecordNumber(value, "default_capacity_teams", event.target.value)
              )
            }
          />
        </label>

        <label className={styles.label}>
          Default entry fee (USD)
          <input
            className={styles.input}
            type="number"
            inputMode="decimal"
            min="0"
            step="0.01"
            value={numberInputValue(value.default_price_usd)}
            disabled={disabled}
            onChange={(event) =>
              onChange(
                setRecordNumber(value, "default_price_usd", event.target.value)
              )
            }
          />
        </label>

        <label className={styles.checkbox}>
          <input
            type="checkbox"
            checked={recordBoolean(value.default_waitlist, true)}
            disabled={disabled}
            onChange={(event) =>
              onChange({ ...value, default_waitlist: event.target.checked })
            }
          />
          Waitlist enabled by default
        </label>

        <label className={styles.checkbox}>
          <input
            type="checkbox"
            checked={recordBoolean(value.default_partner_board, true)}
            disabled={disabled || participantType === "SINGLES"}
            onChange={(event) =>
              onChange({ ...value, default_partner_board: event.target.checked })
            }
          />
          Partner Board enabled by default
        </label>
      </div>

      <article
        style={{
          marginTop: "0.85rem",
          padding: "0.75rem",
          borderRadius: "10px",
          background: "#f8fafc",
          color: "#334155"
        }}
      >
        <strong>{divisionCount} division{divisionCount === 1 ? "" : "s"}</strong>
        <br />
        <small>
          These defaults apply when a new division is created. Existing divisions
          remain editable in Step 3.
        </small>
      </article>

      {issues.length ? (
        <ul id={issueId} className={styles.issues}>
          {issues.map((issue) => (
            <li key={`${issue.path}-${issue.message}`}>{issue.message}</li>
          ))}
        </ul>
      ) : null}
    </fieldset>
  );
}
