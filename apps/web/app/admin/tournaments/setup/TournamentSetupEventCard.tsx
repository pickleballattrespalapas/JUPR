"use client";

import { useId } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import {
  COMPETITION_FORMATS,
  DIVISION_STATUSES,
  GENDER_RESTRICTIONS,
  PARTICIPANT_TYPES,
  SCORING_OPTIONS,
  SKILL_LABEL_OPTIONS,
  cleanString,
  dayLabel,
  dayReference,
  eventDayReference,
  eventDivisionName,
  eventFamilyName,
  eventUsesLabelDayReference,
  numberInputValue,
  recordBoolean,
  setRecordNumber,
  setRecordString,
  type BuilderRow,
  type SetupRecord,
  type ValidationIssue
} from "../../../../tournament-setup/tournamentSetupBuilder";
import styles from "../../../../tournament-setup/TournamentSetupBuilder.module.css";

type Props = {
  row: BuilderRow;
  position: number;
  total: number;
  days: BuilderRow[];
  familyNames: string[];
  disabled: boolean;
  issues: ValidationIssue[];
  onChange: (value: SetupRecord) => void;
  onMove: (direction: -1 | 1) => void;
  onRemove: () => void;
};

type EventMode = "STANDARD" | "COMBINED_RATING_CAP" | "FOUR_PLAYER_TEAM";

const EVENT_MODES: Array<[EventMode, string]> = [
  ["STANDARD", "Standard singles or doubles"],
  ["COMBINED_RATING_CAP", "Combined-rating doubles"],
  ["FOUR_PLAYER_TEAM", "Four-player team"]
];

const TIEBREAK_OPTIONS = [
  ["SINGLES", "One singles game"],
  ["SKINNY_RELAY", "Skinny-singles relay"]
] as const;

const PLAYOFF_OPTIONS = [
  ["NONE", "No playoffs"],
  ["TOP_2_FINAL", "Top two final"],
  ["TOP_4_SEMIFINALS", "Top four semifinals and final"],
  ["TOP_4_SEMIFINALS_WITH_BRONZE", "Top four with bronze match"]
] as const;

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

function eventMode(value: SetupRecord): EventMode {
  if (cleanString(value.competition_format).toUpperCase() === "FOUR_PLAYER_TEAM") {
    return "FOUR_PLAYER_TEAM";
  }
  if (cleanString(value.eligibility_mode).toUpperCase() === "COMBINED_RATING_CAP") {
    return "COMBINED_RATING_CAP";
  }
  return "STANDARD";
}

function applyEventMode(value: SetupRecord, mode: EventMode): SetupRecord {
  if (mode === "COMBINED_RATING_CAP") {
    return {
      ...value,
      eligibility_mode: "COMBINED_RATING_CAP",
      combined_rating_cap:
        Number(value.combined_rating_cap) > 0 ? Number(value.combined_rating_cap) : 8,
      competition_format: "STANDARD",
      team_roster_size: 2,
      team_gender_rule: "NONE",
      team_tiebreak_mode: "SINGLES",
      team_playoff_format: "NONE",
      team_allow_substitutes: false
    };
  }
  if (mode === "FOUR_PLAYER_TEAM") {
    return {
      ...value,
      eligibility_mode: "STANDARD",
      combined_rating_cap: null,
      competition_format: "FOUR_PLAYER_TEAM",
      team_roster_size: 4,
      team_gender_rule: "TWO_MEN_TWO_WOMEN",
      team_tiebreak_mode: ["SINGLES", "SKINNY_RELAY"].includes(
        cleanString(value.team_tiebreak_mode).toUpperCase()
      )
        ? cleanString(value.team_tiebreak_mode).toUpperCase()
        : "SINGLES",
      team_playoff_format: [
        "NONE",
        "TOP_2_FINAL",
        "TOP_4_SEMIFINALS",
        "TOP_4_SEMIFINALS_WITH_BRONZE"
      ].includes(cleanString(value.team_playoff_format).toUpperCase())
        ? cleanString(value.team_playoff_format).toUpperCase()
        : "NONE",
      team_allow_substitutes: recordBoolean(value.team_allow_substitutes, false)
    };
  }
  return {
    ...value,
    eligibility_mode: "STANDARD",
    combined_rating_cap: null,
    competition_format: "STANDARD",
    team_roster_size: 2,
    team_gender_rule: "NONE",
    team_tiebreak_mode: "SINGLES",
    team_playoff_format: "NONE",
    team_allow_substitutes: false
  };
}

function eventModeSummary(value: SetupRecord): string {
  const mode = eventMode(value);
  if (mode === "COMBINED_RATING_CAP") {
    return `Combined-rating doubles · cap ${numberInputValue(value.combined_rating_cap) || "not set"}`;
  }
  if (mode === "FOUR_PLAYER_TEAM") {
    const substitutes = recordBoolean(value.team_allow_substitutes, false)
      ? "substitutes allowed"
      : "no substitutes";
    return `Four-player team · 2 men/2 women · ${optionLabel(
      cleanString(value.team_tiebreak_mode) || "SINGLES"
    )} · ${optionLabel(cleanString(value.team_playoff_format) || "NONE")} · ${substitutes}`;
  }
  return "Standard singles or doubles";
}

export default function TournamentSetupEventCard({
  row,
  position,
  total,
  days,
  familyNames,
  disabled,
  issues,
  onChange,
  onMove,
  onRemove
}: Props) {
  const issueId = useId();
  const value = row.value;
  const name = eventDivisionName(value);
  const family = eventFamilyName(value);
  const currentDay = eventDayReference(value);
  const usesDayLabel = eventUsesLabelDayReference(value);
  const dayOptions = days.map((day) => ({
    value: usesDayLabel ? dayLabel(day.value) : dayReference(day.value),
    label: dayLabel(day.value) || dayReference(day.value),
    enabled: recordBoolean(day.value.enabled, true)
  }));
  const dayOptionValues = new Set(dayOptions.map((option) => option.value));
  const participantType = cleanString(value.event_type || value.participant_type) || "GENDER_DOUBLES";
  const gender = cleanString(value.gender_restriction) || "ANY";
  const skill = cleanString(value.skill_label) || "Open";
  const drawFormat = cleanString(
    value.event_format_override ||
      value.division_format ||
      value.event_format_default
  );
  const scoring = cleanString(
    value.scoring_override || value.division_scoring || value.scoring_default
  );
  const status = cleanString(value.status) || "open";
  const mode = eventMode(value);

  function updateParticipantType(nextType: string) {
    const next = setRecordString(value, ["event_type", "participant_type"], nextType);
    next.partner_required = nextType !== "SINGLES";
    if (nextType === "SINGLES") {
      next.partner_board_enabled = false;
      next.public_partner_board = false;
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
          description="This removes the event from the setup draft. Nothing changes on the published tournament until the final review step."
          confirmLabel="Yes, remove event"
          cancelLabel="No, keep event"
          confirmationText=""
          tone="danger"
          disabled={disabled}
          onConfirm={onRemove}
        />
      </div>

      <div className={styles.grid}>
        <label className={`${styles.label} ${styles.wide}`}>
          Event or division name
          <input
            className={styles.input}
            value={name}
            disabled={disabled}
            onChange={(event) =>
              onChange(
                setRecordString(
                  value,
                  ["division_name", "label"],
                  event.target.value
                )
              )
            }
          />
        </label>

        <label className={styles.label}>
          Event family
          <input
            className={styles.input}
            list={`event-families-${row.key}`}
            value={family}
            disabled={disabled}
            placeholder="Doubles, Singles, Mixed…"
            onChange={(event) =>
              onChange(
                setRecordString(
                  value,
                  ["event_family_label", "event_family"],
                  event.target.value
                )
              )
            }
          />
          <datalist id={`event-families-${row.key}`}>
            {familyNames.map((option) => (
              <option key={option} value={option} />
            ))}
          </datalist>
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
                  usesDayLabel ? ["assigned_day"] : ["registration_day_id"],
                  event.target.value
                )
              )
            }
          >
            {!currentDay ? <option value="">Choose a day</option> : null}
            {currentDay && !dayOptionValues.has(currentDay) ? (
              <option value={currentDay}>{currentDay} (unavailable)</option>
            ) : null}
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

        <label className={`${styles.label} ${styles.wide}`}>
          Event format
          <select
            className={styles.select}
            value={mode}
            disabled={disabled}
            onChange={(event) =>
              onChange(applyEventMode(value, event.target.value as EventMode))
            }
          >
            {EVENT_MODES.map(([option, label]) => (
              <option key={option} value={option}>
                {label}
              </option>
            ))}
          </select>
        </label>

        {mode === "COMBINED_RATING_CAP" ? (
          <label className={styles.label}>
            Maximum combined rating
            <input
              className={styles.input}
              type="number"
              inputMode="decimal"
              min="0.01"
              max="14"
              step="0.01"
              value={numberInputValue(value.combined_rating_cap)}
              disabled={disabled}
              onChange={(event) =>
                onChange(
                  setRecordNumber(value, "combined_rating_cap", event.target.value)
                )
              }
            />
          </label>
        ) : null}

        {mode === "FOUR_PLAYER_TEAM" ? (
          <>
            <label className={styles.label}>
              Tied after four games
              <select
                className={styles.select}
                value={cleanString(value.team_tiebreak_mode) || "SINGLES"}
                disabled={disabled}
                onChange={(event) =>
                  onChange(
                    setRecordString(
                      value,
                      ["team_tiebreak_mode"],
                      event.target.value
                    )
                  )
                }
              >
                {TIEBREAK_OPTIONS.map(([option, label]) => (
                  <option key={option} value={option}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <label className={styles.label}>
              Playoffs
              <select
                className={styles.select}
                value={cleanString(value.team_playoff_format) || "NONE"}
                disabled={disabled}
                onChange={(event) =>
                  onChange(
                    setRecordString(
                      value,
                      ["team_playoff_format"],
                      event.target.value
                    )
                  )
                }
              >
                {PLAYOFF_OPTIONS.map(([option, label]) => (
                  <option key={option} value={option}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <label className={styles.checkbox}>
              <input
                type="checkbox"
                checked={recordBoolean(value.team_allow_substitutes, false)}
                disabled={disabled}
                onChange={(event) =>
                  onChange({
                    ...value,
                    team_allow_substitutes: event.target.checked
                  })
                }
              />
              Allow substitutes
            </label>
          </>
        ) : null}

        <label className={styles.label}>
          Participant type
          <select
            className={styles.select}
            value={participantType}
            disabled={disabled || mode === "FOUR_PLAYER_TEAM"}
            onChange={(event) => updateParticipantType(event.target.value)}
          >
            {optionsWithCurrent(PARTICIPANT_TYPES, participantType).map(
              (option) => (
                <option key={option} value={option}>
                  {optionLabel(option)}
                </option>
              )
            )}
          </select>
        </label>

        <label className={styles.label}>
          Gender
          <select
            className={styles.select}
            value={mode === "FOUR_PLAYER_TEAM" ? "MIXED" : gender}
            disabled={disabled || mode === "FOUR_PLAYER_TEAM"}
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
          Skill level
          <input
            className={styles.input}
            list={`skills-${row.key}`}
            value={skill}
            disabled={disabled}
            onChange={(event) =>
              onChange(
                setRecordString(value, ["skill_label"], event.target.value)
              )
            }
          />
          <datalist id={`skills-${row.key}`}>
            {SKILL_LABEL_OPTIONS.map((option) => (
              <option key={option} value={option} />
            ))}
          </datalist>
        </label>

        <label className={styles.label}>
          Capacity
          <input
            className={styles.input}
            type="number"
            min="1"
            step="1"
            value={numberInputValue(value.capacity_teams)}
            disabled={disabled}
            onChange={(event) =>
              onChange(
                setRecordNumber(value, "capacity_teams", event.target.value)
              )
            }
          />
        </label>

        <label className={styles.label}>
          Entry fee (USD)
          <input
            className={styles.input}
            type="number"
            inputMode="decimal"
            min="0"
            step="0.01"
            value={numberInputValue(value.price_usd)}
            disabled={disabled}
            onChange={(event) =>
              onChange(setRecordNumber(value, "price_usd", event.target.value))
            }
          />
        </label>

        <label className={styles.label}>
          Registration status
          <select
            className={styles.select}
            value={status}
            disabled={disabled}
            onChange={(event) =>
              onChange(
                setRecordString(value, ["status"], event.target.value)
              )
            }
          >
            {optionsWithCurrent(DIVISION_STATUSES, status).map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
      </div>

      <details style={{ marginTop: "0.85rem" }}>
        <summary style={{ cursor: "pointer", fontWeight: 800 }}>
          Scoring and registration options
        </summary>
        <div className={styles.grid} style={{ marginTop: "0.75rem" }}>
          <label className={styles.label}>
            Draw format
            <select
              className={styles.select}
              value={drawFormat}
              disabled={disabled}
              onChange={(event) =>
                onChange(
                  setRecordString(
                    value,
                    [
                      "event_format_override",
                      "division_format",
                      "event_format_default"
                    ],
                    event.target.value
                  )
                )
              }
            >
              <option value="">Use default</option>
              {optionsWithCurrent(COMPETITION_FORMATS, drawFormat)
                .filter(Boolean)
                .map((option) => (
                  <option key={option} value={option}>
                    {optionLabel(option)}
                  </option>
                ))}
            </select>
          </label>

          <label className={styles.label}>
            Scoring
            <select
              className={styles.select}
              value={scoring}
              disabled={disabled}
              onChange={(event) =>
                onChange(
                  setRecordString(
                    value,
                    ["scoring_override", "division_scoring", "scoring_default"],
                    event.target.value
                  )
                )
              }
            >
              <option value="">Use default</option>
              {optionsWithCurrent(SCORING_OPTIONS, scoring)
                .filter(Boolean)
                .map((option) => (
                  <option key={option} value={option}>
                    {optionLabel(option)}
                  </option>
                ))}
            </select>
          </label>

          <label className={styles.checkbox}>
            <input
              type="checkbox"
              checked={recordBoolean(value.waitlist_enabled, true)}
              disabled={disabled}
              onChange={(event) =>
                onChange({ ...value, waitlist_enabled: event.target.checked })
              }
            />
            Waitlist enabled
          </label>

          <label className={styles.checkbox}>
            <input
              type="checkbox"
              checked={recordBoolean(value.partner_board_enabled, true)}
              disabled={disabled || participantType === "SINGLES"}
              onChange={(event) =>
                onChange({
                  ...value,
                  partner_board_enabled: event.target.checked,
                  public_partner_board: event.target.checked
                })
              }
            />
            Partner Board enabled
          </label>

          <label className={styles.checkbox}>
            <input
              type="checkbox"
              checked={recordBoolean(value.enabled, true)}
              disabled={disabled}
              onChange={(event) =>
                onChange({ ...value, enabled: event.target.checked })
              }
            />
            Visible to registrants
          </label>
        </div>
      </details>

      <article
        style={{
          marginTop: "0.85rem",
          padding: "0.75rem",
          borderRadius: "10px",
          background: "#f8fafc",
          color: "#334155"
        }}
      >
        <strong>Format summary</strong>
        <br />
        {eventModeSummary(value)}
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
