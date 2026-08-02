"use client";

import { useId } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import {
  AGE_MODES,
  COMPETITION_FORMATS,
  SCORING_OPTIONS,
  SKILL_LABEL_OPTIONS,
  ageRuleValue,
  cleanString,
  eventAgeMode,
  eventDayReference,
  eventFamilyDefaults,
  eventFamilyName,
  eventDivisionName,
  numberInputValue,
  recordBoolean,
  setAgeRuleNumber,
  setEventAgeMode,
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
  eventFamilies: BuilderRow[];
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

function modeSummary(value: SetupRecord): string {
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

function applyFamily(value: SetupRecord, eventFamilies: BuilderRow[], familyName: string): SetupRecord {
  const defaults = eventFamilyDefaults(eventFamilies, familyName) || {};
  const day = eventDayReference(defaults);
  const next: SetupRecord = {
    ...value,
    event_family_label: familyName,
    event_family: familyName,
    registration_day_id: day || value.registration_day_id,
    assigned_day: day || value.assigned_day,
    event_type: cleanString(defaults.participant_type) || value.event_type,
    participant_type: cleanString(defaults.participant_type) || value.participant_type,
    gender_restriction:
      cleanString(defaults.gender_restriction) || value.gender_restriction,
    waitlist_enabled: recordBoolean(
      value.waitlist_enabled,
      recordBoolean(defaults.default_waitlist, true)
    ),
    partner_board_enabled: recordBoolean(
      value.partner_board_enabled,
      recordBoolean(defaults.default_partner_board, true)
    )
  };
  if (value.capacity_teams == null && defaults.default_capacity_teams != null) {
    next.capacity_teams = defaults.default_capacity_teams;
  }
  if (value.price_usd == null && defaults.default_price_usd != null) {
    next.price_usd = defaults.default_price_usd;
  }
  return next;
}

export default function TournamentSetupDivisionCard({
  row,
  position,
  total,
  eventFamilies,
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
  const mode = eventMode(value);
  const ageMode = eventAgeMode(value);
  const skill = cleanString(value.skill_label) || "Open";
  const drawFormat = cleanString(
    value.event_format_override || value.division_format
  );
  const scoring = cleanString(value.scoring_override || value.division_scoring);
  const familyDefaults = eventFamilyDefaults(eventFamilies, family) || {};
  const eventSummary = [
    family || "No event selected",
    cleanString(familyDefaults.participant_type)
      ? optionLabel(cleanString(familyDefaults.participant_type))
      : "",
    cleanString(familyDefaults.gender_restriction)
      ? optionLabel(cleanString(familyDefaults.gender_restriction))
      : ""
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <fieldset
      className={styles.rowCard}
      aria-describedby={issues.length ? issueId : undefined}
    >
      <legend className={styles.legend}>
        Division {position + 1}: {name || "Untitled division"}
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
          triggerLabel="Remove division"
          title={`Remove ${name || `division ${position + 1}`}?`}
          description="This removes the division from the setup draft. Published tournament data does not change until final review."
          confirmLabel="Yes, remove division"
          cancelLabel="No, keep division"
          confirmationText=""
          tone="danger"
          disabled={disabled}
          onConfirm={onRemove}
        />
      </div>

      <div className={styles.grid}>
        <label className={`${styles.label} ${styles.wide}`}>
          Division name
          <input
            className={styles.input}
            value={name}
            disabled={disabled}
            placeholder="3.5 · 50+"
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
          Event
          <select
            className={styles.select}
            value={family}
            disabled={disabled}
            onChange={(event) =>
              onChange(applyFamily(value, eventFamilies, event.target.value))
            }
          >
            <option value="">Choose an event</option>
            {eventFamilies.map((event) => {
              const option = eventFamilyName(event.value);
              return option ? (
                <option key={event.key} value={option}>
                  {option}
                </option>
              ) : null;
            })}
          </select>
        </label>

        <label className={styles.label}>
          Skill division
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
          Age format
          <select
            className={styles.select}
            value={ageMode}
            disabled={disabled}
            onChange={(event) => onChange(setEventAgeMode(value, event.target.value))}
          >
            {optionsWithCurrent(AGE_MODES, ageMode).map((option) => (
              <option key={option} value={option}>
                {optionLabel(option)}
              </option>
            ))}
          </select>
        </label>

        {ageMode === "FIXED_AGE_BRACKET" ? (
          <label className={styles.label}>
            Age bracket label
            <input
              className={styles.input}
              value={cleanString(value.age_label)}
              disabled={disabled}
              placeholder="50+"
              onChange={(event) =>
                onChange(
                  setRecordString(value, ["age_label"], event.target.value)
                )
              }
            />
          </label>
        ) : null}

        {ageMode === "AUTO_AGE_SPLIT" ? (
          <label className={styles.label}>
            Minimum teams per age group
            <input
              className={styles.input}
              type="number"
              min="2"
              step="1"
              value={numberInputValue(ageRuleValue(value, "min_teams_per_age_group"))}
              disabled={disabled}
              onChange={(event) =>
                onChange(
                  setAgeRuleNumber(
                    value,
                    "min_teams_per_age_group",
                    event.target.value
                  )
                )
              }
            />
          </label>
        ) : null}

        {ageMode === "SPLIT_AGE" ? (
          <label className={styles.label}>
            Split-age threshold
            <input
              className={styles.input}
              type="number"
              min="1"
              step="1"
              value={numberInputValue(ageRuleValue(value, "split_age_threshold"))}
              disabled={disabled}
              onChange={(event) =>
                onChange(
                  setAgeRuleNumber(value, "split_age_threshold", event.target.value)
                )
              }
            />
          </label>
        ) : null}

        <label className={`${styles.label} ${styles.wide}`}>
          Division format
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
      </div>

      <details style={{ marginTop: "0.85rem" }}>
        <summary style={{ cursor: "pointer", fontWeight: 800 }}>
          Division overrides
        </summary>
        <div className={styles.grid} style={{ marginTop: "0.75rem" }}>
          <label className={styles.label}>
            Draw format override
            <select
              className={styles.select}
              value={drawFormat}
              disabled={disabled}
              onChange={(event) =>
                onChange(
                  setRecordString(
                    value,
                    ["event_format_override", "division_format"],
                    event.target.value
                  )
                )
              }
            >
              <option value="">Use event default</option>
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
            Scoring override
            <select
              className={styles.select}
              value={scoring}
              disabled={disabled}
              onChange={(event) =>
                onChange(
                  setRecordString(
                    value,
                    ["scoring_override", "division_scoring"],
                    event.target.value
                  )
                )
              }
            >
              <option value="">Use event default</option>
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
              checked={recordBoolean(value.enabled, true)}
              disabled={disabled}
              onChange={(event) =>
                onChange({ ...value, enabled: event.target.checked })
              }
            />
            Include division in registration
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
        <strong>{eventSummary}</strong>
        <br />
        <small>{modeSummary(value)}</small>
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
