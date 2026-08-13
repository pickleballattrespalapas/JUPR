"use client";

import { useId } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess } from "@/components/interaction";
import {
  AGE_MODES,
  COMPETITION_FORMATS,
  DIVISION_STATUSES,
  GENDER_RESTRICTIONS,
  PARTICIPANT_TYPES,
  SCORING_OPTIONS,
  SKILL_LABEL_OPTIONS,
  cleanString,
  dayLabel,
  dayReference,
  ageRuleValue,
  effectiveGenderRestriction,
  effectiveParticipantType,
  eventDayReference,
  eventAgeMode,
  eventDivisionName,
  eventFamilyName,
  eventUsesLabelDayReference,
  numberInputValue,
  recordBoolean,
  setAgeRuleNumber,
  setEventAgeMode,
  setRecordNumber,
  setRecordString,
  type BuilderRow,
  type SetupRecord,
  type ValidationIssue
} from "./tournamentSetupBuilder";
import styles from "./TournamentSetupBuilder.module.css";

type TournamentSetupDivisionCardProps = {
  row: BuilderRow;
  position: number;
  total: number;
  days: BuilderRow[];
  eventFamilies: BuilderRow[];
  familyNames: string[];
  disabled: boolean;
  issues: ValidationIssue[];
  onChange: (value: SetupRecord) => void;
  onMove: (direction: -1 | 1) => void;
  onRemove: () => void;
};

function optionsWithCurrent(options: readonly string[], current: string): string[] {
  return current && !options.includes(current) ? [current, ...options] : [...options];
}

export function TournamentSetupDivisionCard({
  row,
  position,
  total,
  days,
  eventFamilies,
  familyNames,
  disabled,
  issues,
  onChange,
  onMove,
  onRemove
}: TournamentSetupDivisionCardProps) {
  const issueId = useId();
  const value = row.value;
  const name = eventDivisionName(value);
  const family = eventFamilyName(value);
  const usesDayLabel = eventUsesLabelDayReference(value);
  const currentDay = eventDayReference(value);
  const dayOptions = days.map((day) => ({
    value: usesDayLabel ? dayLabel(day.value) : dayReference(day.value),
    label: dayLabel(day.value) || dayReference(day.value),
    enabled: recordBoolean(day.value.enabled, true)
  }));
  const dayOptionValues = new Set(dayOptions.map((option) => option.value));
  const participantType = effectiveParticipantType(value, eventFamilies);
  const gender = effectiveGenderRestriction(value, eventFamilies);
  const skillLabel = cleanString(value.skill_label) || "Open";
  const ageMode = eventAgeMode(value);
  const ageLabel = cleanString(value.age_label) || "All Ages";
  const minimumTeamsPerAgeGroup = ageRuleValue(value, "min_teams_per_age_group");
  const splitAgeThreshold = ageRuleValue(value, "split_age_threshold");
  const format = cleanString(value.division_format ?? value.event_format_override ?? value.event_format_default);
  const scoring = cleanString(value.division_scoring ?? value.scoring_override ?? value.scoring_default);
  const status = cleanString(value.status) || "open";

  function updateSkill(nextSkill: string) {
    const next = setRecordString(value, ["skill_label"], nextSkill);
    if (Object.prototype.hasOwnProperty.call(value, "skill_mode") || Object.prototype.hasOwnProperty.call(value, "event_type")) {
      next.skill_mode = nextSkill.trim().toLowerCase() === "open" ? "OPEN" : "SKILL_BRACKET";
    }
    onChange(next);
  }

  return (
    <fieldset className={styles.rowCard} aria-describedby={issues.length ? issueId : undefined}>
      <legend className={styles.legend}>Division {position + 1}: {name || "Untitled division"}</legend>
      <div className={styles.rowActions}>
        <button
          type="button"
          className={styles.smallButton}
          disabled={disabled || position === 0}
          onClick={() => onMove(-1)}
          aria-label={`Move ${name || `division ${position + 1}`} earlier`}
        >
          Move up
        </button>
        <button
          type="button"
          className={styles.smallButton}
          disabled={disabled || position === total - 1}
          onClick={() => onMove(1)}
          aria-label={`Move ${name || `division ${position + 1}`} later`}
        >
          Move down
        </button>
        <ConfirmAction
          triggerLabel="Remove division"
          title={`Remove ${name || `division ${position + 1}`}?`}
          description="This removes the division from the local draft. The server is not changed until you save or publish."
          confirmLabel="Yes, remove division"
          cancelLabel="No, keep division"
          confirmationText=""
          tone="danger"
          disabled={disabled}
          onConfirm={async () => {
            onRemove();
            return actionSuccess("Division removed", "The division was removed from the local draft.");
          }}
        />
      </div>
      <div className={styles.grid}>
        <label className={`${styles.label} ${styles.wide}`}>
          Division name
          <input
            className={styles.input}
            value={name}
            disabled={disabled}
            aria-invalid={issues.some((issue) => issue.path.endsWith(".division_name")) || undefined}
            onChange={(event) => onChange(setRecordString(value, ["division_name", "label"], event.target.value))}
          />
        </label>
        <label className={styles.label}>
          Event family
          <select
            className={styles.select}
            value={family}
            disabled={disabled}
            aria-invalid={issues.some((issue) => issue.path.endsWith(".event_family")) || undefined}
            onChange={(event) => onChange(setRecordString(value, ["event_family", "event_family_label"], event.target.value))}
          >
            {!family ? <option value="">Choose an event</option> : null}
            {optionsWithCurrent(familyNames, family).map((option) => <option key={option} value={option}>{option}</option>)}
          </select>
        </label>
        <label className={styles.label}>
          Assigned day
          <select
            className={styles.select}
            value={currentDay}
            disabled={disabled}
            aria-invalid={issues.some((issue) => issue.path.endsWith(".registration_day_id")) || undefined}
            onChange={(event) => onChange(setRecordString(value, usesDayLabel ? ["assigned_day"] : ["registration_day_id"], event.target.value))}
          >
            {!currentDay ? <option value="">Choose a day</option> : null}
            {currentDay && !dayOptionValues.has(currentDay) ? <option value={currentDay}>{currentDay} (unavailable)</option> : null}
            {dayOptions.map((option) => (
              <option key={option.value} value={option.value} disabled={!option.enabled}>
                {option.label}{option.enabled ? "" : " (disabled)"}
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
            onChange={(event) => onChange(setRecordString(value, ["event_type"], event.target.value))}
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
          Skill level
          <input
            className={styles.input}
            list={`skills-${row.key}`}
            value={skillLabel}
            disabled={disabled}
            onChange={(event) => updateSkill(event.target.value)}
          />
          <datalist id={`skills-${row.key}`}>
            {SKILL_LABEL_OPTIONS.map((option) => <option key={option} value={option} />)}
          </datalist>
        </label>
        <label className={styles.label}>
          Age mode
          <select
            className={styles.select}
            value={ageMode}
            disabled={disabled}
            onChange={(event) => onChange(setEventAgeMode(value, event.target.value))}
          >
            {optionsWithCurrent(AGE_MODES, ageMode).map((option) => (
              <option key={option} value={option}>{option.replaceAll("_", " ")}</option>
            ))}
          </select>
        </label>
        <label className={styles.label}>
          Age label
          <input
            className={styles.input}
            value={ageLabel}
            disabled={disabled}
            onChange={(event) => onChange(setRecordString(value, ["age_label"], event.target.value))}
          />
        </label>
        {minimumTeamsPerAgeGroup != null || ageMode === "AUTO_AGE_SPLIT" ? (
          <label className={styles.label}>
            Minimum teams per age group
            <input
              className={styles.input}
              type="number"
              min="1"
              step="1"
              value={numberInputValue(minimumTeamsPerAgeGroup)}
              disabled={disabled || ageMode !== "AUTO_AGE_SPLIT"}
              aria-invalid={issues.some((issue) => issue.path.endsWith(".min_teams_per_age_group")) || undefined}
              onChange={(event) => onChange(setAgeRuleNumber(value, "min_teams_per_age_group", event.target.value))}
            />
          </label>
        ) : null}
        {splitAgeThreshold != null || ageMode === "SPLIT_AGE" ? (
          <label className={styles.label}>
            Split-age threshold
            <input
              className={styles.input}
              type="number"
              min="1"
              step="1"
              value={numberInputValue(splitAgeThreshold)}
              disabled={disabled || ageMode !== "SPLIT_AGE"}
              aria-invalid={issues.some((issue) => issue.path.endsWith(".split_age_threshold")) || undefined}
              onChange={(event) => onChange(setAgeRuleNumber(value, "split_age_threshold", event.target.value))}
            />
          </label>
        ) : null}
        <label className={styles.label}>
          Capacity (teams/players)
          <input
            className={styles.input}
            type="number"
            min="1"
            step="1"
            value={numberInputValue(value.capacity_teams)}
            disabled={disabled}
            aria-invalid={issues.some((issue) => issue.path.endsWith(".capacity_teams")) || undefined}
            onChange={(event) => onChange(setRecordNumber(value, "capacity_teams", event.target.value))}
          />
        </label>
        <label className={styles.label}>
          Price (USD)
          <input
            className={styles.input}
            type="number"
            min="0"
            step="0.01"
            value={numberInputValue(value.price_usd)}
            disabled={disabled}
            aria-invalid={issues.some((issue) => issue.path.endsWith(".price_usd")) || undefined}
            onChange={(event) => onChange(setRecordNumber(value, "price_usd", event.target.value))}
          />
        </label>
        <label className={styles.label}>
          Registration status
          <select
            className={styles.select}
            value={status}
            disabled={disabled}
            onChange={(event) => onChange(setRecordString(value, ["status"], event.target.value))}
          >
            {optionsWithCurrent(DIVISION_STATUSES, status).map((option) => <option key={option} value={option}>{option}</option>)}
          </select>
        </label>
        <label className={styles.label}>
          Format
          <select
            className={styles.select}
            value={format}
            disabled={disabled}
            onChange={(event) => onChange(setRecordString(value, ["division_format", "event_format_override", "event_format_default"], event.target.value))}
          >
            <option value="">Use event default</option>
            {optionsWithCurrent(COMPETITION_FORMATS, format).filter(Boolean).map((option) => (
              <option key={option} value={option}>{option.replaceAll("_", " ")}</option>
            ))}
          </select>
        </label>
        <label className={styles.label}>
          Scoring
          <select
            className={styles.select}
            value={scoring}
            disabled={disabled}
            onChange={(event) => onChange(setRecordString(value, ["division_scoring", "scoring_override", "scoring_default"], event.target.value))}
          >
            <option value="">Use event default</option>
            {optionsWithCurrent(SCORING_OPTIONS, scoring).filter(Boolean).map((option) => (
              <option key={option} value={option}>{option.replaceAll("_", " ")}</option>
            ))}
          </select>
        </label>
        <label className={styles.checkbox}>
          <input
            type="checkbox"
            checked={recordBoolean(value.waitlist_enabled, true)}
            disabled={disabled}
            onChange={(event) => onChange({ ...value, waitlist_enabled: event.target.checked })}
          />
          Waitlist enabled
        </label>
        <label className={styles.checkbox}>
          <input
            type="checkbox"
            checked={recordBoolean(value.partner_board_enabled, true)}
            disabled={disabled}
            onChange={(event) => onChange({ ...value, partner_board_enabled: event.target.checked })}
          />
          Partner Board enabled
        </label>
        <label className={styles.checkbox}>
          <input
            type="checkbox"
            checked={recordBoolean(value.enabled, true)}
            disabled={disabled}
            onChange={(event) => onChange({ ...value, enabled: event.target.checked })}
          />
          Visible to registrants
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
