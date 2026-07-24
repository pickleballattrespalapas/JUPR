"use client";

import { useId } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import {
  cleanString,
  recordBoolean,
  setRecordString,
  type BuilderRow,
  type SetupRecord,
  type ValidationIssue
} from "./tournamentSetupBuilder";
import styles from "./TournamentSetupBuilder.module.css";

type TournamentSetupDayCardProps = {
  row: BuilderRow;
  position: number;
  total: number;
  disabled: boolean;
  issues: ValidationIssue[];
  onChange: (value: SetupRecord) => void;
  onMove: (direction: -1 | 1) => void;
  onRemove: () => void;
};

export function TournamentSetupDayCard({
  row,
  position,
  total,
  disabled,
  issues,
  onChange,
  onMove,
  onRemove
}: TournamentSetupDayCardProps) {
  const issueId = useId();
  const value = row.value;
  const dateValue = cleanString(value.event_date ?? value.date ?? value.start_date);

  return (
    <fieldset className={styles.rowCard} aria-describedby={issues.length ? issueId : undefined}>
      <legend className={styles.legend}>Day {position + 1}: {cleanString(value.label) || "Untitled day"}</legend>
      <div className={styles.rowActions}>
        <button
          type="button"
          className={styles.smallButton}
          disabled={disabled || position === 0}
          onClick={() => onMove(-1)}
          aria-label={`Move ${cleanString(value.label) || `day ${position + 1}`} earlier`}
        >
          Move up
        </button>
        <button
          type="button"
          className={styles.smallButton}
          disabled={disabled || position === total - 1}
          onClick={() => onMove(1)}
          aria-label={`Move ${cleanString(value.label) || `day ${position + 1}`} later`}
        >
          Move down
        </button>
        <ConfirmAction
          triggerLabel="Remove day"
          title={`Remove ${cleanString(value.label) || `day ${position + 1}`}?`}
          description="This removes the day from the local draft. The server is not changed until you save the draft."
          confirmLabel="Yes, remove day"
          cancelLabel="No, keep day"
          confirmationText=""
          tone="danger"
          disabled={disabled}
          onConfirm={onRemove}
        />
      </div>
      <div className={styles.grid}>
        <label className={styles.label}>
          Day label
          <input
            className={styles.input}
            value={cleanString(value.label)}
            disabled={disabled}
            aria-invalid={issues.some((issue) => issue.path.endsWith(".label")) || undefined}
            onChange={(event) => onChange(setRecordString(value, ["label"], event.target.value))}
          />
        </label>
        <label className={styles.label}>
          Date
          <input
            className={styles.input}
            type="date"
            value={dateValue}
            disabled={disabled}
            aria-invalid={issues.some((issue) => issue.path.endsWith(".event_date")) || undefined}
            onChange={(event) => onChange(setRecordString(value, ["event_date", "date", "start_date"], event.target.value))}
          />
        </label>
        <label className={styles.checkbox}>
          <input
            type="checkbox"
            checked={recordBoolean(value.enabled, true)}
            disabled={disabled}
            onChange={(event) => onChange({ ...value, enabled: event.target.checked })}
          />
          Available for registration
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
