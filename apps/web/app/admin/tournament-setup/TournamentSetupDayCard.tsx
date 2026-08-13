"use client";

import { useId } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess } from "@/components/interaction";
import {
  FACILITY_COURT_LIMIT,
  cleanString,
  dayCourtLabels,
  defaultCourtLabels,
  setRecordNumber,
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
  structureLocked?: boolean;
  onChange: (value: SetupRecord) => void;
  onMove?: (direction: -1 | 1) => void;
  onRemove?: () => void;
};

export function TournamentSetupDayCard({
  row,
  position,
  total,
  disabled,
  issues,
  structureLocked = false,
  onChange,
  onMove,
  onRemove
}: TournamentSetupDayCardProps) {
  const issueId = useId();
  const value = row.value;
  const dateValue = cleanString(value.event_date ?? value.date ?? value.start_date);
  const courtCount = Math.max(1, Math.min(FACILITY_COURT_LIMIT, Number(value.court_count) || FACILITY_COURT_LIMIT));
  const labels = dayCourtLabels(value);

  function updateCourtCount(raw: string) {
    const parsed = Math.max(1, Math.min(FACILITY_COURT_LIMIT, Math.trunc(Number(raw) || 1)));
    const current = labels.length ? labels : defaultCourtLabels(courtCount);
    const nextLabels = Array.from({ length: parsed }, (_, index) => current[index] || `Court ${index + 1}`);
    onChange({ ...setRecordNumber(value, "court_count", String(parsed)), court_labels: nextLabels });
  }

  function updateCourtLabel(index: number, label: string) {
    const current = Array.from({ length: courtCount }, (_, courtIndex) => labels[courtIndex] || `Court ${courtIndex + 1}`);
    current[index] = label;
    onChange({ ...value, court_labels: current });
  }

  return (
    <fieldset className={styles.rowCard} aria-describedby={issues.length ? issueId : undefined}>
      <legend className={styles.legend}>Day {position + 1}: {cleanString(value.label) || "Untitled day"}</legend>
      {!structureLocked ? (
        <div className={styles.rowActions}>
          <button
            type="button"
            className={styles.smallButton}
            disabled={disabled || position === 0 || !onMove}
            onClick={() => onMove?.(-1)}
            aria-label={`Move ${cleanString(value.label) || `day ${position + 1}`} earlier`}
          >
            Move up
          </button>
          <button
            type="button"
            className={styles.smallButton}
            disabled={disabled || position === total - 1 || !onMove}
            onClick={() => onMove?.(1)}
            aria-label={`Move ${cleanString(value.label) || `day ${position + 1}`} later`}
          >
            Move down
          </button>
          <ConfirmAction
            triggerLabel="Remove day"
            title={`Remove ${cleanString(value.label) || `day ${position + 1}`}?`}
            description="This removes the day from the unpublished setup draft. Published tournament data is unchanged until final review and publication."
            confirmLabel="Yes, remove day"
            cancelLabel="No, keep day"
            confirmationText=""
            tone="danger"
            disabled={disabled || !onRemove}
            onConfirm={async () => {
              onRemove?.();
              return actionSuccess("Day removed", "The day was removed from the unpublished local draft.");
            }}
          />
        </div>
      ) : null}
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
          Fixed tournament date
          <input
            className={styles.input}
            type="date"
            value={dateValue}
            disabled={disabled || structureLocked}
            readOnly={structureLocked}
            aria-invalid={issues.some((issue) => issue.path.endsWith(".event_date")) || undefined}
            onChange={(event) => onChange(setRecordString(value, ["event_date", "date", "start_date"], event.target.value))}
          />
          {structureLocked ? <small>Dates are generated from the tournament start and end dates in Step 1.</small> : null}
        </label>
        <label className={styles.label}>
          Available courts
          <input
            className={styles.input}
            type="number"
            min="1"
            max={FACILITY_COURT_LIMIT}
            step="1"
            value={courtCount}
            disabled={disabled}
            aria-invalid={issues.some((issue) => issue.path.endsWith(".court_count")) || undefined}
            onChange={(event) => updateCourtCount(event.target.value)}
          />
          <small>Maximum facility capacity: {FACILITY_COURT_LIMIT} courts.</small>
        </label>
        <label className={styles.label}>
          Courts open
          <input
            className={styles.input}
            type="time"
            value={cleanString(value.court_open_time) || "08:00"}
            disabled={disabled}
            onChange={(event) => onChange(setRecordString(value, ["court_open_time"], event.target.value))}
          />
        </label>
        <label className={styles.label}>
          Courts close
          <input
            className={styles.input}
            type="time"
            value={cleanString(value.court_close_time) || "20:00"}
            disabled={disabled}
            aria-invalid={issues.some((issue) => issue.path.endsWith(".court_hours")) || undefined}
            onChange={(event) => onChange(setRecordString(value, ["court_close_time"], event.target.value))}
          />
        </label>
        <p className={styles.checkbox} style={{ margin: 0, color: "#475569" }}>
          This date is part of the tournament and remains available for
          registration and scheduling. Change the tournament date range in Step
          1 to add or remove tournament days.
        </p>
        <fieldset className={`${styles.wide} ${styles.rowCard}`} style={{ padding: "0.75rem" }}>
          <legend style={{ fontWeight: 800 }}>Court labels</legend>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.55rem" }}>
            {Array.from({ length: courtCount }, (_, index) => (
              <label key={index} className={styles.label}>
                Court {index + 1}
                <input
                  className={styles.input}
                  value={labels[index] || `Court ${index + 1}`}
                  disabled={disabled}
                  onChange={(event) => updateCourtLabel(index, event.target.value)}
                />
              </label>
            ))}
          </div>
        </fieldset>
        <label className={`${styles.label} ${styles.wide}`}>
          Court notes
          <textarea
            className={styles.textarea}
            value={cleanString(value.court_notes)}
            disabled={disabled}
            placeholder="Optional setup, maintenance, or court-allocation notes"
            onChange={(event) => onChange(setRecordString(value, ["court_notes"], event.target.value))}
          />
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
