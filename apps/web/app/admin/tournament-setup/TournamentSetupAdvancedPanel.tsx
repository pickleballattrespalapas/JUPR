"use client";

import { useId, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import {
  formatAdvancedConfiguration,
  parseAdvancedConfiguration,
  type SetupConfiguration,
  type SetupPayload
} from "./tournamentSetupBuilder";
import styles from "./TournamentSetupBuilder.module.css";

type TournamentSetupAdvancedPanelProps = {
  configuration: SetupConfiguration;
  disabled: boolean;
  onApply: (payload: SetupPayload) => void;
};

export function TournamentSetupAdvancedPanel({
  configuration,
  disabled,
  onApply
}: TournamentSetupAdvancedPanelProps) {
  const exportId = useId();
  const importId = useId();
  const [importText, setImportText] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const exportText = formatAdvancedConfiguration(configuration);

  async function copyExport() {
    setError(null);
    setMessage(null);
    try {
      await navigator.clipboard.writeText(exportText);
      setMessage("Current configuration copied.");
    } catch {
      setError("Copy was unavailable. Select the export text and copy it manually.");
    }
  }

  function applyImport() {
    setError(null);
    setMessage(null);
    try {
      const payload = parseAdvancedConfiguration(importText);
      onApply(payload);
      setMessage("Imported configuration applied to the local draft. Review the guided fields before saving.");
    } catch (importError) {
      setError(importError instanceof Error ? importError.message : "Unable to import this configuration.");
      throw importError;
    }
  }

  return (
    <details className={styles.advanced}>
      <summary>Advanced JSON import/export</summary>
      <div className={styles.advancedContent}>
        <p className={styles.sectionDescription}>
          The guided builder is the primary editor. Use this panel only to copy a complete payload or import a known-good configuration.
          Importing replaces the local draft only; it does not write to the server.
        </p>
        <label className={styles.label} htmlFor={exportId}>Current configuration export</label>
        <textarea
          id={exportId}
          className={`${styles.textarea} ${styles.code}`}
          value={exportText}
          readOnly
          rows={16}
          spellCheck={false}
        />
        <div className={styles.advancedActions}>
          <button type="button" className={styles.secondaryButton} onClick={() => void copyExport()}>
            Copy current JSON
          </button>
          <button
            type="button"
            className={styles.secondaryButton}
            disabled={disabled}
            onClick={() => {
              setImportText(exportText);
              setMessage("Current configuration loaded into the import editor.");
              setError(null);
            }}
          >
            Load current JSON for editing
          </button>
        </div>
        <label className={styles.label} htmlFor={importId}>Configuration to import</label>
        <textarea
          id={importId}
          className={`${styles.textarea} ${styles.code}`}
          value={importText}
          disabled={disabled}
          rows={16}
          spellCheck={false}
          placeholder={'{\n  "days": [],\n  "event_families": [],\n  "event_options": []\n}'}
          aria-invalid={Boolean(error) || undefined}
          onChange={(event) => {
            setImportText(event.target.value);
            setError(null);
            setMessage(null);
          }}
        />
        <div className={styles.advancedActions}>
          <ConfirmAction
            triggerLabel="Apply imported JSON"
            title="Replace the local builder draft with this JSON?"
            description="The imported days, event defaults, and divisions will replace unsaved builder edits. This does not write to the server."
            confirmLabel="Yes, apply import"
            confirmationText=""
            disabled={disabled || !importText.trim()}
            onConfirm={applyImport}
          />
          <button
            type="button"
            className={styles.secondaryButton}
            disabled={disabled || !importText}
            onClick={() => {
              setImportText("");
              setError(null);
              setMessage(null);
            }}
          >
            Clear import
          </button>
        </div>
        {error ? <p className={styles.error} role="alert">{error}</p> : null}
        {message ? <p className={styles.success} role="status">{message}</p> : null}
      </div>
    </details>
  );
}
