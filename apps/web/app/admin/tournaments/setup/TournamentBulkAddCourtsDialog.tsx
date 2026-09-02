"use client";

import { useMemo, useRef, useState, type CSSProperties } from "react";
import {
  FormDialog,
  InteractionActionError,
  useOpenDialogInitializer,
  type ActionCompletion
} from "@/components/interaction";

const MAX_VENUE_COURTS = 100;
const PRESET_COUNTS = [4, 6, 8, 10] as const;

type Props = {
  open: boolean;
  existingCount: number;
  disabled?: boolean;
  onCancel: () => void;
  onConfirm: (count: number) => Promise<ActionCompletion>;
  onAcknowledge?: () => void;
};

const inputStyle: CSSProperties = {
  width: "100%",
  minWidth: 0,
  boxSizing: "border-box",
  padding: "0.6rem",
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  font: "inherit"
};

const ghostButtonStyle: CSSProperties = {
  padding: "0.58rem 0.85rem",
  borderRadius: "999px",
  border: "1px solid #64748b",
  background: "white",
  color: "#0f172a",
  fontWeight: 800,
  cursor: "pointer"
};

export default function TournamentBulkAddCourtsDialog({
  open,
  existingCount,
  disabled = false,
  onCancel,
  onConfirm,
  onAcknowledge
}: Props) {
  const safeExistingCount = Math.max(0, Math.min(MAX_VENUE_COURTS, Math.trunc(existingCount || 0)));
  const maximumAddition = Math.max(0, MAX_VENUE_COURTS - safeExistingCount);
  const defaultCount = Math.min(4, maximumAddition);
  const [count, setCount] = useState(defaultCount);
  const countRef = useRef<HTMLInputElement>(null);

  useOpenDialogInitializer(open, () => {
    if (!open) return;
    setCount(Math.min(4, Math.max(0, MAX_VENUE_COURTS - safeExistingCount)));
  });

  const preview = useMemo(() => {
    const safeCount = Math.max(0, Math.min(maximumAddition, Math.trunc(count || 0)));
    return Array.from({ length: Math.min(safeCount, 12) }, (_, index) => `Court ${safeExistingCount + index + 1}`);
  }, [count, maximumAddition, safeExistingCount]);

  if (!open) return null;

  async function submit(): Promise<ActionCompletion> {
    const safeCount = Math.trunc(Number(count));
    if (!Number.isInteger(safeCount) || safeCount < 1) {
      throw new InteractionActionError("Choose at least one court to add.", { kind: "validation", fieldErrors: { "Courts to add": "Choose at least one court to add." } });
    }
    if (safeCount > maximumAddition) {
      throw new InteractionActionError(`A venue can have no more than ${MAX_VENUE_COURTS} courts.`, { kind: "validation", fieldErrors: { "Courts to add": `Choose ${maximumAddition} or fewer courts.` } });
    }
    return onConfirm(safeCount);
  }

  return (
    <FormDialog
      open={open}
      mode="bulk"
      size="wide"
      title="Bulk add venue courts"
      description="Add several stable court records at once. Court titles remain optional and can be edited after the courts are created."
      dirty={count !== defaultCount}
      submitLabel={`Add ${count || 0} court${count === 1 ? "" : "s"}`}
      submitDisabled={disabled || count < 1 || count > maximumAddition}
      workingLabel="Adding courts…"
      initialFocusRef={countRef}
      getFirstInvalidField={() => countRef.current}
      onSubmit={submit}
      onCancel={onCancel}
      onAcknowledge={onAcknowledge}
    >
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem" }}>
          <div style={{ padding: "0.75rem", border: "1px solid #e2e8f0", borderRadius: "12px", background: "#f8fafc" }}>
            <strong>Current inventory</strong><br />
            {safeExistingCount} court{safeExistingCount === 1 ? "" : "s"}
          </div>
          <div style={{ padding: "0.75rem", border: "1px solid #bfdbfe", borderRadius: "12px", background: "#eff6ff" }}>
            <strong>Resulting inventory</strong><br />
            {safeExistingCount + Math.max(0, Math.min(maximumAddition, Math.trunc(count || 0)))} courts
          </div>
        </div>

        <fieldset style={{ marginTop: "0.9rem", padding: "0.8rem", border: "1px solid #cbd5e1", borderRadius: "12px" }}>
          <legend style={{ fontWeight: 800 }}>Courts to add</legend>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
            {PRESET_COUNTS.map((preset) => (
              <button
                key={preset}
                type="button"
                disabled={disabled || preset > maximumAddition}
                style={{
                  ...ghostButtonStyle,
                  borderColor: count === preset ? "#2563eb" : "#64748b",
                  background: count === preset ? "#eff6ff" : "white"
                }}
                onClick={() => {
                  setCount(preset);
                }}
              >
                Add {preset}
              </button>
            ))}
          </div>
          <label style={{ display: "block", marginTop: "0.75rem" }}>
            <strong>Custom number</strong><br />
            <input
              ref={countRef}
              type="number"
              min="1"
              max={maximumAddition || 1}
              step="1"
              value={count || ""}
              disabled={disabled || maximumAddition < 1}
              style={inputStyle}
              onChange={(event) => {
                setCount(Math.trunc(Number(event.target.value) || 0));
              }}
            />
            <small>Maximum venue inventory: {MAX_VENUE_COURTS} courts.</small>
          </label>
        </fieldset>

        <article style={{ marginTop: "0.9rem", padding: "0.8rem", border: "1px solid #e2e8f0", borderRadius: "12px", background: "#f8fafc" }}>
          <h3 style={{ margin: "0 0 0.45rem" }}>Preview</h3>
          {preview.length ? (
            <>
              <p style={{ margin: "0 0 0.4rem", color: "#475569" }}>
                The new courts will be added after the existing inventory with stable IDs and these fallback display names:
              </p>
              <ul style={{ margin: 0, columns: preview.length > 6 ? 2 : 1 }}>
                {preview.map((name) => <li key={name}>{name}</li>)}
              </ul>
              {count > preview.length ? <p style={{ marginBottom: 0, color: "#64748b" }}>…and {count - preview.length} more.</p> : null}
            </>
          ) : (
            <p style={{ margin: 0, color: "#64748b" }}>Choose how many courts to add.</p>
          )}
        </article>

        {maximumAddition < 1 ? <p role="alert" style={{ color: "#b91c1c" }}>This venue already has the maximum court inventory.</p> : null}
    </FormDialog>
  );
}
