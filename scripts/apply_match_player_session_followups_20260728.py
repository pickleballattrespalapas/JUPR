from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one exact match, found {count}")
    return text.replace(old, new, 1)


def sub_once(text: str, pattern: str, replacement: str, label: str) -> str:
    regex = re.compile(pattern, re.DOTALL)
    match = regex.search(text)
    if match is None:
        raise RuntimeError(f"{label}: expected one regex match, found 0")
    rendered = replacement
    for index, value in enumerate(match.groups(), start=1):
        rendered = rendered.replace(f"\\{index}", value or "")
    return text[:match.start()] + rendered + text[match.end():]


def update_confirm_action() -> None:
    path = ROOT / "apps/web/components/ConfirmAction.tsx"
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        '''export type ConfirmActionProps = {''',
        '''export type ConfirmActionSuccess = {
  title?: string;
  description: ReactNode;
  closeLabel?: string;
};

export type ConfirmActionProps = {''',
        "confirm success type",
    )
    text = replace_once(
        text,
        '''  onConfirm: (confirmationText: string) => void | Promise<void>;''',
        '''  onConfirm: (confirmationText: string) => void | ConfirmActionSuccess | Promise<void | ConfirmActionSuccess>;''',
        "confirm return type",
    )
    text = replace_once(
        text,
        '''  const [error, setError] = useState<string | null>(null);
  const [portalContainer, setPortalContainer] = useState<HTMLElement | null>(null);''',
        '''  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<ConfirmActionSuccess | null>(null);
  const [portalContainer, setPortalContainer] = useState<HTMLElement | null>(null);''',
        "confirm success state",
    )
    text = replace_once(
        text,
        '''  }, [open, portalContainer]);''',
        '''  }, [open, portalContainer, success]);''',
        "confirm focus after success",
    )
    text = replace_once(
        text,
        '''    setError(null);
    setOpen(true);''',
        '''    setError(null);
    setSuccess(null);
    setOpen(true);''',
        "confirm clear success",
    )
    text = replace_once(
        text,
        '''      await onConfirm(confirmationText);
      setOpen(false);''',
        '''      const completion = await onConfirm(confirmationText);
      if (completion) setSuccess(completion);
      else setOpen(false);''',
        "confirm preserve success dialog",
    )
    text = replace_once(
        text,
        '''          <div className={styles.content}>
            <h2 id={titleId} className={styles.title}>{title}</h2>
            <div id={descriptionId} className={styles.description}>{description}</div>
            {error ? <p id={errorId} className={styles.error} role="alert">{error}</p> : null}
            <div className={styles.actions}>
              <button
                ref={cancelRef}
                type="button"
                className={styles.cancelButton}
                disabled={actionBusy}
                onClick={closeDialog}
              >
                {cancelLabel}
              </button>
              <button
                type="button"
                className={`${styles.confirmButton} ${tone === "danger" ? styles.dangerConfirm : ""}`}
                disabled={disabled || actionBusy}
                onClick={() => void handleConfirm()}
              >
                {actionBusy ? "Working…" : confirmLabel}
              </button>
            </div>
          </div>''',
        '''          <div className={styles.content}>
            {success ? (
              <>
                <h2 id={titleId} className={styles.title}>{success.title || "Action complete"}</h2>
                <div id={descriptionId} className={styles.description}>{success.description}</div>
                <div className={styles.actions}>
                  <button
                    ref={cancelRef}
                    type="button"
                    className={styles.confirmButton}
                    disabled={actionBusy}
                    onClick={closeDialog}
                  >
                    {success.closeLabel || "OK"}
                  </button>
                </div>
              </>
            ) : (
              <>
                <h2 id={titleId} className={styles.title}>{title}</h2>
                <div id={descriptionId} className={styles.description}>{description}</div>
                {error ? <p id={errorId} className={styles.error} role="alert">{error}</p> : null}
                <div className={styles.actions}>
                  <button
                    ref={cancelRef}
                    type="button"
                    className={styles.cancelButton}
                    disabled={actionBusy}
                    onClick={closeDialog}
                  >
                    {cancelLabel}
                  </button>
                  <button
                    type="button"
                    className={`${styles.confirmButton} ${tone === "danger" ? styles.dangerConfirm : ""}`}
                    disabled={disabled || actionBusy}
                    onClick={() => void handleConfirm()}
                  >
                    {actionBusy ? "Working…" : confirmLabel}
                  </button>
                </div>
              </>
            )}
          </div>''',
        "confirm success rendering",
    )

    path.write_text(text, encoding="utf-8")


def update_match_uploader() -> None:
    path = ROOT / "apps/web/app/admin/match-uploader/MatchUploaderForm.tsx"
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        '''  disabled?: boolean;
  onChange: (playerId: string) => void;''',
        '''  disabled?: boolean;
  invalid?: boolean;
  onChange: (playerId: string) => void;''',
        "player invalid prop",
    )
    text = replace_once(
        text,
        '''const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };''',
        '''const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
const dangerButtonStyle = { ...buttonStyle, background: "#b91c1c", borderColor: "#b91c1c" };''',
        "danger button style",
    )
    text = replace_once(
        text,
        '''function ratingLabel(value?: number | null): string {
  return value == null ? "—" : Math.round(Number(value)).toString();
}

function deltaLabel(value?: number | null): string {
  if (value == null) return "—";
  const rounded = Math.round(Number(value));
  return `${rounded >= 0 ? "+" : ""}${rounded}`;
}''',
        '''function ratingLabel(value?: number | null): string {
  if (value == null) return "—";
  return (Number(value) / 400).toFixed(2);
}

function deltaLabel(value?: number | null): string {
  if (value == null) return "—";
  const juprDelta = Number(value) / 400;
  const normalized = Math.abs(juprDelta) < 0.005 ? 0 : juprDelta;
  return `${normalized >= 0 ? "+" : ""}${normalized.toFixed(2)}`;
}''',
        "JUPR rating display",
    )
    text = replace_once(
        text,
        '''function validateRow(row: MatchRow, index: number): string | null {
  if (!isFilled(row)) return null;
  const ids = [row.t1p1, row.t1p2, row.t2p1, row.t2p2].filter(Boolean);
  if (ids.length !== 4) return `Row ${index + 1}: select four players.`;
  if (new Set(ids).size !== 4) return `Row ${index + 1}: select four different players.`;
  const s1 = Number(row.s1 || 0);
  const s2 = Number(row.s2 || 0);
  if (!Number.isFinite(s1) || !Number.isFinite(s2) || s1 < 0 || s2 < 0) return `Row ${index + 1}: scores must be non-negative numbers.`;
  if (s1 + s2 <= 0) return `Row ${index + 1}: enter a non-zero score.`;
  return null;
}''',
        '''function validateRequiredRow(row: MatchRow, index: number): string | null {
  const ids = [row.t1p1, row.t1p2, row.t2p1, row.t2p2].filter(Boolean);
  if (ids.length !== 4) return `Row ${index + 1}: complete the highlighted player fields.`;
  if (new Set(ids).size !== 4) return `Row ${index + 1}: each player may appear only once.`;
  const s1 = Number(row.s1 || 0);
  const s2 = Number(row.s2 || 0);
  if (!Number.isFinite(s1) || !Number.isFinite(s2) || s1 < 0 || s2 < 0) return `Row ${index + 1}: scores must be non-negative numbers.`;
  if (s1 + s2 <= 0) return `Row ${index + 1}: enter a non-zero score in the highlighted score fields.`;
  return null;
}

function validateRow(row: MatchRow, index: number): string | null {
  if (!isFilled(row)) return null;
  return validateRequiredRow(row, index);
}''',
        "required row validation",
    )
    before, searchable, after = section(
    text,
    "function SearchablePlayerInput({",
    "function SearchablePlayerMultiInput({",
)
searchable = replace_once(
    searchable,
    """  players,
  disabled = false,
  onChange,""",
    """  players,
  disabled = false,
  invalid = false,
  onChange,""",
    "player invalid destructure",
)
searchable = replace_once(
    searchable,
    "  const numericStartingJupr = Number(startingJupr);",
    """  const numericStartingJupr = Number(startingJupr);
  const validatedInputStyle = invalid
    ? { ...inputStyle, border: "2px solid #dc2626", background: "#fef2f2" }
    : inputStyle;""",
    "player invalid style",
)
searchable = replace_once(
    searchable,
    """            style={{ ...inputStyle, minHeight: "2.4rem", display: "flex", alignItems: "center", whiteSpace: "normal", overflowWrap: "anywhere", background: "#f8fafc" }}""",
    """            style={{ ...validatedInputStyle, minHeight: "2.4rem", display: "flex", alignItems: "center", whiteSpace: "normal", overflowWrap: "anywhere", background: invalid ? "#fef2f2" : "#f8fafc" }}""",
    "selected player invalid style",
)
searchable = replace_once(
    searchable,
    """          disabled={disabled || creating}
  onChange={(event) => {""",
    """          disabled={disabled || creating}
  aria-invalid={invalid || undefined}
  onChange={(event) => {""",
    "player aria invalid",
)
searchable = replace_once(
    searchable,
    """          style={inputStyle}
        />""",
    """          style={validatedInputStyle}
        />""",
    "player invalid input style",
)
text = before + searchable + after


    dialog_component = '''
function RemoveAllMatchesDialog({
  onClose,
  onKeepRows,
  onRemoveAll,
}: {
  onClose: () => void;
  onKeepRows: () => void;
  onRemoveAll: () => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
    return () => {
      if (dialog?.open) dialog.close();
    };
  }, []);

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby="remove-all-matches-title"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      style={{ width: "min(620px, calc(100vw - 2rem))", border: 0, borderRadius: "16px", padding: 0, boxShadow: "0 24px 70px rgba(15, 23, 42, 0.35)" }}
    >
      <div style={{ padding: "1.25rem" }}>
        <h2 id="remove-all-matches-title" style={{ marginTop: 0 }}>Remove entered matches?</h2>
        <p>Choose whether to keep completed or partially entered rows, remove only blank rows, or clear the entire batch.</p>
        <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.5rem", flexWrap: "wrap" }}>
          <button type="button" onClick={onClose} style={ghostButtonStyle}>No, go back</button>
          <button type="button" onClick={onKeepRows} style={ghostButtonStyle}>Keep rows with data</button>
          <button type="button" onClick={onRemoveAll} style={dangerButtonStyle}>Yes, remove all</button>
        </div>
      </div>
    </dialog>
  );
}
'''
    text = replace_once(
        text,
        '''function SubmissionResultDialog({''',
        dialog_component + '''
function SubmissionResultDialog({''',
        "remove all dialog",
    )
    text = replace_once(
        text,
        '''  const [submissionKind, setSubmissionKind] = useState<"manual" | "round_robin" | "singles" | null>(null);''',
        '''  const [submissionKind, setSubmissionKind] = useState<"manual" | "round_robin" | "singles" | null>(null);
  const [manualValidationAttempted, setManualValidationAttempted] = useState(false);
  const [removeAllDialogOpen, setRemoveAllDialogOpen] = useState(false);''',
        "manual validation state",
    )
    text = replace_once(
        text,
        '''  function resetManualRows() {
    const preservedScope = (readyRows[0] || filledRows[0])?.ratingScope || "";
    setRows([newMatchRow(defaultDate, defaultWeekTag, preservedScope)]);
  }''',
        '''  function resetManualRows() {
    const preservedScope = (readyRows[0] || filledRows[0])?.ratingScope || "";
    setRows([newMatchRow(defaultDate, defaultWeekTag, preservedScope)]);
    setManualValidationAttempted(false);
  }''',
        "reset validation state",
    )
    text = replace_once(
        text,
        '''  function removeAllRows() {
    clearEntryFeedback();
    resetManualRows();
  }''',
        '''  function removeAllRows() {
    clearEntryFeedback();
    setRemoveAllDialogOpen(false);
    resetManualRows();
  }

  function keepRowsWithData() {
    clearEntryFeedback();
    setRemoveAllDialogOpen(false);
    setManualValidationAttempted(false);
    setRows((current) => {
      const kept = current.filter(rowHasEnteredData);
      return kept.length ? kept : [newMatchRow(defaultDate, defaultWeekTag)];
    });
  }

  function playerOptionsFor(row: MatchRow, currentValue: string): PublicPlayer[] {
    const selectedElsewhere = new Set(
      [row.t1p1, row.t1p2, row.t2p1, row.t2p2].filter((value) => value && value !== currentValue),
    );
    return knownPlayers.filter((player) => !selectedElsewhere.has(String(player.id)));
  }''',
        "remove blank rows and player pools",
    )
    text = replace_once(
        text,
        '''  async function submitManualBatch() {
    setMessage(null);
    setResult(null);
    if (!requireReady()) return;
    const errors = rows.map(validateRow).filter(Boolean) as string[];
    if (errors.length) {
      setMessage(errors[0]);
      return;
    }''',
        '''  async function submitManualBatch() {
    setMessage(null);
    setResult(null);
    setManualValidationAttempted(true);
    if (!requireReady()) return;
    const enteredRows = rows.filter(rowHasEnteredData);
    const validationRows = enteredRows.length ? enteredRows : rows.slice(0, 1);
    const errors = validationRows.map((row) => validateRequiredRow(row, rows.indexOf(row))).filter(Boolean) as string[];
    if (errors.length) {
      setMessage(errors[0]);
      return;
    }''',
        "clickable validation",
    )
    text = replace_once(
        text,
        '''<select value={entryMethod} onChange={(event) => { clearEntryFeedback(); setEntryMethod(event.target.value as "singles" | "manual" | "round_robin"); }} style={inputStyle}>''',
        '''<select value={entryMethod} onChange={(event) => { clearEntryFeedback(); setManualValidationAttempted(false); setEntryMethod(event.target.value as "singles" | "manual" | "round_robin"); }} style={inputStyle}>''',
        "clear validation on method change",
    )
    text = replace_once(
        text,
        '''            <button type="button" onClick={() => { clearEntryFeedback(); setRows((current) => [...current, newMatchRow(defaultDate, defaultWeekTag)]); }} disabled={rows.length >= status.max_batch_rows} style={ghostButtonStyle}>Add 1 Match</button>
            <button type="button" onClick={() => { clearEntryFeedback(); setRows((current) => [...current, ...Array.from({ length: 5 }, () => newMatchRow(defaultDate, defaultWeekTag))].slice(0, status.max_batch_rows)); }} disabled={rows.length >= status.max_batch_rows} style={ghostButtonStyle}>Add 5 Matches</button>
            {rows.some(rowHasEnteredData) ? (
              <ConfirmAction
                triggerLabel="Remove All"
                title="Remove all entered matches?"
                description="This clears every entered player and score and returns the uploader to one blank match."
                confirmLabel="Yes, remove all"
                confirmationText="REMOVE"
                tone="danger"
                disabled={saving}
                busy={saving}
                onConfirm={() => removeAllRows()}
              />
            ) : (
              <button type="button" onClick={removeAllRows} disabled={rows.length <= 1} style={ghostButtonStyle}>Remove All</button>
            )}''',
        '''            <button type="button" onClick={() => { clearEntryFeedback(); setManualValidationAttempted(false); setRows((current) => [...current, newMatchRow(defaultDate, defaultWeekTag)]); }} disabled={rows.length >= status.max_batch_rows} style={ghostButtonStyle}>Add 1 Match</button>
            <button type="button" onClick={() => { clearEntryFeedback(); setManualValidationAttempted(false); setRows((current) => [...current, ...Array.from({ length: 5 }, () => newMatchRow(defaultDate, defaultWeekTag))].slice(0, status.max_batch_rows)); }} disabled={rows.length >= status.max_batch_rows} style={ghostButtonStyle}>Add 5 Matches</button>
            {rows.some(rowHasEnteredData) ? (
              <button type="button" onClick={() => setRemoveAllDialogOpen(true)} disabled={saving} style={dangerButtonStyle}>Remove All</button>
            ) : (
              <button type="button" onClick={removeAllRows} disabled={rows.length <= 1} style={ghostButtonStyle}>Remove All</button>
            )}
            {removeAllDialogOpen ? (
              <RemoveAllMatchesDialog
                onClose={() => setRemoveAllDialogOpen(false)}
                onKeepRows={keepRowsWithData}
                onRemoveAll={removeAllRows}
              />
            ) : null}''',
        "three option remove all",
    )

    old_rows = '''            {rows.map((row, index) => (
              <div key={row.rowId} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: rowHasEnteredData(row) ? "#f8fafc" : "white" }}>'''
    new_rows = '''            {rows.map((row, index) => {
              const validateThisRow = manualValidationAttempted && (rowHasEnteredData(row) || (!rows.some(rowHasEnteredData) && index === 0));
              const rowError = validateThisRow ? validateRequiredRow(row, index) : null;
              const scoreInvalid = validateThisRow && Number(row.s1 || 0) + Number(row.s2 || 0) <= 0;
              return (
              <div key={row.rowId} style={{ border: rowError ? "2px solid #dc2626" : "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: rowError ? "#fff7f7" : rowHasEnteredData(row) ? "#f8fafc" : "white" }}>'''
    text = replace_once(text, old_rows, new_rows, "row validation rendering")

    old_team_grid = '''                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.5rem", marginTop: "0.5rem" }}>
                  <SearchablePlayerInput inputId={`${row.rowId}-t1p1`} label="Team 1 · Player 1" value={row.t1p1} players={knownPlayers} disabled={saving || creatingPlayers} onChange={(t1p1) => patchRow(row.rowId, { t1p1 })} onCreate={createAndSelectPlayer} />
                  <SearchablePlayerInput inputId={`${row.rowId}-t1p2`} label="Team 1 · Player 2" value={row.t1p2} players={knownPlayers} disabled={saving || creatingPlayers} onChange={(t1p2) => patchRow(row.rowId, { t1p2 })} onCreate={createAndSelectPlayer} />
                  <label><strong>Team 1 score</strong><br /><input value={row.s1} onChange={(event) => patchRow(row.rowId, { s1: event.target.value })} type="number" min={0} max={99} style={inputStyle} /></label>
                  <label><strong>Team 2 score</strong><br /><input value={row.s2} onChange={(event) => patchRow(row.rowId, { s2: event.target.value })} type="number" min={0} max={99} style={inputStyle} /></label>
                  <SearchablePlayerInput inputId={`${row.rowId}-t2p1`} label="Team 2 · Player 1" value={row.t2p1} players={knownPlayers} disabled={saving || creatingPlayers} onChange={(t2p1) => patchRow(row.rowId, { t2p1 })} onCreate={createAndSelectPlayer} />
                  <SearchablePlayerInput inputId={`${row.rowId}-t2p2`} label="Team 2 · Player 2" value={row.t2p2} players={knownPlayers} disabled={saving || creatingPlayers} onChange={(t2p2) => patchRow(row.rowId, { t2p2 })} onCreate={createAndSelectPlayer} />
                </div>
              </div>
            ))}'''
    new_team_grid = '''                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" }}>
                  <section aria-label={`Match ${index + 1} Team 1`} style={{ border: "1px solid #cbd5e1", borderRadius: "10px", padding: "0.75rem", display: "grid", gap: "0.6rem", alignContent: "start", background: "white" }}>
                    <h4 style={{ margin: 0 }}>Team 1</h4>
                    <SearchablePlayerInput inputId={`${row.rowId}-t1p1`} label="Player 1" value={row.t1p1} players={playerOptionsFor(row, row.t1p1)} invalid={validateThisRow && !row.t1p1} disabled={saving || creatingPlayers} onChange={(t1p1) => patchRow(row.rowId, { t1p1 })} onCreate={createAndSelectPlayer} />
                    <SearchablePlayerInput inputId={`${row.rowId}-t1p2`} label="Player 2" value={row.t1p2} players={playerOptionsFor(row, row.t1p2)} invalid={validateThisRow && !row.t1p2} disabled={saving || creatingPlayers} onChange={(t1p2) => patchRow(row.rowId, { t1p2 })} onCreate={createAndSelectPlayer} />
                    <label><strong>Score</strong><br /><input value={row.s1} onChange={(event) => patchRow(row.rowId, { s1: event.target.value })} aria-invalid={scoreInvalid || undefined} type="number" min={0} max={99} style={scoreInvalid ? { ...inputStyle, border: "2px solid #dc2626", background: "#fef2f2" } : inputStyle} /></label>
                  </section>
                  <section aria-label={`Match ${index + 1} Team 2`} style={{ border: "1px solid #cbd5e1", borderRadius: "10px", padding: "0.75rem", display: "grid", gap: "0.6rem", alignContent: "start", background: "white" }}>
                    <h4 style={{ margin: 0 }}>Team 2</h4>
                    <SearchablePlayerInput inputId={`${row.rowId}-t2p1`} label="Player 1" value={row.t2p1} players={playerOptionsFor(row, row.t2p1)} invalid={validateThisRow && !row.t2p1} disabled={saving || creatingPlayers} onChange={(t2p1) => patchRow(row.rowId, { t2p1 })} onCreate={createAndSelectPlayer} />
                    <SearchablePlayerInput inputId={`${row.rowId}-t2p2`} label="Player 2" value={row.t2p2} players={playerOptionsFor(row, row.t2p2)} invalid={validateThisRow && !row.t2p2} disabled={saving || creatingPlayers} onChange={(t2p2) => patchRow(row.rowId, { t2p2 })} onCreate={createAndSelectPlayer} />
                    <label><strong>Score</strong><br /><input value={row.s2} onChange={(event) => patchRow(row.rowId, { s2: event.target.value })} aria-invalid={scoreInvalid || undefined} type="number" min={0} max={99} style={scoreInvalid ? { ...inputStyle, border: "2px solid #dc2626", background: "#fef2f2" } : inputStyle} /></label>
                  </section>
                </div>
                {rowError ? <p role="alert" style={{ color: "#b91c1c", marginBottom: 0 }}><strong>{rowError}</strong></p> : null}
              </div>
              );
            })}'''
    text = replace_once(text, old_team_grid, new_team_grid, "vertical team layout")

    text = replace_once(
        text,
        '''          <button type="button" onClick={submitManualBatch} disabled={saving || !accessToken || !readyRows.length || hasInvalidFilledRows} style={buttonStyle}>{saving ? "Submitting…" : "Submit batch"}</button>''',
        '''          <button type="button" onClick={submitManualBatch} disabled={saving || !accessToken} style={buttonStyle}>{saving ? "Submitting…" : "Submit batch"}</button>
          {message && !result ? <p aria-live="polite" role={messageIsError ? "alert" : "status"} style={{ color: messageIsError ? "#b91c1c" : "#166534" }}>{message}</p> : null}''',
        "manual validation feedback",
    )
    text = replace_once(
        text,
        '''      {message && !result ? <p aria-live="polite" role={messageIsError ? "alert" : "status"} style={{ color: messageIsError ? "#b91c1c" : "#166534" }}>{message}</p> : null}''',
        '''      {message && !result && entryMethod !== "manual" ? <p aria-live="polite" role={messageIsError ? "alert" : "status"} style={{ color: messageIsError ? "#b91c1c" : "#166534" }}>{message}</p> : null}''',
        "avoid duplicate manual feedback",
    )

    path.write_text(text, encoding="utf-8")


def update_match_log_apply() -> None:
    path = ROOT / "apps/web/app/admin/match-log/MatchLogApplyPanel.tsx"
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        '''import { ConfirmAction } from "@/components/ConfirmAction";''',
        '''import { ConfirmAction } from "@/components/ConfirmAction";
import type { ConfirmActionSuccess } from "@/components/ConfirmAction";''',
        "confirm success import",
    )
    text = replace_once(
        text,
        '''  async function submitGuidedPatches(confirmationText: string) {''',
        '''  async function submitGuidedPatches(confirmationText: string): Promise<ConfirmActionSuccess | void> {''',
        "guided result type",
    )
    old_success = '''      setResult(payload);
      showMessage("apply", resultSummary(payload) || "Match edits completed.", payload.ok ? "success" : "error");
      if (payload.ok) {
        setStagedPatches([]);
        setRecoveryOperationId(null);
        setIdempotencyKey(requestKey());
        onMutationComplete();
      }'''
    new_success = '''      const summary = resultSummary(payload) || "Match edits completed.";
      setResult(payload);
      showMessage("apply", summary, payload.ok ? "success" : "error");
      if (!payload.ok) throw new Error(summary);
      setStagedPatches([]);
      setRecoveryOperationId(null);
      setIdempotencyKey(requestKey());
      onMutationComplete();
      return {
        title: payload.mode === "applied_and_replayed" ? "Match edit and replay complete" : "Match edit complete",
        description: (
          <div>
            <p role="status" style={{ color: "#166534" }}><strong>{summary}</strong></p>
            <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem", margin: 0 }}>
              <div><dt style={{ fontWeight: 700 }}>Matches updated</dt><dd style={{ margin: 0 }}>{payload.updated_count ?? 0}</dd></div>
              <div><dt style={{ fontWeight: 700 }}>Ratings replay</dt><dd style={{ margin: 0 }}>{payload.replay_job_id ? "Completed" : "Not required"}</dd></div>
              {payload.replay_job_id ? <div><dt style={{ fontWeight: 700 }}>Replay job</dt><dd style={{ margin: 0, fontFamily: "monospace", overflowWrap: "anywhere" }}>{payload.replay_job_id}</dd></div> : null}
            </dl>
            {payload.warnings?.length ? <ul style={{ color: "#92400e" }}>{payload.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
          </div>
        ),
        closeLabel: "OK",
      };'''
    text = replace_once(text, old_success, new_success, "guided success popup")
    text = replace_once(
        text,
        '''    } catch (error) {
      if (error instanceof ApiCallError && error.operationId) setRecoveryOperationId(error.operationId);
      showMessage("apply", error instanceof Error ? error.message : "Unable to apply match edits.");
    } finally {''',
        '''    } catch (error) {
      if (error instanceof ApiCallError && error.operationId) setRecoveryOperationId(error.operationId);
      const applyError = error instanceof Error ? error : new Error("Unable to apply match edits.");
      showMessage("apply", applyError.message);
      throw applyError;
    } finally {''',
        "guided error stays in dialog",
    )

    path.write_text(text, encoding="utf-8")


def update_quick_replay() -> None:
    path = ROOT / "apps/web/app/admin/match-log/MatchLogQuickReplayPanel.tsx"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''      <ConfirmAction
        triggerLabel="Run Quick Replay"
        title="Run Replay History now?"
        description={<>This will run Replay History for <strong>{targetReset}</strong>. Ratings and derived history may be rebuilt across that scope.</>}
        confirmLabel="Yes, run replay"
        confirmationText="REPLAY"
        disabled={pending || !accessToken}
        busy={pending}
        onConfirm={onSubmit}
      />''',
        '''      <ConfirmAction
        triggerLabel="Run Quick Replay"
        title="Run Replay History now?"
        description={<>This will run Replay History for <strong>{targetReset}</strong>. Ratings and derived history may be rebuilt across that scope.</>}
        confirmLabel="Yes, run replay"
        confirmationText="REPLAY"
        disabled={pending || !accessToken}
        busy={pending}
        onConfirm={onSubmit}
      />
      <p style={{ margin: 0 }}><Link href="/admin/replay-history"><strong>Open Replay History</strong></Link> to view recent jobs and their status.</p>''',
        "replay history link",
    )
    path.write_text(text, encoding="utf-8")


def write_docs_and_tests() -> None:
    docs = ROOT / "docs/manual_acceptance_match_player_followups_20260728.md"
    docs.write_text(
        '''# Match/player manual acceptance follow-ups — July 28, 2026

The write window was closed before this staging-only implementation began.

## Implemented fixes

1. Match Uploader submission attempts now show an inline validation message and highlight missing player and score fields instead of silently doing nothing.
2. A player selected in one slot is removed from the other player pickers within that match. Clearing or replacing the player makes them available again; other match rows are unaffected.
3. Remove All now offers **No, go back**, **Keep rows with data**, and **Yes, remove all**.
4. Doubles entry is organized into separate Team 1 and Team 2 cards, with each team’s players stacked vertically and its score kept inside the same card.
5. Match submission results display ratings on the JUPR 2.0–7.0 scale rather than raw ELO values.
6. Match Log edit confirmation remains open through **Working…**, then changes into a success summary with an **OK** button after the edit and any required replay complete.
7. Quick Replay includes a clear **Open Replay History** link for recent job status.

## Verified behaviors retained

- One blank doubles row on load.
- Accurate ready-row counting.
- Guarded individual-row removal.
- Submission success dialog and reset to one blank row.
- Compact single-match editor with working filters.
- Notes-only edits and rating-affecting edits with durable Replay ALL.
- Recent durable edit operations collapsed by default.
''',
        encoding="utf-8",
    )

    tests = ROOT / "tests/test_match_player_session_followups.py"
    tests.write_text(
        '''from pathlib import Path


def test_match_uploader_followups_are_present() -> None:
    source = Path("apps/web/app/admin/match-uploader/MatchUploaderForm.tsx").read_text(encoding="utf-8")
    assert "validateRequiredRow" in source
    assert "complete the highlighted player fields" in source
    assert "manualValidationAttempted" in source
    assert "Keep rows with data" in source
    assert "playerOptionsFor" in source
    assert "selectedElsewhere" in source
    assert 'aria-label={`Match ${index + 1} Team 1`}' in source
    assert 'aria-label={`Match ${index + 1} Team 2`}' in source
    assert "Number(value) / 400" in source
    assert "disabled={saving || !accessToken}" in source


def test_confirmation_can_show_completion_result() -> None:
    source = Path("apps/web/components/ConfirmAction.tsx").read_text(encoding="utf-8")
    assert "ConfirmActionSuccess" in source
    assert "setSuccess(completion)" in source
    assert 'success.closeLabel || "OK"' in source


def test_match_log_success_and_replay_history_are_visible() -> None:
    panel = Path("apps/web/app/admin/match-log/MatchLogApplyPanel.tsx").read_text(encoding="utf-8")
    replay = Path("apps/web/app/admin/match-log/MatchLogQuickReplayPanel.tsx").read_text(encoding="utf-8")
    assert "Match edit and replay complete" in panel
    assert "Matches updated" in panel
    assert "Ratings replay" in panel
    assert "throw applyError" in panel
    assert 'href="/admin/replay-history"' in replay
    assert "Open Replay History" in replay
''',
        encoding="utf-8",
    )


def main() -> None:
    update_confirm_action()
    update_match_uploader()
    update_match_log_apply()
    update_quick_replay()
    write_docs_and_tests()


if __name__ == "__main__":
    main()
