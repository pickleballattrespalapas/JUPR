"use client";

import { useId, useMemo, useRef, useState } from "react";
import SearchablePlayerSelect from "@/components/SearchablePlayerSelect";
import StartingRatingReview from "@/components/StartingRatingReview";
import { buildCourtReserveSelection, matchCourtReserveRows, parseCourtReserveReport, MAX_COURTRESERVE_FILE_BYTES, type CourtReserveMatchedRow, type CourtReservePlayer, type CourtReserveRosterSelection } from "@/lib/courtReserveRosterImport";
import styles from "./StartingRatingReview.module.css";

export type CourtReserveRosterImportProps = {
  apiBase: string;
  clubId: string;
  accessToken: string | null | undefined;
  players: CourtReservePlayer[];
  matchFormat?: "doubles" | "singles";
  requireRatingReview?: boolean;
  onImport: (selection: CourtReserveRosterSelection) => void | Promise<void>;
  onCancel?: () => void;
  disabled?: boolean;
  submitLabel?: string;
};

export function CourtReserveRosterImport({ apiBase, clubId, accessToken, players, matchFormat = "doubles", requireRatingReview = true, onImport, onCancel, disabled = false, submitLabel = "Use these players" }: CourtReserveRosterImportProps) {
  const titleId = useId();
  const [text, setText] = useState("");
  const [fileName, setFileName] = useState("");
  const [rows, setRows] = useState<CourtReserveMatchedRow[]>([]);
  const [choices, setChoices] = useState<Record<number, number | null>>({});
  const [skipped, setSkipped] = useState<Set<number>>(new Set());
  const [error, setError] = useState("");
  const [reading, setReading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [readyKey, setReadyKey] = useState("");
  const readVersion = useRef(0);
  const availableIds = useMemo(() => new Set(players.map(player => player.id)), [players]);
  const playerOptions = useMemo(() => [...players].sort((a, b) => a.name.localeCompare(b.name)), [players]);
  const selection = useMemo(() => {
    if (!rows.length) return { value: null, error: "" };
    try {
      const value = buildCourtReserveSelection(rows, choices, skipped);
      if (value.playerIds.some(id => !availableIds.has(id))) throw new Error("A selected player is no longer available. Choose a current PCS profile.");
      return { value, error: "" };
    } catch (cause) { return { value: null, error: cause instanceof Error ? cause.message : "Check the selected players." }; }
  }, [rows, choices, skipped, availableIds]);
  const selectionKey = selection.value ? JSON.stringify([clubId, matchFormat, selection.value]) : "";
  const ready = Boolean(selection.value) && (!requireRatingReview || readyKey === selectionKey);
  const selectedCounts = new Map<number, number>();
  for (const row of rows) {
    const selected = choices[row.rowNumber];
    if (selected && !skipped.has(row.rowNumber)) selectedCounts.set(selected, (selectedCounts.get(selected) || 0) + 1);
  }

  function loadReport(raw: string) {
    setReadyKey(""); setError(""); setSkipped(new Set());
    try {
      const matched = matchCourtReserveRows(parseCourtReserveReport(raw), players);
      setRows(matched); setChoices(Object.fromEntries(matched.map(row => [row.rowNumber, row.playerId])));
    } catch (cause) {
      setRows([]); setChoices({}); setError(cause instanceof Error ? cause.message : "The report could not be read.");
    }
  }

  async function chooseFile(file?: File) {
    const version = ++readVersion.current;
    setError(""); setReadyKey(""); setRows([]); setChoices({});
    if (!file) return;
    if (file.size > MAX_COURTRESERVE_FILE_BYTES) { setError("Choose a report smaller than 2 MB."); return; }
    if (!/\.(csv|tsv|txt)$/i.test(file.name)) { setError("Export the report from CourtReserve, Numbers, or Excel as CSV, or copy its cells into the box below."); return; }
    setReading(true); setFileName(file.name);
    try {
      const raw = await file.text();
      if (version !== readVersion.current) return;
      setText(raw); loadReport(raw);
    } catch { if (version === readVersion.current) setError("The file could not be opened. Try exporting it as CSV again or paste its cells."); }
    finally { if (version === readVersion.current) setReading(false); }
  }

  async function submit() {
    if (!selection.value || !ready || disabled || reading || saving) return;
    setSaving(true); setError("");
    try { await onImport(selection.value); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "The roster could not be updated. Check the roster before trying again."); }
    finally { setSaving(false); }
  }

  return <section className={styles.panel} aria-labelledby={titleId} aria-busy={reading || saving}>
    <h3 id={titleId}>Import CourtReserve roster</h3>
    <p className={styles.help}>Upload a CSV report or paste the name and skill columns from Numbers or Excel, including their headings. Review the matching PCS profiles before adding them.</p>
    <label className={styles.label}>CourtReserve report
      <input className={styles.file} type="file" accept=".csv,.tsv,.txt,text/csv,text/tab-separated-values" disabled={disabled || reading || saving}
        aria-label="Upload CourtReserve report" onChange={event => { void chooseFile(event.target.files?.[0]); event.target.value = ""; }} />
    </label>
    {fileName ? <p className={styles.quiet}>Selected report: {fileName}</p> : null}
    <label className={styles.label}>Or paste report cells
      <textarea className={styles.textarea} value={text} disabled={disabled || reading || saving} aria-label="CourtReserve report cells"
        placeholder={"Enter Your Skill Level\tFull Name\n3.5\tPlayer name"}
        onChange={event => { ++readVersion.current; setText(event.target.value); setRows([]); setChoices({}); setReadyKey(""); setError(""); setFileName(""); }} />
    </label>
    <div className={styles.actions}><button className={styles.secondary} type="button" disabled={disabled || reading || saving || !text.trim()} onClick={() => loadReport(text)}>Preview report</button></div>
    {reading ? <p className={styles.status} role="status">Reading report…</p> : null}
    {error ? <p className={styles.error} role="alert">{error}</p> : null}
    {rows.length ? <>
      <p className={styles.help}>{rows.length} {rows.length === 1 ? "registrant" : "registrants"} found. Check name matches. Missing or ambiguous names need a selected profile; you can skip rows you do not want to add.</p>
      <div className={styles.tableWrap}><table className={styles.table}>
        <thead><tr><th scope="col">Report name</th><th scope="col">Requested skill</th><th scope="col">PCS player</th><th scope="col">Include</th></tr></thead>
        <tbody>{rows.map(row => {
          const selected = choices[row.rowNumber];
          const isSkipped = skipped.has(row.rowNumber);
          const duplicate = !isSkipped && Boolean(selected && (selectedCounts.get(selected) || 0) > 1);
          return <tr key={row.rowNumber}>
            <td>{row.name}<span className={styles.detail}>Report row {row.rowNumber}</span></td>
            <td>{row.requestedRating || <span className={styles.quiet}>Not supplied</span>}</td>
            <td><SearchablePlayerSelect aria-label={`PCS profile for report row ${row.rowNumber}: ${row.name}`} value={selected ? String(selected) : ""}
              disabled={disabled || saving || isSkipped} onValueChange={value => { setChoices(previous => ({ ...previous, [row.rowNumber]: value ? Number(value) : null })); setReadyKey(""); }}>
              <option value="">Choose a PCS profile</option>
              {playerOptions.map(player => <option key={player.id} value={player.id}>{player.name} · #{player.id}</option>)}
            </SearchablePlayerSelect>
              {!isSkipped && !selected ? <span className={styles.detail}>{row.matchStatus === "ambiguous" ? "More than one profile matches this name. Choose the correct person." : "No exact profile match. Choose a profile or skip this row."}</span> : null}
              {duplicate ? <span className={styles.error}>This PCS player is selected more than once.</span> : null}
            </td>
            <td><label className={styles.skip}><input type="checkbox" checked={!isSkipped} disabled={disabled || saving}
              aria-label={`Include report row ${row.rowNumber}: ${row.name}`} onChange={event => { setSkipped(previous => { const next = new Set(previous); if (event.target.checked) next.delete(row.rowNumber); else next.add(row.rowNumber); return next; }); setReadyKey(""); }} />Include</label></td>
          </tr>;
        })}</tbody>
      </table></div>
      {selection.error ? <p className={styles.warning} role="status">{selection.error}</p> : null}
      <p className={styles.quiet}>If a player has no PCS profile yet, create one in Players, then reload the roster list and import the report again.</p>
      {selection.value && requireRatingReview ? <StartingRatingReview key={selectionKey} apiBase={apiBase} clubId={clubId} accessToken={accessToken}
        playerIds={selection.value.playerIds} requestedRatings={selection.value.requestedRatings} matchFormat={matchFormat} disabled={disabled || saving}
        onReady={value => setReadyKey(value ? selectionKey : "")} /> : null}
      {selection.value && !requireRatingReview ? <p className={styles.status}>The report will add names to this unrated roster. Starting ratings will be reviewed before any rated play.</p> : null}
      <div className={styles.actions}>
        <button className={styles.button} type="button" disabled={disabled || reading || saving || !ready} onClick={() => void submit()}>{saving ? "Adding players…" : `${submitLabel}${selection.value ? ` (${selection.value.playerIds.length})` : ""}`}</button>
        {onCancel ? <button className={styles.secondary} type="button" disabled={saving} onClick={onCancel}>Cancel import</button> : null}
      </div>
    </> : onCancel ? <div className={styles.actions}><button className={styles.secondary} type="button" disabled={saving} onClick={onCancel}>Cancel import</button></div> : null}
  </section>;
}

export default CourtReserveRosterImport;
