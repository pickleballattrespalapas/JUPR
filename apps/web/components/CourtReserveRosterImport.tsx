"use client";

import { useCallback, useId, useMemo, useRef, useState } from "react";
import GeneratorPlayerCreate from "@/app/admin/play-generators/submissions/GeneratorPlayerCreate";
import SearchablePlayerSelect from "@/components/SearchablePlayerSelect";
import StartingRatingReview, { type StartingRatingReviewRow } from "@/components/StartingRatingReview";
import type { PublicPlayer } from "@/lib/api";
import { buildCourtReserveSelection, courtReserveStartingLevel, matchCourtReserveRows, parseCourtReserveReport, MAX_COURTRESERVE_FILE_BYTES, type CourtReserveMatchedRow, type CourtReservePlayer, type CourtReserveRosterSelection } from "@/lib/courtReserveRosterImport";
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
  const [additionalPlayers, setAdditionalPlayers] = useState<PublicPlayer[]>([]);
  const [pendingCreations, setPendingCreations] = useState<Set<string>>(new Set());
  const [creationBusy, setCreationBusy] = useState(false);
  const reviewedPlayers = useRef<StartingRatingReviewRow[]>([]);
  const readVersion = useRef(0);
  const playerOptions = useMemo(() => [...new Map([...players, ...additionalPlayers].map(player => [Number(player.id), { id: Number(player.id), name: player.name }])).values()].sort((a, b) => a.name.localeCompare(b.name)), [players, additionalPlayers]);
  const availableIds = useMemo(() => new Set(playerOptions.map(player => player.id)), [playerOptions]);
  const creationPending = pendingCreations.size > 0;
  const editingBlocked = disabled || reading || saving || creationBusy || creationPending;
  const handleCreationPending = useCallback((id: string, pending: boolean) => {
    setPendingCreations(previous => { const next = new Set(previous); if (pending) next.add(id); else next.delete(id); return next; });
    setReadyKey("");
  }, []);
  const selection = useMemo(() => {
    if (!rows.length) return { value: null, error: "" };
    try {
      const value = buildCourtReserveSelection(rows, choices, skipped);
      if (value.playerIds.some(id => !availableIds.has(id))) throw new Error("A selected player is no longer available. Choose a current PCS profile.");
      return { value, error: "" };
    } catch (cause) { return { value: null, error: cause instanceof Error ? cause.message : "Check the selected players." }; }
  }, [rows, choices, skipped, availableIds]);
  const selectionKey = selection.value ? JSON.stringify([clubId, matchFormat, selection.value]) : "";
  const ready = Boolean(selection.value) && !creationPending && !creationBusy && (!requireRatingReview || readyKey === selectionKey);
  const selectedCounts = new Map<number, number>();
  for (const row of rows) {
    const selected = choices[row.rowNumber];
    if (selected && !skipped.has(row.rowNumber)) selectedCounts.set(selected, (selectedCounts.get(selected) || 0) + 1);
  }

  function loadReport(raw: string) {
    setReadyKey(""); setError(""); setSkipped(new Set());
    try {
      const matched = matchCourtReserveRows(parseCourtReserveReport(raw), playerOptions);
      setRows(matched); setChoices(Object.fromEntries(matched.map(row => [row.rowNumber, row.playerId])));
    } catch (cause) {
      setRows([]); setChoices({}); setError(cause instanceof Error ? cause.message : "The report could not be read.");
    }
  }

  async function selectCreatedPlayer(rowNumber: number, playerId: number) {
    const response = await fetch(`${apiBase.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/players/editor/players/${playerId}`, {
      cache: "no-store", headers: { Authorization: `Bearer ${accessToken}` }
    });
    if (!response.ok) throw new Error("The player could not be loaded. Check player creation to try again.");
    const payload = await response.json() as { player?: PublicPlayer & { club_id?: string } };
    const player = payload.player;
    if (!player || Number(player.id) !== playerId || !player.name?.trim() || (player.club_id && player.club_id !== clubId)) throw new Error("Could not confirm the player's PCS profile.");
    setAdditionalPlayers(current => [...current.filter(existing => Number(existing.id) !== playerId), player]);
    setChoices(current => ({ ...current, [rowNumber]: playerId }));
    setReadyKey("");
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
    if (!selection.value || !ready || editingBlocked) return;
    setSaving(true); setError("");
    try {
      const selectedAdditionalPlayers = additionalPlayers.filter(player => selection.value!.playerIds.includes(Number(player.id))).map(player => {
        const review = reviewedPlayers.current.find(row => row.player_id === Number(player.id));
        if (!requireRatingReview || !review || review.rating_jupr == null) return player;
        return matchFormat === "singles"
          ? { ...player, singles_rating: review.rating_jupr * 400, singles_rating_jupr: review.rating_jupr }
          : { ...player, rating: review.rating_jupr * 400, rating_jupr: review.rating_jupr };
      });
      await onImport({ ...selection.value, ...(selectedAdditionalPlayers.length ? { additionalPlayers: selectedAdditionalPlayers } : {}) });
    }
    catch (cause) { setError(cause instanceof Error ? cause.message : "The roster could not be updated. Check the roster before trying again."); }
    finally { setSaving(false); }
  }

  return <section className={styles.panel} aria-labelledby={titleId} aria-busy={reading || saving || creationBusy}>
    <h3 id={titleId}>Import CourtReserve roster</h3>
    <p className={styles.help}>Upload a CSV report or paste the name and skill columns from Numbers or Excel, including their headings. Review the matching PCS profiles before adding them.</p>
    <label className={styles.label}>CourtReserve report
      <input className={styles.file} type="file" accept=".csv,.tsv,.txt,text/csv,text/tab-separated-values" disabled={editingBlocked}
        aria-label="Upload CourtReserve report" onChange={event => { void chooseFile(event.target.files?.[0]); event.target.value = ""; }} />
    </label>
    {fileName ? <p className={styles.quiet}>Selected report: {fileName}</p> : null}
    <label className={styles.label}>Or paste report cells
      <textarea className={styles.textarea} value={text} disabled={editingBlocked} aria-label="CourtReserve report cells"
        placeholder={"Enter Your Skill Level\tFull Name\n3.5\tPlayer name"}
        onChange={event => { ++readVersion.current; setText(event.target.value); setRows([]); setChoices({}); setReadyKey(""); setError(""); setFileName(""); }} />
    </label>
    <div className={styles.actions}><button className={styles.secondary} type="button" disabled={editingBlocked || !text.trim()} onClick={() => loadReport(text)}>Preview report</button></div>
    {reading ? <p className={styles.status} role="status">Reading report…</p> : null}
    {error ? <p className={styles.error} role="alert">{error}</p> : null}
    {rows.length ? <>
      <p className={styles.help}>{rows.length} {rows.length === 1 ? "registrant" : "registrants"} found. Choose an existing PCS profile or create a new player beside an unmatched name. Uncheck Include to skip someone.</p>
      <div className={styles.tableWrap}><table className={styles.table}>
        <thead><tr><th scope="col">Report name</th><th scope="col">Requested skill</th><th scope="col">PCS player</th><th scope="col">Include</th></tr></thead>
        <tbody>{rows.map(row => {
          const selected = choices[row.rowNumber];
          const isSkipped = skipped.has(row.rowNumber);
          const duplicate = !isSkipped && Boolean(selected && (selectedCounts.get(selected) || 0) > 1);
          return <tr key={`${row.rowNumber}:${row.name}`}>
            <td>{row.name}<span className={styles.detail}>Report row {row.rowNumber}</span></td>
            <td>{row.requestedRating || <span className={styles.quiet}>Not supplied</span>}</td>
            <td><SearchablePlayerSelect aria-label={`PCS profile for report row ${row.rowNumber}: ${row.name}`} value={selected ? String(selected) : ""}
              disabled={editingBlocked || isSkipped} onValueChange={value => { setChoices(previous => ({ ...previous, [row.rowNumber]: value ? Number(value) : null })); setReadyKey(""); }}>
              <option value="">Choose a PCS profile</option>
              {playerOptions.map(player => <option key={player.id} value={player.id}>{player.name} · #{player.id}</option>)}
            </SearchablePlayerSelect>
              {!isSkipped && !selected ? <span className={styles.detail}>{row.matchStatus === "ambiguous" ? "More than one profile matches this name. Choose the correct person." : "No exact profile match. Choose an existing profile or create a new player below."}</span> : null}
              {duplicate ? <span className={styles.error}>This PCS player is selected more than once.</span> : null}
              {!isSkipped && accessToken ? <GeneratorPlayerCreate context="courtreserve" clubId={clubId}
                sessionKey={`courtreserve:${row.rowNumber}:${row.name}`} participantId={String(row.rowNumber)} participantName={row.name}
                initialStartingLevel={courtReserveStartingLevel(row.requestedRating)} apiBase={apiBase} accessToken={accessToken}
                players={playerOptions} disabled={disabled || reading || saving || creationBusy || (creationPending && !pendingCreations.has(String(row.rowNumber)))} matched={Boolean(selected)}
                onPendingChange={handleCreationPending} onBusyChange={setCreationBusy}
                onCreated={playerId => selectCreatedPlayer(row.rowNumber, playerId)} /> : null}
            </td>
            <td><label className={styles.skip}><input type="checkbox" checked={!isSkipped} disabled={editingBlocked}
              aria-label={`Include report row ${row.rowNumber}: ${row.name}`} onChange={event => { setSkipped(previous => { const next = new Set(previous); if (event.target.checked) next.delete(row.rowNumber); else next.add(row.rowNumber); return next; }); setReadyKey(""); }} />Include</label></td>
          </tr>;
        })}</tbody>
      </table></div>
      {selection.error ? <p className={styles.warning} role="status">{selection.error}</p> : null}
      {creationPending ? <p className={styles.warning} role="status">Finish or check the player creation above before continuing the import.</p> : null}
      {selection.value && requireRatingReview && !creationPending && !creationBusy ? <StartingRatingReview key={selectionKey} apiBase={apiBase} clubId={clubId} accessToken={accessToken}
        playerIds={selection.value.playerIds} requestedRatings={selection.value.requestedRatings} matchFormat={matchFormat} disabled={editingBlocked}
        onReady={(value, reviewed) => { reviewedPlayers.current = reviewed; setReadyKey(value ? selectionKey : ""); }} /> : null}
      {selection.value && !requireRatingReview ? <p className={styles.status}>The report will add names to this unrated roster. Starting ratings will be reviewed before any rated play.</p> : null}
      <div className={styles.actions}>
        <button className={styles.button} type="button" disabled={editingBlocked || !ready} onClick={() => void submit()}>{saving ? "Adding players…" : `${submitLabel}${selection.value ? ` (${selection.value.playerIds.length})` : ""}`}</button>
        {onCancel ? <button className={styles.secondary} type="button" disabled={saving || creationBusy || creationPending} onClick={onCancel}>Cancel import</button> : null}
      </div>
    </> : onCancel ? <div className={styles.actions}><button className={styles.secondary} type="button" disabled={saving} onClick={onCancel}>Cancel import</button></div> : null}
  </section>;
}

export default CourtReserveRosterImport;
