"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import styles from "./StartingRatingReview.module.css";

export type StartingRatingReviewRow = {
  player_id: number;
  name: string;
  rating_jupr: number | null;
  starting_jupr: number | null;
  rated_matches: number;
  has_rated_history: boolean;
  review_required: boolean;
  requested_rating: string | null;
  requested_jupr: number | null;
  requested_rating_differs: boolean;
  state_fingerprint: string;
  confirmed_at: string | null;
  can_edit_starting_rating?: boolean;
  review_note?: string | null;
};

export type StartingRatingReviewProps = {
  apiBase: string;
  clubId: string;
  accessToken: string | null | undefined;
  playerIds: number[];
  requestedRatings?: Record<string, string>;
  matchFormat?: "doubles" | "singles";
  onReady: (ready: boolean, rows: StartingRatingReviewRow[]) => void;
  onConfirmed?: (rows: StartingRatingReviewRow[]) => void;
  disabled?: boolean;
  title?: string;
};

function errorText(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") return fallback;
  const value = payload as { detail?: unknown; message?: unknown };
  if (typeof value.detail === "string") return value.detail;
  if (value.detail && typeof value.detail === "object" && typeof (value.detail as { message?: unknown }).message === "string") return String((value.detail as { message: string }).message);
  return typeof value.message === "string" ? value.message : fallback;
}

function startingValue(row: StartingRatingReviewRow): string {
  if (row.can_edit_starting_rating === false) return row.starting_jupr == null ? "" : String(row.starting_jupr);
  if (row.requested_rating_differs && row.requested_jupr == null) return "";
  const value = row.requested_jupr ?? row.starting_jupr ?? row.rating_jupr;
  return value == null ? "" : String(value);
}

function validRating(value: string): boolean {
  return value.trim() !== "" && Number.isFinite(Number(value)) && Number(value) >= 1 && Number(value) <= 7;
}

function displayRating(value: number | null): string {
  return value == null || !Number.isFinite(value) ? "Not set" : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, ".0");
}

export function StartingRatingReview({ apiBase, clubId, accessToken, playerIds, requestedRatings = {}, matchFormat = "doubles", onReady, onConfirmed, disabled = false, title = "Review starting ratings" }: StartingRatingReviewProps) {
  const sectionId = useId();
  const idsKey = JSON.stringify([...new Set(playerIds)].sort((a, b) => a - b));
  const requestsKey = JSON.stringify(Object.fromEntries(Object.entries(requestedRatings).filter(([id]) => playerIds.includes(Number(id))).sort(([a], [b]) => a.localeCompare(b))));
  const ids: number[] = useMemo(() => JSON.parse(idsKey), [idsKey]);
  const requests: Record<string, string> = useMemo(() => JSON.parse(requestsKey), [requestsKey]);
  const contextKey = JSON.stringify([apiBase, clubId, matchFormat, idsKey, requestsKey, accessToken]);
  const currentContext = useRef(contextKey);
  currentContext.current = contextKey;
  const callbacks = useRef({ onReady, onConfirmed });
  callbacks.current = { onReady, onConfirmed };
  const [rows, setRows] = useState<StartingRatingReviewRow[]>([]);
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [loadedContext, setLoadedContext] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [savedNotice, setSavedNotice] = useState(false);
  const requestVersion = useRef(0);
  const base = `${apiBase.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/starting-rating-review`;

  useEffect(() => {
    const version = ++requestVersion.current;
    const controller = new AbortController();
    setRows([]); setDrafts({}); setError(""); setLoadedContext(""); setSaving(false);
    callbacks.current.onReady(false, []);
    if (!ids.length) { setLoading(false); return () => controller.abort(); }
    if (!accessToken) { setLoading(false); setError("Sign in as an administrator to review starting ratings."); return () => controller.abort(); }
    if (ids.some(id => !Number.isSafeInteger(id) || id <= 0)) { setLoading(false); setError("Choose saved PCS player profiles before reviewing starting ratings."); return () => controller.abort(); }
    setLoading(true);
    async function preview() {
      try {
        const response = await fetch(`${base}/preview`, {
          method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${accessToken}` },
          body: JSON.stringify({ player_ids: ids, match_format: matchFormat, requested_ratings: requests }), signal: controller.signal
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(errorText(payload, "Starting ratings could not be checked. Try again."));
        const next = payload?.players as StartingRatingReviewRow[] | undefined;
        if (!Array.isArray(next) || next.length !== ids.length || new Set(next.map(row => row.player_id)).size !== ids.length || next.some(row => !ids.includes(row.player_id))) throw new Error("The rating review did not return every selected player. Recheck before continuing.");
        if (controller.signal.aborted || requestVersion.current !== version || currentContext.current !== contextKey) return;
        setRows(next); setDrafts(Object.fromEntries(next.filter(row => row.review_required && !row.has_rated_history).map(row => [row.player_id, startingValue(row)])));
        setLoadedContext(contextKey);
        callbacks.current.onReady(!next.some(row => row.review_required), next);
      } catch (cause) {
        if (controller.signal.aborted || requestVersion.current !== version || currentContext.current !== contextKey) return;
        setError(cause instanceof Error ? cause.message : "Starting ratings could not be checked. Try again.");
        callbacks.current.onReady(false, []);
      } finally { if (!controller.signal.aborted && requestVersion.current === version && currentContext.current === contextKey) setLoading(false); }
    }
    void preview();
    return () => controller.abort();
  }, [base, accessToken, contextKey, ids, requests, matchFormat, refresh]);

  const currentRows = loadedContext === contextKey ? rows : [];
  const pending = currentRows.filter(row => row.review_required && !row.has_rated_history);
  const allValid = pending.every(row => validRating(drafts[row.player_id] || ""));
  const estimates = currentRows.filter(row => !row.has_rated_history);
  const established = currentRows.filter(row => row.has_rated_history);

  async function confirm() {
    if (disabled || loading || saving || !accessToken || !pending.length || !allValid || loadedContext !== contextKey) return;
    const version = requestVersion.current;
    setSaving(true); setError(""); setSavedNotice(false); callbacks.current.onReady(false, currentRows);
    try {
      const response = await fetch(`${base}/confirm`, {
        method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${accessToken}` },
        body: JSON.stringify({ match_format: matchFormat, reviews: pending.map(row => ({ player_id: row.player_id, rating_jupr: Number(drafts[row.player_id]), expected_state_fingerprint: row.state_fingerprint, ...(row.requested_rating ? { requested_rating: row.requested_rating } : {}) })) })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(errorText(payload, "Starting ratings could not be confirmed. Recheck ratings before trying again."));
      if (requestVersion.current !== version || currentContext.current !== contextKey) return;
      setSavedNotice(true); callbacks.current.onConfirmed?.(Array.isArray(payload?.players) ? payload.players : currentRows);
      setRefresh(value => value + 1);
    } catch (cause) {
      if (requestVersion.current !== version || currentContext.current !== contextKey) return;
      setError(cause instanceof Error ? cause.message : "Starting ratings could not be confirmed. Recheck ratings before trying again.");
      callbacks.current.onReady(false, currentRows);
    } finally { if (requestVersion.current === version && currentContext.current === contextKey) setSaving(false); }
  }

  if (!ids.length) return null;
  return <section className={styles.panel} aria-labelledby={sectionId} aria-busy={loading || saving}>
    <h3 id={sectionId}>{title}</h3>
    <p className={styles.help}>Players with no rated {matchFormat} history need a confirmed starting estimate before rated play. Players with rated history use their calculated PCS JUPR.</p>
    {loading ? <p role="status" className={styles.status}>Checking {matchFormat} rating history…</p> : null}
    {error ? <p role="alert" className={styles.error}>{error}</p> : null}
    {savedNotice && !loading && !error ? <p role="status" className={styles.status}>Starting estimates confirmed.</p> : null}
    {established.length ? <details><summary>{established.length} {established.length === 1 ? "player has" : "players have"} rated history — calculated JUPR will be used</summary>
      <ul>{established.map(row => <li key={row.player_id}>{row.name}: JUPR {displayRating(row.rating_jupr)} · {row.rated_matches} rated {row.rated_matches === 1 ? "game" : "games"}</li>)}</ul>
    </details> : null}
    <div className={styles.rows}>{estimates.map(row => <div className={styles.row} key={row.player_id}>
      <div>
        <p className={styles.name}>{row.name}</p>
        <span className={styles.estimate}>No rated history</span>
        <span className={styles.detail}>Saved starting estimate: {displayRating(row.starting_jupr)}</span>
        {row.requested_rating ? <span className={styles.detail}>CourtReserve skill level: {row.requested_rating}</span> : null}
        {row.requested_rating_differs && row.review_required ? <p className={styles.warning}>{row.requested_jupr == null ? "Choose one starting rating for this request." : "The requested level differs from the saved estimate. Confirm the starting rating to use."}</p> : null}
        {row.review_note ? <span className={styles.detail}>{row.review_note}</span> : null}
        {!row.review_required ? <span className={styles.detail}>Starting estimate confirmed{row.confirmed_at ? ` · ${new Date(row.confirmed_at).toLocaleDateString()}` : ""}</span> : null}
      </div>
      {row.review_required ? <label className={styles.label}>Starting {matchFormat} JUPR
        <input className={styles.input} type="number" min="1" max="7" step="0.01" inputMode="decimal"
          aria-label={`${row.name} ${matchFormat} starting rating`} value={drafts[row.player_id] ?? ""}
          disabled={disabled || saving || row.can_edit_starting_rating === false}
          aria-invalid={drafts[row.player_id] !== undefined && !validRating(drafts[row.player_id]) || undefined}
          onChange={event => { setDrafts(values => ({ ...values, [row.player_id]: event.target.value })); setSavedNotice(false); callbacks.current.onReady(false, currentRows); }} />
      </label> : <span className={styles.detail}>Confirmed estimate: {displayRating(row.starting_jupr)}</span>}
    </div>)}</div>
    {!loading && !error && currentRows.length && !pending.length ? <p role="status" className={styles.status}>All selected players are ready for rated {matchFormat} play.</p> : null}
    <div className={styles.actions}>
      {pending.length ? <button className={styles.button} type="button" disabled={disabled || loading || saving || !allValid || Boolean(error)} onClick={() => void confirm()}>{saving ? "Confirming…" : `Confirm starting ${pending.length === 1 ? "rating" : "ratings"} (${pending.length})`}</button> : null}
      <button className={styles.secondary} type="button" disabled={disabled || loading || saving} onClick={() => { setSavedNotice(false); setRefresh(value => value + 1); }}>Recheck ratings</button>
    </div>
  </section>;
}

export default StartingRatingReview;
