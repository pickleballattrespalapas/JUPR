"use client";

import { useEffect, useRef, useState } from "react";
import { InteractionDialog } from "@/components/interaction";
import type { AdminPlayerEditorDetailResponse, AdminPlayerEditorOperationResponse, AdminPlayerEditorWriteResponse } from "@/lib/adminPlayerEditorApi";

type Props = {
  playerId: number;
  clubId: string;
  leagueName: string;
  request: <T>(path: string, options?: RequestInit) => Promise<T>;
  onSaved: (playerId: number, rating: number, scope: "club" | "league") => void;
  onClose: () => void;
};
type Pending = { key: string; path: string; body: Record<string, unknown>; scope: "club" | "league" };
const button = { padding: "0.55rem 0.8rem", border: "1px solid #cbd5e1", borderRadius: 8, background: "white", font: "inherit", cursor: "pointer" };
const input = { padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: 8, font: "inherit" };

export function LeagueLiveRatingEditor({ playerId, clubId, leagueName, request, onSaved, onClose }: Props) {
  const [detail, setDetail] = useState<AdminPlayerEditorDetailResponse | null>(null);
  const [rating, setRating] = useState("");
  const [correctStarting, setCorrectStarting] = useState(false);
  const [confirmation, setConfirmation] = useState("");
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const mounted = useRef(true);
  const latest = useRef({ request, onSaved });
  latest.current = { request, onSaved };
  const base = `/admin/clubs/${encodeURIComponent(clubId)}/players/editor`;
  const storageKey = `live-rating-edit:${clubId}:${playerId}`;
  const leagueRating = detail?.league_ratings.find(row => row.league_name === leagueName);
  const scope = leagueRating ? "league" : "club";
  const row = leagueRating ?? detail?.player;

  useEffect(() => {
    mounted.current = true;
    try {
      const stored = JSON.parse(sessionStorage.getItem(storageKey) || "null") as Pending | null;
      if (stored?.key && stored.path && stored.body) setPending(stored);
    } catch { /* Recovery remains available in memory when storage is unavailable. */ }
    void latest.current.request<AdminPlayerEditorDetailResponse>(`${base}/players/${playerId}`).then(payload => {
      if (!mounted.current) return;
      setDetail(payload);
      const target = payload.league_ratings.find(item => item.league_name === leagueName) ?? payload.player;
      setRating(target.rating_jupr == null ? "" : String(target.rating_jupr));
      setCorrectStarting(Number(target.matches_played || 0) === 0 && Number(payload.match_reference_counts?.total || 0) === 0);
    }).catch(reason => { if (mounted.current) setError(reason instanceof Error ? reason.message : "Unable to load this player."); })
      .finally(() => { if (mounted.current) setBusy(false); });
    return () => { mounted.current = false; };
  }, [base, playerId, leagueName, storageKey]);

  function retain(value: Pending | null) {
    setPending(value);
    try { if (value) sessionStorage.setItem(storageKey, JSON.stringify(value)); else sessionStorage.removeItem(storageKey); } catch { /* In-memory recovery still holds the exact request. */ }
  }

  function finish(payload: AdminPlayerEditorWriteResponse, targetScope: "club" | "league") {
    const saved = targetScope === "league" ? payload.league_rating ?? payload.league_ratings?.find(item => item.league_name === leagueName) : payload.player;
    if (!payload.ok || saved?.rating == null) throw new Error("The saved rating could not be verified. Check the saved edit before retrying.");
    retain(null);
    latest.current.onSaved(playerId, Number(saved.rating), targetScope);
  }

  async function save() {
    const value = Number(rating);
    if (busy || pending || !row?.state_fingerprint || !rating.trim() || !Number.isFinite(value) || value < 1 || value > 7) return;
    if (scope === "league" && confirmation.trim().toUpperCase() !== "SAVE LEAGUE RATING") return;
    const operation: Pending = {
      key: `live-rating:${crypto.randomUUID()}`,
      path: `${base}/players/${playerId}${leagueRating ? `/league-ratings/${leagueRating.id}` : ""}`,
      scope,
      body: { rating_jupr: value, ...(correctStarting ? { starting_jupr: value } : {}), expected_state_fingerprint: row.state_fingerprint, source: "next_live_court_rating", ...(leagueRating ? { confirmation_text: confirmation.trim().toUpperCase() } : {}) },
    };
    retain(operation);
    setBusy(true); setError(null);
    try {
      const payload = await request<AdminPlayerEditorWriteResponse>(operation.path, { method: "PATCH", body: JSON.stringify({ ...operation.body, idempotency_key: operation.key }) });
      if (mounted.current) finish(payload, operation.scope);
    } catch (reason) {
      if (!mounted.current) return;
      const failure = reason as { status?: number; uncertain?: boolean };
      if (failure.status && failure.status < 500 && ![408, 425, 429].includes(failure.status) && !failure.uncertain) retain(null);
      setError(reason instanceof Error ? reason.message : "Unable to confirm the rating edit. Check the saved edit before retrying.");
    } finally { if (mounted.current) setBusy(false); }
  }

  async function checkSavedEdit() {
    if (!pending || busy) return;
    setBusy(true); setError(null);
    try {
      const operation = await request<AdminPlayerEditorOperationResponse>(`${base}/operations/${encodeURIComponent(pending.key)}`);
      if (!mounted.current) return;
      if (operation.status === "completed" && operation.result) finish(operation.result, pending.scope);
      else if (operation.status === "failed" && !operation.recovery_required) { retain(null); setError("The edit did not save. Close and reopen the rating editor to reload the latest rating."); setDetail(null); }
      else {
        const payload = await request<AdminPlayerEditorWriteResponse>(`${base}/operations/${encodeURIComponent(pending.key)}/reconcile`, { method: "POST", body: JSON.stringify({ confirmation_text: "RECONCILE PLAYER OPERATION", source: "next_live_court_rating" }) });
        if (mounted.current) finish(payload, pending.scope);
      }
    } catch (reason) { if (mounted.current) setError(reason instanceof Error ? reason.message : "Unable to verify the saved edit."); }
    finally { if (mounted.current) setBusy(false); }
  }

  const valid = Boolean(rating.trim()) && Number.isFinite(Number(rating)) && Number(rating) >= 1 && Number(rating) <= 7;
  return <InteractionDialog open phase={busy ? "working" : "ready"} title={`Edit rating${detail ? ` · ${detail.player.name}` : ""}`} onRequestClose={() => { if (!busy) onClose(); }}
    actions={<><button type="button" style={button} disabled={busy} onClick={onClose}>Close</button>{pending ? <button type="button" style={button} disabled={busy} onClick={() => void checkSavedEdit()}>Check saved rating edit</button> : <button type="button" style={button} disabled={busy || !row || !valid || (scope === "league" && confirmation.trim().toUpperCase() !== "SAVE LEAGUE RATING")} onClick={() => void save()}>Save rating</button>}</>}>
    {busy && !detail ? <p>Loading player rating…</p> : null}
    {detail ? <div style={{ display: "grid", gap: "0.8rem" }}>
      <p style={{ margin: 0 }}>{scope === "league" ? `Updates the rating for ${leagueName}.` : "Updates this player's club rating."} Court positions stay as you arranged them.</p>
      <label>Rating <input aria-label="Player rating" type="number" min={1} max={7} step={0.01} style={input} value={rating} onChange={event => setRating(event.target.value)} disabled={busy || Boolean(pending)} /></label>
      <label><input type="checkbox" checked={correctStarting} disabled={busy || Boolean(pending)} onChange={event => setCorrectStarting(event.target.checked)} /> Also correct the starting rating</label>
      {Number(row?.matches_played || 0) > 0 ? <p>Recorded matches are not recalculated by this edit.</p> : null}
      {scope === "league" ? <label>Type SAVE LEAGUE RATING to confirm <input aria-label="Confirm league rating edit" style={input} value={confirmation} onChange={event => setConfirmation(event.target.value)} disabled={busy || Boolean(pending)} /></label> : null}
    </div> : null}
    {pending ? <p role="status">This edit is awaiting verification. Check its saved result before making another change.</p> : null}
    {error ? <p role="alert">{error}</p> : null}
  </InteractionDialog>;
}
