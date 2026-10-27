"use client";

import SearchablePlayerSelect from "@/components/SearchablePlayerSelect";

import { useCallback, useEffect, useRef, useState } from "react";
import { useAdminSession } from "@/lib/useAdminSession";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, InteractionActionError } from "@/components/interaction";
import GeneratorPlayerCreate from "./GeneratorPlayerCreate";

type ReviewRequest = { expected_version: number; action: "approve" | "reject"; player_ids: Record<string, number>; match_date: string; reason: string };
type MatchFormat = "doubles" | "singles";
type PlayerRating = { rating_jupr: number | null; matches_played: number; can_set_starting_rating: boolean; state_fingerprint: string };
type Submission = {
  id: string; session_key: string; title: string; version: number; status: string;
  organizer_name: string; rating_mode: "rated" | "unrated"; match_date: string; match_count: number; approved_mode?: string;
  review?: { request: ReviewRequest };
  participants: Array<{ id: string; name: string; player_id?: number | null }>;
  matches: Array<{ id: string; round_number: number; sideA?: string[]; sideB?: string[]; teamA?: string[]; teamB?: string[]; scoreA: number; scoreB: number }>;
};
type Queue = { submissions: Submission[]; players: Array<{ id: number; name: string; ratings?: Record<MatchFormat, PlayerRating> }> };
const field = { padding: "0.6rem", border: "1px solid #cbd5e1", borderRadius: 8, font: "inherit", maxWidth: "100%" };
const scoreKey = (match: Submission["matches"][number]) => `${match.round_number}:${match.id}`;
const validScore = (value: string) => /^\d{1,2}$/.test(value);

function playerFormats(item: Submission, participantId: string): MatchFormat[] {
  const formats = new Set<MatchFormat>();
  for (const match of item.matches) {
    const a = match.sideA || match.teamA || [];
    const b = match.sideB || match.teamB || [];
    if ([...a, ...b].includes(participantId)) formats.add(a.length === 1 && b.length === 1 ? "singles" : "doubles");
  }
  return [...formats];
}

function PlayerProfileChoice({ name, value, players, disabled, onChange }: {
  name: string; value?: number; players: Queue["players"]; disabled: boolean; onChange: (value: number) => void;
}) {
  return <div style={{ display: "grid", gap: 4 }}>
    <strong>{name}</strong>
    <SearchablePlayerSelect aria-label={`Club profile for ${name}`} value={value || ""} disabled={disabled} style={field} onValueChange={playerValue => onChange(Number(playerValue))}>
      <option value="">Choose a club player</option>{players.map(player => <option key={player.id} value={player.id}>{player.name}</option>)}
    </SearchablePlayerSelect>
  </div>;
}

export default function GeneratorSubmissions({ clubId, apiBase }: { clubId: string; apiBase: string }) {
  const { accessToken } = useAdminSession();
  const [queue, setQueue] = useState<Queue>({ submissions: [], players: [] });
  const [status, setStatus] = useState("pending");
  const [selected, setSelected] = useState<Submission | null>(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const [editingScores, setEditingScores] = useState(false);
  const [scoreDrafts, setScoreDrafts] = useState<Record<string, { a: string; b: string }>>({});
  const [scoreRound, setScoreRound] = useState("");
  const [scorePlayer, setScorePlayer] = useState("");
  const [sessionRating, setSessionRating] = useState<"rated" | "unrated">("unrated");
  const [titleDraft, setTitleDraft] = useState("");
  const [mapping, setMapping] = useState<Record<string, number>>({});
  const [ratingDrafts, setRatingDrafts] = useState<Record<string, string>>({});
  const [pendingCreations, setPendingCreations] = useState<Record<string, boolean>>({});
  const markCreationPending = useCallback((participantId: string, pending: boolean) => {
    setPendingCreations(old => old[participantId] === pending ? old : { ...old, [participantId]: pending });
  }, []);
  const [matchDate, setMatchDate] = useState("");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const generation = useRef(0);
  const queueScope = useRef("");
  const followedNotification = useRef("");
  const focusReview = useRef(false);
  const reviewPanel = useRef<HTMLElement | null>(null);
  const request = useCallback(async (path: string, body?: unknown, signal?: AbortSignal, method?: string) => {
    if (!apiBase || !accessToken) throw new InteractionActionError("Sign in to review generator results.", { kind: "forbidden" });
    const response = await fetch(`${apiBase.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/play-generators/${path}`, {
      method: method || (body ? "POST" : "GET"), headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined, cache: "no-store", signal
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      // Only deliberate client-facing API failures are safe to show in the
      // confirmation dialog. Network failures and server errors stay generic.
      if (response.status >= 400 && response.status < 500 && typeof payload?.detail === "string" && payload.detail.trim()) {
        throw new InteractionActionError(payload.detail, {
          kind: response.status === 409 ? "conflict" : response.status === 401 || response.status === 403 ? "forbidden" : "validation"
        });
      }
      throw new Error("Could not load generator results. Refresh and try again.");
    }
    if (!payload) throw new Error("Could not read generator results. Refresh and try again.");
    return payload;
  }, [accessToken, apiBase, clubId]);
  const refresh = useCallback(async () => {
    const current = ++generation.current;
    setBusy(true);
    try { const payload = await request(`submissions?status=${status}`); if (current === generation.current) { queueScope.current = `${clubId}:${accessToken}`; setQueue(payload); setSelected(null); } }
    catch (error) { if (current === generation.current) setMessage(error instanceof Error ? error.message : "Could not refresh results."); }
    finally { if (current === generation.current) setBusy(false); }
  }, [accessToken, clubId, request, status]);
  useEffect(() => {
    const controller = new AbortController();
    const current = ++generation.current;
    queueScope.current = "";
    setQueue({ submissions: [], players: [] }); setSelected(null); setMessage("");
    if (!accessToken) return;
    setBusy(true);
    request(`submissions?status=${status}`, undefined, controller.signal).then(payload => {
      if (!controller.signal.aborted && current === generation.current) { queueScope.current = `${clubId}:${accessToken}`; setQueue(payload); }
    }).catch(error => { if (!controller.signal.aborted) setMessage(error.message); }).finally(() => { if (current === generation.current) setBusy(false); });
    return () => { controller.abort(); };
  }, [accessToken, clubId, request, status]);
  const select = useCallback((item: Submission) => {
    setEditingTitle(false); setTitleDraft(item.title); setSessionRating(item.rating_mode);
    setEditingScores(false); setScoreDrafts({}); setScoreRound(""); setScorePlayer("");
    setRatingDrafts({}); setPendingCreations({}); setMessage("");
    setSelected(item); setReason(item.review?.request.reason || ""); setMatchDate(item.review?.request.match_date || item.match_date);
    const ids: Record<string, number> = {};
    for (const participant of item.participants) {
      const exact = queue.players.filter(p => p.name.trim().toLocaleLowerCase() === participant.name.trim().toLocaleLowerCase());
      const chosen = participant.player_id || (exact.length === 1 ? exact[0].id : 0);
      if (chosen) ids[participant.id] = chosen;
    }
    setMapping(item.review?.request.player_ids || ids);
  }, [queue.players]);
  useEffect(() => {
    if (typeof window === "undefined" || !accessToken || queueScope.current !== `${clubId}:${accessToken}`) return;
    const session = new URLSearchParams(window.location.search).get("session");
    const linkKey = `${clubId}:${accessToken}:${session}`;
    if (!session || followedNotification.current === linkKey) return;
    const item = queue.submissions.find(submission => submission.session_key === session);
    if (!item) return;
    followedNotification.current = linkKey;
    focusReview.current = true;
    select(item);
  }, [accessToken, clubId, queue.submissions, select]);
  useEffect(() => {
    if (!selected || !focusReview.current || !reviewPanel.current) return;
    focusReview.current = false;
    reviewPanel.current.scrollIntoView({ block: "start" });
    reviewPanel.current.focus({ preventScroll: true });
  }, [selected]);

  async function saveTitle() {
    if (!selected || selected.status !== "pending" || busy || editingScores) return;
    const title = titleDraft.trim().replace(/\s+/g, " ");
    if (!title || title.length > 160 || (title === selected.title && sessionRating === selected.rating_mode)) return;
    const scope = clubId + ":" + accessToken;
    const current = generation.current;
    setBusy(true); setMessage("");
    try {
      const payload = await request(`sessions/${encodeURIComponent(selected.session_key)}/details`, {
        expected_version: selected.version, title, rating_mode: sessionRating
      }, undefined, "PATCH");
      if (queueScope.current !== scope || generation.current !== current) return;
      const updated = { ...selected, title: payload.title, rating_mode: payload.rating_mode, version: payload.version };
      setSelected(updated);
      setQueue(old => ({ ...old, submissions: old.submissions.map(item => item.session_key === updated.session_key ? updated : item) }));
      setTitleDraft(payload.title); setEditingTitle(false); setMessage("Session details saved.");
    } catch (error) {
      if (queueScope.current === scope && generation.current === current) setMessage(error instanceof Error ? error.message : "Could not save session details. Try again.");
    } finally {
      if (queueScope.current === scope && generation.current === current) setBusy(false);
    }
  }
  const scoreChanges = (selected?.matches || []).flatMap(match => {
    const draft = scoreDrafts[scoreKey(match)];
    if (!draft || (validScore(draft.a) && validScore(draft.b) && Number(draft.a) === match.scoreA && Number(draft.b) === match.scoreB)) return [];
    return [{ match, draft, valid: validScore(draft.a) && validScore(draft.b) && Number(draft.a) !== Number(draft.b) }];
  });
  const scoresValid = scoreChanges.every(change => change.valid);
  async function saveScoreCorrections() {
    if (!selected || selected.status !== "pending" || busy || editingTitle || !editingScores || !scoreChanges.length || !scoresValid) return;
    const scope = `${clubId}:${accessToken}`;
    const current = generation.current;
    setBusy(true); setMessage("");
    try {
      const payload = await request(`sessions/${encodeURIComponent(selected.session_key)}/submission/scores`, {
        expected_version: selected.version,
        scores: scoreChanges.map(({ match, draft }) => ({ round_number: match.round_number, match_id: match.id, score_a: Number(draft.a), score_b: Number(draft.b) }))
      }, undefined, "PATCH");
      if (queueScope.current !== scope || generation.current !== current) return;
      const updated: Submission = payload.submission;
      if (!updated || updated.session_key !== selected.session_key || !Array.isArray(updated.matches)) throw new Error("Could not confirm the saved scores.");
      setSelected(updated);
      setQueue(old => ({ ...old, submissions: old.submissions.map(item => item.session_key === updated.session_key ? updated : item) }));
      setEditingScores(false); setScoreDrafts({});
      setMessage("Score corrections saved. This session is still awaiting approval.");
    } catch (error) {
      if (queueScope.current === scope && generation.current === current) setMessage(`${error instanceof Error ? error.message : "Could not confirm the saved scores."} Your edits are still here. Refresh to check the saved scores before trying again.`);
    } finally {
      if (queueScope.current === scope && generation.current === current) setBusy(false);
    }
  }
  async function saveStartingRating(participantId: string, matchFormat: MatchFormat) {
    if (!selected) return;
    const scope = clubId + ":" + accessToken;
    const player = queue.players.find(p => p.id === mapping[participantId]);
    const rating = player?.ratings?.[matchFormat];
    if (!player || !rating) return;
    const key = player.id + ":" + matchFormat;
    const value = Number(ratingDrafts[key]);
    if (!Number.isFinite(value) || value < 1 || value > 7) return;
    setBusy(true); setMessage("");
    try {
      const payload = await request("sessions/" + encodeURIComponent(selected.session_key) + "/starting-ratings/" + player.id, {
        participant_id: participantId, match_format: matchFormat, rating_jupr: value, expected_state_fingerprint: rating.state_fingerprint
      }, undefined, "PATCH");
      if (queueScope.current !== scope) return;
      setQueue(old => ({ ...old, players: old.players.map(p => p.id === player.id ? payload.player : p) }));
      setRatingDrafts(old => { const next = { ...old }; delete next[key]; return next; });
      setMessage(player.name + "’s doubles starting level is saved at " + value.toFixed(2) + ". Approve the results when all levels are ready.");
    } catch (error) {
      if (queueScope.current === scope) setMessage((error instanceof Error ? error.message : "Could not confirm the starting level.") + " Refresh to check the saved level before continuing.");
    } finally { if (queueScope.current === scope) setBusy(false); }
  }
  async function selectCreatedPlayer(participantId: string, playerId: number) {
    const scope = clubId + ":" + accessToken;
    const payload: Queue = await request(`submissions?status=${status}`);
    if (queueScope.current !== scope) throw new Error("The club workspace changed. Reopen the submission in its original club.");
    const player = payload.players.find(p => p.id === playerId);
    if (!player) throw new Error("The player profile is not available yet. Refresh the player list and try again.");
    setQueue(old => ({ ...old, players: payload.players }));
    setMapping(old => ({ ...old, [participantId]: player.id }));
    setMessage(player.name + " is selected. Review the remaining players, then approve the results.");
  }
  async function decide(action: ReviewRequest["action"]) {
    if (!selected) throw new InteractionActionError("Choose a submission first.", { kind: "validation" });
    if (editingTitle) throw new InteractionActionError("Save or cancel the session details edit before reviewing results.", { kind: "validation" });
    if (editingScores) throw new InteractionActionError("Save or cancel score corrections before reviewing results.", { kind: "validation" });
    setBusy(true); setMessage("");
    try {
      const body = selected.status === "processing" && selected.review ? selected.review.request : { expected_version: selected.version, action, player_ids: mapping, match_date: matchDate, reason };
      const payload = await request(`sessions/${encodeURIComponent(selected.session_key)}/review`, { ...body, expected_version: selected.version });
      if (payload.status !== "approved" && payload.status !== "rejected") throw new Error("Review result was not confirmed.");
      const text = payload.status === "rejected" ? "Submission rejected. Club records are unchanged." : `Approved as ${selected.rating_mode}. Games are now included in player stats and available for weekly recaps.`;
      await refresh(); setMessage(text);
      return actionSuccess("Review saved", text);
    } catch (error) {
      const detail = error instanceof InteractionActionError ? error.message : `We couldn’t confirm whether the ${action === "approve" ? "approval" : "rejection"} finished.`;
      const recovery = `Close this dialog and refresh the submissions before trying again. If this submission shows “Approval interrupted”, open it and choose “Resume ${action === "reject" ? "rejection" : `${selected.rating_mode} approval`}”.`;
      const failure = new InteractionActionError(`${detail} ${recovery}`, {
        kind: error instanceof InteractionActionError ? error.kind : "failed",
        cause: error
      });
      setMessage(failure.message);
      throw failure;
    } finally { setBusy(false); }
  }
  const editable = selected?.status === "pending";
  const normalizedTitle = titleDraft.trim().replace(/\s+/g, " ");
  const allMapped = selected?.participants.every(p => queue.players.some(player => player.id === mapping[p.id]));
  const ratingChecks = selected?.rating_mode === "rated" ? selected.participants.flatMap(p => {
    const player = queue.players.find(player => player.id === mapping[p.id]);
    return playerFormats(selected, p.id).filter(format => format === "doubles").map(format => {
      const rating = player?.ratings?.[format];
      const draft = ratingDrafts[player?.id + ":" + format];
      return { missing: !!player && (!rating || rating.rating_jupr == null || rating.rating_jupr <= 0),
        dirty: draft !== undefined && (draft.trim() === "" || Number(draft) !== rating?.rating_jupr) };
    });
  }) : [];
  const ratingsReady = ratingChecks.every(check => !check.missing && !check.dirty);
  const creationsReady = !Object.values(pendingCreations).some(Boolean);
  const names = new Map(selected?.participants.map(p => [p.id, p.name]) || []);
  const scoreRounds = [...new Set(selected?.matches.map(match => match.round_number) || [])].sort((a, b) => a - b);
  const visibleMatches = (selected?.matches || []).filter(match => (!scoreRound || match.round_number === Number(scoreRound))
    && [...(match.sideA || match.teamA || []), ...(match.sideB || match.teamB || [])].some(id => (names.get(id) || "Unknown player").toLocaleLowerCase().includes(scorePlayer.trim().toLocaleLowerCase())));
  return <div style={{ display: "grid", gap: "1rem", minWidth: 0 }}>
    <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
      <label>Show <select value={status} disabled={busy} onChange={e => setStatus(e.target.value)} style={field}><option value="pending">Awaiting approval</option><option value="approved">Approved</option><option value="rejected">Rejected</option></select></label>
      <button onClick={() => void refresh()} disabled={busy || !accessToken} style={field}>Refresh</button>
    </div>
    {message ? <p role="status">{message}</p> : null}
    {busy ? <p role="status">Loading…</p> : null}
    {!busy && !queue.submissions.length ? <p>No submissions in this view.</p> : null}
    <div style={{ display: "grid", gap: "0.75rem", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 260px), 1fr))" }}>
      {queue.submissions.map(item => <button key={item.id} onClick={() => select(item)} disabled={busy} style={{ ...field, textAlign: "left", background: selected?.id === item.id ? "#eff6ff" : "white" }}>
        <strong>{item.title}</strong><br />{item.match_date} · {item.match_count} games · {item.rating_mode === "rated" ? "Rated" : "Unrated"}<br />Submitted by {item.organizer_name}<br />{item.status === "processing" ? "Approval interrupted — resume review" : item.approved_mode || item.status}
      </button>)}
    </div>
    {selected ? <article id="generator-submission-review" ref={reviewPanel} tabIndex={-1} style={{ padding: "1rem", border: "1px solid #2563eb", borderRadius: 14, scrollMarginTop: "1rem", minWidth: 0 }}>
      <h2>{selected.title} · {selected.rating_mode === "rated" ? "Rated" : "Unrated"}</h2>
      {editable ? editingTitle ? <form aria-label="Edit session details" onSubmit={event => { event.preventDefault(); void saveTitle(); }} style={{ display: "grid", gap: "0.6rem", maxWidth: 560 }}>
        <label style={{ display: "grid", gap: 4 }}>Session title
          <input aria-label="Session title" value={titleDraft} maxLength={160} required autoFocus disabled={busy} style={field} onChange={event => setTitleDraft(event.target.value)} />
        </label>
        <label style={{ display: "grid", gap: 4 }}>Session rating
          <select aria-label="Session rating" value={sessionRating} disabled={busy} style={field} onChange={event => setSessionRating(event.target.value as "rated" | "unrated")}>
            <option value="unrated">Unrated</option><option value="rated">Rated</option>
          </select>
        </label>
        <p style={{ margin: 0 }}>{sessionRating === "rated" ? "These games will update club ratings when approved." : "These games will not change player ratings."}</p>
        {!normalizedTitle ? <small role="alert">Enter a title.</small> : null}
        <div style={{ display: "flex", gap: "0.6rem" }}>
          <button type="submit" style={field} disabled={busy || !normalizedTitle || normalizedTitle.length > 160 || (normalizedTitle === selected.title && sessionRating === selected.rating_mode)}>Save session details</button>
          <button type="button" style={field} disabled={busy} onClick={() => { setTitleDraft(selected.title); setEditingTitle(false); }}>Cancel</button>
        </div>
      </form> : <button type="button" style={field} disabled={busy || editingScores} onClick={() => { setTitleDraft(selected.title); setSessionRating(selected.rating_mode); setEditingTitle(true); setMessage(""); }}>Edit session details</button> : null}
      <p>{selected.match_count} scored games · Submitted by {selected.organizer_name}</p>
      <label>Date played <input type="date" value={matchDate} disabled={!editable || busy} onChange={e => setMatchDate(e.target.value)} style={field} /></label>
      <h3>Match players{selected.rating_mode === "rated" ? " and review ratings" : " to club profiles"}</h3>
      <p>Choose an existing club profile for each player. If someone is missing, use <strong>Create new player</strong> below their name and enter their starting level.</p>
      {selected.rating_mode === "rated" ? <p>Set and save a doubles starting level for players who have no rated doubles games. Saving updates their club profile immediately; approving applies these scores from that level. Players with rated games use their current rating. Singles ratings are shown separately.</p> : null}
      <div style={{ display: "grid", gap: "0.75rem", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 260px), 1fr))" }}>
        {selected.participants.map(p => {
          const player = queue.players.find(player => player.id === mapping[p.id]);
          return <div key={selected.id + ":" + p.id} style={{ border: "1px solid #e2e8f0", borderRadius: 10, padding: "0.75rem", display: "grid", gap: "0.65rem", alignContent: "start" }}>
            <PlayerProfileChoice name={p.name} value={mapping[p.id]} players={queue.players} disabled={!editable || busy}
              onChange={value => setMapping(old => { const next = { ...old }; if (value) next[p.id] = value; else delete next[p.id]; return next; })} />
            {editable && accessToken ? <GeneratorPlayerCreate clubId={clubId} sessionKey={selected.session_key} participantId={p.id} participantName={p.name}
              apiBase={apiBase} accessToken={accessToken} players={queue.players} disabled={busy} matched={Boolean(player)} onPendingChange={markCreationPending}
              onBusyChange={value => { if (queueScope.current === clubId + ":" + accessToken) setBusy(value); }}
              onCreated={playerId => selectCreatedPlayer(p.id, playerId)} /> : null}
            {selected.rating_mode === "rated" && player ? playerFormats(selected, p.id).map(format => {
              const rating = player.ratings?.[format];
              const key = player.id + ":" + format;
              const draft = ratingDrafts[key];
              const value = draft ?? String(rating?.rating_jupr ?? "");
              const changed = draft !== undefined && (draft.trim() === "" || Number(draft) !== rating?.rating_jupr);
              const valid = value.trim() !== "" && Number.isFinite(Number(value)) && Number(value) >= 1 && Number(value) <= 7;
              return <div key={format} style={{ display: "grid", gap: 6 }}>
                {editable && format === "doubles" && rating?.can_set_starting_rating ? <>
                  <label style={{ display: "grid", gap: 4 }}>Doubles starting level
                    <input aria-label={p.name + " doubles starting level"} type="number" min={1} max={7} step="any" inputMode="decimal" placeholder="e.g. 3.50" value={value} disabled={busy} style={field}
                      onChange={e => setRatingDrafts(old => ({ ...old, [key]: e.target.value }))} />
                  </label>
                  <small>{changed ? "Unsaved starting level" : "No rated doubles games yet · check this level before approval"}</small>
                  {changed ? <button type="button" style={field} disabled={busy || !valid} onClick={() => void saveStartingRating(p.id, format)}>Save doubles starting level</button> : null}
                  {changed && !valid ? <small role="alert">Enter a level from 1.0 to 7.0.</small> : null}
                </> : <small>{format === "singles" ? "Singles" : "Doubles"} rating: <strong>{rating?.rating_jupr == null ? "Not set" : rating.rating_jupr.toFixed(3)}</strong>{rating ? " · " + rating.matches_played + " rated games" : " · refresh to load ratings"}</small>}
              </div>;
            }) : null}
          </div>;
        })}
      </div>
      <h3>Saved scores</h3>
      <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap", alignItems: "end", marginBottom: "0.75rem" }}>
        <label style={{ display: "grid", gap: 4 }}>Round
          <select aria-label="Filter scores by round" value={scoreRound} style={field} onChange={event => setScoreRound(event.target.value)}>
            <option value="">All rounds</option>{scoreRounds.map(round => <option key={round} value={round}>Round {round}</option>)}
          </select>
        </label>
        <label style={{ display: "grid", gap: 4 }}>Find a player
          <input type="search" aria-label="Find a player in saved scores" value={scorePlayer} placeholder="Player name" style={field} onChange={event => setScorePlayer(event.target.value)} />
        </label>
        {editable && !editingScores ? <button type="button" style={field} disabled={busy || editingTitle} onClick={() => { setScoreDrafts({}); setEditingScores(true); setMessage(""); }}>Edit scores</button> : null}
      </div>
      {editingScores ? <p>Correct the scores below, then save before approving. Enter both scores from 0 to 99; games cannot end in a tie.</p> : null}
      <div style={{ overflowX: "auto" }}><table style={{ width: "100%", minWidth: 580, textAlign: "left", borderSpacing: "0 0.5rem" }}><thead><tr><th>Round</th><th>Team 1</th><th>Score</th><th>Team 2</th></tr></thead><tbody>
        {visibleMatches.map(m => {
          const key = scoreKey(m);
          const draft = scoreDrafts[key] || { a: String(m.scoreA), b: String(m.scoreB) };
          const teams = [(m.sideA || m.teamA || []), (m.sideB || m.teamB || [])].map(side => side.map(id => names.get(id) || "Unknown player").join(" & "));
          const changed = scoreChanges.some(change => scoreKey(change.match) === key);
          return <tr key={key}><td>{m.round_number}</td><td>{teams[0]}</td><td style={{ whiteSpace: "nowrap" }}>
            {editingScores ? <>
              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                {(["a", "b"] as const).map((side, index) => <label key={side} style={{ display: "grid", gap: 3 }}>
                  <small>Team {index + 1}</small>
                  <input type="number" inputMode="numeric" min={0} max={99} step={1} aria-label={`Round ${m.round_number}, game ${m.id}, ${teams[index]} score`} value={draft[side]} disabled={busy}
                    style={{ ...field, width: 76 }} onChange={event => setScoreDrafts(old => ({ ...old, [key]: { ...(old[key] || { a: String(m.scoreA), b: String(m.scoreB) }), [side]: event.target.value } }))} />
                </label>)}
              </div>
              {changed ? <small>Saved: {m.scoreA} – {m.scoreB}</small> : null}
            </> : <>{m.scoreA} – {m.scoreB}</>}
          </td><td>{teams[1]}</td></tr>;
        })}
      </tbody></table></div>
      {!visibleMatches.length ? <p>No games match these filters.</p> : null}
      {editingScores ? <div style={{ display: "grid", gap: "0.6rem", marginTop: "0.75rem" }}>
        <p role="status" style={{ margin: 0 }}>{scoreChanges.length} game{scoreChanges.length === 1 ? "" : "s"} changed across all rounds. Save or cancel score corrections before reviewing results.</p>
        {!scoresValid ? <p role="alert" style={{ margin: 0 }}>Enter two different whole-number scores from 0 to 99 for every changed game.</p> : null}
        <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
          <button type="button" style={field} disabled={busy || !scoreChanges.length || !scoresValid} onClick={() => void saveScoreCorrections()}>Save score corrections</button>
          <button type="button" style={field} disabled={busy} onClick={() => { setScoreDrafts({}); setEditingScores(false); }}>Cancel score edits</button>
        </div>
      </div> : null}
      {editable ? <>
        <p>This session is marked {selected.rating_mode}. {selected.rating_mode === "rated" ? "Approval updates the appropriate singles or doubles ratings." : "Approval leaves ratings unchanged."} Approved games count toward stats and recaps.</p>
        {!allMapped ? <p>Choose a club profile for every player before approving.</p> : null}
        {!ratingsReady ? <p role="status">Save the missing or edited doubles starting levels before approving.</p> : null}
        {!creationsReady ? <p role="status">Check unfinished player creation before approving.</p> : null}
        {editingTitle ? <p role="status">Save or cancel the session details edit before reviewing results.</p> : null}
        <label style={{ display: "grid", gap: 4 }}>Reason if rejecting (optional)<textarea value={reason} disabled={busy} maxLength={500} onChange={e => setReason(e.target.value)} style={field} /></label>
        <div style={{ display: "flex", gap: "0.7rem", flexWrap: "wrap", marginTop: "1rem" }}>
          {(["approve", "reject"] as const).map(action => <ConfirmAction key={action}
            triggerLabel={action === "reject" ? "Reject" : `Approve ${selected.rating_mode} results`} title={action === "reject" ? "Reject these results?" : `Approve ${selected.match_count} ${selected.rating_mode} games?`}
            description={action === "reject" ? "These games will stay out of club records." : selected.rating_mode === "rated" ? "These games will update player ratings and count toward stats and recaps." : "These games will count toward stats and recaps. Ratings stay unchanged."}
            confirmLabel={action === "reject" ? "Reject results" : `Approve ${selected.rating_mode} results`} confirmationText="" disabled={busy || editingTitle || editingScores || (action !== "reject" && (!allMapped || !matchDate || !ratingsReady || !creationsReady))} onConfirm={() => decide(action)} />)}
        </div>
      </> : null}
      {selected.status === "processing" && selected.review ? <button disabled={busy} style={field} onClick={() => void decide(selected.review!.request.action).catch(() => {})}>Resume {selected.review.request.action === "reject" ? "rejection" : `${selected.rating_mode} approval`}</button> : null}
    </article> : null}
  </div>;
}
