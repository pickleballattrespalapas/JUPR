"use client";

import { useMemo, useState } from "react";
import type { AdminLeagueManagerDetailResponse, AdminLeagueManagerListResponse, AdminLeagueManagerStatusResponse } from "@/lib/adminLeagueManagerApi";
import type { AdminMatchUploaderRoundRobinCourt, AdminMatchUploaderRoundRobinPreview, AdminMatchUploaderStatusResponse, AdminMatchUploaderWriteResult } from "@/lib/adminMatchUploaderApi";
import type { PublicPlayer } from "@/lib/api";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  leagueStatus: AdminLeagueManagerStatusResponse;
  uploaderStatus: AdminMatchUploaderStatusResponse;
  players: PublicPlayer[];
};

type CourtDraft = { court: string; formatType: string; playerNames: string };
type ScoreDraft = { scoreT1: string; scoreT2: string };
type LeagueLiveSession = {
  id: string;
  league_name: string;
  week_tag: string;
  status: string;
  total_rounds: number;
  current_round: number;
  roster_json?: Array<Record<string, unknown>>;
  current_court_state_json?: Array<Record<string, unknown>>;
  notes?: string | null;
  updated_at?: string | null;
};
type LeagueLiveCourt = { round_number: number; court_number: number; format_type: string; player_names: string[]; players_json?: Array<Record<string, unknown>> };
type LeagueLiveRound = { round_number: number; round_label?: string | null; status: string; submitted_match_count?: number | null; match_date?: string | null; updated_at?: string | null };
type LeagueLiveListResponse = { ok: boolean; sessions: LeagueLiveSession[]; count: number };
type LeagueLiveDetailResponse = { ok: boolean; session: LeagueLiveSession; rounds: LeagueLiveRound[]; courts: LeagueLiveCourt[] };
type LeagueLiveWriteResponse = { ok: boolean; session: LeagueLiveSession; round?: LeagueLiveRound; courts?: LeagueLiveCourt[]; warnings?: string[] };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

function rosterToCourtDrafts(detail: AdminLeagueManagerDetailResponse | null): CourtDraft[] {
  const active = (detail?.roster || []).filter((row) => row.in_league).sort((a, b) => Number(b.rating || 0) - Number(a.rating || 0));
  if (!active.length) return [{ court: "1", formatType: "4-player", playerNames: "" }];
  const courts: CourtDraft[] = [];
  for (let i = 0; i < active.length; i += 4) {
    const chunk = active.slice(i, i + 4);
    courts.push({ court: String(courts.length + 1), formatType: "4-player", playerNames: chunk.map((row) => row.player_name).join("\n") });
  }
  return courts;
}

function splitNames(value: string): string[] {
  return String(value || "").replace(/,/g, "\n").split("\n").map((item) => item.trim()).filter(Boolean);
}

function scoreIsValid(score: ScoreDraft): boolean {
  const a = Number(score.scoreT1);
  const b = Number(score.scoreT2);
  return Number.isInteger(a) && Number.isInteger(b) && a >= 0 && b >= 0 && a !== b && a + b > 0;
}

function courtsToPayload(courts: CourtDraft[], currentRound: number) {
  return courts.map((court, index) => ({
    round_number: currentRound,
    court_number: Number(court.court) || index + 1,
    court: Number(court.court) || index + 1,
    format_type: court.formatType,
    player_names: splitNames(court.playerNames)
  }));
}

function courtsFromPersisted(courts: LeagueLiveCourt[], currentRound: number, fallback: Array<Record<string, unknown>> = []): CourtDraft[] {
  const scoped = courts.filter((court) => Number(court.round_number) === Number(currentRound));
  const rows = scoped.length ? scoped : fallback.map((row, index) => ({
    round_number: currentRound,
    court_number: Number(row.court_number || row.court || index + 1),
    format_type: String(row.format_type || row.formatType || "4-player"),
    player_names: Array.isArray(row.player_names) ? row.player_names.map(String) : []
  }));
  if (!rows.length) return [{ court: "1", formatType: "4-player", playerNames: "" }];
  return rows
    .sort((a, b) => Number(a.court_number) - Number(b.court_number))
    .map((row) => ({ court: String(row.court_number), formatType: String(row.format_type || "4-player"), playerNames: (row.player_names || []).join("\n") }));
}

function activeRosterPayload(detail: AdminLeagueManagerDetailResponse | null) {
  return (detail?.roster || []).filter((row) => row.in_league).map((row) => ({
    player_id: row.player_id,
    player_name: row.player_name,
    rating: row.rating,
    rating_jupr: row.rating_jupr,
    wins: row.wins,
    losses: row.losses,
    matches_played: row.matches_played
  }));
}

export default function LeagueLiveRoundPanel({ apiBase, clubId, leagueStatus, uploaderStatus, players }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [leagues, setLeagues] = useState<string[]>([]);
  const [leagueName, setLeagueName] = useState("");
  const [weekTag, setWeekTag] = useState("Week 1");
  const [roundNumber, setRoundNumber] = useState("1");
  const [totalRounds, setTotalRounds] = useState("5");
  const [roundLabel, setRoundLabel] = useState("Round 1");
  const [matchDate, setMatchDate] = useState(todayIso());
  const [detail, setDetail] = useState<AdminLeagueManagerDetailResponse | null>(null);
  const [liveSessions, setLiveSessions] = useState<LeagueLiveSession[]>([]);
  const [sessionId, setSessionId] = useState("");
  const [sessionStatus, setSessionStatus] = useState("active");
  const [sessionNotes, setSessionNotes] = useState("");
  const [createConfirm, setCreateConfirm] = useState("");
  const [saveConfirm, setSaveConfirm] = useState("");
  const [roundHistory, setRoundHistory] = useState<LeagueLiveRound[]>([]);
  const [courts, setCourts] = useState<CourtDraft[]>([{ court: "1", formatType: "4-player", playerNames: "" }]);
  const [preview, setPreview] = useState<AdminMatchUploaderRoundRobinPreview | null>(null);
  const [scores, setScores] = useState<Record<string, ScoreDraft>>({});
  const [confirmation, setConfirmation] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const currentRound = Math.max(1, Number(roundNumber) || 1);
  const safeTotalRounds = Math.max(currentRound, Number(totalRounds) || currentRound);
  const playerOptions = useMemo(() => players.map((player) => player.name).filter(Boolean).sort((a, b) => a.localeCompare(b)), [players]);
  const allPreviewMatches = (preview?.courts || []).flatMap((court) => court.matches || []);
  const validScoreCount = allPreviewMatches.filter((match) => scoreIsValid(scores[match.row_id] || { scoreT1: "", scoreT2: "" })).length;

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using League Live.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  function applySession(sessionRow: LeagueLiveSession, courtsRows: LeagueLiveCourt[] = [], rounds: LeagueLiveRound[] = []) {
    setSessionId(sessionRow.id);
    setLeagueName(sessionRow.league_name);
    setWeekTag(sessionRow.week_tag || "Week 1");
    setSessionStatus(sessionRow.status || "active");
    setTotalRounds(String(sessionRow.total_rounds || 1));
    setRoundNumber(String(sessionRow.current_round || 1));
    setRoundLabel(`Round ${sessionRow.current_round || 1}`);
    setSessionNotes(sessionRow.notes || "");
    setRoundHistory(rounds || []);
    setCourts(courtsFromPersisted(courtsRows, sessionRow.current_round || 1, sessionRow.current_court_state_json || []));
    setPreview(null);
    setScores({});
  }

  async function loadLeagues() {
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`);
      const names = (payload.leagues || []).map((league) => league.league_name).filter(Boolean);
      setLeagues(names);
      if (!leagueName && names.length) setLeagueName(names[0]);
      setMessage(`Loaded ${names.length} league(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load leagues.");
    } finally {
      setBusy(false);
    }
  }

  async function loadSessions() {
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions?limit=100`);
      setLiveSessions(payload.sessions || []);
      setMessage(`Loaded ${payload.count ?? payload.sessions?.length ?? 0} persisted live session(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load live sessions.");
    } finally {
      setBusy(false);
    }
  }

  async function loadLeagueDetail() {
    if (!leagueName) {
      setMessage("Select a league first.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(leagueName)}`);
      setDetail(payload);
      setCourts(rosterToCourtDrafts(payload));
      setPreview(null);
      setScores({});
      setMessage("League roster loaded into court draft.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load league detail.");
    } finally {
      setBusy(false);
    }
  }

  async function createSession() {
    if (!leagueName) {
      setMessage("Select a league before creating a persisted session.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions`, {
        method: "POST",
        body: JSON.stringify({
          league_name: leagueName,
          week_tag: weekTag,
          total_rounds: safeTotalRounds,
          current_round: currentRound,
          roster: activeRosterPayload(detail),
          courts: courtsToPayload(courts, currentRound),
          notes: sessionNotes,
          confirmation_text: createConfirm,
          source: "next_league_live_session_create"
        })
      });
      applySession(payload.session, payload.courts || [], []);
      setCreateConfirm("");
      await loadSessions();
      setMessage("Persisted League Live session created. You can now resume it later from this page.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to create persisted session.");
    } finally {
      setBusy(false);
    }
  }

  async function loadSessionDetail() {
    if (!sessionId) {
      setMessage("Select a persisted session first.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(sessionId)}`);
      applySession(payload.session, payload.courts || [], payload.rounds || []);
      setMessage("Persisted League Live session loaded.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load persisted session.");
    } finally {
      setBusy(false);
    }
  }

  async function saveSessionSnapshot() {
    if (!sessionId) {
      setMessage("Create or load a persisted session before saving a snapshot.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(sessionId)}/snapshot`, {
        method: "PATCH",
        body: JSON.stringify({
          status: sessionStatus,
          week_tag: weekTag,
          total_rounds: safeTotalRounds,
          current_round: currentRound,
          roster: activeRosterPayload(detail),
          courts: courtsToPayload(courts, currentRound),
          notes: sessionNotes,
          confirmation_text: saveConfirm,
          source: "next_league_live_session_snapshot"
        })
      });
      applySession(payload.session, payload.courts || [], roundHistory);
      setSaveConfirm("");
      setMessage("League Live session snapshot saved.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to save session snapshot.");
    } finally {
      setBusy(false);
    }
  }

  function updateCourt(index: number, patch: Partial<CourtDraft>) {
    setCourts((current) => current.map((row, idx) => idx === index ? { ...row, ...patch } : row));
    setPreview(null);
    setScores({});
  }

  function addCourt() {
    setCourts((current) => [...current, { court: String(current.length + 1), formatType: "4-player", playerNames: "" }]);
    setPreview(null);
    setScores({});
  }

  function removeCourt(index: number) {
    setCourts((current) => current.filter((_, idx) => idx !== index).map((row, idx) => ({ ...row, court: String(idx + 1) })));
    setPreview(null);
    setScores({});
  }

  async function generatePreview() {
    setBusy(true);
    setMessage(null);
    try {
      const courtPayload = courtsToPayload(courts, currentRound).map((court) => ({ court: court.court, format_type: court.format_type, player_names: court.player_names }));
      const payload = await requestJson<AdminMatchUploaderRoundRobinPreview>(`/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/round-robin/preview`, {
        method: "POST",
        body: JSON.stringify({ courts: courtPayload, schedule_mode: "full", source: "next_league_manager_live_preview" })
      });
      setPreview(payload);
      const nextScores: Record<string, ScoreDraft> = {};
      for (const match of (payload.courts || []).flatMap((court) => court.matches || [])) nextScores[match.row_id] = { scoreT1: "", scoreT2: "" };
      setScores(nextScores);
      if (payload.missing_players?.length) setMessage(`Missing players: ${payload.missing_players.join(", ")}`);
      else setMessage(`Generated ${payload.match_count || 0} match slot(s). Save the session snapshot before leaving this page.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to generate round preview.");
    } finally {
      setBusy(false);
    }
  }

  async function persistSubmittedRound(matches: Array<Record<string, unknown>>, submitResult: AdminMatchUploaderWriteResult) {
    if (!sessionId) return null;
    return requestJson<LeagueLiveWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(sessionId)}/rounds/${encodeURIComponent(String(currentRound))}`, {
      method: "POST",
      body: JSON.stringify({
        round_label: roundLabel,
        match_date: matchDate,
        preview: preview || {},
        matches,
        submitted_match_count: submitResult.submitted_count ?? matches.length,
        submitted_match_ids: submitResult.feedback?.latest_match_id ? [submitResult.feedback.latest_match_id] : [],
        courts: courtsToPayload(courts, currentRound),
        advance_after_save: true,
        confirmation_text: "SAVE ROUND",
        source: "next_league_live_round_persist"
      })
    });
  }

  async function submitRound() {
    if (!sessionId) {
      setMessage("Create or load a persisted League Live session before submitting official scores.");
      return;
    }
    if (confirmation.trim().toUpperCase() !== "SUBMIT LEAGUE ROUND") {
      setMessage("Type SUBMIT LEAGUE ROUND to publish these scored league matches.");
      return;
    }
    const matches = allPreviewMatches
      .filter((match) => scoreIsValid(scores[match.row_id] || { scoreT1: "", scoreT2: "" }))
      .map((match) => ({
        date: matchDate,
        league: leagueName,
        week_tag: weekTag,
        match_type: "League Manager Live",
        context_type: "league_live_session",
        context_id: `${sessionId}:round-${currentRound}:court-${match.court}:match-${match.match_index}`,
        t1_p1: match.t1_p1,
        t1_p2: match.t1_p2,
        t2_p1: match.t2_p1,
        t2_p2: match.t2_p2,
        score_t1: Number(scores[match.row_id]?.scoreT1 || 0),
        score_t2: Number(scores[match.row_id]?.scoreT2 || 0)
      }));
    if (!matches.length) {
      setMessage("Enter at least one valid non-tied score before submitting.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminMatchUploaderWriteResult>(`/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/batch`, {
        method: "POST",
        body: JSON.stringify({ matches, source: "next_league_manager_live_submit" })
      });
      const saved = await persistSubmittedRound(matches, payload);
      if (saved?.session) {
        setRoundNumber(String(saved.session.current_round || currentRound));
        setRoundLabel(`Round ${saved.session.current_round || currentRound}`);
      }
      setConfirmation("");
      setPreview(null);
      setScores({});
      if (sessionId) await loadSessionDetail();
      setMessage(`Submitted ${payload.submitted_count ?? matches.length} league match(es), saved round state, and advanced session. Latest match: ${payload.feedback?.latest_match_id ?? "—"}.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to submit league round.");
    } finally {
      setBusy(false);
    }
  }

  if (!leagueStatus.enabled || !uploaderStatus.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>League Live is disabled</h2>
        <p style={{ color: "#475569" }}>Enable both League Manager and Match Uploader on FastAPI before using League Live round entry.</p>
        <ul style={{ color: "#475569" }}>
          <li>League Manager: {leagueStatus.status}</li>
          <li>Match Uploader: {uploaderStatus.status}</li>
        </ul>
      </article>
    );
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Admin session</h2>
        <p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>
        {sessionLoading ? <p>Checking session…</p> : null}
        {sessionMessage ? <p style={{ color: "#64748b" }}>{sessionMessage}</p> : null}
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>1. League and persisted session</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>League<br /><select value={leagueName} onChange={(event) => setLeagueName(event.target.value)} style={inputStyle}>{leagues.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
          <label>Week<br /><input value={weekTag} onChange={(event) => setWeekTag(event.target.value)} style={inputStyle} /></label>
          <label>Round #<br /><input value={roundNumber} onChange={(event) => { setRoundNumber(event.target.value); setRoundLabel(`Round ${event.target.value || 1}`); }} style={inputStyle} /></label>
          <label>Total rounds<br /><input value={totalRounds} onChange={(event) => setTotalRounds(event.target.value)} style={inputStyle} /></label>
          <label>Date<br /><input type="date" value={matchDate} onChange={(event) => setMatchDate(event.target.value)} style={inputStyle} /></label>
          <button type="button" onClick={loadLeagues} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Loading…" : "Load leagues"}</button>
          <button type="button" onClick={loadLeagueDetail} disabled={busy || !leagueName} style={buttonStyle}>Load roster</button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "0.75rem", marginTop: "0.75rem", alignItems: "end" }}>
          <label>Existing sessions<br />
            <select value={sessionId} onChange={(event) => setSessionId(event.target.value)} style={inputStyle}>
              <option value="">Select session…</option>
              {liveSessions.map((row) => <option key={row.id} value={row.id}>{row.league_name} · {row.week_tag} · R{row.current_round}/{row.total_rounds} · {row.status}</option>)}
            </select>
          </label>
          <button type="button" onClick={loadSessions} disabled={busy || !accessToken} style={ghostButtonStyle}>Load sessions</button>
          <button type="button" onClick={loadSessionDetail} disabled={busy || !sessionId} style={ghostButtonStyle}>Resume selected</button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", marginTop: "0.75rem", alignItems: "end" }}>
          <label>Session status<br /><select value={sessionStatus} onChange={(event) => setSessionStatus(event.target.value)} style={inputStyle}><option>active</option><option>paused</option><option>complete</option><option>archived</option></select></label>
          <label>Round label<br /><input value={roundLabel} onChange={(event) => setRoundLabel(event.target.value)} style={inputStyle} /></label>
          <label>Notes<br /><input value={sessionNotes} onChange={(event) => setSessionNotes(event.target.value)} style={inputStyle} /></label>
          <label>Create confirmation<br /><input value={createConfirm} onChange={(event) => setCreateConfirm(event.target.value)} placeholder="CREATE LIVE SESSION" style={inputStyle} /></label>
          <button type="button" onClick={createSession} disabled={busy || !leagueName} style={buttonStyle}>Create persisted session</button>
        </div>
        {sessionId ? <p style={{ color: "#475569" }}><strong>Active persisted session:</strong> {sessionId}</p> : <p style={{ color: "#92400e" }}>Create or resume a persisted session before submitting official scores.</p>}
        {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("missing") || message.toLowerCase().includes("type") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>2. Courts</h2>
        <p style={{ color: "#475569" }}>Roster players are seeded from the selected league. Edit courts, then save a session snapshot so the night can be resumed.</p>
        <datalist id="league-live-players">{playerOptions.map((name) => <option key={name} value={name} />)}</datalist>
        {courts.map((court, index) => (
          <div key={index} style={{ borderTop: index ? "1px solid #e2e8f0" : undefined, paddingTop: index ? "0.75rem" : 0, marginTop: index ? "0.75rem" : 0 }}>
            <div style={{ display: "grid", gridTemplateColumns: "120px 180px 1fr auto", gap: "0.75rem", alignItems: "start" }}>
              <label>Court<br /><input value={court.court} onChange={(event) => updateCourt(index, { court: event.target.value })} style={inputStyle} /></label>
              <label>Format<br /><select value={court.formatType} onChange={(event) => updateCourt(index, { formatType: event.target.value })} style={inputStyle}>{(uploaderStatus.round_robin_format_options || ["4-player"]).map((option) => <option key={option} value={option}>{option}</option>)}</select></label>
              <label>Players, one per line<br /><textarea value={court.playerNames} onChange={(event) => updateCourt(index, { playerNames: event.target.value })} rows={4} style={inputStyle} /></label>
              <button type="button" onClick={() => removeCourt(index)} style={ghostButtonStyle}>Remove</button>
            </div>
          </div>
        ))}
        <p><button type="button" onClick={addCourt} style={ghostButtonStyle}>Add court</button> <button type="button" onClick={generatePreview} disabled={busy || !leagueName} style={buttonStyle}>Generate match slots</button></p>
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "0.75rem", alignItems: "end" }}>
          <label>Save snapshot confirmation<br /><input value={saveConfirm} onChange={(event) => setSaveConfirm(event.target.value)} placeholder="SAVE SESSION" style={inputStyle} /></label>
          <button type="button" onClick={saveSessionSnapshot} disabled={busy || !sessionId} style={ghostButtonStyle}>Save session snapshot</button>
        </div>
      </article>

      {roundHistory.length ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Round history</h2>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr><th align="left">Round</th><th align="left">Label</th><th align="left">Status</th><th align="right">Submitted</th><th align="left">Date</th></tr></thead>
              <tbody>{roundHistory.map((round) => <tr key={`${round.round_number}-${round.status}`}><td>{round.round_number}</td><td>{round.round_label || "—"}</td><td>{round.status}</td><td align="right">{round.submitted_match_count ?? 0}</td><td>{round.match_date || "—"}</td></tr>)}</tbody>
            </table>
          </div>
        </article>
      ) : null}

      {preview?.courts?.length ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>3. Enter scores</h2>
          <p style={{ color: "#475569" }}>Only rows with valid non-tied scores will be submitted. Corrections after publish should go through Match Log and Replay History.</p>
          <div style={{ display: "grid", gap: "0.75rem" }}>
            {(preview.courts as AdminMatchUploaderRoundRobinCourt[]).map((court) => (
              <section key={court.court} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem" }}>
                <h3 style={{ marginTop: 0 }}>Court {court.court} · {court.format_type}</h3>
                {(court.matches || []).map((match) => (
                  <div key={match.row_id} style={{ display: "grid", gridTemplateColumns: "1fr 90px 90px", gap: "0.75rem", alignItems: "center", borderTop: "1px solid #f1f5f9", padding: "0.5rem 0" }}>
                    <div><strong>{match.label}</strong><br />{match.t1.map((p) => p.name).join(" / ")} vs {match.t2.map((p) => p.name).join(" / ")}</div>
                    <input value={scores[match.row_id]?.scoreT1 || ""} onChange={(event) => setScores((current) => ({ ...current, [match.row_id]: { ...(current[match.row_id] || { scoreT1: "", scoreT2: "" }), scoreT1: event.target.value } }))} placeholder="Team 1" style={inputStyle} />
                    <input value={scores[match.row_id]?.scoreT2 || ""} onChange={(event) => setScores((current) => ({ ...current, [match.row_id]: { ...(current[match.row_id] || { scoreT1: "", scoreT2: "" }), scoreT2: event.target.value } }))} placeholder="Team 2" style={inputStyle} />
                  </div>
                ))}
              </section>
            ))}
          </div>
          <p style={{ color: "#475569" }}>Valid scored matches: {validScoreCount} / {allPreviewMatches.length}</p>
          <label>Confirmation<br /><input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} placeholder="SUBMIT LEAGUE ROUND" style={inputStyle} /></label>
          <button type="button" onClick={submitRound} disabled={busy || !validScoreCount || !sessionId} style={{ ...buttonStyle, marginTop: "0.75rem" }}>{busy ? "Submitting…" : "Submit scored league round"}</button>
        </article>
      ) : null}
    </div>
  );
}
