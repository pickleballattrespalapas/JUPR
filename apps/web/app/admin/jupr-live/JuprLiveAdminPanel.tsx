"use client";

import { useState } from "react";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type LiveMatch = { id: string; desc?: string; teamA?: string[]; teamB?: string[]; scoreA?: number | null; scoreB?: number | null; round?: number | null };
type LiveParticipant = { id: string; name: string; player_id?: number | null };
type LiveEvent = { type?: string; name?: string; participants?: LiveParticipant[]; rounds?: Array<{ number?: number; matches?: LiveMatch[] }> };
type LiveState = { event_type?: string; page_state?: { event?: LiveEvent }; official_publish?: Record<string, unknown> };
type LiveSession = {
  session_key: string;
  title?: string | null;
  status: string;
  source?: string | null;
  event_type?: string | null;
  created_by_email?: string | null;
  updated_at?: string | null;
  expires_at?: string | null;
  state?: LiveState;
};

type StatusResponse = { enabled: boolean; status: string; counts?: Record<string, number>; warnings?: string[] };
type ListResponse = { ok: boolean; sessions: LiveSession[]; count: number };
type WriteResponse = { ok: boolean; session: LiveSession; changed_scores?: number; published_count?: number; result?: Record<string, unknown> };
type Props = { apiBase: string | null; clubId: string; status: StatusResponse | null };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

async function apiError(response: Response): Promise<string> {
  const text = await response.text().catch(() => "");
  if (!text) return `API error (${response.status}).`;
  try {
    const payload = JSON.parse(text) as { detail?: unknown };
    return String(payload.detail || text);
  } catch {
    return text.slice(0, 240);
  }
}

function eventFromSession(row: LiveSession): LiveEvent {
  return row.state?.page_state?.event || {};
}
function participantLabel(event: LiveEvent, id: string): string {
  const hit = (event.participants || []).find((p) => String(p.id) === String(id));
  if (!hit) return String(id);
  return hit.player_id ? `${hit.name} (#${hit.player_id})` : hit.name;
}
function flattenMatches(event: LiveEvent): LiveMatch[] {
  const rows: LiveMatch[] = [];
  for (const round of event.rounds || []) {
    for (const match of round.matches || []) rows.push({ ...match, round: round.number ?? null });
  }
  return rows;
}
function defaultScoreDraft(row: LiveSession): Record<string, { score_a: string; score_b: string }> {
  const event = eventFromSession(row);
  const draft: Record<string, { score_a: string; score_b: string }> = {};
  for (const match of flattenMatches(event)) {
    draft[match.id] = { score_a: match.scoreA == null ? "" : String(match.scoreA), score_b: match.scoreB == null ? "" : String(match.scoreB) };
  }
  return draft;
}

export default function JuprLiveAdminPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [sessions, setSessions] = useState<LiveSession[]>([]);
  const [filter, setFilter] = useState("active");
  const [title, setTitle] = useState(`JUPR Live ${new Date().toISOString().slice(0, 10)}`);
  const [eventType, setEventType] = useState("round_robin");
  const [participantNames, setParticipantNames] = useState("");
  const [participantIds, setParticipantIds] = useState("");
  const [createConfirm, setCreateConfirm] = useState("");
  const [sessionConfirm, setSessionConfirm] = useState<Record<string, string>>({});
  const [scoreDrafts, setScoreDrafts] = useState<Record<string, Record<string, { score_a: string; score_b: string }>>>({});
  const [scoreConfirm, setScoreConfirm] = useState<Record<string, string>>({});
  const [publishConfirm, setPublishConfirm] = useState<Record<string, string>>({});
  const [publishDate, setPublishDate] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<WriteResponse | null>(null);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("Missing JUPR API base URL.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using JUPR Live Admin.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    if (!response.ok) throw new Error(await apiError(response));
    return (await response.json()) as T;
  }

  function rememberScores(rows: LiveSession[]) {
    const next: Record<string, Record<string, { score_a: string; score_b: string }>> = {};
    for (const row of rows) next[row.session_key] = scoreDrafts[row.session_key] || defaultScoreDraft(row);
    setScoreDrafts(next);
  }

  async function loadSessions() {
    setBusy(true); setMessage(null);
    try {
      const query = filter ? `?status=${encodeURIComponent(filter)}&limit=100` : `?limit=100`;
      const payload = await requestJson<ListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/jupr-live/sessions${query}`);
      setSessions(payload.sessions || []);
      rememberScores(payload.sessions || []);
      setMessage(`Loaded ${payload.count ?? payload.sessions?.length ?? 0} live session(s).`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to load live sessions."); }
    finally { setBusy(false); }
  }

  async function createSession() {
    setBusy(true); setMessage(null);
    try {
      const names = participantNames.replace(/,/g, "\n").split("\n").map((x) => x.trim()).filter(Boolean);
      const ids = participantIds.replace(/,/g, "\n").split("\n").map((x) => x.trim()).filter(Boolean).map(Number).filter((x) => Number.isInteger(x));
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/jupr-live/sessions`, {
        method: "POST",
        body: JSON.stringify({ title, event_type: eventType, participant_names: names, player_ids: ids, confirmation_text: createConfirm })
      });
      setCreateConfirm("");
      setLastResult(payload);
      setMessage(`Created live session ${payload.session.session_key}.`);
      await loadSessions();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to create live session."); }
    finally { setBusy(false); }
  }

  async function updateSession(row: LiveSession, nextStatus: string) {
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/jupr-live/sessions/${encodeURIComponent(row.session_key)}`, {
        method: "PATCH",
        body: JSON.stringify({ status: nextStatus, title: row.title || undefined, confirmation_text: sessionConfirm[row.session_key] || "" })
      });
      setLastResult(payload);
      setMessage(`${row.title || row.session_key} marked ${nextStatus}.`);
      await loadSessions();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to update live session."); }
    finally { setBusy(false); }
  }

  async function saveScores(row: LiveSession) {
    if ((scoreConfirm[row.session_key] || "").trim().toUpperCase() !== "SAVE LIVE SCORES") { setMessage("Type SAVE LIVE SCORES to save session scores."); return; }
    setBusy(true); setMessage(null);
    try {
      const scores = Object.entries(scoreDrafts[row.session_key] || {}).map(([match_id, score]) => ({ match_id, score_a: score.score_a === "" ? null : Number(score.score_a), score_b: score.score_b === "" ? null : Number(score.score_b) }));
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/jupr-live/sessions/${encodeURIComponent(row.session_key)}/scores`, {
        method: "PATCH",
        body: JSON.stringify({ scores, confirmation_text: scoreConfirm[row.session_key] || "" })
      });
      setLastResult(payload);
      setMessage(`Saved ${payload.changed_scores || 0} score row(s).`);
      setScoreConfirm((current) => ({ ...current, [row.session_key]: "" }));
      await loadSessions();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to save live scores."); }
    finally { setBusy(false); }
  }

  async function publishMatches(row: LiveSession) {
    if ((publishConfirm[row.session_key] || "").trim().toUpperCase() !== "PUBLISH LIVE MATCHES") { setMessage("Type PUBLISH LIVE MATCHES to publish official rated matches."); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/jupr-live/sessions/${encodeURIComponent(row.session_key)}/publish`, {
        method: "POST",
        body: JSON.stringify({ match_date: publishDate[row.session_key] || new Date().toISOString(), confirmation_text: publishConfirm[row.session_key] || "" })
      });
      setLastResult(payload);
      setMessage(`Published ${payload.published_count || 0} official JUPR Live match(es).`);
      setPublishConfirm((current) => ({ ...current, [row.session_key]: "" }));
      await loadSessions();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to publish live matches."); }
    finally { setBusy(false); }
  }

  if (!status?.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>JUPR Live Admin is disabled</h2><p>{status?.warnings?.[0] || "Enable JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE on FastAPI."}</p></article>;

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Admin session</h2>
        <p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>
        {sessionLoading ? <p>Checking session…</p> : null}
        {sessionMessage ? <p>{sessionMessage}</p> : null}
        <p style={{ color: "#475569" }}>Active: {status.counts?.active ?? "—"} · Completed: {status.counts?.completed ?? "—"} · Abandoned: {status.counts?.abandoned ?? "—"}</p>
      </article>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Create durable live session</h2>
        <p style={{ color: "#475569" }}>JUPR Live is for one-off round robins and quick league/ladder sessions. Tournament draws stay in Tournament Live/Ops.</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Title<br /><input value={title} onChange={(e) => setTitle(e.target.value)} style={inputStyle} /></label>
          <label>Event type<br /><select value={eventType} onChange={(e) => setEventType(e.target.value)} style={inputStyle}><option value="round_robin">Round Robin</option><option value="league_ladder">League / Ladder shell</option></select></label>
          <label>Confirmation<br /><input value={createConfirm} onChange={(e) => setCreateConfirm(e.target.value)} placeholder="CREATE LIVE SESSION" style={inputStyle} /></label>
        </div>
        <label>Participant names, one per line<br /><textarea value={participantNames} onChange={(e) => setParticipantNames(e.target.value)} rows={4} style={inputStyle} /></label>
        <label>Official player IDs, one per line, optional but required for official publish<br /><textarea value={participantIds} onChange={(e) => setParticipantIds(e.target.value)} rows={3} style={inputStyle} /></label>
        <p><button type="button" onClick={createSession} disabled={busy} style={buttonStyle}>Create session</button></p>
      </article>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Manage sessions</h2>
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "0.75rem", alignItems: "end" }}>
          <label>Status filter<br /><select value={filter} onChange={(e) => setFilter(e.target.value)} style={inputStyle}><option value="active">active</option><option value="completed">completed</option><option value="abandoned">abandoned</option><option value="archived">archived</option><option value="">all</option></select></label>
          <button type="button" onClick={loadSessions} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Loading…" : "Load sessions"}</button>
        </div>
        {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("type") || message.toLowerCase().includes("requires") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>
      {sessions.map((row) => {
        const event = eventFromSession(row);
        const matches = flattenMatches(event);
        const draft = scoreDrafts[row.session_key] || defaultScoreDraft(row);
        return (
          <article key={row.session_key} style={cardStyle}>
            <h3 style={{ marginTop: 0 }}>{row.title || row.session_key}</h3>
            <p style={{ color: "#475569" }}>{row.session_key} · {row.status} · {row.event_type || "event"} · updated {row.updated_at ? String(row.updated_at).slice(0, 19) : "—"}</p>
            <p><a href={`/clubs/tres-palapas/live/${encodeURIComponent(row.session_key)}`}>Public view</a></p>
            {matches.length ? <section style={{ marginTop: "1rem" }}>
              <h4>Score round robin</h4>
              <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}><thead><tr><th align="left">Match</th><th align="left">Team A</th><th align="left">A</th><th align="left">B</th><th align="left">Team B</th></tr></thead><tbody>{matches.map((match) => <tr key={match.id}><td>{match.desc || match.id}</td><td>{(match.teamA || []).map((id) => participantLabel(event, id)).join(" / ")}</td><td><input value={draft[match.id]?.score_a || ""} onChange={(e) => setScoreDrafts((current) => ({ ...current, [row.session_key]: { ...(current[row.session_key] || {}), [match.id]: { ...(current[row.session_key]?.[match.id] || { score_b: "" }), score_a: e.target.value } } }))} style={{ ...inputStyle, maxWidth: "80px" }} /></td><td><input value={draft[match.id]?.score_b || ""} onChange={(e) => setScoreDrafts((current) => ({ ...current, [row.session_key]: { ...(current[row.session_key] || {}), [match.id]: { ...(current[row.session_key]?.[match.id] || { score_a: "" }), score_b: e.target.value } } }))} style={{ ...inputStyle, maxWidth: "80px" }} /></td><td>{(match.teamB || []).map((id) => participantLabel(event, id)).join(" / ")}</td></tr>)}</tbody></table></div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", alignItems: "end", marginTop: "1rem" }}>
                <label>Score confirmation<br /><input value={scoreConfirm[row.session_key] || ""} onChange={(e) => setScoreConfirm((current) => ({ ...current, [row.session_key]: e.target.value }))} placeholder="SAVE LIVE SCORES" style={inputStyle} /></label>
                <button type="button" onClick={() => saveScores(row)} disabled={busy} style={buttonStyle}>Save scores</button>
                <label>Publish date ISO<br /><input value={publishDate[row.session_key] || new Date().toISOString()} onChange={(e) => setPublishDate((current) => ({ ...current, [row.session_key]: e.target.value }))} style={inputStyle} /></label>
                <label>Publish confirmation<br /><input value={publishConfirm[row.session_key] || ""} onChange={(e) => setPublishConfirm((current) => ({ ...current, [row.session_key]: e.target.value }))} placeholder="PUBLISH LIVE MATCHES" style={inputStyle} /></label>
                <button type="button" onClick={() => publishMatches(row)} disabled={busy} style={ghostButtonStyle}>Publish official matches</button>
              </div>
              {row.state?.official_publish ? <pre style={{ whiteSpace: "pre-wrap", background: "#f8fafc", padding: "0.75rem", borderRadius: "10px" }}>{JSON.stringify(row.state.official_publish, null, 2)}</pre> : null}
            </section> : <p style={{ color: "#64748b" }}>No generated scoring grid on this session yet. Create a round-robin with 4–20 participants to score/publish here.</p>}
            <hr style={{ border: 0, borderTop: "1px solid #e2e8f0", margin: "1rem 0" }} />
            <label>Session confirmation<br /><input value={sessionConfirm[row.session_key] || ""} onChange={(e) => setSessionConfirm((current) => ({ ...current, [row.session_key]: e.target.value }))} placeholder="SAVE LIVE SESSION" style={inputStyle} /></label>
            <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}><button type="button" onClick={() => updateSession(row, "completed")} disabled={busy} style={buttonStyle}>Complete</button><button type="button" onClick={() => updateSession(row, "abandoned")} disabled={busy} style={ghostButtonStyle}>Abandon</button><button type="button" onClick={() => updateSession(row, "archived")} disabled={busy} style={ghostButtonStyle}>Archive</button></p>
          </article>
        );
      })}
      {lastResult ? <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Last result</h2><pre style={{ whiteSpace: "pre-wrap", background: "#0f172a", color: "white", padding: "1rem", borderRadius: "12px", overflowX: "auto" }}>{JSON.stringify(lastResult, null, 2)}</pre></article> : null}
    </div>
  );
}
