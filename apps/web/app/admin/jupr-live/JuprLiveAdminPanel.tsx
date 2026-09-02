"use client";

import { useRef, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, type ActionCompletion } from "@/components/interaction";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";
import { deriveLiveLadderOperationKey, idempotencyKeyFor, rotateIdempotencyKey } from "@/lib/liveLadderOperations";

type LiveMatch = { id: string; desc?: string; teamA?: string[]; teamB?: string[]; scoreA?: number | null; scoreB?: number | null; round?: number | null; court?: number | null; mini_round?: number | null };
type LiveParticipant = { id: string; name: string; player_id?: number | null };
type LiveRound = { number?: number; matches?: LiveMatch[]; courts?: Array<{ courtNumber?: number; miniRounds?: Array<{ number?: number; matches?: LiveMatch[] }> }> };
type LiveEvent = { type?: string; name?: string; participants?: LiveParticipant[]; rounds?: LiveRound[]; currentRoundNumber?: number; totalRounds?: number };
type LiveState = { event_type?: string; page_state?: { event?: LiveEvent }; official_publish?: Record<string, unknown> };
type LiveSession = { session_key: string; title?: string | null; status: string; version: string; event_type?: string | null; current_round_number?: number | null; total_rounds?: number | null; updated_at?: string | null; expires_at?: string | null; state?: LiveState };
type StatusResponse = { enabled: boolean; writes_enabled?: boolean; status: string; counts?: Record<string, number>; warnings?: string[] };
type ListResponse = { ok: boolean; sessions: LiveSession[]; count: number };
type Recovery = { match_log_url?: string; replay_history_url?: string; instructions?: string };
type WriteResponse = { ok: boolean; session?: LiveSession; changed_scores?: number; published_count?: number; operation_key?: string; idempotent_replay?: boolean; recovery?: Recovery; correction?: Recovery; outcome?: string };
type Props = { apiBase: string | null; clubId: string; clubSlug: string; status: StatusResponse | null };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
async function apiError(response: Response): Promise<string> { const text = await response.text().catch(() => ""); if (!text) return `API error (${response.status}).`; try { return String((JSON.parse(text) as { detail?: unknown }).detail || text); } catch { return text.slice(0, 240); } }
function eventFromSession(row: LiveSession): LiveEvent { return row.state?.page_state?.event || {}; }
function participantLabel(event: LiveEvent, id: string): string { const hit = (event.participants || []).find((participant) => String(participant.id) === String(id)); return hit ? `${hit.name}${hit.player_id ? ` (#${hit.player_id})` : ""}` : String(id); }
function flattenMatches(event: LiveEvent): LiveMatch[] { const rows: LiveMatch[] = []; for (const round of event.rounds || []) { for (const match of round.matches || []) rows.push({ ...match, round: round.number ?? null }); for (const court of round.courts || []) for (const mini of court.miniRounds || []) for (const match of mini.matches || []) rows.push({ ...match, round: round.number ?? null, court: court.courtNumber ?? null, mini_round: mini.number ?? null }); } return rows; }
function defaultScoreDraft(row: LiveSession): Record<string, { score_a: string; score_b: string }> { const draft: Record<string, { score_a: string; score_b: string }> = {}; for (const match of flattenMatches(eventFromSession(row))) draft[match.id] = { score_a: match.scoreA == null ? "" : String(match.scoreA), score_b: match.scoreB == null ? "" : String(match.scoreB) }; return draft; }
function parseIds(value: string): number[] { return value.replace(/,/g, "\n").split("\n").map((token) => token.trim()).filter(Boolean).map(Number).filter(Number.isInteger); }

export default function JuprLiveAdminPanel({ apiBase, clubId, clubSlug, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const operationKeys = useRef<Record<string, string>>({});
  const [sessions, setSessions] = useState<LiveSession[]>([]);
  const [filter, setFilter] = useState("active");
  const [title, setTitle] = useState(`JUPR Live ${new Date().toISOString().slice(0, 10)}`);
  const [eventType, setEventType] = useState("round_robin");
  const [participantNames, setParticipantNames] = useState("");
  const [participantIds, setParticipantIds] = useState("");
  const [totalRounds, setTotalRounds] = useState("3");
  const [courtSizes, setCourtSizes] = useState("");
  const [scoreDrafts, setScoreDrafts] = useState<Record<string, Record<string, { score_a: string; score_b: string }>>>({});
  const [publishDate, setPublishDate] = useState<Record<string, string>>({});
  const [lastOperationKey, setLastOperationKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<WriteResponse | null>(null);
  const sessionsRequest = useLatestRequestGuard(accessToken, () => {
    operationKeys.current = {};
    setBusy(false); setMessage(null); setSessions([]); setScoreDrafts({}); setPublishDate({});
    setLastOperationKey(""); setLastResult(null);
  });
  const actionRequest = useLatestRequestGuard(accessToken);
  const writesEnabled = status?.writes_enabled === true;

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> { if (!apiBase) throw new Error("Missing JUPR API base URL."); if (!accessToken) throw new Error("Sign in at /admin/login before using JUPR Live Admin."); const headers = new Headers(options?.headers); headers.set("Authorization", `Bearer ${accessToken}`); if (options?.body) headers.set("Content-Type", "application/json"); const response = await fetch(apiUrl(apiBase, path), { ...options, headers }); if (!response.ok) throw new Error(await apiError(response)); return (await response.json()) as T; }
  function rememberScores(rows: LiveSession[]) { const next: Record<string, Record<string, { score_a: string; score_b: string }>> = {}; for (const row of rows) next[row.session_key] = scoreDrafts[row.session_key] || defaultScoreDraft(row); setScoreDrafts(next); }
  async function durableFields(scope: string, operationType: string, entityId: string, expectedVersion: string) { if (!writesEnabled) throw new Error("Next writes are guarded off; use Streamlit JUPR Live Admin."); const idempotencyKey = idempotencyKeyFor(operationKeys.current, scope); const operationKey = await deriveLiveLadderOperationKey({ clubId, surface: "jupr_live_admin", operationType, entityId, idempotencyKey }); return { idempotency_key: idempotencyKey, expected_version: expectedVersion, operationKey }; }
  function completeScope(scope: string) { rotateIdempotencyKey(operationKeys.current, scope); }

  async function loadSessions() { const generation = sessionsRequest.begin(); setBusy(true); setMessage(null); setSessions([]); try { const query = filter ? `?status=${encodeURIComponent(filter)}&limit=100` : "?limit=100"; const payload = await requestJson<ListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/jupr-live/sessions${query}`); if (!sessionsRequest.isCurrent(generation)) return; setSessions(payload.sessions || []); rememberScores(payload.sessions || []); setMessage(payload.sessions?.length ? `Loaded ${payload.count ?? payload.sessions.length} one-off session(s).` : `No ${filter || "matching"} sessions.`); } catch (error) { if (sessionsRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load live sessions."); } finally { if (sessionsRequest.isCurrent(generation)) setBusy(false); } }
  async function createSession(confirmationText: string): Promise<ActionCompletion> {
    if (!writesEnabled) { const error = new Error("Next writes are guarded off; use Streamlit JUPR Live Admin."); setMessage(error.message); throw error; }
    const generation = actionRequest.begin();
    const scope = "create:new";
    setBusy(true); setMessage(null);
    try {
      const fields = await durableFields(scope, "create_session", "new", "new");
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before live-session creation could continue.");
      setLastOperationKey(fields.operationKey);
      const names = participantNames.replace(/,/g, "\n").split("\n").map((token) => token.trim()).filter(Boolean);
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/jupr-live/sessions`, { method: "POST", body: JSON.stringify({ title, event_type: eventType, participant_names: names, player_ids: parseIds(participantIds), total_rounds: Number(totalRounds), court_sizes: parseIds(courtSizes), confirmation_text: confirmationText, expected_version: fields.expected_version, idempotency_key: fields.idempotency_key }) });
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the created session response was applied.");
      completeScope(scope); setLastResult(payload); setMessage(`Created durable live session ${payload.session?.session_key || "(recovered)"}.`);
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the live sessions could be refreshed.");
      await loadSessions();
      return actionSuccess("Live session created", `Durable live session ${payload.session?.session_key || "(recovered)"} was created.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(`${error instanceof Error ? error.message : "Create outcome is uncertain."} Reconcile the operation before creating another session.`);
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function updateSession(row: LiveSession, nextStatus: string, confirmationText: string): Promise<ActionCompletion> {
    const generation = actionRequest.begin();
    const scope = `status:${row.session_key}:${nextStatus}`;
    setBusy(true); setMessage(null);
    try {
      const fields = await durableFields(scope, "update_session", row.session_key, row.version);
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the live-session update could continue.");
      setLastOperationKey(fields.operationKey);
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/jupr-live/sessions/${encodeURIComponent(row.session_key)}`, { method: "PATCH", body: JSON.stringify({ status: nextStatus, title: row.title || undefined, confirmation_text: confirmationText, expected_version: fields.expected_version, idempotency_key: fields.idempotency_key }) });
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the live-session update response was applied.");
      completeScope(scope); setLastResult(payload); setMessage(`${row.title || row.session_key} marked ${nextStatus}.`);
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the live sessions could be refreshed.");
      await loadSessions();
      return actionSuccess("Live session updated", `${row.title || row.session_key} was marked ${nextStatus}.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(`${error instanceof Error ? error.message : "Session update is uncertain."} Reconcile before retrying.`);
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveScores(row: LiveSession, confirmationText: string): Promise<ActionCompletion> {
    const generation = actionRequest.begin();
    const scope = `scores:${row.session_key}`;
    setBusy(true); setMessage(null);
    try {
      const fields = await durableFields(scope, "save_scores", row.session_key, row.version);
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the score save could continue.");
      setLastOperationKey(fields.operationKey);
      const scores = Object.entries(scoreDrafts[row.session_key] || {}).map(([match_id, score]) => ({ match_id, score_a: score.score_a === "" ? null : Number(score.score_a), score_b: score.score_b === "" ? null : Number(score.score_b) }));
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/jupr-live/sessions/${encodeURIComponent(row.session_key)}/scores`, { method: "PATCH", body: JSON.stringify({ scores, confirmation_text: confirmationText, expected_version: fields.expected_version, idempotency_key: fields.idempotency_key }) });
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the saved scores response was applied.");
      completeScope(scope); setLastResult(payload); setMessage(`Saved ${payload.changed_scores || 0} score row(s).`);
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the live sessions could be refreshed.");
      await loadSessions();
      return actionSuccess("Live scores saved", `${payload.changed_scores || 0} score row(s) were saved.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(`${error instanceof Error ? error.message : "Score save is uncertain."} Reconcile before retrying.`);
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function advanceRound(row: LiveSession, confirmationText: string): Promise<ActionCompletion> {
    const generation = actionRequest.begin();
    const scope = `advance:${row.session_key}`;
    setBusy(true); setMessage(null);
    try {
      const fields = await durableFields(scope, "advance_round", row.session_key, row.version);
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the round advance could continue.");
      setLastOperationKey(fields.operationKey);
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/jupr-live/sessions/${encodeURIComponent(row.session_key)}/advance`, { method: "POST", body: JSON.stringify({ confirmation_text: confirmationText, expected_version: fields.expected_version, idempotency_key: fields.idempotency_key }) });
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the round advance response was applied.");
      completeScope(scope); setLastResult(payload); setMessage("Advanced to the next Python-generated live round.");
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the live sessions could be refreshed.");
      await loadSessions();
      return actionSuccess("Live round advanced", "The next Python-generated live round is now active.");
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(`${error instanceof Error ? error.message : "Round advance is uncertain."} Reconcile before retrying.`);
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function publishMatches(row: LiveSession, confirmationText: string): Promise<ActionCompletion> {
    const generation = actionRequest.begin();
    const scope = `publish:${row.session_key}`;
    const matchDate = publishDate[row.session_key] || new Date().toISOString();
    if (!publishDate[row.session_key]) setPublishDate((current) => ({ ...current, [row.session_key]: matchDate }));
    setBusy(true); setMessage(null);
    try {
      const fields = await durableFields(scope, "official_publish", row.session_key, row.version);
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before official publication could continue.");
      setLastOperationKey(fields.operationKey);
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/jupr-live/sessions/${encodeURIComponent(row.session_key)}/publish`, { method: "POST", body: JSON.stringify({ match_date: matchDate, confirmation_text: confirmationText, expected_version: fields.expected_version, idempotency_key: fields.idempotency_key }) });
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the publication response was applied.");
      completeScope(scope); setLastResult(payload); setMessage(`${payload.idempotent_replay ? "Recovered" : "Published"} ${payload.published_count || 0} official JUPR Live match(es).`);
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the live sessions could be refreshed.");
      await loadSessions();
      return actionSuccess(payload.idempotent_replay ? "Official matches recovered" : "Official matches published", `${payload.published_count || 0} official JUPR Live match(es) were ${payload.idempotent_replay ? "recovered" : "published"}.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(`${error instanceof Error ? error.message : "Publish outcome is uncertain."} Do not publish again; reconcile and inspect Match Log/Replay History.`);
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function reconcileOperation(confirmationText: string): Promise<ActionCompletion> {
    if (!lastOperationKey) throw new Error("No live operation is available to reconcile.");
    const generation = actionRequest.begin();
    const requestedOperationKey = lastOperationKey;
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/jupr-live/operations/${encodeURIComponent(requestedOperationKey)}/reconcile`, { method: "POST", body: JSON.stringify({ confirmation_text: confirmationText }) });
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the reconciliation response was applied.");
      setLastResult(payload);
      if (!payload.ok) throw new Error("Outcome remains uncertain. Follow Match Log/Replay History recovery.");
      setMessage("Recovered the durable response without replaying the write.");
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the live sessions could be refreshed.");
      await loadSessions();
      return actionSuccess("Live operation reconciled", "The durable response was recovered without replaying the write.");
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to reconcile live operation.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(status?.enabled ? accessToken : "", loadSessions, filter || "all");

  if (!status?.enabled) {
    return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>JUPR Live Admin is disabled</h2><p>{status?.warnings?.[0] || "Enable JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE on FastAPI."}</p></article>;
  }

  return <div style={{ display: "grid", gap: "1rem" }}>
    <article style={{ ...cardStyle, background: "#f8fafc" }}>
      <h2 style={{ marginTop: 0 }}>Admin session</h2>
      <p>{adminSessionLabel(session)}</p>
      {sessionLoading ? <p>Checking session…</p> : null}
      {sessionMessage ? <p>{sessionMessage}</p> : null}
      <p>Active: {status.counts?.active ?? "—"} · Completed: {status.counts?.completed ?? "—"} · Abandoned: {status.counts?.abandoned ?? "—"}</p>
      {!writesEnabled ? <p role="alert" style={{ color: "#92400e" }}>{status.warnings?.[0]} Streamlit remains the operational fallback.</p> : null}
    </article>

    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Create durable one-off session</h2>
      <p style={{ color: "#475569" }}>Only Round Robin and League / Ladder belong here. Tournament sessions are explicitly rejected; use Tournament Live/Ops.</p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
        <label>Title<br /><input value={title} onChange={(event) => setTitle(event.target.value)} style={inputStyle} /></label>
        <label>Event type<br /><select value={eventType} onChange={(event) => setEventType(event.target.value)} style={inputStyle}><option value="round_robin">Round Robin</option><option value="league_ladder">League / Ladder</option></select></label>
        <label>Total rounds<br /><input value={totalRounds} onChange={(event) => setTotalRounds(event.target.value)} inputMode="numeric" style={inputStyle} /></label>
        <label>Court sizes optional<br /><input value={courtSizes} onChange={(event) => setCourtSizes(event.target.value)} placeholder="4,4 or 5,5" style={inputStyle} /></label>
      </div>
      <label>Participant names, one per line<br /><textarea value={participantNames} onChange={(event) => setParticipantNames(event.target.value)} rows={4} style={inputStyle} /></label>
      <label>Official player IDs, one per line (required for official publish)<br /><textarea value={participantIds} onChange={(event) => setParticipantIds(event.target.value)} rows={3} style={inputStyle} /></label>
      <ConfirmAction
        triggerLabel="Create session"
        title="Create this live session?"
        description="This creates a durable one-off live session using the participant and round settings above."
        confirmLabel="Yes, create session"
        confirmationText="CREATE LIVE SESSION"
        disabled={!writesEnabled}
        busy={busy}
        onConfirm={createSession}
      />
    </article>

    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Manage sessions</h2>
      <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "0.75rem", alignItems: "end" }}>
        <label>Status filter<br /><select value={filter} onChange={(event) => setFilter(event.target.value)} disabled={busy} style={inputStyle}><option value="active">active</option><option value="completed">completed</option><option value="abandoned">abandoned</option><option value="archived">archived</option><option value="">all</option></select></label>
        <button type="button" onClick={loadSessions} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Refreshing…" : "Refresh sessions"}</button>
      </div>
      {message ? <p role="status" aria-live="polite" style={{ color: /unable|type|uncertain|guarded|requires/i.test(message) ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </article>

    {sessions.map((row) => {
      const event = eventFromSession(row);
      const matches = flattenMatches(event);
      const draft = scoreDrafts[row.session_key] || defaultScoreDraft(row);
      const isLeague = event.type === "league";
      return <article key={row.session_key} style={cardStyle}>
        <h3 style={{ marginTop: 0 }}>{row.title || row.session_key}</h3>
        <p>{row.session_key} · {row.status} · {event.type || row.event_type || "event"}{isLeague ? ` · round ${event.currentRoundNumber || row.current_round_number || 1} of ${event.totalRounds || row.total_rounds || "?"}` : ""} · version <code>{row.version || "missing"}</code></p>
        <p><a href={`/clubs/${encodeURIComponent(clubSlug)}/live/${encodeURIComponent(row.session_key)}`}>Verify public state</a></p>
        {matches.length ? <section>
          <h4>Scores</h4>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <caption style={{ textAlign: "left" }}>One-off live scoring grid</caption>
              <thead><tr><th scope="col" align="left">Match</th><th scope="col" align="left">Team A</th><th scope="col">A</th><th scope="col">B</th><th scope="col" align="left">Team B</th></tr></thead>
              <tbody>{matches.map((match) => <tr key={match.id}>
                <th scope="row" align="left">{match.desc || `R${match.round || ""} C${match.court || ""}`}</th>
                <td>{(match.teamA || []).map((id) => participantLabel(event, id)).join(" / ")}</td>
                <td><label><span style={{ position: "absolute", width: 1, height: 1, overflow: "hidden" }}>{match.id} team A score</span><input value={draft[match.id]?.score_a || ""} onChange={(event_) => setScoreDrafts((current) => ({ ...current, [row.session_key]: { ...(current[row.session_key] || {}), [match.id]: { ...(current[row.session_key]?.[match.id] || { score_b: "" }), score_a: event_.target.value } } }))} inputMode="numeric" style={{ ...inputStyle, maxWidth: 80 }} /></label></td>
                <td><label><span style={{ position: "absolute", width: 1, height: 1, overflow: "hidden" }}>{match.id} team B score</span><input value={draft[match.id]?.score_b || ""} onChange={(event_) => setScoreDrafts((current) => ({ ...current, [row.session_key]: { ...(current[row.session_key] || {}), [match.id]: { ...(current[row.session_key]?.[match.id] || { score_a: "" }), score_b: event_.target.value } } }))} inputMode="numeric" style={{ ...inputStyle, maxWidth: 80 }} /></label></td>
                <td>{(match.teamB || []).map((id) => participantLabel(event, id)).join(" / ")}</td>
              </tr>)}</tbody>
            </table>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", marginTop: "1rem" }}>
            <ConfirmAction
              triggerLabel="Save scores"
              title="Save these live scores?"
              description="This saves every score currently shown in this session's scoring grid."
              confirmLabel="Yes, save scores"
              confirmationText="SAVE LIVE SCORES"
              disabled={!writesEnabled}
              busy={busy}
              onConfirm={(confirmationText) => saveScores(row, confirmationText)}
            />
            {isLeague ? <ConfirmAction
              triggerLabel="Advance round"
              title="Advance to the next live round?"
              description="This generates and activates the next round for this league or ladder session."
              confirmLabel="Yes, advance round"
              confirmationText="ADVANCE LIVE ROUND"
              disabled={!writesEnabled}
              busy={busy}
              onConfirm={(confirmationText) => advanceRound(row, confirmationText)}
            /> : null}
            <label>Publish date ISO<br /><input value={publishDate[row.session_key] || ""} onChange={(event_) => setPublishDate((current) => ({ ...current, [row.session_key]: event_.target.value }))} placeholder="Defaults to publish time" style={inputStyle} /></label>
            <ConfirmAction
              triggerLabel="Publish official matches"
              title="Publish these official matches?"
              description="This creates official rated match records from the live scores. Verify every score and player first."
              confirmLabel="Yes, publish matches"
              confirmationText="PUBLISH LIVE MATCHES"
              tone="danger"
              disabled={!writesEnabled}
              busy={busy}
              onConfirm={(confirmationText) => publishMatches(row, confirmationText)}
            />
          </div>
        </section> : <p>No generated scoring grid. Four or more participants are required.</p>}
        <hr style={{ border: 0, borderTop: "1px solid #e2e8f0" }} />
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <ConfirmAction
            triggerLabel="Complete"
            title="Complete this live session?"
            description="This marks the live session as completed."
            confirmLabel="Yes, complete session"
            confirmationText="SAVE LIVE SESSION"
            disabled={!writesEnabled}
            busy={busy}
            onConfirm={(confirmationText) => updateSession(row, "completed", confirmationText)}
          />
          <ConfirmAction
            triggerLabel="Abandon"
            title="Abandon this live session?"
            description="This marks the live session as abandoned and removes it from the active workflow."
            confirmLabel="Yes, abandon session"
            confirmationText="SAVE LIVE SESSION"
            tone="danger"
            disabled={!writesEnabled}
            busy={busy}
            onConfirm={(confirmationText) => updateSession(row, "abandoned", confirmationText)}
          />
          <ConfirmAction
            triggerLabel="Archive"
            title="Archive this live session?"
            description="This archives the live session and removes it from active operations."
            confirmLabel="Yes, archive session"
            confirmationText="SAVE LIVE SESSION"
            tone="danger"
            disabled={!writesEnabled}
            busy={busy}
            onConfirm={(confirmationText) => updateSession(row, "archived", confirmationText)}
          />
        </div>
      </article>;
    })}

    {lastOperationKey ? <article style={{ ...cardStyle, background: "#fff7ed" }}>
      <h2 style={{ marginTop: 0 }}>Audit and recovery</h2>
      <p>A timeout is not proof of failure. Operation <code>{lastOperationKey}</code>.</p>
      <ConfirmAction
        triggerLabel="Reconcile stored response"
        title="Reconcile this live operation?"
        description="This checks the durable operation record and recovers its stored response without replaying the write."
        confirmLabel="Yes, reconcile operation"
        confirmationText="RECONCILE LIVE OPERATION"
        disabled={!writesEnabled}
        busy={busy}
        onConfirm={reconcileOperation}
      />
      <p><a href={lastResult?.correction?.match_log_url || lastResult?.recovery?.match_log_url || "/admin/match-log"}>Match Log correction</a> · <a href={lastResult?.correction?.replay_history_url || lastResult?.recovery?.replay_history_url || "/admin/replay-history"}>Replay History verification</a></p>
    </article> : null}
  </div>;
}
