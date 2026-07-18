"use client";

import { useState } from "react";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type StatusResponse = { enabled: boolean; status: string; roles?: string[]; retention_days?: number; retention_cutoff?: string };
type OverviewResponse = { ok: boolean; roles: Array<Record<string, unknown>>; activity: Array<Record<string, unknown>>; activity_warning?: string | null; health: Record<string, unknown>; role_options: string[]; retention_days: number; retention_cutoff: string };
type RoleResponse = { ok: boolean; roles: Array<Record<string, unknown>>; audit_warning?: string | null };
type WorkerResponse = { ok: boolean; mode?: string; result?: Record<string, unknown>; summary?: Record<string, unknown>; worker_status?: Record<string, unknown>; audit_warning?: string | null };
type TournamentBackfillPreviewResponse = { ok: boolean; mode: "tournament_match_backfill_preview"; read_only: true; preview_fingerprint: string; confirmation_text: string; summary: Record<string, unknown>; candidates: Array<Record<string, unknown>>; warnings: string[] };
type TournamentBackfillApplyResponse = { ok: boolean; mode: "tournament_match_backfill_apply"; operation_id: string; selected_game_ids: string[]; inserted_count: number; warnings: string[] };
type Props = { apiBase: string | null; clubId: string; status: StatusResponse | null };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
async function apiError(response: Response): Promise<string> { const text = await response.text().catch(() => ""); try { return String((JSON.parse(text) as { detail?: unknown }).detail || text); } catch { return text || `API error (${response.status}).`; } }
function table(rows: Array<Record<string, unknown>>, keys: string[]) {
  if (!rows.length) return <p style={{ color: "#64748b" }}>No rows.</p>;
  return <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}><thead><tr>{keys.map((key) => <th key={key} align="left" style={{ borderBottom: "1px solid #e2e8f0", padding: "0.4rem" }}>{key}</th>)}</tr></thead><tbody>{rows.slice(0, 100).map((row, idx) => <tr key={idx}>{keys.map((key) => <td key={key} style={{ borderBottom: "1px solid #f1f5f9", padding: "0.4rem", verticalAlign: "top" }}>{typeof row[key] === "object" && row[key] !== null ? JSON.stringify(row[key]) : String(row[key] ?? "")}</td>)}</tr>)}</tbody></table>{rows.length > 100 ? <p style={{ color: "#64748b" }}>Showing first 100 of {rows.length} rows.</p> : null}</div>;
}
function Pre({ value }: { value: unknown }) { return <pre style={{ whiteSpace: "pre-wrap", background: "#0f172a", color: "white", padding: "1rem", borderRadius: "12px", overflowX: "auto", fontSize: "0.82rem" }}>{JSON.stringify(value, null, 2)}</pre>; }

export default function AdminToolsPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [flaggedOnly, setFlaggedOnly] = useState(false);
  const [targetEmail, setTargetEmail] = useState("");
  const [targetRole, setTargetRole] = useState("read_only");
  const [targetUserId, setTargetUserId] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [workerMode, setWorkerMode] = useState("batch");
  const [workerMaxJobs, setWorkerMaxJobs] = useState("25");
  const [workerBudget, setWorkerBudget] = useState("15");
  const [workerConfirmation, setWorkerConfirmation] = useState("");
  const [recomputeMode, setRecomputeMode] = useState("dry-run");
  const [recomputePlayerId, setRecomputePlayerId] = useState("");
  const [recomputeBadgeId, setRecomputeBadgeId] = useState("");
  const [recomputeLeagueId, setRecomputeLeagueId] = useState("");
  const [recomputeContextId, setRecomputeContextId] = useState("");
  const [recomputeSince, setRecomputeSince] = useState("");
  const [recomputeUntil, setRecomputeUntil] = useState("");
  const [recomputeStrictGlobal, setRecomputeStrictGlobal] = useState(false);
  const [recomputeConfirmation, setRecomputeConfirmation] = useState("");
  const [lastWorkerResult, setLastWorkerResult] = useState<WorkerResponse | null>(null);
  const [tournamentBackfillPreview, setTournamentBackfillPreview] = useState<TournamentBackfillPreviewResponse | null>(null);
  const [selectedTournamentBackfillGameIds, setSelectedTournamentBackfillGameIds] = useState<string[]>([]);
  const [tournamentBackfillConfirmation, setTournamentBackfillConfirmation] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("Missing JUPR API base URL.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Admin Tools.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    if (!response.ok) throw new Error(await apiError(response));
    return (await response.json()) as T;
  }

  async function loadOverview() {
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<OverviewResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tools/overview?flagged_only=${flaggedOnly ? "true" : "false"}&limit=200`);
      setOverview(payload); setMessage(`Loaded ${payload.roles?.length || 0} role assignment(s) and ${payload.activity?.length || 0} activity row(s).`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to load Admin Tools overview."); }
    finally { setBusy(false); }
  }

  async function saveRole(action: "upsert" | "revoke") {
    const expected = action === "upsert" ? "SAVE ROLE" : "REVOKE ROLE";
    if (confirmation.trim().toUpperCase() !== expected) { setMessage(`Type ${expected} to continue.`); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<RoleResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tools/roles`, {
        method: "PATCH",
        body: JSON.stringify({ email: targetEmail, role: targetRole, user_id: targetUserId || null, action, confirmation_text: confirmation })
      });
      setOverview((current) => current ? { ...current, roles: payload.roles } : current);
      setMessage(payload.audit_warning ? `Saved, but audit warning: ${payload.audit_warning}` : (action === "upsert" ? "Role assignment saved." : "Role assignment revoked."));
      setConfirmation("");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to update role assignment."); }
    finally { setBusy(false); }
  }

  async function runQueueWorker() {
    const expected = workerMode === "drain" ? "DRAIN BADGE QUEUE" : "PROCESS BADGE QUEUE";
    if (workerConfirmation.trim().toUpperCase() !== expected) { setMessage(`Type ${expected} to run the badge queue worker.`); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<WorkerResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tools/workers/badge-queue`, {
        method: "POST",
        body: JSON.stringify({ mode: workerMode, max_jobs: Number(workerMaxJobs), time_budget_seconds: Number(workerBudget), confirmation_text: workerConfirmation })
      });
      setLastWorkerResult(payload);
      setMessage(payload.audit_warning ? `Badge queue completed with audit warning: ${payload.audit_warning}` : "Badge queue worker completed.");
      setWorkerConfirmation("");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to run badge queue worker."); }
    finally { setBusy(false); }
  }

  async function loadTournamentBackfillPreview() {
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<TournamentBackfillPreviewResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tools/backfills/tournament-matches/preview?limit=500`);
      setTournamentBackfillPreview(payload);
      setSelectedTournamentBackfillGameIds([]);
      setTournamentBackfillConfirmation("");
      setMessage(`Tournament backfill preview found ${String(payload.summary.ready_count ?? 0)} ready and ${String(payload.summary.blocked_count ?? 0)} blocked missing match candidate(s). No rows were written.`);
    } catch (error) { setTournamentBackfillPreview(null); setMessage(error instanceof Error ? error.message : "Unable to preview missing tournament matches."); }
    finally { setBusy(false); }
  }

  async function applyTournamentMatchBackfill() {
    if (!tournamentBackfillPreview) { setMessage("Load and review a current tournament backfill preview first."); return; }
    if (!selectedTournamentBackfillGameIds.length) { setMessage("Select at least one ready tournament game to backfill."); return; }
    if (tournamentBackfillConfirmation.trim().toUpperCase() !== tournamentBackfillPreview.confirmation_text) { setMessage(`Type ${tournamentBackfillPreview.confirmation_text} to apply the selected backfill.`); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<TournamentBackfillApplyResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tools/backfills/tournament-matches/apply`, {
        method: "POST",
        body: JSON.stringify({
          game_ids: selectedTournamentBackfillGameIds,
          preview_fingerprint: tournamentBackfillPreview.preview_fingerprint,
          preview_limit: Number(tournamentBackfillPreview.summary.candidate_limit ?? 500),
          confirmation_text: tournamentBackfillConfirmation,
          source: "next_admin_tools_tournament_match_backfill"
        })
      });
      setTournamentBackfillPreview(null);
      setSelectedTournamentBackfillGameIds([]);
      setTournamentBackfillConfirmation("");
      setMessage(`Backfilled ${payload.inserted_count} reviewed tournament match(es). Operation ${payload.operation_id}. Reload the preview and verify Match Log before any further write.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to apply the tournament match backfill."); }
    finally { setBusy(false); }
  }

  async function runBadgeRecompute() {
    if (recomputeMode !== "dry-run" && recomputeConfirmation.trim().toUpperCase() !== "RUN BADGE RECOMPUTE") { setMessage("Type RUN BADGE RECOMPUTE to apply badge recompute changes."); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<WorkerResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tools/workers/badge-recompute`, {
        method: "POST",
        body: JSON.stringify({
          mode: recomputeMode,
          player_id: recomputePlayerId ? Number(recomputePlayerId) : null,
          badge_id: recomputeBadgeId || null,
          league_id: recomputeLeagueId || null,
          context_id: recomputeContextId || null,
          since: recomputeSince || null,
          until: recomputeUntil || null,
          allow_strict_global: recomputeStrictGlobal,
          match_limit: 50000,
          confirmation_text: recomputeConfirmation
        })
      });
      setLastWorkerResult(payload);
      setMessage(payload.audit_warning ? `Badge recompute finished with audit warning: ${payload.audit_warning}` : "Badge recompute finished.");
      setRecomputeConfirmation("");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to run badge recompute."); }
    finally { setBusy(false); }
  }

  if (status && !status.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Disabled</h2><p>Set <code>JUPR_ENABLE_NEXT_ADMIN_TOOLS=1</code> on FastAPI to enable guarded Admin Tools.</p></article>;

  const roleOptions = overview?.role_options || status?.roles || ["read_only", "scorekeeper", "organizer", "club_owner", "super_admin"];
  const health = overview?.health || {};
  const workerStatus = (health as { workers?: unknown }).workers;
  return <div style={{ display: "grid", gap: "1rem" }}>
    <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Admin session</h2><p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>{sessionLoading ? <p>Checking session…</p> : null}{sessionMessage ? <p style={{ color: "#64748b" }}>{sessionMessage}</p> : null}</article>
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Overview</h2>
      <label style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center" }}><input type="checkbox" checked={flaggedOnly} onChange={(event) => setFlaggedOnly(event.target.checked)} /> Flagged activity only</label>
      <p><button type="button" onClick={loadOverview} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Loading…" : "Load Admin Tools"}</button></p>
      {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("type") || message.toLowerCase().includes("blocked") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </article>
    {overview ? <>
      <article style={cardStyle}><h2 style={{ marginTop: 0 }}>System health</h2><Pre value={health} /></article>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Workers and backfills</h2>
        <p style={{ color: "#475569" }}>Badge worker/recompute controls run through FastAPI/Python with <code>run_replay</code> permission. Tournament match backfill uses a read-only preview followed by a selected, stale-preview-guarded apply.</p>
        {workerStatus ? <Pre value={workerStatus} /> : null}
        <section style={{ border: "1px solid #bfdbfe", background: "#eff6ff", borderRadius: "12px", padding: "1rem", margin: "1rem 0" }}>
          <h3 style={{ marginTop: 0 }}>Tournament match backfill</h3>
          <p style={{ color: "#475569" }}>Start with a read-only scan for finalized tournament games that have no club-scoped official match with the same <code>tournament_game_id</code>. Select only reviewed ready games; apply rechecks the preview fingerprint, player scope, and duplicate state before using the Python match service.</p>
          <button type="button" onClick={loadTournamentBackfillPreview} disabled={busy} style={ghostButtonStyle}>Preview missing tournament matches</button>
          {tournamentBackfillPreview ? <div style={{ marginTop: "1rem" }}>
            <Pre value={tournamentBackfillPreview.summary} />
            {tournamentBackfillPreview.warnings.map((warning) => <p key={warning} style={{ color: "#92400e" }}>{warning}</p>)}
            {table(tournamentBackfillPreview.candidates, ["tournament_name", "game_id", "score_a", "score_b", "status", "reason"])}
            {(() => {
              const readyGameIds = tournamentBackfillPreview.candidates
                .filter((candidate) => candidate.status === "ready")
                .map((candidate) => String(candidate.game_id || ""))
                .filter(Boolean);
              const applyLimit = Math.max(1, Number(tournamentBackfillPreview.summary.apply_limit ?? 100));
              return readyGameIds.length ? <div style={{ borderTop: "1px solid #bfdbfe", marginTop: "1rem", paddingTop: "1rem" }}>
                <p><strong>Reviewed ready games</strong> — maximum {applyLimit} per apply.</p>
                <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                  <button type="button" onClick={() => setSelectedTournamentBackfillGameIds(readyGameIds.slice(0, applyLimit))} disabled={busy} style={ghostButtonStyle}>Select all ready shown</button>
                  <button type="button" onClick={() => setSelectedTournamentBackfillGameIds([])} disabled={busy} style={ghostButtonStyle}>Clear selection</button>
                </p>
                <div style={{ display: "grid", gap: "0.4rem", maxHeight: "240px", overflowY: "auto" }}>
                  {readyGameIds.map((gameId) => {
                    const checked = selectedTournamentBackfillGameIds.includes(gameId);
                    return <label key={gameId} style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}><input type="checkbox" checked={checked} disabled={busy || (!checked && selectedTournamentBackfillGameIds.length >= applyLimit)} onChange={(event) => setSelectedTournamentBackfillGameIds((current) => event.target.checked ? [...new Set([...current, gameId])].slice(0, applyLimit) : current.filter((value) => value !== gameId))} /> <code>{gameId}</code></label>;
                  })}
                </div>
                <p><label>Confirmation<br /><input value={tournamentBackfillConfirmation} onChange={(event) => setTournamentBackfillConfirmation(event.target.value)} placeholder={tournamentBackfillPreview.confirmation_text} style={inputStyle} /></label></p>
                <p style={{ color: "#991b1b" }}>After apply, verify the exact rows in Match Log. If counts or ratings disagree, stop further writes and recover through Replay History.</p>
                <button type="button" onClick={applyTournamentMatchBackfill} disabled={busy || !selectedTournamentBackfillGameIds.length} style={buttonStyle}>Apply selected tournament matches</button>
              </div> : <p style={{ color: "#64748b" }}>No ready tournament games are available to select.</p>;
            })()}
          </div> : null}
        </section>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Queue mode<br /><select value={workerMode} onChange={(event) => setWorkerMode(event.target.value)} style={inputStyle}><option value="batch">Batch</option><option value="drain">Drain until empty / limit</option></select></label>
          <label>Max jobs<br /><input value={workerMaxJobs} onChange={(event) => setWorkerMaxJobs(event.target.value)} style={inputStyle} /></label>
          <label>Time budget seconds<br /><input value={workerBudget} onChange={(event) => setWorkerBudget(event.target.value)} style={inputStyle} /></label>
          <label>Confirmation<br /><input value={workerConfirmation} onChange={(event) => setWorkerConfirmation(event.target.value)} placeholder={workerMode === "drain" ? "DRAIN BADGE QUEUE" : "PROCESS BADGE QUEUE"} style={inputStyle} /></label>
          <button type="button" onClick={runQueueWorker} disabled={busy} style={buttonStyle}>Run badge queue</button>
        </div>
        <hr style={{ border: 0, borderTop: "1px solid #e2e8f0", margin: "1rem 0" }} />
        <h3>Badge recompute</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Mode<br /><select value={recomputeMode} onChange={(event) => setRecomputeMode(event.target.value)} style={inputStyle}><option value="dry-run">dry-run</option><option value="append-only">append-only</option><option value="strict">strict</option></select></label>
          <label>Player ID<br /><input value={recomputePlayerId} onChange={(event) => setRecomputePlayerId(event.target.value)} style={inputStyle} /></label>
          <label>Badge ID<br /><input value={recomputeBadgeId} onChange={(event) => setRecomputeBadgeId(event.target.value)} style={inputStyle} /></label>
          <label>League<br /><input value={recomputeLeagueId} onChange={(event) => setRecomputeLeagueId(event.target.value)} style={inputStyle} /></label>
          <label>Context ID<br /><input value={recomputeContextId} onChange={(event) => setRecomputeContextId(event.target.value)} style={inputStyle} /></label>
          <label>Since<br /><input type="date" value={recomputeSince} onChange={(event) => setRecomputeSince(event.target.value)} style={inputStyle} /></label>
          <label>Until<br /><input type="date" value={recomputeUntil} onChange={(event) => setRecomputeUntil(event.target.value)} style={inputStyle} /></label>
          <label style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center" }}><input type="checkbox" checked={recomputeStrictGlobal} onChange={(event) => setRecomputeStrictGlobal(event.target.checked)} /> Allow strict global</label>
          <label>Confirmation<br /><input value={recomputeConfirmation} onChange={(event) => setRecomputeConfirmation(event.target.value)} placeholder="RUN BADGE RECOMPUTE" style={inputStyle} /></label>
          <button type="button" onClick={runBadgeRecompute} disabled={busy} style={ghostButtonStyle}>Run badge recompute</button>
        </div>
        {lastWorkerResult ? <><h3>Last worker result</h3><Pre value={lastWorkerResult} /></> : null}
      </article>
      <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Role assignments</h2>{table(overview.roles || [], ["email", "role", "user_id", "created_at", "updated_at"])}<div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", marginTop: "1rem", alignItems: "end" }}><label>Email<br /><input value={targetEmail} onChange={(event) => setTargetEmail(event.target.value)} style={inputStyle} /></label><label>Role<br /><select value={targetRole} onChange={(event) => setTargetRole(event.target.value)} style={inputStyle}>{roleOptions.map((role) => <option key={role} value={role}>{role}</option>)}</select></label><label>User ID optional<br /><input value={targetUserId} onChange={(event) => setTargetUserId(event.target.value)} style={inputStyle} /></label><label>Confirmation<br /><input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} placeholder="SAVE ROLE or REVOKE ROLE" style={inputStyle} /></label><button type="button" onClick={() => saveRole("upsert")} disabled={busy} style={buttonStyle}>Save role</button><button type="button" onClick={() => saveRole("revoke")} disabled={busy} style={ghostButtonStyle}>Revoke role</button></div></article>
      <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Admin activity</h2><p style={{ color: "#475569" }}>Retention guidance: {overview.retention_days} days. Suggested cutoff: {String(overview.retention_cutoff || "").slice(0, 10)}.</p>{overview.activity_warning ? <p style={{ color: "#b91c1c" }}>{overview.activity_warning}</p> : null}{table(overview.activity || [], ["created_at", "actor_email", "actor_role", "action_type", "entity_type", "entity_id", "source_page", "flagged_for_review", "note"])}</article>
    </> : null}
  </div>;
}
