"use client";

import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type StatusResponse = { enabled: boolean; status: string; roles?: string[]; retention_days?: number; retention_cutoff?: string };
type OverviewResponse = { ok: boolean; roles: Array<Record<string, unknown>>; activity: Array<Record<string, unknown>>; activity_warning?: string | null; health: Record<string, unknown>; role_options: string[]; retention_days: number; retention_cutoff: string };
type RoleResponse = { ok: boolean; operation_key?: string; roles: Array<Record<string, unknown>>; audit_warning?: string | null };
type WorkerResponse = { ok: boolean; mode?: string; operation_key?: string; read_only?: boolean; result?: Record<string, unknown>; summary?: Record<string, unknown>; worker_status?: Record<string, unknown>; audit_warning?: string | null };
type RatingReportResponse = { ok: boolean; mode: "admin_rating_report"; read_only: true; scope: string; available_scopes: string[]; generated_on: string; summary: Record<string, unknown>; rows: Array<Record<string, unknown>>; csv_text: string; csv_filename: string; csv_formula_neutralized: true; warnings: string[] };
type TournamentBackfillPreviewResponse = { ok: boolean; mode: "tournament_match_backfill_preview"; read_only: true; preview_fingerprint: string; confirmation_text: string; summary: Record<string, unknown>; candidates: Array<Record<string, unknown>>; warnings: string[] };
type TournamentBackfillApplyResponse = { ok: boolean; mode: "tournament_match_backfill_apply"; operation_id: string; operation_key: string; selected_game_ids: string[]; inserted_count: number; warnings: string[] };
type SocialSubmission = { id: string; name: string; event_type: string; event_date: unknown; status: string; submission_mode: string; submitted_by_name: string; summary_json: Record<string, unknown>; raw_event_json: Record<string, unknown>; created_at: unknown; updated_at: unknown; rejection_reason?: string | null; moderated_at?: unknown; moderated_by?: string | null };
type SocialSubmissionListResponse = { ok: boolean; mode: "admin_social_submission_review"; read_only: true; status: string; statuses: string[]; confirmation_text: Record<"approve" | "reject", string>; summary: Record<string, unknown>; submissions: SocialSubmission[]; warnings: string[] };
type SocialSubmissionModerationResponse = { ok: boolean; mode: "admin_social_submission_moderation"; operation_key: string; action: "approve" | "reject"; submission: SocialSubmission; warnings: string[] };
type GuardedOperationResponse = { ok: boolean; workflow?: string; operation_key: string; status?: string; result?: Record<string, unknown>; error?: string | null; recovery?: Record<string, string>; persisted_game_ids?: string[] };
type Props = { apiBase: string | null; clubId: string; status: StatusResponse | null };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
const mutedTextStyle = { color: "#475569" };
const warningTextStyle = { color: "#92400e" };
const dangerTextStyle = { color: "#991b1b" };
function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
async function apiError(response: Response): Promise<string> {
  const text = await response.text().catch(() => "");
  try {
    const detail = (JSON.parse(text) as { detail?: unknown }).detail;
    if (detail && typeof detail === "object" && "message" in detail) return String((detail as { message?: unknown }).message || text);
    return String(detail || text);
  } catch { return text || `API error (${response.status}).`; }
}
function table(rows: Array<Record<string, unknown>>, keys: string[]) {
  if (!rows.length) return <p style={{ color: "#64748b" }}>No rows.</p>;
  return <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}><thead><tr>{keys.map((key) => <th key={key} align="left" style={{ borderBottom: "1px solid #e2e8f0", padding: "0.4rem" }}>{key}</th>)}</tr></thead><tbody>{rows.slice(0, 100).map((row, idx) => <tr key={idx}>{keys.map((key) => <td key={key} style={{ borderBottom: "1px solid #f1f5f9", padding: "0.4rem", verticalAlign: "top" }}>{typeof row[key] === "object" && row[key] !== null ? JSON.stringify(row[key]) : String(row[key] ?? "")}</td>)}</tr>)}</tbody></table>{rows.length > 100 ? <p style={{ color: "#64748b" }}>Showing first 100 of {rows.length} rows.</p> : null}</div>;
}
function Pre({ value }: { value: unknown }) { return <pre style={{ whiteSpace: "pre-wrap", background: "#0f172a", color: "white", padding: "1rem", borderRadius: "12px", overflowX: "auto", fontSize: "0.82rem" }}>{JSON.stringify(value, null, 2)}</pre>; }
function downloadRatingReport(report: RatingReportResponse): void {
  const blob = new Blob(["\uFEFF", report.csv_text], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = report.csv_filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

export default function AdminToolsPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [flaggedOnly, setFlaggedOnly] = useState(false);
  const [ratingReportScope, setRatingReportScope] = useState("OVERALL");
  const [ratingReportScopes, setRatingReportScopes] = useState<string[]>(["OVERALL"]);
  const [ratingReport, setRatingReport] = useState<RatingReportResponse | null>(null);
  const [targetEmail, setTargetEmail] = useState("");
  const [targetRole, setTargetRole] = useState("read_only");
  const [targetUserId, setTargetUserId] = useState("");
  const [workerMode, setWorkerMode] = useState("batch");
  const [workerMaxJobs, setWorkerMaxJobs] = useState("25");
  const [workerBudget, setWorkerBudget] = useState("15");
  const [recomputeMode, setRecomputeMode] = useState("dry-run");
  const [recomputePlayerId, setRecomputePlayerId] = useState("");
  const [recomputeBadgeId, setRecomputeBadgeId] = useState("");
  const [recomputeLeagueId, setRecomputeLeagueId] = useState("");
  const [recomputeContextId, setRecomputeContextId] = useState("");
  const [recomputeSince, setRecomputeSince] = useState("");
  const [recomputeUntil, setRecomputeUntil] = useState("");
  const [recomputeStrictGlobal, setRecomputeStrictGlobal] = useState(false);
  const [lastWorkerResult, setLastWorkerResult] = useState<WorkerResponse | null>(null);
  const [tournamentBackfillPreview, setTournamentBackfillPreview] = useState<TournamentBackfillPreviewResponse | null>(null);
  const [selectedTournamentBackfillGameIds, setSelectedTournamentBackfillGameIds] = useState<string[]>([]);
  const [socialSubmissionStatus, setSocialSubmissionStatus] = useState("pending");
  const [socialSubmissionQueue, setSocialSubmissionQueue] = useState<SocialSubmissionListResponse | null>(null);
  const [selectedSocialSubmissionId, setSelectedSocialSubmissionId] = useState("");
  const [socialSubmissionAction, setSocialSubmissionAction] = useState<"approve" | "reject">("approve");
  const [socialSubmissionReason, setSocialSubmissionReason] = useState("");
  const [roleOperationKey, setRoleOperationKey] = useState("");
  const [workerOperationKey, setWorkerOperationKey] = useState("");
  const [recomputeOperationKey, setRecomputeOperationKey] = useState("");
  const [backfillOperationKey, setBackfillOperationKey] = useState("");
  const [socialOperationKey, setSocialOperationKey] = useState("");
  const [operationLookupKey, setOperationLookupKey] = useState("");
  const [operationLookup, setOperationLookup] = useState<GuardedOperationResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [overviewLoading, setOverviewLoading] = useState(false);
  const [overviewMessage, setOverviewMessage] = useState<string | null>(null);
  const [socialQueueLoading, setSocialQueueLoading] = useState(false);
  const [socialQueueMessage, setSocialQueueMessage] = useState<string | null>(null);
  const overviewRequest = useLatestRequestGuard(accessToken, clearProtectedAdminToolsState);
  const socialQueueRequest = useLatestRequestGuard(accessToken);
  const actionRequest = useLatestRequestGuard(accessToken);

  function clearProtectedAdminToolsState() {
    socialQueueRequest.invalidate();
    actionRequest.invalidate();
    setOverview(null);
    setRatingReport(null);
    setRatingReportScopes(["OVERALL"]);
    setLastWorkerResult(null);
    setTournamentBackfillPreview(null);
    setSelectedTournamentBackfillGameIds([]);
    setSocialSubmissionQueue(null);
    setSelectedSocialSubmissionId("");
    setSocialSubmissionReason("");
    setOperationLookup(null);
    setBusy(false);
    setOverviewLoading(false);
    setSocialQueueLoading(false);
    setMessage(null);
    setOverviewMessage(null);
    setSocialQueueMessage(null);
  }

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
    const generation = overviewRequest.begin();
    setOverviewLoading(true); setOverviewMessage(null);
    setOverview(null);
    try {
      const payload = await requestJson<OverviewResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tools/overview?flagged_only=${flaggedOnly ? "true" : "false"}&limit=200`);
      if (!overviewRequest.isCurrent(generation)) return;
      setOverview(payload); setOverviewMessage(`Loaded ${payload.roles?.length || 0} role assignment(s) and ${payload.activity?.length || 0} activity row(s).`);
    } catch (error) { if (overviewRequest.isCurrent(generation)) setOverviewMessage(error instanceof Error ? error.message : "Unable to load Admin Tools overview."); }
    finally { if (overviewRequest.isCurrent(generation)) setOverviewLoading(false); }
  }

  async function loadRatingReport() {
    const generation = actionRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<RatingReportResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tools/reports/ratings?league=${encodeURIComponent(ratingReportScope)}`);
      if (!actionRequest.isCurrent(generation)) return;
      setRatingReport(payload);
      setRatingReportScopes(payload.available_scopes);
      setRatingReportScope(payload.scope);
      setMessage(`Loaded ${String(payload.summary.row_count ?? payload.rows.length)} ${payload.scope} rating report row(s). No rows were written.`);
    } catch (error) { if (actionRequest.isCurrent(generation)) { setRatingReport(null); setMessage(error instanceof Error ? error.message : "Unable to load the rating report."); } }
    finally { if (actionRequest.isCurrent(generation)) setBusy(false); }
  }

  async function saveRole(action: "upsert" | "revoke", confirmationText: string) {
    const generation = actionRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const key = roleOperationKey || `admin-role:${Date.now()}:${crypto.randomUUID()}`;
      if (!roleOperationKey) setRoleOperationKey(key);
      const payload = await requestJson<RoleResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tools/roles`, {
        method: "PATCH",
        body: JSON.stringify({ email: targetEmail, role: targetRole, user_id: targetUserId || null, action, confirmation_text: confirmationText, operation_key: key })
      });
      if (!actionRequest.isCurrent(generation)) return;
      setOverview((current) => current ? { ...current, roles: payload.roles } : current);
      setMessage(payload.audit_warning ? `Saved, but audit warning: ${payload.audit_warning}` : (action === "upsert" ? "Role assignment saved." : "Role assignment revoked."));
      setRoleOperationKey("");
    } catch (error) { if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to update role assignment."); }
    finally { if (actionRequest.isCurrent(generation)) setBusy(false); }
  }

  async function runQueueWorker(confirmationText: string) {
    const generation = actionRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const key = workerOperationKey || `badge-queue:${Date.now()}:${crypto.randomUUID()}`;
      if (!workerOperationKey) setWorkerOperationKey(key);
      const payload = await requestJson<WorkerResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tools/workers/badge-queue`, {
        method: "POST",
        body: JSON.stringify({ mode: workerMode, max_jobs: Number(workerMaxJobs), time_budget_seconds: Number(workerBudget), confirmation_text: confirmationText, operation_key: key })
      });
      if (!actionRequest.isCurrent(generation)) return;
      setLastWorkerResult(payload);
      setMessage(payload.audit_warning ? `Badge queue completed with audit warning: ${payload.audit_warning}` : "Badge queue worker completed.");
      setWorkerOperationKey("");
    } catch (error) { if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to run badge queue worker."); }
    finally { if (actionRequest.isCurrent(generation)) setBusy(false); }
  }

  async function loadTournamentBackfillPreview() {
    const generation = actionRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<TournamentBackfillPreviewResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tools/backfills/tournament-matches/preview?limit=500`);
      if (!actionRequest.isCurrent(generation)) return;
      setTournamentBackfillPreview(payload);
      setSelectedTournamentBackfillGameIds([]);
      setMessage(`Tournament backfill preview found ${String(payload.summary.ready_count ?? 0)} ready and ${String(payload.summary.blocked_count ?? 0)} blocked missing match candidate(s). No rows were written.`);
    } catch (error) { if (actionRequest.isCurrent(generation)) { setTournamentBackfillPreview(null); setMessage(error instanceof Error ? error.message : "Unable to preview missing tournament matches."); } }
    finally { if (actionRequest.isCurrent(generation)) setBusy(false); }
  }

  async function loadSocialSubmissionQueue() {
    const generation = socialQueueRequest.begin();
    setSocialQueueLoading(true); setSocialQueueMessage(null);
    setSocialSubmissionQueue(null);
    setSelectedSocialSubmissionId("");
    setSocialSubmissionReason("");
    try {
      const payload = await requestJson<SocialSubmissionListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tools/social-submissions?status=${encodeURIComponent(socialSubmissionStatus)}&limit=100`);
      if (!socialQueueRequest.isCurrent(generation)) return;
      setSocialSubmissionQueue(payload);
      setSocialQueueMessage(`Loaded ${payload.submissions.length} ${payload.status} Club Social submission(s). No rows were written.`);
    } catch (error) { if (socialQueueRequest.isCurrent(generation)) setSocialQueueMessage(error instanceof Error ? error.message : "Unable to load the Club Social review queue."); }
    finally { if (socialQueueRequest.isCurrent(generation)) setSocialQueueLoading(false); }
  }

  function selectSocialSubmission(submission: SocialSubmission) {
    setSelectedSocialSubmissionId(submission.id);
    setSocialSubmissionAction(submission.status === "saved" ? "reject" : "approve");
    setSocialSubmissionReason("");
  }

  async function moderateSocialSubmission(confirmationText: string) {
    const selected = socialSubmissionQueue?.submissions.find((submission) => submission.id === selectedSocialSubmissionId);
    if (!selected || !socialSubmissionQueue) { setMessage("Select and review one Club Social submission first."); return; }
    if (socialSubmissionAction === "reject" && !socialSubmissionReason.trim()) { setMessage("Enter a rejection reason before rejecting this submission."); return; }
    const generation = actionRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const key = socialOperationKey || `social-moderation:${Date.now()}:${crypto.randomUUID()}`;
      if (!socialOperationKey) setSocialOperationKey(key);
      const payload = await requestJson<SocialSubmissionModerationResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tools/social-submissions/${encodeURIComponent(selected.id)}/moderate`, {
        method: "POST",
        body: JSON.stringify({
          action: socialSubmissionAction,
          expected_status: selected.status,
          rejection_reason: socialSubmissionReason,
          confirmation_text: confirmationText,
          operation_key: key,
          source: "next_admin_tools_social_review"
        })
      });
      if (!actionRequest.isCurrent(generation)) return;
      setSocialSubmissionQueue((current) => current ? { ...current, submissions: current.submissions.filter((submission) => submission.id !== selected.id), summary: { ...current.summary, returned_count: Math.max(0, current.submissions.length - 1) } } : current);
      setSelectedSocialSubmissionId("");
      setSocialSubmissionReason("");
      setSocialOperationKey("");
      const warning = payload.warnings.length ? ` Audit warning: ${payload.warnings.join(" ")}` : "";
      setMessage(`Submission ${payload.action === "approve" ? "approved" : "rejected"}.${warning}`);
    } catch (error) { if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to moderate the Club Social submission."); }
    finally { if (actionRequest.isCurrent(generation)) setBusy(false); }
  }

  async function applyTournamentMatchBackfill(confirmationText: string) {
    if (!tournamentBackfillPreview) { setMessage("Load and review a current tournament backfill preview first."); return; }
    if (!selectedTournamentBackfillGameIds.length) { setMessage("Select at least one ready tournament game to backfill."); return; }
    const generation = actionRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const key = backfillOperationKey || `tournament-backfill:${Date.now()}:${crypto.randomUUID()}`;
      if (!backfillOperationKey) setBackfillOperationKey(key);
      const payload = await requestJson<TournamentBackfillApplyResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tools/backfills/tournament-matches/apply`, {
        method: "POST",
        body: JSON.stringify({
          game_ids: selectedTournamentBackfillGameIds,
          preview_fingerprint: tournamentBackfillPreview.preview_fingerprint,
          preview_limit: Number(tournamentBackfillPreview.summary.candidate_limit ?? 500),
          confirmation_text: confirmationText,
          operation_key: key,
          source: "next_admin_tools_tournament_match_backfill"
        })
      });
      if (!actionRequest.isCurrent(generation)) return;
      setTournamentBackfillPreview(null);
      setSelectedTournamentBackfillGameIds([]);
      setBackfillOperationKey("");
      setMessage(`Backfilled ${payload.inserted_count} reviewed tournament match(es). Operation ${payload.operation_id}. Reload the preview and verify Match Log before any further write.`);
    } catch (error) { if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to apply the tournament match backfill."); }
    finally { if (actionRequest.isCurrent(generation)) setBusy(false); }
  }

  async function runBadgeRecompute(confirmationText = "") {
    const generation = actionRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const key = recomputeMode === "dry-run" ? "" : (recomputeOperationKey || `tools-badge-recompute:${Date.now()}:${crypto.randomUUID()}`);
      if (recomputeMode !== "dry-run" && !recomputeOperationKey) setRecomputeOperationKey(key);
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
          confirmation_text: confirmationText,
          operation_key: key
        })
      });
      if (!actionRequest.isCurrent(generation)) return;
      setLastWorkerResult(payload);
      setMessage(payload.read_only ? "Read-only badge recompute preview finished; no rows were written." : "Badge recompute finished.");
      if (recomputeMode !== "dry-run") setRecomputeOperationKey("");
    } catch (error) { if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to run badge recompute."); }
    finally { if (actionRequest.isCurrent(generation)) setBusy(false); }
  }

  async function inspectOperation() {
    if (!operationLookupKey.trim()) { setMessage("Enter the exact guarded operation key first."); return; }
    const generation = actionRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<GuardedOperationResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tools/operations/${encodeURIComponent(operationLookupKey.trim())}`);
      if (!actionRequest.isCurrent(generation)) return;
      setOperationLookup(payload);
      setMessage(`Operation ${payload.operation_key} is ${payload.status || "available"}.`);
    } catch (error) { if (actionRequest.isCurrent(generation)) { setOperationLookup(null); setMessage(error instanceof Error ? error.message : "Unable to inspect operation."); } }
    finally { if (actionRequest.isCurrent(generation)) setBusy(false); }
  }

  async function recoverTournamentBackfill(confirmationText: string) {
    if (!operationLookup || operationLookup.workflow !== "tournament_match_backfill") { setMessage("Inspect a tournament backfill operation first."); return; }
    const generation = actionRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<GuardedOperationResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tools/backfills/tournament-matches/operations/${encodeURIComponent(operationLookup.operation_key)}/recover`, {
        method: "POST",
        body: JSON.stringify({ confirmation_text: confirmationText, source: "next_admin_tools_tournament_match_backfill_recovery" })
      });
      if (!actionRequest.isCurrent(generation)) return;
      setOperationLookup(payload);
      setMessage("Tournament backfill rows are reconciled. Verify Replay History before further writes.");
    } catch (error) { if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to recover tournament backfill."); }
    finally { if (actionRequest.isCurrent(generation)) setBusy(false); }
  }

  useAuthenticatedAutoLoad(
    status?.enabled !== false ? accessToken : "",
    loadOverview,
    flaggedOnly ? "flagged" : "all"
  );
  useAuthenticatedAutoLoad(
    status?.enabled !== false ? accessToken : "",
    loadSocialSubmissionQueue,
    socialSubmissionStatus
  );

  if (status && !status.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Disabled</h2><p>Set <code>JUPR_ENABLE_NEXT_ADMIN_TOOLS=1</code> on FastAPI to enable guarded Admin Tools.</p></article>;

  const roleOptions = overview?.role_options || status?.roles || ["read_only", "scorekeeper", "organizer", "club_owner", "super_admin"];
  const health = overview?.health || {};
  const workerStatus = (health as { workers?: unknown }).workers;
  const pendingOperationKeys = [roleOperationKey, workerOperationKey, recomputeOperationKey, backfillOperationKey, socialOperationKey].filter(Boolean);
  return <div style={{ display: "grid", gap: "1rem" }}>
    <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Admin session</h2><p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>{sessionLoading ? <p>Checking session…</p> : null}{sessionMessage ? <p style={{ color: "#64748b" }}>{sessionMessage}</p> : null}</article>
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Overview</h2>
      <label style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center" }}><input type="checkbox" checked={flaggedOnly} onChange={(event) => setFlaggedOnly(event.target.checked)} disabled={busy || overviewLoading} /> Flagged activity only</label>
      <p><button type="button" onClick={loadOverview} disabled={busy || overviewLoading || !accessToken} style={buttonStyle}>{overviewLoading ? "Refreshing…" : "Refresh overview"}</button></p>
      {overviewMessage ? <p role="status" aria-live="polite" style={{ color: overviewMessage.toLowerCase().includes("unable") ? "#b91c1c" : "#166534" }}>{overviewMessage}</p> : null}
      {message ? <p role="status" aria-live="polite" style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("type") || message.toLowerCase().includes("blocked") || message.toLowerCase().includes("critical") || message.toLowerCase().includes("recovery") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </article>
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Guarded operation recovery</h2>
      <p style={mutedTextStyle}>Every applying action has a durable operation key. If the outcome is uncertain, stop further writes, keep the exact key shown below, and inspect it here. A completed key replays its saved result; an incomplete key never runs the mutation twice.</p>
      {pendingOperationKeys.length ? <p style={warningTextStyle}><strong>Retained operation key{pendingOperationKeys.length === 1 ? "" : "s"} after an incomplete request:</strong> {pendingOperationKeys.map((key) => <code key={key} style={{ display: "block", overflowWrap: "anywhere" }}>{key}</code>)}</p> : null}
      <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) auto", gap: "0.75rem", alignItems: "end" }}>
        <label>Exact operation key<br /><input value={operationLookupKey} onChange={(event) => { setOperationLookupKey(event.target.value); setOperationLookup(null); }} style={inputStyle} /></label>
        <button type="button" onClick={inspectOperation} disabled={busy || !accessToken} style={ghostButtonStyle}>Inspect operation</button>
      </div>
      {operationLookup ? <div style={{ marginTop: "1rem" }}>
        <Pre value={operationLookup} />
        {operationLookup.status === "recovery_required" ? <p style={dangerTextStyle}><strong>Stop:</strong> do not blindly retry this operation. Follow the workflow-specific recovery below, then verify Match Log and Replay History before another write.</p> : null}
        {operationLookup.workflow === "tournament_match_backfill" && operationLookup.status === "recovery_required" ? <section style={{ border: "1px solid #fecaca", background: "#fef2f2", borderRadius: "12px", padding: "1rem" }}>
          <h3 style={{ marginTop: 0 }}>Reconcile tournament backfill</h3>
          <p>Recovery only succeeds when every selected game now has exactly one official match. It does not create or delete matches.</p>
          <p><ConfirmAction triggerLabel="Reconcile existing rows" title="Reconcile this tournament backfill?" description="This verifies that every selected game now has exactly one official match. It does not create or delete matches." confirmLabel="Yes, reconcile rows" confirmationText="RECOVER TOURNAMENT BACKFILL" disabled={busy} busy={busy} onConfirm={recoverTournamentBackfill} /></p>
        </section> : null}
      </div> : null}
      <p><a href="/admin/match-log">Open Match Log</a> · <a href="/admin/replay-history">Open Replay History</a> · <a href="/admin/guide">Open Admin Guide</a></p>
      <p style={mutedTextStyle}>If FastAPI is disabled or recovery cannot prove the exact result, stop and use the existing Streamlit Admin fallback with the operation key and audit activity visible.</p>
    </article>
    {overview ? <>
      <article style={cardStyle}><h2 style={{ marginTop: 0 }}>System health</h2><Pre value={health} /></article>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Reports &amp; exports</h2>
        <p style={mutedTextStyle}>Generate the Streamlit-parity overall or league rating report. This is a club-scoped, read-only query; FastAPI creates a formula-neutralized CSV and the browser only downloads those server-produced bytes.</p>
        <div style={{ display: "grid", gridTemplateColumns: "minmax(200px, 1fr) auto", gap: "0.75rem", alignItems: "end" }}>
          <label>Report scope<br /><select value={ratingReportScope} onChange={(event) => { setRatingReportScope(event.target.value); setRatingReport(null); }} style={inputStyle}>{ratingReportScopes.map((scope) => <option key={scope} value={scope}>{scope}</option>)}</select></label>
          <button type="button" onClick={loadRatingReport} disabled={busy} style={ghostButtonStyle}>Generate report</button>
        </div>
        {!ratingReport ? <p style={{ color: "#64748b" }}>Generate the selected scope to view and download it. Available league scopes load after the first OVERALL report.</p> : <div style={{ marginTop: "1rem" }}>
          {ratingReport.warnings.map((warning) => <p key={warning} style={{ color: "#92400e" }}>{warning}</p>)}
          {table(ratingReport.rows, ["name", "jupr", "wins", "losses", "matches_played", "win_percent", "gain"])}
          <p><button type="button" onClick={() => downloadRatingReport(ratingReport)} disabled={!ratingReport.rows.length} style={buttonStyle}>Download {ratingReport.scope} CSV</button></p>
        </div>}
      </article>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Club Social submission review</h2>
        <p style={{ color: "#475569" }}>Review unrated Club Social event submissions for this club. Loading a queue is read-only. Approve or reject requires <code>manage_matches</code> permission, a current expected status, and a Yes/No confirmation dialog; the dialog supplies the internal API safeguard.</p>
        <div style={{ display: "grid", gridTemplateColumns: "minmax(180px, 1fr) auto", gap: "0.75rem", alignItems: "end" }}>
          <label>Queue<br /><select value={socialSubmissionStatus} onChange={(event) => { setSocialSubmissionStatus(event.target.value); setSocialSubmissionQueue(null); setSelectedSocialSubmissionId(""); }} disabled={busy || socialQueueLoading} style={inputStyle}><option value="pending">pending</option><option value="saved">saved</option><option value="rejected">rejected</option></select></label>
          <button type="button" onClick={loadSocialSubmissionQueue} disabled={busy || socialQueueLoading || !accessToken} style={ghostButtonStyle}>{socialQueueLoading ? "Refreshing…" : "Refresh review queue"}</button>
        </div>
        {socialQueueMessage ? <p role="status" aria-live="polite" style={{ color: socialQueueMessage.toLowerCase().includes("unable") ? "#b91c1c" : "#166534" }}>{socialQueueMessage}</p> : null}
        {socialSubmissionQueue ? <div style={{ marginTop: "1rem" }}>
          {socialSubmissionQueue.warnings.map((warning) => <p key={warning} style={{ color: "#92400e" }}>{warning}</p>)}
          {!socialSubmissionQueue.submissions.length ? <p style={{ color: "#64748b" }}>No {socialSubmissionQueue.status} Club Social submissions found.</p> : <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}><thead><tr>{["name", "event", "date", "submitted by", "participants", "matches", "updated", "review"].map((heading) => <th key={heading} align="left" style={{ borderBottom: "1px solid #e2e8f0", padding: "0.4rem" }}>{heading}</th>)}</tr></thead><tbody>{socialSubmissionQueue.submissions.map((submission) => <tr key={submission.id} style={{ background: selectedSocialSubmissionId === submission.id ? "#eff6ff" : "transparent" }}><td style={{ borderBottom: "1px solid #f1f5f9", padding: "0.4rem" }}>{submission.name}</td><td style={{ borderBottom: "1px solid #f1f5f9", padding: "0.4rem" }}>{submission.event_type}</td><td style={{ borderBottom: "1px solid #f1f5f9", padding: "0.4rem" }}>{String(submission.event_date ?? "")}</td><td style={{ borderBottom: "1px solid #f1f5f9", padding: "0.4rem" }}>{submission.submitted_by_name}</td><td style={{ borderBottom: "1px solid #f1f5f9", padding: "0.4rem" }}>{String(submission.summary_json.participant_count ?? 0)}</td><td style={{ borderBottom: "1px solid #f1f5f9", padding: "0.4rem" }}>{String(submission.summary_json.match_count ?? 0)}</td><td style={{ borderBottom: "1px solid #f1f5f9", padding: "0.4rem" }}>{String(submission.updated_at ?? "")}</td><td style={{ borderBottom: "1px solid #f1f5f9", padding: "0.4rem" }}><button type="button" onClick={() => selectSocialSubmission(submission)} disabled={busy} style={ghostButtonStyle}>Review</button></td></tr>)}</tbody></table></div>}
          {(() => {
            const selected = socialSubmissionQueue.submissions.find((submission) => submission.id === selectedSocialSubmissionId);
            if (!selected) return null;
            const targetStatus = socialSubmissionAction === "approve" ? "saved" : "rejected";
            const isNoOp = selected.status === targetStatus;
            const expectedConfirmation = socialSubmissionQueue.confirmation_text[socialSubmissionAction];
            return <section style={{ border: "1px solid #bfdbfe", background: "#eff6ff", borderRadius: "12px", padding: "1rem", marginTop: "1rem" }}>
              <h3 style={{ marginTop: 0 }}>Review: {selected.name}</h3>
              <p><strong>Status:</strong> {selected.status} · <strong>Submitted by:</strong> {selected.submitted_by_name} · <strong>Mode:</strong> {selected.submission_mode}</p>
              {selected.rejection_reason ? <p><strong>Previous rejection reason:</strong> {selected.rejection_reason}</p> : null}
              <details><summary>Summary JSON</summary><Pre value={selected.summary_json} /></details>
              <details><summary>Raw event JSON</summary><Pre value={selected.raw_event_json} /></details>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem", alignItems: "end", marginTop: "1rem" }}>
                <label>Action<br /><select value={socialSubmissionAction} onChange={(event) => { setSocialSubmissionAction(event.target.value as "approve" | "reject"); setSocialSubmissionReason(""); }} style={inputStyle}><option value="approve">approve → saved</option><option value="reject">reject → rejected</option></select></label>
                {socialSubmissionAction === "reject" ? <label>Rejection reason<br /><input value={socialSubmissionReason} onChange={(event) => setSocialSubmissionReason(event.target.value)} maxLength={1200} style={inputStyle} /></label> : null}
                <ConfirmAction triggerLabel={socialSubmissionAction === "approve" ? "Approve selected submission" : "Reject selected submission"} title={`${socialSubmissionAction === "approve" ? "Approve" : "Reject"} ${selected.name}?`} description={socialSubmissionAction === "approve" ? "This marks the reviewed Club Social submission as saved." : "This rejects the reviewed submission and records the rejection reason."} confirmLabel={socialSubmissionAction === "approve" ? "Yes, approve submission" : "Yes, reject submission"} confirmationText={expectedConfirmation} tone={socialSubmissionAction === "reject" ? "danger" : "default"} disabled={busy || isNoOp || (socialSubmissionAction === "reject" && !socialSubmissionReason.trim())} busy={busy} onConfirm={moderateSocialSubmission} />
              </div>
              {isNoOp ? <p style={{ color: "#92400e" }}>This submission is already {targetStatus}; choose the other action or reload another queue.</p> : <p style={{ color: "#991b1b" }}>Only the reviewed submission status and moderation metadata will change. Reload the queue after any stale-status warning.</p>}
            </section>;
          })()}
        </div> : <p style={{ color: "#64748b" }}>{socialQueueLoading ? "Loading review queue…" : "No review queue is available."}</p>}
      </article>
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
                <p style={{ color: "#991b1b" }}>After apply, verify the exact rows in Match Log. If counts or ratings disagree, stop further writes and recover through Replay History.</p>
                <ConfirmAction triggerLabel="Apply selected tournament matches" title="Apply this tournament match backfill?" description={`This writes official matches for ${selectedTournamentBackfillGameIds.length} reviewed game(s) after rechecking the preview and duplicate state.`} confirmLabel="Yes, apply backfill" confirmationText={tournamentBackfillPreview.confirmation_text} tone="danger" disabled={busy || !selectedTournamentBackfillGameIds.length} busy={busy} onConfirm={applyTournamentMatchBackfill} />
              </div> : <p style={{ color: "#64748b" }}>No ready tournament games are available to select.</p>;
            })()}
          </div> : null}
        </section>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Queue mode<br /><select value={workerMode} onChange={(event) => setWorkerMode(event.target.value)} style={inputStyle}><option value="batch">Batch</option><option value="drain">Drain until empty / limit</option></select></label>
          <label>Max jobs<br /><input value={workerMaxJobs} onChange={(event) => setWorkerMaxJobs(event.target.value)} style={inputStyle} /></label>
          <label>Time budget seconds<br /><input value={workerBudget} onChange={(event) => setWorkerBudget(event.target.value)} style={inputStyle} /></label>
          <ConfirmAction triggerLabel="Run badge queue" title={`${workerMode === "drain" ? "Drain" : "Process"} the badge queue?`} description={`This runs up to ${workerMaxJobs} queued job(s) within the ${workerBudget}-second budget.`} confirmLabel={workerMode === "drain" ? "Yes, drain queue" : "Yes, process queue"} confirmationText={workerMode === "drain" ? "DRAIN BADGE QUEUE" : "PROCESS BADGE QUEUE"} tone={workerMode === "drain" ? "danger" : "default"} disabled={busy} busy={busy} onConfirm={runQueueWorker} />
        </div>
        <hr style={{ border: 0, borderTop: "1px solid #e2e8f0", margin: "1rem 0" }} />
        <h3>Badge recompute</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Mode<br /><select value={recomputeMode} onChange={(event) => { setRecomputeMode(event.target.value); setRecomputeOperationKey(""); }} style={inputStyle}><option value="dry-run">dry-run (no writes)</option><option value="append-only">append-only</option><option value="strict">strict</option></select></label>
          <label>Player ID<br /><input value={recomputePlayerId} onChange={(event) => setRecomputePlayerId(event.target.value)} style={inputStyle} /></label>
          <label>Badge ID<br /><input value={recomputeBadgeId} onChange={(event) => setRecomputeBadgeId(event.target.value)} style={inputStyle} /></label>
          <label>League<br /><input value={recomputeLeagueId} onChange={(event) => setRecomputeLeagueId(event.target.value)} style={inputStyle} /></label>
          <label>Context ID<br /><input value={recomputeContextId} onChange={(event) => setRecomputeContextId(event.target.value)} style={inputStyle} /></label>
          <label>Since<br /><input type="date" value={recomputeSince} onChange={(event) => setRecomputeSince(event.target.value)} style={inputStyle} /></label>
          <label>Until<br /><input type="date" value={recomputeUntil} onChange={(event) => setRecomputeUntil(event.target.value)} style={inputStyle} /></label>
          <label style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center" }}><input type="checkbox" checked={recomputeStrictGlobal} onChange={(event) => setRecomputeStrictGlobal(event.target.checked)} /> Allow strict global</label>
          {recomputeMode === "dry-run" ? <button type="button" onClick={() => void runBadgeRecompute()} disabled={busy} style={ghostButtonStyle}>Run badge recompute preview</button> : <ConfirmAction triggerLabel="Run badge recompute" title={`Run ${recomputeMode} badge recompute?`} description="This applies badge changes for the selected scope through the guarded worker." confirmLabel="Yes, run recompute" confirmationText="RUN BADGE RECOMPUTE" tone={recomputeMode === "strict" ? "danger" : "default"} disabled={busy} busy={busy} onConfirm={runBadgeRecompute} />}
        </div>
        {lastWorkerResult ? <><h3>Last worker result</h3><Pre value={lastWorkerResult} /></> : null}
      </article>
      <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Role assignments</h2>{table(overview.roles || [], ["email", "role", "user_id", "created_at", "updated_at"])}<div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", marginTop: "1rem", alignItems: "end" }}><label>Email<br /><input value={targetEmail} onChange={(event) => setTargetEmail(event.target.value)} style={inputStyle} /></label><label>Role<br /><select value={targetRole} onChange={(event) => setTargetRole(event.target.value)} style={inputStyle}>{roleOptions.map((role) => <option key={role} value={role}>{role}</option>)}</select></label><label>User ID optional<br /><input value={targetUserId} onChange={(event) => setTargetUserId(event.target.value)} style={inputStyle} /></label><ConfirmAction triggerLabel="Save role" title={`Save ${targetRole} role for ${targetEmail || "this account"}?`} description="This creates or updates the club-scoped role assignment and records the guarded operation." confirmLabel="Yes, save role" confirmationText="SAVE ROLE" disabled={busy || !targetEmail.trim()} busy={busy} onConfirm={(confirmationText) => saveRole("upsert", confirmationText)} /><ConfirmAction triggerLabel="Revoke role" title={`Revoke the role for ${targetEmail || "this account"}?`} description="This removes the club-scoped role assignment for the selected email." confirmLabel="Yes, revoke role" confirmationText="REVOKE ROLE" tone="danger" disabled={busy || !targetEmail.trim()} busy={busy} onConfirm={(confirmationText) => saveRole("revoke", confirmationText)} /></div></article>
      <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Admin activity</h2><p style={{ color: "#475569" }}>Retention guidance: {overview.retention_days} days. Suggested cutoff: {String(overview.retention_cutoff || "").slice(0, 10)}.</p>{overview.activity_warning ? <p style={{ color: "#b91c1c" }}>{overview.activity_warning}</p> : null}{table(overview.activity || [], ["created_at", "actor_email", "actor_role", "action_type", "entity_type", "entity_id", "source_page", "flagged_for_review", "note"])}</article>
    </> : null}
  </div>;
}
