"use client";

import { useMemo, useState } from "react";
import type {
  AdminBadgeAuditResponse,
  AdminBadgeDebugResponse,
  AdminBadgeDiagnosticsStatusResponse,
  AdminBadgeOption,
  AdminBadgeOptionsResponse,
  AdminBadgePlayerOption
} from "@/lib/adminBadgeDiagnosticsApi";
import { ConfirmAction } from "@/components/ConfirmAction";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = { apiBase: string | null; clubId: string; status: AdminBadgeDiagnosticsStatusResponse };
type RepairResponse = { ok: boolean; mode?: string; recompute_mode?: string; operation_key?: string; read_only?: boolean; summary?: Record<string, unknown>; revoked_count?: number; rows?: Array<Record<string, unknown>>; audit_warning?: string | null };
type BadgeStateUpdateResponse = { ok: boolean; mode: "badge_definition_state_update"; operation_key?: string; badge: { badge_id: string; name: string; state: "live" | "frozen" | "deprecated"; state_changed_at?: string | null; state_change_reason?: string | null }; force: boolean; audit_warning?: string | null };
type BadgeOperationResponse = { ok: boolean; workflow: string; operation_key: string; status: string; result: Record<string, unknown>; error?: string | null; recovery?: Record<string, string> };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function shortValue(value: unknown): string {
  if (value == null || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value).slice(0, 260);
  return String(value);
}

function countValue(report: Record<string, unknown>, key: string): number | string {
  const counts = report.counts as Record<string, unknown> | undefined;
  const value = counts?.[key];
  return typeof value === "number" || typeof value === "string" ? value : "—";
}

function reportRows(report: Record<string, unknown>, key: string): Array<Record<string, unknown>> {
  const value = report[key];
  return Array.isArray(value) ? value.filter((item) => item && typeof item === "object") as Array<Record<string, unknown>> : [];
}

function messageColor(message: string | null): string {
  const text = (message || "").toLowerCase();
  if (text.includes("unable") || text.includes("error") || text.includes("disabled") || text.includes("missing") || text.includes("type ") || text.includes("critical") || text.includes("recovery")) return "#b91c1c";
  return "#166534";
}

export default function BadgeDiagnosticsPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [players, setPlayers] = useState<AdminBadgePlayerOption[]>([]);
  const [badges, setBadges] = useState<AdminBadgeOption[]>([]);
  const [playerId, setPlayerId] = useState("");
  const [badgeId, setBadgeId] = useState("high_roller");
  const [leagueId, setLeagueId] = useState("");
  const [contextId, setContextId] = useState("");
  const [matchLimit, setMatchLimit] = useState("5000");
  const [since, setSince] = useState("");
  const [until, setUntil] = useState("");
  const [includeNonLive, setIncludeNonLive] = useState(false);
  const [includeRevoked, setIncludeRevoked] = useState(false);
  const [recomputeMode, setRecomputeMode] = useState("dry-run");
  const [playerBadgeId, setPlayerBadgeId] = useState("");
  const [revokeReason, setRevokeReason] = useState("");
  const [badgeStateTarget, setBadgeStateTarget] = useState<"live" | "frozen" | "deprecated">("frozen");
  const [badgeStateReason, setBadgeStateReason] = useState("");
  const [badgeStateForce, setBadgeStateForce] = useState(false);
  const [recomputeOperationKey, setRecomputeOperationKey] = useState("");
  const [revokeOperationKey, setRevokeOperationKey] = useState("");
  const [stateOperationKey, setStateOperationKey] = useState("");
  const [operationLookupKey, setOperationLookupKey] = useState("");
  const [operationLookup, setOperationLookup] = useState<BadgeOperationResponse | null>(null);
  const [debugReport, setDebugReport] = useState<Record<string, unknown> | null>(null);
  const [auditReport, setAuditReport] = useState<Record<string, unknown> | null>(null);
  const [repairResult, setRepairResult] = useState<RepairResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const optionsRequest = useLatestRequestGuard(accessToken, clearProtectedBadgeState);
  const reportRequest = useLatestRequestGuard(accessToken);
  const actionRequest = useLatestRequestGuard(accessToken);

  const selectedPlayerName = useMemo(() => players.find((player) => String(player.id) === String(playerId))?.name || "", [players, playerId]);
  const selectedBadge = useMemo(() => badges.find((badge) => String(badge.badge_id) === String(badgeId)), [badges, badgeId]);
  const selectedBadgeName = selectedBadge?.name || badgeId;

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Badge Diagnostics.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = payload?.detail;
      const text = detail && typeof detail === "object" ? String(detail.message || JSON.stringify(detail)) : String(detail || `API error (${response.status})`);
      throw new Error(text);
    }
    return payload as T;
  }

  function clearProtectedBadgeState() {
    reportRequest.invalidate();
    setBusy(false); setMessage(null);
    setPlayers([]); setBadges([]); setPlayerId(""); setBadgeId("");
    setRecomputeOperationKey(""); setRevokeOperationKey(""); setStateOperationKey(""); setOperationLookupKey("");
    setDebugReport(null); setAuditReport(null); setRepairResult(null); setOperationLookup(null);
  }

  async function loadOptions() {
    const generation = optionsRequest.begin();
    reportRequest.invalidate();
    setBusy(true); setMessage(null);
    setDebugReport(null); setAuditReport(null); setRepairResult(null);
    try {
      const payload = await requestJson<AdminBadgeOptionsResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/badges/options`);
      if (!optionsRequest.isCurrent(generation)) return;
      const nextPlayers = payload.players || [];
      const nextBadges = payload.badges || [];
      setPlayers(nextPlayers);
      setBadges(nextBadges);
      setPlayerId(nextPlayers.some((row) => String(row.id) === playerId) ? playerId : String(nextPlayers[0]?.id || ""));
      setBadgeId(nextBadges.some((row) => row.badge_id === badgeId) ? badgeId : String(nextBadges[0]?.badge_id || ""));
      setMessage(`Loaded ${payload.player_count} player option(s) and ${payload.badge_count} badge option(s).`);
    } catch (error) { if (optionsRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load badge diagnostic options."); }
    finally { if (optionsRequest.isCurrent(generation)) setBusy(false); }
  }

  async function runDebug() {
    if (!playerId || !badgeId) { setMessage("Choose a player and badge before running Badge Debug."); return; }
    const generation = reportRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const params = new URLSearchParams({ player_id: String(playerId), badge_id: badgeId, match_limit: String(matchLimit || 5000) });
      if (leagueId.trim()) params.set("league_id", leagueId.trim());
      const payload = await requestJson<AdminBadgeDebugResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/badges/debug?${params.toString()}`);
      if (!reportRequest.isCurrent(generation)) return;
      setDebugReport(payload.report || {});
      setMessage(`Badge Debug complete for ${selectedPlayerName || playerId} / ${selectedBadgeName}.`);
    } catch (error) { if (reportRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to run Badge Debug."); }
    finally { if (reportRequest.isCurrent(generation)) setBusy(false); }
  }

  async function runAudit() {
    const generation = reportRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const params = new URLSearchParams({ match_limit: String(matchLimit || 5000), include_non_live: String(includeNonLive), include_revoked: String(includeRevoked) });
      if (playerId) params.set("player_id", String(playerId));
      if (badgeId) params.set("badge_id", badgeId);
      if (leagueId.trim()) params.set("league_id", leagueId.trim());
      if (contextId.trim()) params.set("context_id", contextId.trim());
      if (since) params.set("since", since);
      if (until) params.set("until", until);
      const payload = await requestJson<AdminBadgeAuditResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/badges/audit?${params.toString()}`);
      if (!reportRequest.isCurrent(generation)) return;
      setAuditReport(payload.report || {});
      setMessage("Badge Audit complete.");
    } catch (error) { if (reportRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to run Badge Audit."); }
    finally { if (reportRequest.isCurrent(generation)) setBusy(false); }
  }

  async function runRecompute(confirmationText = "") {
    const generation = actionRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const key = recomputeMode === "dry-run" ? "" : (recomputeOperationKey || `badge-recompute:${Date.now()}:${crypto.randomUUID()}`);
      if (recomputeMode !== "dry-run" && !recomputeOperationKey) setRecomputeOperationKey(key);
      const payload = await requestJson<RepairResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/badges/recompute`, {
        method: "POST",
        body: JSON.stringify({
          mode: recomputeMode,
          player_id: playerId ? Number(playerId) : null,
          badge_id: badgeId || null,
          league_id: leagueId || null,
          context_id: contextId || null,
          since: since || null,
          until: until || null,
          include_non_live: includeNonLive,
          match_limit: Number(matchLimit || 5000),
          revoke_reason: revokeReason || null,
          confirmation_text: confirmationText,
          operation_key: key
        })
      });
      if (!actionRequest.isCurrent(generation)) return;
      setRepairResult(payload);
      if (recomputeMode !== "dry-run") setRecomputeOperationKey("");
      setMessage(payload.read_only ? "Read-only badge recompute preview complete; no rows were written." : `Badge recompute ${payload.recompute_mode || recomputeMode} complete.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to run badge recompute.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function revokeBadge(confirmationText: string) {
    const generation = actionRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const key = revokeOperationKey || `badge-revoke:${Date.now()}:${crypto.randomUUID()}`;
      if (!revokeOperationKey) setRevokeOperationKey(key);
      const payload = await requestJson<RepairResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/badges/revoke`, {
        method: "PATCH",
        body: JSON.stringify({
          player_badge_id: playerBadgeId || null,
          player_id: playerId ? Number(playerId) : null,
          badge_id: badgeId || null,
          context_id: contextId || null,
          revoke_reason: revokeReason || null,
          confirmation_text: confirmationText,
          operation_key: key
        })
      });
      if (!actionRequest.isCurrent(generation)) return;
      setRepairResult(payload); setRevokeOperationKey(""); setMessage(`Revoked ${payload.revoked_count || 0} badge row(s).`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to revoke badge row.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function updateBadgeState(confirmationText: string) {
    if (!selectedBadge || !selectedBadge.definition_found) { setMessage("Choose a badge definition loaded from the staging badges table."); return; }
    if (!badgeStateReason.trim()) { setMessage("Enter a reason for the badge state change."); return; }
    if (selectedBadge.state === badgeStateTarget) { setMessage(`Badge state is already ${badgeStateTarget}.`); return; }
    const generation = actionRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const key = stateOperationKey || `badge-state:${Date.now()}:${crypto.randomUUID()}`;
      if (!stateOperationKey) setStateOperationKey(key);
      const payload = await requestJson<BadgeStateUpdateResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/badges/${encodeURIComponent(selectedBadge.badge_id)}/state`, {
        method: "PATCH",
        body: JSON.stringify({
          expected_state: selectedBadge.state || "live",
          target_state: badgeStateTarget,
          reason: badgeStateReason,
          force: badgeStateForce,
          confirmation_text: confirmationText,
          operation_key: key,
          source: "next_badge_definition_state"
        })
      });
      if (!actionRequest.isCurrent(generation)) return;
      setBadges((current) => current.map((badge) => badge.badge_id === payload.badge.badge_id ? { ...badge, ...payload.badge, definition_found: true } : badge));
      setBadgeStateReason("");
      setBadgeStateForce(false);
      setStateOperationKey("");
      setMessage(payload.audit_warning ? `Badge state updated with audit warning: ${payload.audit_warning}` : `Badge state updated to ${payload.badge.state}.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to update badge definition state.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function inspectBadgeOperation() {
    if (!operationLookupKey.trim()) { setMessage("Enter the exact badge operation key first."); return; }
    const generation = actionRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<BadgeOperationResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/badges/operations/${encodeURIComponent(operationLookupKey.trim())}`);
      if (!actionRequest.isCurrent(generation)) return;
      setOperationLookup(payload);
      setMessage(`Badge operation ${payload.operation_key} is ${payload.status}.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) { setOperationLookup(null); setMessage(error instanceof Error ? error.message : "Unable to inspect badge operation."); }
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadOptions);

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Badge Diagnostics is disabled</h2>
        <p style={{ color: "#475569" }}>Enable <code>JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS</code> on FastAPI before using this workflow.</p>
        {status.warnings?.map((warning) => <p key={warning} style={{ color: "#92400e" }}>{warning}</p>)}
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
        <h2 style={{ marginTop: 0 }}>1. Scope</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Player<br /><select value={playerId} onChange={(event) => setPlayerId(event.target.value)} style={inputStyle}><option value="">All players for audit</option>{players.map((player) => <option key={player.id} value={player.id}>{player.name} #{player.id}</option>)}</select></label>
          <label>Badge<br /><select value={badgeId} onChange={(event) => setBadgeId(event.target.value)} style={inputStyle}><option value="">All badges for audit</option>{badges.map((badge) => <option key={badge.badge_id} value={badge.badge_id}>{badge.name} · {badge.badge_id}</option>)}</select></label>
          <label>League<br /><input value={leagueId} onChange={(event) => setLeagueId(event.target.value)} placeholder="Optional" style={inputStyle} /></label>
          <label>Context ID<br /><input value={contextId} onChange={(event) => setContextId(event.target.value)} placeholder="Optional exact badge context" style={inputStyle} /></label>
          <label>Match limit<br /><input value={matchLimit} onChange={(event) => setMatchLimit(event.target.value)} style={inputStyle} /></label>
          <button type="button" onClick={loadOptions} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Refreshing…" : "Refresh options"}</button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" }}>
          <label>Since<br /><input type="date" value={since} onChange={(event) => setSince(event.target.value)} style={inputStyle} /></label>
          <label>Until<br /><input type="date" value={until} onChange={(event) => setUntil(event.target.value)} style={inputStyle} /></label>
          <label><input type="checkbox" checked={includeNonLive} onChange={(event) => setIncludeNonLive(event.target.checked)} /> Include non-live badge rules</label>
          <label><input type="checkbox" checked={includeRevoked} onChange={(event) => setIncludeRevoked(event.target.checked)} /> Include revoked badge rows</label>
        </div>
        {message ? <p role="status" aria-live="polite" style={{ color: messageColor(message) }}>{message}</p> : null}
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>2. Badge Debug</h2>
        <p style={{ color: "#475569" }}>Use this for one player and one badge. It shows evaluator candidates, filter audit steps, and badge-specific diagnostics.</p>
        <button type="button" onClick={runDebug} disabled={busy || !playerId || !badgeId} style={buttonStyle}>Run Badge Debug</button>
        {debugReport ? <ReportBlock report={debugReport} /> : null}
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>3. Badge Audit</h2>
        <p style={{ color: "#475569" }}>Expected-vs-actual badge audit. Use the scope above to narrow to a player, badge, league, context, or date range.</p>
        <button type="button" onClick={runAudit} disabled={busy} style={buttonStyle}>Run Badge Audit</button>
        {auditReport ? (
          <div style={{ marginTop: "1rem", display: "grid", gap: "0.75rem" }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
              <article style={cardStyle}><strong>Expected</strong><br />{countValue(auditReport, "expected_exact_count")}</article>
              <article style={cardStyle}><strong>Active actual</strong><br />{countValue(auditReport, "actual_active_exact_count")}</article>
              <article style={cardStyle}><strong>Missing</strong><br />{countValue(auditReport, "missing_exact_count")}</article>
              <article style={cardStyle}><strong>Stale</strong><br />{countValue(auditReport, "stale_exact_count")}</article>
              <article style={cardStyle}><strong>Duplicates</strong><br />{countValue(auditReport, "duplicate_count")}</article>
              <article style={cardStyle}><strong>Context drift</strong><br />{countValue(auditReport, "context_drift_soft_key_count")}</article>
            </div>
            {reportRows(auditReport, "per_badge_summary").length ? <Table title="Per-badge summary" rows={reportRows(auditReport, "per_badge_summary").slice(0, 20)} /> : null}
            {reportRows(auditReport, "missing_rows").length ? <Table title="Missing rows" rows={reportRows(auditReport, "missing_rows").slice(0, 25)} /> : null}
            {reportRows(auditReport, "stale_rows").length ? <Table title="Stale rows" rows={reportRows(auditReport, "stale_rows").slice(0, 25)} /> : null}
            {reportRows(auditReport, "duplicate_rows").length ? <Table title="Duplicate rows" rows={reportRows(auditReport, "duplicate_rows").slice(0, 25)} /> : null}
          </div>
        ) : null}
      </article>

      <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
        <h2 style={{ marginTop: 0 }}>4. Badge Definition State</h2>
        <p style={{ color: "#475569" }}>Control whether the selected badge definition can continue awarding. Normal transitions are <code>live → frozen → deprecated</code>. This staging-only definition change requires <code>run_replay</code>, a reason, current-state locking, durable retry key, strict audit intent, and an exact confirmation.</p>
        {!selectedBadge ? <p style={{ color: "#64748b" }}>{busy ? "Loading badge options…" : "Select a badge to manage its state."}</p> : <>
          <p><strong>{selectedBadge.name}</strong> · <code>{selectedBadge.badge_id}</code><br />Current state: <strong>{selectedBadge.state || "live"}</strong>{selectedBadge.state_changed_at ? ` · changed ${selectedBadge.state_changed_at}` : ""}</p>
          {selectedBadge.state_change_reason ? <p><strong>Previous reason:</strong> {selectedBadge.state_change_reason}</p> : null}
          {!selectedBadge.definition_found ? <p style={{ color: "#92400e" }}>This badge exists in the code registry but not in the staging <code>badges</code> table, so its state cannot be changed here.</p> : <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
            <label>Target state<br /><select value={badgeStateTarget} onChange={(event) => setBadgeStateTarget(event.target.value as "live" | "frozen" | "deprecated")} style={inputStyle}><option value="live">live</option><option value="frozen">frozen</option><option value="deprecated">deprecated</option></select></label>
            <label>Reason<br /><input value={badgeStateReason} onChange={(event) => setBadgeStateReason(event.target.value)} maxLength={500} style={inputStyle} /></label>
            <label style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center" }}><input type="checkbox" checked={badgeStateForce} onChange={(event) => setBadgeStateForce(event.target.checked)} /> Force nonstandard transition</label>
            <ConfirmAction
              triggerLabel="Update badge state"
              title={`Update ${selectedBadge.name} to ${badgeStateTarget}?`}
              description={badgeStateForce ? "This uses a forced nonstandard transition. The reason and current-state lock will be recorded." : "This changes whether the selected badge definition can award new badges."}
              confirmLabel="Yes, update badge state"
              confirmationText={status.confirmation_text?.state || "UPDATE BADGE STATE"}
              tone={badgeStateTarget === "deprecated" || badgeStateForce ? "danger" : "default"}
              disabled={busy || !badgeStateReason.trim() || selectedBadge.state === badgeStateTarget}
              busy={busy}
              onConfirm={updateBadgeState}
            />
          </div>}
          {badgeStateForce ? <p style={{ color: "#991b1b" }}>Force override permits a transition outside the normal lifecycle. Re-review the badge, target state, and reason before confirming.</p> : null}
        </>}
      </article>

      <article style={{ ...cardStyle, background: "#fff7ed", borderColor: "#fed7aa" }}>
        <h2 style={{ marginTop: 0 }}>5. Badge Repair / Recompute</h2>
        <p style={{ color: "#9a3412" }}>Super-admin repair controls for staging validation. Dry-run previews do not mutate badges; append-only and strict modes write through the Python badge recompute path. Revoke marks matched badge rows as revoked.</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Recompute mode<br /><select value={recomputeMode} onChange={(event) => { setRecomputeMode(event.target.value); setRecomputeOperationKey(""); }} style={inputStyle}><option value="dry-run">dry-run</option><option value="append-only">append-only</option><option value="strict">strict</option></select></label>
          {recomputeMode === "dry-run" ? <button type="button" onClick={() => void runRecompute()} disabled={busy} style={buttonStyle}>Run recompute preview</button> : <ConfirmAction
            triggerLabel="Run applying recompute"
            title={`Run ${recomputeMode} badge recompute?`}
            description="This applies badge changes for the selected scope through the guarded recompute service."
            confirmLabel="Yes, recompute badges"
            confirmationText="RECOMPUTE BADGES"
            tone={recomputeMode === "strict" ? "danger" : "default"}
            disabled={busy}
            busy={busy}
            onConfirm={runRecompute}
          />}
        </div>
        {recomputeOperationKey || revokeOperationKey || stateOperationKey ? <div style={{ color: "#92400e" }}><p><strong>Uncertain-operation guard:</strong> do not change inputs or retry with a new key. Inspect Badge Audit and operation status first.</p>{[recomputeOperationKey, revokeOperationKey, stateOperationKey].filter(Boolean).map((key) => <code key={key} style={{ display: "block", overflowWrap: "anywhere" }}>{key}</code>)}</div> : null}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", alignItems: "end", marginTop: "1rem" }}>
          <label>Player badge row ID optional<br /><input value={playerBadgeId} onChange={(event) => setPlayerBadgeId(event.target.value)} placeholder="Use exact row id when known" style={inputStyle} /></label>
          <label>Revoke reason<br /><input value={revokeReason} onChange={(event) => setRevokeReason(event.target.value)} placeholder="Required for operator clarity" style={inputStyle} /></label>
          <ConfirmAction
            triggerLabel="Revoke matched badge row(s)"
            title="Revoke the matched badge rows?"
            description="This marks the selected badge rows as revoked and records the guarded operation."
            confirmLabel="Yes, revoke badges"
            confirmationText="REVOKE BADGE"
            tone="danger"
            disabled={busy || !revokeReason.trim()}
            busy={busy}
            onConfirm={revokeBadge}
          />
        </div>
        {repairResult ? <pre style={{ whiteSpace: "pre-wrap", background: "#0f172a", color: "white", padding: "1rem", borderRadius: "12px", overflowX: "auto" }}>{JSON.stringify(repairResult, null, 2)}</pre> : null}
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>6. Guarded operation recovery</h2>
        <p style={{ color: "#475569" }}>If an applying response is interrupted, preserve the exact key shown above and inspect it here. Completed keys return their saved result; incomplete keys never rerun the write.</p>
        <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) auto", gap: "0.75rem", alignItems: "end" }}>
          <label>Exact operation key<br /><input value={operationLookupKey} onChange={(event) => { setOperationLookupKey(event.target.value); setOperationLookup(null); }} style={inputStyle} /></label>
          <button type="button" onClick={inspectBadgeOperation} disabled={busy || !accessToken} style={ghostButtonStyle}>Inspect badge operation</button>
        </div>
        {operationLookup ? <div style={{ marginTop: "1rem" }}>
          <pre style={{ whiteSpace: "pre-wrap", background: "#0f172a", color: "white", padding: "1rem", borderRadius: "12px", overflowX: "auto" }}>{JSON.stringify(operationLookup, null, 2)}</pre>
          {operationLookup.status === "recovery_required" || operationLookup.status === "intent_recorded" || operationLookup.status === "running" ? <p style={{ color: "#991b1b" }}><strong>Stop further badge writes.</strong> Run Badge Audit and inspect the relevant definitions/rows or eval run before any retry.</p> : null}
        </div> : null}
        <p><a href="/admin/badges">Badge Audit</a> · <a href="/admin/replay-history">Replay History</a> · <a href="/admin/guide">Admin Guide</a></p>
        <p style={{ color: "#475569" }}>If recovery cannot prove the exact outcome, keep Streamlit Badge Debug/Audit as the fallback and carry the operation key into the incident note.</p>
      </article>
    </div>
  );
}

function ReportBlock({ report }: { report: Record<string, unknown> }) {
  return (
    <div style={{ marginTop: "1rem", display: "grid", gap: "0.75rem" }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
        <article style={cardStyle}><strong>Raw matches</strong><br />{Array.isArray(report.matches_raw) ? report.matches_raw.length : "—"}</article>
        <article style={cardStyle}><strong>Filtered matches</strong><br />{Array.isArray(report.matches_filtered) ? report.matches_filtered.length : "—"}</article>
        <article style={cardStyle}><strong>Candidates</strong><br />{Array.isArray(report.candidates) ? report.candidates.length : "—"}</article>
        <article style={cardStyle}><strong>Errors</strong><br />{Array.isArray(report.errors) ? report.errors.length : "—"}</article>
      </div>
      {reportRows(report, "candidates").length ? <Table title="Evaluator candidates" rows={reportRows(report, "candidates").slice(0, 10)} /> : null}
      {reportRows(report, "filter_audit_steps").length ? <Table title="Filter audit steps" rows={reportRows(report, "filter_audit_steps").slice(0, 10)} /> : null}
      <details><summary>Diagnostics detail</summary><pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(report.diagnostics || {}, null, 2)}</pre></details>
    </div>
  );
}

function Table({ title, rows }: { title: string; rows: Array<Record<string, unknown>> }) {
  const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row)))).slice(0, 8);
  return (
    <section>
      <h3>{title}</h3>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
          <thead><tr>{columns.map((column) => <th key={column} align="left" style={{ borderBottom: "1px solid #e2e8f0", padding: "0.4rem" }}>{column}</th>)}</tr></thead>
          <tbody>{rows.map((row, idx) => <tr key={idx}>{columns.map((column) => <td key={column} style={{ borderBottom: "1px solid #f1f5f9", padding: "0.4rem", verticalAlign: "top" }}>{shortValue(row[column])}</td>)}</tr>)}</tbody>
        </table>
      </div>
    </section>
  );
}
