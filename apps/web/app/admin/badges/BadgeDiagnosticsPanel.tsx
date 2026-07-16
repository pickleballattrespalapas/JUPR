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
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = { apiBase: string | null; clubId: string; status: AdminBadgeDiagnosticsStatusResponse };
type RepairResponse = { ok: boolean; mode?: string; recompute_mode?: string; summary?: Record<string, unknown>; revoked_count?: number; rows?: Array<Record<string, unknown>>; audit_warning?: string | null };

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
  if (text.includes("unable") || text.includes("error") || text.includes("disabled") || text.includes("missing") || text.includes("type ")) return "#b91c1c";
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
  const [recomputeConfirm, setRecomputeConfirm] = useState("");
  const [playerBadgeId, setPlayerBadgeId] = useState("");
  const [revokeReason, setRevokeReason] = useState("");
  const [revokeConfirm, setRevokeConfirm] = useState("");
  const [debugReport, setDebugReport] = useState<Record<string, unknown> | null>(null);
  const [auditReport, setAuditReport] = useState<Record<string, unknown> | null>(null);
  const [repairResult, setRepairResult] = useState<RepairResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const selectedPlayerName = useMemo(() => players.find((player) => String(player.id) === String(playerId))?.name || "", [players, playerId]);
  const selectedBadgeName = useMemo(() => badges.find((badge) => String(badge.badge_id) === String(badgeId))?.name || badgeId, [badges, badgeId]);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Badge Diagnostics.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  async function loadOptions() {
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<AdminBadgeOptionsResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/badges/options`);
      setPlayers(payload.players || []);
      setBadges(payload.badges || []);
      if (!playerId && payload.players?.length) setPlayerId(String(payload.players[0].id));
      if (!badgeId && payload.badges?.length) setBadgeId(payload.badges[0].badge_id);
      setMessage(`Loaded ${payload.player_count} player option(s) and ${payload.badge_count} badge option(s).`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to load badge diagnostic options."); }
    finally { setBusy(false); }
  }

  async function runDebug() {
    if (!playerId || !badgeId) { setMessage("Choose a player and badge before running Badge Debug."); return; }
    setBusy(true); setMessage(null);
    try {
      const params = new URLSearchParams({ player_id: String(playerId), badge_id: badgeId, match_limit: String(matchLimit || 5000) });
      if (leagueId.trim()) params.set("league_id", leagueId.trim());
      const payload = await requestJson<AdminBadgeDebugResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/badges/debug?${params.toString()}`);
      setDebugReport(payload.report || {});
      setMessage(`Badge Debug complete for ${selectedPlayerName || playerId} / ${selectedBadgeName}.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to run Badge Debug."); }
    finally { setBusy(false); }
  }

  async function runAudit() {
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
      setAuditReport(payload.report || {});
      setMessage("Badge Audit complete.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to run Badge Audit."); }
    finally { setBusy(false); }
  }

  async function runRecompute() {
    if (recomputeConfirm.trim().toUpperCase() !== "RECOMPUTE BADGES") { setMessage("Type RECOMPUTE BADGES to run badge recompute."); return; }
    setBusy(true); setMessage(null);
    try {
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
          confirmation_text: recomputeConfirm
        })
      });
      setRepairResult(payload); setRecomputeConfirm(""); setMessage(`Badge recompute ${payload.recompute_mode || recomputeMode} complete.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to run badge recompute."); }
    finally { setBusy(false); }
  }

  async function revokeBadge() {
    if (revokeConfirm.trim().toUpperCase() !== "REVOKE BADGE") { setMessage("Type REVOKE BADGE to revoke badge rows."); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<RepairResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/badges/revoke`, {
        method: "PATCH",
        body: JSON.stringify({
          player_badge_id: playerBadgeId || null,
          player_id: playerId ? Number(playerId) : null,
          badge_id: badgeId || null,
          context_id: contextId || null,
          revoke_reason: revokeReason || null,
          confirmation_text: revokeConfirm
        })
      });
      setRepairResult(payload); setRevokeConfirm(""); setMessage(`Revoked ${payload.revoked_count || 0} badge row(s).`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to revoke badge row."); }
    finally { setBusy(false); }
  }

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
          <button type="button" onClick={loadOptions} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Working…" : "Load options"}</button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" }}>
          <label>Since<br /><input type="date" value={since} onChange={(event) => setSince(event.target.value)} style={inputStyle} /></label>
          <label>Until<br /><input type="date" value={until} onChange={(event) => setUntil(event.target.value)} style={inputStyle} /></label>
          <label><input type="checkbox" checked={includeNonLive} onChange={(event) => setIncludeNonLive(event.target.checked)} /> Include non-live badge rules</label>
          <label><input type="checkbox" checked={includeRevoked} onChange={(event) => setIncludeRevoked(event.target.checked)} /> Include revoked badge rows</label>
        </div>
        {message ? <p style={{ color: messageColor(message) }}>{message}</p> : null}
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

      <article style={{ ...cardStyle, background: "#fff7ed", borderColor: "#fed7aa" }}>
        <h2 style={{ marginTop: 0 }}>4. Badge Repair / Recompute</h2>
        <p style={{ color: "#9a3412" }}>Super-admin repair controls for staging validation. Dry-run previews do not mutate badges; append-only and strict modes write through the Python badge recompute path. Revoke marks matched badge rows as revoked.</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Recompute mode<br /><select value={recomputeMode} onChange={(event) => setRecomputeMode(event.target.value)} style={inputStyle}><option value="dry-run">dry-run</option><option value="append-only">append-only</option><option value="strict">strict</option></select></label>
          <label>Recompute confirmation<br /><input value={recomputeConfirm} onChange={(event) => setRecomputeConfirm(event.target.value)} placeholder="RECOMPUTE BADGES" style={inputStyle} /></label>
          <button type="button" onClick={runRecompute} disabled={busy} style={buttonStyle}>Run recompute</button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", alignItems: "end", marginTop: "1rem" }}>
          <label>Player badge row ID optional<br /><input value={playerBadgeId} onChange={(event) => setPlayerBadgeId(event.target.value)} placeholder="Use exact row id when known" style={inputStyle} /></label>
          <label>Revoke reason<br /><input value={revokeReason} onChange={(event) => setRevokeReason(event.target.value)} placeholder="Required for operator clarity" style={inputStyle} /></label>
          <label>Revoke confirmation<br /><input value={revokeConfirm} onChange={(event) => setRevokeConfirm(event.target.value)} placeholder="REVOKE BADGE" style={inputStyle} /></label>
          <button type="button" onClick={revokeBadge} disabled={busy} style={ghostButtonStyle}>Revoke matched badge row(s)</button>
        </div>
        {repairResult ? <pre style={{ whiteSpace: "pre-wrap", background: "#0f172a", color: "white", padding: "1rem", borderRadius: "12px", overflowX: "auto" }}>{JSON.stringify(repairResult, null, 2)}</pre> : null}
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
