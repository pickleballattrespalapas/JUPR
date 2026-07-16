"use client";

import { useState } from "react";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type StatusResponse = { enabled: boolean; status: string; confirmation_text?: string };
type PlayerOption = { player_id: number; player_name: string };
type OptionsResponse = { ok: boolean; players: PlayerOption[]; leagues: string[]; schema_degraded?: boolean; schema_degraded_reason?: string | null };
type AuditReport = {
  counts?: Record<string, number>;
  scope?: Record<string, unknown>;
  excluded_only_in_profile?: Array<Record<string, unknown>>;
  only_in_canonical?: string[];
  shared_ids?: string[];
  exclusion_reasons_summary?: Array<Record<string, unknown>>;
};
type AuditResponse = { ok: boolean; report: AuditReport; schema_degraded?: boolean; schema_degraded_reason?: string | null };
type NormalizeResponse = { ok: boolean; result: Record<string, unknown> };

type Props = { apiBase: string | null; clubId: string; status: StatusResponse | null };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
async function apiError(response: Response): Promise<string> {
  const text = await response.text().catch(() => "");
  if (!text) return `API error (${response.status}).`;
  try { return String((JSON.parse(text) as { detail?: unknown }).detail || text); } catch { return text.slice(0, 240); }
}
function smallTable(rows: Array<Record<string, unknown>>, keys: string[]) {
  if (!rows.length) return <p style={{ color: "#64748b" }}>No rows.</p>;
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
        <thead><tr>{keys.map((key) => <th key={key} align="left" style={{ borderBottom: "1px solid #e2e8f0", padding: "0.4rem" }}>{key}</th>)}</tr></thead>
        <tbody>{rows.slice(0, 100).map((row, idx) => <tr key={idx}>{keys.map((key) => <td key={key} style={{ borderBottom: "1px solid #f1f5f9", padding: "0.4rem", verticalAlign: "top" }}>{Array.isArray(row[key]) ? (row[key] as unknown[]).join(", ") : typeof row[key] === "object" && row[key] !== null ? JSON.stringify(row[key]) : String(row[key] ?? "")}</td>)}</tr>)}</tbody>
      </table>
      {rows.length > 100 ? <p style={{ color: "#64748b" }}>Showing first 100 of {rows.length} rows.</p> : null}
    </div>
  );
}

export default function MatchCanonicalAuditPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [playerId, setPlayerId] = useState("");
  const [leagueId, setLeagueId] = useState("");
  const [limit, setLimit] = useState("1200");
  const [report, setReport] = useState<AuditReport | null>(null);
  const [matchIds, setMatchIds] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [normalizeResult, setNormalizeResult] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("Missing JUPR API base URL.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Match Canonical Audit.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    if (!response.ok) throw new Error(await apiError(response));
    return (await response.json()) as T;
  }

  async function loadOptions() {
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<OptionsResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/match-canonical-audit/options`);
      setOptions(payload);
      if (!playerId && payload.players?.length) setPlayerId(String(payload.players[0].player_id));
      setMessage(`Loaded ${payload.players?.length || 0} player option(s).`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to load options."); }
    finally { setBusy(false); }
  }

  async function runAudit() {
    if (!playerId) { setMessage("Select a player first."); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<AuditResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/match-canonical-audit/run`, {
        method: "POST",
        body: JSON.stringify({ player_id: Number(playerId), league_id: leagueId || null, limit: Number(limit) || 1200 })
      });
      setReport(payload.report);
      setNormalizeResult(null);
      const counts = payload.report?.counts || {};
      setMessage(`Audit complete. Only in profile: ${counts.only_in_profile ?? 0}; only in canonical: ${counts.only_in_canonical ?? 0}.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to run audit."); }
    finally { setBusy(false); }
  }

  async function normalize(dryRun: boolean) {
    if (!playerId) { setMessage("Select a player first."); return; }
    const ids = matchIds.split(/[\s,]+/).map((value) => Number(value.trim())).filter((value) => Number.isInteger(value) && value > 0);
    if (!dryRun && confirmation.trim().toUpperCase() !== "APPLY NORMALIZE") { setMessage("Type APPLY NORMALIZE to apply canonical normalization updates."); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<NormalizeResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/match-canonical-audit/normalize`, {
        method: "POST",
        body: JSON.stringify({ player_id: Number(playerId), match_ids: ids, dry_run: dryRun, confirmation_text: confirmation, source: dryRun ? "next_match_canonical_audit_dry_run" : "next_match_canonical_audit_apply" })
      });
      setNormalizeResult(payload.result || {});
      setMessage(dryRun ? "Dry-run normalization plan generated." : "Normalization updates applied. Re-run this audit, then use Badge Diagnostics and Replay History if needed.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to normalize rows."); }
    finally { setBusy(false); }
  }

  if (status && !status.enabled) {
    return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Disabled</h2><p>Set <code>JUPR_ENABLE_NEXT_ADMIN_MATCH_CANONICAL_AUDIT=1</code> on FastAPI to enable this guarded workflow.</p></article>;
  }

  const counts = report?.counts || {};
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
        <p style={{ color: "#475569" }}>Run this for one player at a time. Applying normalization is explicit and audit-flagged.</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <button type="button" onClick={loadOptions} disabled={busy || !accessToken} style={ghostButtonStyle}>Load players/leagues</button>
          <label>Player<br /><select value={playerId} onChange={(event) => setPlayerId(event.target.value)} style={inputStyle}>{(options?.players || []).map((player) => <option key={player.player_id} value={player.player_id}>{player.player_name} · #{player.player_id}</option>)}</select></label>
          <label>League<br /><select value={leagueId} onChange={(event) => setLeagueId(event.target.value)} style={inputStyle}><option value="">All leagues</option>{(options?.leagues || []).map((league) => <option key={league} value={league}>{league}</option>)}</select></label>
          <label>Limit<br /><input value={limit} onChange={(event) => setLimit(event.target.value)} style={inputStyle} /></label>
          <button type="button" onClick={runAudit} disabled={busy || !playerId} style={buttonStyle}>Run audit</button>
        </div>
        {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("type") || message.toLowerCase().includes("missing") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>

      {report ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>2. Audit results</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem" }}>
          <div><strong>Profile-visible</strong><br />{counts.profile_visible ?? 0}</div>
          <div><strong>Canonical-visible</strong><br />{counts.canonical_visible ?? 0}</div>
          <div><strong>Only in Profile</strong><br />{counts.only_in_profile ?? 0}</div>
          <div><strong>Only in Canonical</strong><br />{counts.only_in_canonical ?? 0}</div>
          <div><strong>Shared</strong><br />{counts.shared ?? 0}</div>
        </div>
        <h3>Exclusion reasons</h3>
        {smallTable((report.exclusion_reasons_summary || []) as Array<Record<string, unknown>>, ["reason", "count"])}
        <h3>Only in profile</h3>
        {smallTable((report.excluded_only_in_profile || []) as Array<Record<string, unknown>>, ["match_id", "date", "league", "match_type", "context_type", "score_t1", "score_t2", "exclusion_reasons", "suggested_normalization_action"])}
      </article> : null}

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>3. Normalize legacy rows</h2>
        <p style={{ color: "#475569" }}>Leave match IDs blank to target the audit's candidate set, or paste IDs separated by commas/spaces.</p>
        <label>Match IDs<br /><textarea value={matchIds} onChange={(event) => setMatchIds(event.target.value)} rows={3} style={inputStyle} /></label>
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto auto", gap: "0.75rem", alignItems: "end", marginTop: "0.75rem" }}>
          <label>Apply confirmation<br /><input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} placeholder="APPLY NORMALIZE" style={inputStyle} /></label>
          <button type="button" onClick={() => normalize(true)} disabled={busy || !playerId} style={ghostButtonStyle}>Dry run normalize</button>
          <button type="button" onClick={() => normalize(false)} disabled={busy || !playerId} style={buttonStyle}>Apply normalize</button>
        </div>
        {normalizeResult ? <pre style={{ whiteSpace: "pre-wrap", background: "#0f172a", color: "white", padding: "1rem", borderRadius: "12px", overflowX: "auto" }}>{JSON.stringify(normalizeResult, null, 2)}</pre> : null}
      </article>
    </div>
  );
}
