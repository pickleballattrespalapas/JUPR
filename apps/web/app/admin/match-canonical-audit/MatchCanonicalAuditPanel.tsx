"use client";

import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, type ActionCompletion } from "@/components/interaction";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type StatusResponse = { enabled: boolean; status: string; confirmation_text?: string; write_environment?: string; recovery_routes?: Record<string, string> };
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
type NormalizeResponse = { ok: boolean; result?: Record<string, unknown>; preview_fingerprint?: string; operation_key?: string; updated_count?: number; recovery?: Record<string, string>; readback_verified?: boolean };
type OperationResponse = { ok: boolean; operation_key: string; status: string; result: Record<string, unknown>; error?: string | null; recovery?: Record<string, string> };

type Props = { apiBase: string | null; clubId: string; status: StatusResponse | null };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
async function apiError(response: Response): Promise<string> {
  const text = await response.text().catch(() => "");
  if (!text) return `API error (${response.status}).`;
  try {
    const detail = (JSON.parse(text) as { detail?: unknown }).detail;
    if (detail && typeof detail === "object" && "message" in detail) return String((detail as { message?: unknown }).message || text);
    return String(detail || text);
  } catch { return text.slice(0, 240); }
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
  const [previewFingerprint, setPreviewFingerprint] = useState("");
  const [operationKey, setOperationKey] = useState("");
  const [operationLookupKey, setOperationLookupKey] = useState("");
  const [operationLookup, setOperationLookup] = useState<OperationResponse | null>(null);
  const [normalizeResult, setNormalizeResult] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const optionsRequest = useLatestRequestGuard(accessToken, clearProtectedAuditState);
  const auditRequest = useLatestRequestGuard(accessToken);
  const actionRequest = useLatestRequestGuard(accessToken);

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

  function clearProtectedAuditState() {
    auditRequest.invalidate();
    setBusy(false); setMessage(null);
    setOptions(null); setPlayerId(""); setLeagueId(""); setReport(null); setMatchIds("");
    setPreviewFingerprint(""); setOperationKey(""); setOperationLookupKey("");
    setNormalizeResult(null); setOperationLookup(null);
  }

  async function loadOptions() {
    const generation = optionsRequest.begin();
    auditRequest.invalidate();
    setBusy(true); setMessage(null);
    setReport(null); setMatchIds(""); setPreviewFingerprint(""); setNormalizeResult(null);
    try {
      const payload = await requestJson<OptionsResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/match-canonical-audit/options`);
      if (!optionsRequest.isCurrent(generation)) return;
      setOptions(payload);
      setPlayerId(payload.players?.some((row) => String(row.player_id) === playerId) ? playerId : String(payload.players?.[0]?.player_id || ""));
      setLeagueId(payload.leagues?.includes(leagueId) ? leagueId : "");
      setMessage(`Loaded ${payload.players?.length || 0} player option(s).`);
    } catch (error) { if (optionsRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load options."); }
    finally { if (optionsRequest.isCurrent(generation)) setBusy(false); }
  }

  async function runAudit() {
    if (!playerId) { setMessage("Select a player first."); return; }
    const generation = auditRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<AuditResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/match-canonical-audit/run`, {
        method: "POST",
        body: JSON.stringify({ player_id: Number(playerId), league_id: leagueId || null, limit: Number(limit) || 1200 })
      });
      if (!auditRequest.isCurrent(generation)) return;
      setReport(payload.report);
      setNormalizeResult(null);
      const counts = payload.report?.counts || {};
      setMessage(`Audit complete. Only in profile: ${counts.only_in_profile ?? 0}; only in canonical: ${counts.only_in_canonical ?? 0}.`);
    } catch (error) { if (auditRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to run audit."); }
    finally { if (auditRequest.isCurrent(generation)) setBusy(false); }
  }

  async function normalize(dryRun: boolean, confirmationText = ""): Promise<ActionCompletion> {
    if (!playerId) { const error = new Error("Select a player first."); setMessage(error.message); throw error; }
    const ids = matchIds.split(/[\s,]+/).map((value) => Number(value.trim())).filter((value) => Number.isInteger(value) && value > 0);
    if (!dryRun && !ids.length) { const error = new Error("Select the exact IDs from the current dry run before applying."); setMessage(error.message); throw error; }
    if (!dryRun && !previewFingerprint) { const error = new Error("Run and review a current dry run before applying."); setMessage(error.message); throw error; }
    const generation = actionRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const key = dryRun ? "" : (operationKey || `canonical:${Date.now()}:${crypto.randomUUID()}`);
      if (!dryRun && !operationKey) setOperationKey(key);
      const payload = await requestJson<NormalizeResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/match-canonical-audit/normalize`, {
        method: "POST",
        body: JSON.stringify({
          player_id: Number(playerId),
          match_ids: ids,
          dry_run: dryRun,
          confirmation_text: confirmationText,
          preview_fingerprint: dryRun ? "" : previewFingerprint,
          operation_key: key,
          source: dryRun ? "next_match_canonical_audit_dry_run" : "next_match_canonical_audit_apply"
        })
      });
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the normalization response was applied.");
      const result: Record<string, unknown> = payload.result || (payload as unknown as Record<string, unknown>);
      setNormalizeResult(result);
      if (dryRun) {
        const fingerprint = String(payload.preview_fingerprint || result.preview_fingerprint || "");
        const proposals = Array.isArray(result.proposals) ? result.proposals as Array<Record<string, unknown>> : [];
        const proposedIds = proposals.map((row) => Number(row.match_id)).filter((value) => Number.isInteger(value) && value > 0);
        setPreviewFingerprint(fingerprint);
        setMatchIds(proposedIds.join(", "));
        setOperationKey("");
        setMessage(`Read-only dry run prepared ${proposedIds.length} exact match ID(s). Review every patch before applying.`);
        return actionSuccess("Normalization preview ready", `The read-only dry run prepared ${proposedIds.length} exact match ID(s).`);
      } else {
        setPreviewFingerprint("");
        setOperationKey("");
        setMessage(`Atomic normalization updated ${payload.updated_count ?? ids.length} match(es) and verified readback. Re-run the audit.`);
        return actionSuccess("Canonical normalization complete", `Atomic normalization updated ${payload.updated_count ?? ids.length} match(es) and verified readback.`);
      }
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to normalize rows.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function inspectOperation() {
    if (!operationLookupKey.trim()) { setMessage("Enter the exact canonical operation key first."); return; }
    const generation = actionRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<OperationResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/match-canonical-audit/operations/${encodeURIComponent(operationLookupKey.trim())}`);
      if (!actionRequest.isCurrent(generation)) return;
      setOperationLookup(payload);
      setMessage(`Canonical operation ${payload.operation_key} is ${payload.status}.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) { setOperationLookup(null); setMessage(error instanceof Error ? error.message : "Unable to inspect canonical operation."); }
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(status?.enabled !== false ? accessToken : "", loadOptions);

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
          <button type="button" onClick={loadOptions} disabled={busy || !accessToken} style={ghostButtonStyle}>{busy ? "Refreshing…" : "Refresh players/leagues"}</button>
          <label>Player<br /><select value={playerId} onChange={(event) => { setPlayerId(event.target.value); setPreviewFingerprint(""); setMatchIds(""); }} style={inputStyle}>{(options?.players || []).map((player) => <option key={player.player_id} value={player.player_id}>{player.player_name} · #{player.player_id}</option>)}</select></label>
          <label>League<br /><select value={leagueId} onChange={(event) => setLeagueId(event.target.value)} style={inputStyle}><option value="">All leagues</option>{(options?.leagues || []).map((league) => <option key={league} value={league}>{league}</option>)}</select></label>
          <label>Limit<br /><input value={limit} onChange={(event) => setLimit(event.target.value)} style={inputStyle} /></label>
          <button type="button" onClick={runAudit} disabled={busy || !playerId} style={buttonStyle}>Run audit</button>
        </div>
        {message ? <p role="status" aria-live="polite" style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("type") || message.toLowerCase().includes("missing") || message.toLowerCase().includes("stop") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
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
        <p style={{ color: "#475569" }}>Run a read-only dry run first. Apply is bound to its fingerprint and exact proposed IDs, runs atomically through FastAPI/Supabase, and cannot silently replay after an uncertain response.</p>
        <label>Match IDs<br /><textarea value={matchIds} onChange={(event) => { setMatchIds(event.target.value); setPreviewFingerprint(""); }} rows={3} style={inputStyle} /></label>
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "end", flexWrap: "wrap", marginTop: "0.75rem" }}>
          <button type="button" onClick={() => void normalize(true).catch(() => undefined)} disabled={busy || !playerId} style={ghostButtonStyle}>Dry run normalize</button>
          <ConfirmAction
            triggerLabel="Apply exact reviewed plan"
            title="Apply this canonical normalization plan?"
            description={<>This will update the exact reviewed match IDs bound to preview fingerprint <code>{previewFingerprint || "not available"}</code>. Stop if the preview is stale or the IDs are not exact.</>}
            confirmLabel="Yes, apply reviewed plan"
            confirmationText="APPLY NORMALIZE"
            disabled={busy || !playerId || !previewFingerprint || !matchIds.trim()}
            busy={busy}
            onConfirm={(confirmationText) => normalize(false, confirmationText)}
          />
        </div>
        {previewFingerprint ? <p><strong>Current preview fingerprint:</strong> <code>{previewFingerprint}</code></p> : null}
        {operationKey ? <p style={{ color: "#92400e" }}><strong>Retry guard:</strong> keep operation key <code>{operationKey}</code>. If the response is uncertain, inspect status before retrying.</p> : null}
        <p style={{ color: "#475569" }}>Stop on any stale preview, count mismatch, or uncertain response. Recover through <a href="/admin/match-log">Match Log</a> and <a href="/admin/replay-history">Replay History</a>; Streamlit remains the fallback.</p>
        {normalizeResult ? <pre style={{ whiteSpace: "pre-wrap", background: "#0f172a", color: "white", padding: "1rem", borderRadius: "12px", overflowX: "auto" }}>{JSON.stringify(normalizeResult, null, 2)}</pre> : null}
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>4. Inspect or recover an uncertain apply</h2>
        <p style={{ color: "#475569" }}>If an apply response is interrupted, keep its operation key and inspect the server record. Never create a new key or rerun patches while status is incomplete.</p>
        {operationKey ? <p style={{ color: "#92400e" }}><strong>Retained key:</strong> <code style={{ overflowWrap: "anywhere" }}>{operationKey}</code></p> : null}
        <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) auto", gap: "0.75rem", alignItems: "end" }}>
          <label>Exact operation key<br /><input value={operationLookupKey} onChange={(event) => { setOperationLookupKey(event.target.value); setOperationLookup(null); }} style={inputStyle} /></label>
          <button type="button" onClick={inspectOperation} disabled={busy || !accessToken} style={ghostButtonStyle}>Inspect canonical operation</button>
        </div>
        {operationLookup ? <div style={{ marginTop: "1rem" }}>
          <pre style={{ whiteSpace: "pre-wrap", background: "#0f172a", color: "white", padding: "1rem", borderRadius: "12px", overflowX: "auto" }}>{JSON.stringify(operationLookup, null, 2)}</pre>
          {operationLookup.status !== "completed" ? <p style={{ color: "#991b1b" }}><strong>Stop further writes.</strong> Compare the exact IDs in Match Log, then use Replay History. Do not blindly retry this operation.</p> : <p style={{ color: "#166534" }}>The atomic operation completed. Verify its saved IDs and read models before continuing.</p>}
        </div> : null}
        <p><a href="/admin/match-log">Match Log</a> · <a href="/admin/replay-history">Replay History</a> · <a href="/admin/guide">Admin Guide</a></p>
      </article>
    </div>
  );
}
