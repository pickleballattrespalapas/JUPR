"use client";

import { useState } from "react";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type StatusResponse = { enabled: boolean; status: string; roles?: string[]; retention_days?: number; retention_cutoff?: string };
type OverviewResponse = { ok: boolean; roles: Array<Record<string, unknown>>; activity: Array<Record<string, unknown>>; activity_warning?: string | null; health: Record<string, unknown>; role_options: string[]; retention_days: number; retention_cutoff: string };
type RoleResponse = { ok: boolean; roles: Array<Record<string, unknown>>; audit_warning?: string | null };
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

export default function AdminToolsPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [flaggedOnly, setFlaggedOnly] = useState(false);
  const [targetEmail, setTargetEmail] = useState("");
  const [targetRole, setTargetRole] = useState("read_only");
  const [targetUserId, setTargetUserId] = useState("");
  const [confirmation, setConfirmation] = useState("");
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

  if (status && !status.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Disabled</h2><p>Set <code>JUPR_ENABLE_NEXT_ADMIN_TOOLS=1</code> on FastAPI to enable guarded Admin Tools.</p></article>;

  const roleOptions = overview?.role_options || status?.roles || ["read_only", "scorekeeper", "organizer", "club_owner", "super_admin"];
  return <div style={{ display: "grid", gap: "1rem" }}>
    <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Admin session</h2><p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>{sessionLoading ? <p>Checking session…</p> : null}{sessionMessage ? <p style={{ color: "#64748b" }}>{sessionMessage}</p> : null}</article>
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Overview</h2>
      <label style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center" }}><input type="checkbox" checked={flaggedOnly} onChange={(event) => setFlaggedOnly(event.target.checked)} /> Flagged activity only</label>
      <p><button type="button" onClick={loadOverview} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Loading…" : "Load Admin Tools"}</button></p>
      {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("type") || message.toLowerCase().includes("blocked") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </article>
    {overview ? <>
      <article style={cardStyle}><h2 style={{ marginTop: 0 }}>System health</h2><pre style={{ whiteSpace: "pre-wrap", background: "#0f172a", color: "white", padding: "1rem", borderRadius: "12px", overflowX: "auto" }}>{JSON.stringify(overview.health, null, 2)}</pre></article>
      <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Role assignments</h2>{table(overview.roles || [], ["email", "role", "user_id", "created_at", "updated_at"])}<div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", marginTop: "1rem", alignItems: "end" }}><label>Email<br /><input value={targetEmail} onChange={(event) => setTargetEmail(event.target.value)} style={inputStyle} /></label><label>Role<br /><select value={targetRole} onChange={(event) => setTargetRole(event.target.value)} style={inputStyle}>{roleOptions.map((role) => <option key={role} value={role}>{role}</option>)}</select></label><label>User ID optional<br /><input value={targetUserId} onChange={(event) => setTargetUserId(event.target.value)} style={inputStyle} /></label><label>Confirmation<br /><input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} placeholder="SAVE ROLE or REVOKE ROLE" style={inputStyle} /></label><button type="button" onClick={() => saveRole("upsert")} disabled={busy} style={buttonStyle}>Save role</button><button type="button" onClick={() => saveRole("revoke")} disabled={busy} style={ghostButtonStyle}>Revoke role</button></div></article>
      <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Admin activity</h2><p style={{ color: "#475569" }}>Retention guidance: {overview.retention_days} days. Suggested cutoff: {String(overview.retention_cutoff || "").slice(0, 10)}.</p>{overview.activity_warning ? <p style={{ color: "#b91c1c" }}>{overview.activity_warning}</p> : null}{table(overview.activity || [], ["created_at", "actor_email", "actor_role", "action_type", "entity_type", "entity_id", "source_page", "flagged_for_review", "note"])}</article>
    </> : null}
  </div>;
}
