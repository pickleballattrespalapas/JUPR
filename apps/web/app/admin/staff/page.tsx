"use client";

import Link from "next/link";
import { ConfirmAction } from "@/components/ConfirmAction";
import { useEffect, useState } from "react";
import { useAdminSession } from "@/lib/useAdminSession";
import { getAdminPlayerEditorApiBaseUrl } from "@/lib/adminPlayerEditorApi";

type Scope = { kind: string; program_type: string; resource_id: string };
type Staff = { email: string; role: string; scopes: Scope[]; expires_at: string | null; revoked_at: string | null };
const programs = ["leagues", "tournaments", "round_robin", "ladder", "challenge_ladder", "live_play", "moneyball"];

export default function StaffPage() {
  const { session, accessToken, loading } = useAdminSession();
  const assignments = session?.capabilities?.assignments || [];
  const clubs = assignments.filter(a => ["super_admin", "club_owner", "administrator"].includes(a.role));
  const [club, setClub] = useState("");
  const clubId = club || clubs[0]?.club_id || "";
  const [rows, setRows] = useState<Staff[]>([]);
  const [targets, setTargets] = useState<{program_type: string; resource_id: string; label: string}[]>([]);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("operator");
  const [scopes, setScopes] = useState<Scope[]>([{ kind: "program_type", program_type: "leagues", resource_id: "" }]);
  const [expires, setExpires] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [revision, setRevision] = useState(0);
  const api = getAdminPlayerEditorApiBaseUrl();
  useEffect(() => {
    if (!accessToken || !clubId || !api) return;
    const controller = new AbortController();
    setRows([]); setTargets([]);
    fetch(`${api}/admin/clubs/${encodeURIComponent(clubId)}/staff`, { headers: { Authorization: `Bearer ${accessToken}` }, signal: controller.signal })
      .then(async response => { const data = await response.json(); if (!response.ok) throw new Error(data.detail || "Unable to load staff."); setRows(data.staff); })
      .catch(error => { if (error.name !== "AbortError") setMessage(error.message); });
    fetch(`${api}/admin/clubs/${encodeURIComponent(clubId)}/staff/targets`, { headers: { Authorization: `Bearer ${accessToken}` }, signal: controller.signal })
      .then(async response => { const data = await response.json(); if (!response.ok) throw new Error("Unable to load program choices."); setTargets(data.targets); })
      .catch(error => { if (error.name !== "AbortError") setMessage(error.message); });
    return () => controller.abort();
  }, [accessToken, clubId, api, revision]);
  async function save(target?: Staff) {
    if (!api || busy) { if (target) throw new Error("Staff service is not ready."); return; }
    setBusy(true); setMessage("");
    try {
      const response = await fetch(`${api}/admin/clubs/${encodeURIComponent(clubId)}/staff`, {
        method: "PUT", headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
        body: JSON.stringify(target ? { ...target, role: target.role === "club_owner" ? "administrator" : target.role, revoke: true } : {
          email, role, scopes: role === "operator" ? scopes : [], expires_at: role === "operator" && expires ? new Date(expires).toISOString() : null
        })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Check the email and scope fields.");
      setMessage(target ? "Staff access removed." : "Staff access saved. They can sign in using this email.");
      setEmail(""); setRevision(n => n + 1);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to save staff."); if (target) throw error; }
    finally { setBusy(false); }
  }
  if (loading) return <p>Loading staff access…</p>;
  if (!clubs.length) return <p>Club administrator access is required. <Link href="/admin/login">Sign in</Link></p>;
  return <section style={{ maxWidth: 960, margin: "0 auto" }}>
    <h1>Club staff</h1><p>Assign administrators and operators. Each person signs in with their own account.</p>
    <label>Club <select value={clubId} onChange={e => { setClub(e.target.value); setEmail(""); setMessage(""); }} disabled={busy}>{clubs.map(a => <option key={a.club_id} value={a.club_id}>{a.club_id.replaceAll("_", " ")}</option>)}</select></label>
    <form onSubmit={e => { e.preventDefault(); void save(); }} style={{ display: "grid", gap: 12, margin: "24px 0", padding: 20, border: "1px solid #cbd5e1", borderRadius: 12 }}>
      <h2>{rows.some(r => r.email === email) ? "Edit staff access" : "Add staff"}</h2>
      <label>Email <input type="email" required value={email} onChange={e => setEmail(e.target.value)} disabled={busy}/></label>
      <label>Role <select value={role} onChange={e => setRole(e.target.value)} disabled={busy}><option value="operator">Operator</option><option value="administrator">Administrator</option></select></label>
      {role === "administrator" ? <p>Full club control, including staff access and all programs.</p> : <>
        {scopes.map((scope, index) => <fieldset key={index} disabled={busy} style={{ display: "flex", gap: 12, flexWrap: "wrap" }}><legend>Assignment {index + 1}</legend>
          <label>Scope <select value={scope.kind} onChange={e => setScopes(old => old.map((s, i) => i === index ? { ...s, kind: e.target.value } : s))}><option value="program_type">Program type</option><option value="resource">One program or session</option><option value="club">All club programs</option></select></label>
          {scope.kind !== "club" && <label>Program <select value={scope.program_type} onChange={e => setScopes(old => old.map((s, i) => i === index ? { ...s, program_type: e.target.value, resource_id: "" } : s))}>{programs.map(p => <option key={p} value={p}>{p.replaceAll("_", " ")}</option>)}</select></label>}
          {scope.kind === "resource" && <label>Assigned program or session <select required value={scope.resource_id} onChange={e => setScopes(old => old.map((s, i) => i === index ? { ...s, resource_id: e.target.value } : s))}><option value="">Choose a program or session</option>{targets.filter(t => t.program_type === scope.program_type).map(t => <option key={t.resource_id} value={t.resource_id}>{t.label}</option>)}</select></label>}
          {scopes.length > 1 && <button type="button" onClick={() => setScopes(old => old.filter((_, i) => i !== index))}>Remove assignment</button>}
        </fieldset>)}
        <button type="button" disabled={busy} onClick={() => setScopes(old => [...old, { kind: "program_type", program_type: "leagues", resource_id: "" }])}>Add another assignment</button>
        <label>Access ends (leave blank for ongoing access) <input type="datetime-local" value={expires} onChange={e => setExpires(e.target.value)} disabled={busy}/></label>
      </>}
      <button disabled={busy} type="submit">{busy ? "Saving…" : "Save staff access"}</button>
    </form>
    {message && <p role="status">{message}</p>}
    <h2>Current staff</h2>
    {rows.map(row => <article key={row.email} style={{ padding: 16, borderBottom: "1px solid #cbd5e1" }}>
      <strong>{row.email}</strong> · {row.role.replaceAll("_", " ")} {row.revoked_at ? "— access removed" : row.expires_at && new Date(row.expires_at) <= new Date() ? "— expired" : ""}
      <p>{row.scopes?.map(s => s.kind === "club" ? "All club programs" : `${s.program_type.replaceAll("_", " ")}${s.kind === "resource" ? `: ${s.resource_id}` : ""}`).join("; ")}</p>
      {row.expires_at && <p>Expires {new Date(row.expires_at).toLocaleString()}</p>}
      {["administrator", "operator", "club_owner"].includes(row.role) && <div style={{ display: "flex", gap: 12 }}>
        <button disabled={busy} onClick={() => { setEmail(row.email); setRole(row.role === "operator" ? "operator" : "administrator"); setScopes(row.scopes?.length ? row.scopes : [{ kind: "program_type", program_type: "leagues", resource_id: "" }]); setExpires(row.expires_at ? new Date(new Date(row.expires_at).getTime() - new Date(row.expires_at).getTimezoneOffset() * 60000).toISOString().slice(0,16) : ""); }}>Edit access</button>
        {!row.revoked_at && <ConfirmAction disabled={busy} triggerLabel="Remove access" title="Remove staff access?" description={`Remove access for ${row.email} at this club.`} confirmLabel="Remove access" confirmationText="" tone="danger" onConfirm={async () => { await save(row); return { status: "success", title: "Staff access removed", description: `${row.email} no longer has this club assignment.` }; }}/>}
      </div>}
    </article>)}
    <p><Link href="/admin">Back to club operations</Link></p>
  </section>;
}
