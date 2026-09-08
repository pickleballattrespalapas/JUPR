"use client";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";

import Link from "next/link";
import { ConfirmAction } from "@/components/ConfirmAction";
import { useEffect, useRef, useState } from "react";
import { useAdminSession } from "@/lib/useAdminSession";
import { getAdminPlayerEditorApiBaseUrl } from "@/lib/adminPlayerEditorApi";
import { StaffInvitation, staffInvitationPath, describeStaffScopes } from "@/lib/staffInvitations";

type Scope = { kind: string; program_type: string; resource_id: string };
type Staff = { email: string; role: string; scopes: Scope[]; expires_at: string | null; revoked_at: string | null };
const programs = ["leagues", "tournaments", "round_robin", "ladder", "challenge_ladder", "live_play", "moneyball"];

export default function StaffPage() {
  const { session } = useAdminSession();
  const { clubId } = useAdminWorkspace();
  return <StaffWorkspace key={`${clubId}:${session?.user?.id || session?.user?.email || ""}`} />;
}

function StaffWorkspace() {
  const { session, accessToken, loading } = useAdminSession();
  const assignments = session?.capabilities?.assignments || [];
  const clubs = assignments.filter(a => ["super_admin", "club_owner", "administrator"].includes(a.role));
  const { clubId } = useAdminWorkspace();
  const canManage = clubs.some(club => club.club_id === clubId);
  const [rows, setRows] = useState<Staff[]>([]);
  const [invitations, setInvitations] = useState<StaffInvitation[]>([]);
  const [editingEmail, setEditingEmail] = useState("");
  const [targets, setTargets] = useState<{program_type: string; resource_id: string; label: string}[]>([]);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("operator");
  const [scopes, setScopes] = useState<Scope[]>([{ kind: "program_type", program_type: "leagues", resource_id: "" }]);
  const [expires, setExpires] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [revision, setRevision] = useState(0);
  const pending = useRef(false);
  const mutation = useRef<AbortController | null>(null);
  const invitationRequest = useRef({ payload: "", id: "" });
  const api = getAdminPlayerEditorApiBaseUrl();
  useEffect(() => () => { mutation.current?.abort(); }, []);
  useEffect(() => {
    if (!accessToken || !clubId || !api || !canManage) return;
    const controller = new AbortController();
    fetch(`${api}/admin/clubs/${encodeURIComponent(clubId)}/staff`, { headers: { Authorization: `Bearer ${accessToken}` }, signal: controller.signal })
      .then(async response => { const data = await response.json(); if (!response.ok) throw new Error(data.detail || "Unable to load staff."); if (!controller.signal.aborted) setRows(data.staff); })
      .catch(error => { if (!controller.signal.aborted) setMessage(error.message); });
    fetch(`${api}/admin/clubs/${encodeURIComponent(clubId)}/staff/targets`, { headers: { Authorization: `Bearer ${accessToken}` }, signal: controller.signal })
      .then(async response => { const data = await response.json(); if (!response.ok) throw new Error("Unable to load program choices."); if (!controller.signal.aborted) setTargets(data.targets); })
      .catch(error => { if (!controller.signal.aborted) setMessage(error.message); });
    fetch(`${api}/admin/clubs/${encodeURIComponent(clubId)}/staff/invitations`, { headers: { Authorization: `Bearer ${accessToken}` }, signal: controller.signal })
      .then(async response => { const data = await response.json(); if (!response.ok) throw new Error("Unable to load invitations."); if (!controller.signal.aborted) setInvitations(data.invitations); })
      .catch(error => { if (!controller.signal.aborted) setMessage(error.message); });
    return () => controller.abort();
  }, [accessToken, clubId, api, revision, canManage]);
  async function save(target?: Staff) {
    if (!api || !accessToken || pending.current) { if (target) throw new Error("Staff service is not ready."); return; }
    pending.current = true; setBusy(true); setMessage("");
    const controller = new AbortController(); mutation.current = controller;
    try {
      const inviting = !target && !editingEmail;
      const end = role === "operator" && expires ? new Date(expires).toISOString() : null;
      const payload = target ? { ...target, role: target.role === "club_owner" ? "administrator" : target.role, revoke: true } : {
        email: email.trim().toLowerCase(), role, scopes: role === "operator" ? scopes : [], [inviting ? "access_expires_at" : "expires_at"]: end
      };
      const serialized = JSON.stringify(payload);
      if (inviting && invitationRequest.current.payload !== serialized) invitationRequest.current = { payload: serialized, id: crypto.randomUUID() };
      const response = await fetch(`${api}/admin/clubs/${encodeURIComponent(clubId)}/staff${inviting ? "/invitations" : ""}`, {
        method: inviting ? "POST" : "PUT", headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" }, signal: controller.signal,
        body: JSON.stringify(inviting ? { ...payload, invitation_id: invitationRequest.current.id } : payload)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Check the email and scope fields.");
      if (controller.signal.aborted) return;
      setMessage(target ? "Staff access removed." : inviting ? "Invitation created. Copy the link below and share it with the invited person. Access starts after they sign in and accept." : "Staff access saved.");
      invitationRequest.current = { payload: "", id: "" };
      setEmail(""); setEditingEmail(""); setRevision(n => n + 1);
    } catch (error) { if (!controller.signal.aborted) setMessage(error instanceof Error ? error.message : "Unable to save staff."); if (target) throw error; }
    finally { pending.current = false; if (!controller.signal.aborted) setBusy(false); }
  }
  async function cancelInvitation(invitation: StaffInvitation) {
    if (!api || pending.current) throw new Error("Staff service is not ready.");
    pending.current = true; setBusy(true);
    const controller = new AbortController(); mutation.current = controller;
    try {
      const response = await fetch(`${api}/admin/clubs/${encodeURIComponent(clubId)}/staff/invitations/${invitation.id}/cancel`, {
        method: "POST", headers: { Authorization: `Bearer ${accessToken}` }, signal: controller.signal
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Unable to cancel invitation.");
      if (!controller.signal.aborted) { setMessage("Invitation cancelled."); setRevision(n => n + 1); }
    } finally { pending.current = false; if (!controller.signal.aborted) setBusy(false); }
  }
  if (loading) return <p>Loading staff access…</p>;
  if (!canManage) return <p>Club administrator access is required. <Link href="/admin/login">Sign in</Link></p>;
  return <section style={{ maxWidth: 960, margin: "0 auto" }}>
    <h1>Club staff</h1><p>Invite administrators and operators to this club. Each person signs in with the invited email and accepts before access begins.</p>
    <form onSubmit={e => { e.preventDefault(); void save(); }} style={{ display: "grid", gap: 12, margin: "24px 0", padding: 20, border: "1px solid #cbd5e1", borderRadius: 12 }}>
      <h2>{editingEmail ? "Edit staff access" : "Invite staff"}</h2>
      <label>Email <input type="email" required value={email} onChange={e => setEmail(e.target.value)} disabled={busy || Boolean(editingEmail)}/></label>
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
      <button disabled={busy} type="submit">{busy ? "Saving…" : editingEmail ? "Save staff access" : "Create invitation"}</button>
      {editingEmail && <button type="button" disabled={busy} onClick={() => { setEditingEmail(""); setEmail(""); }}>Cancel edit</button>}
    </form>
    {message && <p role="status">{message}</p>}
    <h2>Invitations</h2>
    <p>Links are valid for seven days and work only with the invited email. Creating a link does not send an email.</p>
    <button disabled={busy} onClick={() => setRevision(n => n + 1)}>Refresh staff and invitations</button>
    {!invitations.length && <p>No invitations yet.</p>}
    {invitations.map(invitation => <article key={invitation.id} style={{ padding: 16, borderBottom: "1px solid #cbd5e1" }}>
      <strong>{invitation.email}</strong> · {invitation.role} · {invitation.status}
      <p>{invitation.role === "administrator" ? "Full club control" : describeStaffScopes(invitation.scopes)}</p>
      {invitation.accepted_at ? <p>Accepted {new Date(invitation.accepted_at).toLocaleString()}</p> : <p>Invitation ends {new Date(invitation.expires_at).toLocaleString()}</p>}
      {invitation.access_expires_at && <p>Access ends {new Date(invitation.access_expires_at).toLocaleString()}</p>}
      {invitation.status === "pending" && <div style={{ display: "grid", gap: 12 }}>
        <label>Invitation link <input readOnly style={{ width: "100%" }} value={`${typeof window === "undefined" ? "" : window.location.origin}${staffInvitationPath(invitation.id)}`} onFocus={e => e.target.select()} /></label>
        <button disabled={busy} onClick={() => { if (!navigator.clipboard) { setMessage("Select and copy the invitation link above."); return; } void navigator.clipboard.writeText(`${window.location.origin}${staffInvitationPath(invitation.id)}`).then(() => setMessage("Invitation link copied. Share it with the invited person.")).catch(() => setMessage("Select and copy the invitation link above.")); }}>Copy invitation link</button>
        <ConfirmAction disabled={busy} triggerLabel="Cancel invitation" title="Cancel this invitation?" description={`The unused link for ${invitation.email} will stop working.`} confirmLabel="Cancel invitation" confirmationText="" tone="danger" onConfirm={async () => { await cancelInvitation(invitation); return { status: "success", title: "Invitation cancelled", description: "The link can no longer grant access." }; }} />
      </div>}
    </article>)}
    {invitations.length === 500 && <p>Showing the latest 500 invitations.</p>}
    <h2>Current staff</h2>
    {rows.map(row => <article key={row.email} style={{ padding: 16, borderBottom: "1px solid #cbd5e1" }}>
      <strong>{row.email}</strong> · {row.role.replaceAll("_", " ")} {row.revoked_at ? "— access removed" : row.expires_at && new Date(row.expires_at) <= new Date() ? "— expired" : ""}
      <p>{row.scopes?.map(s => s.kind === "club" ? "All club programs" : `${s.program_type.replaceAll("_", " ")}${s.kind === "resource" ? `: ${s.resource_id}` : ""}`).join("; ")}</p>
      {row.expires_at && <p>Expires {new Date(row.expires_at).toLocaleString()}</p>}
      {["administrator", "operator", "club_owner"].includes(row.role) && <div style={{ display: "flex", gap: 12 }}>
        <button disabled={busy} onClick={() => { const active = !row.revoked_at && (!row.expires_at || new Date(row.expires_at) > new Date()); setEditingEmail(active ? row.email : ""); setEmail(row.email); setRole(row.role === "operator" ? "operator" : "administrator"); setScopes(row.scopes?.length ? row.scopes : [{ kind: "program_type", program_type: "leagues", resource_id: "" }]); setExpires(active && row.expires_at ? new Date(new Date(row.expires_at).getTime() - new Date(row.expires_at).getTimezoneOffset() * 60000).toISOString().slice(0,16) : ""); }}>{row.revoked_at || (row.expires_at && new Date(row.expires_at) <= new Date()) ? "Invite again" : "Edit access"}</button>
        {!row.revoked_at && <ConfirmAction disabled={busy} triggerLabel="Remove access" title="Remove staff access?" description={`Remove access for ${row.email} at this club.`} confirmLabel="Remove access" confirmationText="" tone="danger" onConfirm={async () => { await save(row); return { status: "success", title: "Staff access removed", description: `${row.email} no longer has this club assignment.` }; }}/>}
      </div>}
    </article>)}
    <p><Link href="/admin">Back to club operations</Link></p>
  </section>;
}
