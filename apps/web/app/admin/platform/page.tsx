"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { useAdminSession } from "@/lib/useAdminSession";
import { getAdminPlayerEditorApiBaseUrl } from "@/lib/adminPlayerEditorApi";

type Club = { id: string; slug: string; name: string; is_active: boolean; plan_status: string; onboarding_status: string };
export default function PlatformPage() {
  const { accessToken, loading } = useAdminSession();
  const api = getAdminPlayerEditorApiBaseUrl();
  const [clubs, setClubs] = useState<Club[]>([]);
  const [allowed, setAllowed] = useState(false);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [name, setName] = useState("");
  const [slug, setSlug] = useState("");
  const [email, setEmail] = useState("");
  const [offset, setOffset] = useState(0);
  const [next, setNext] = useState<number | null>(null);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    if (!api || !accessToken) return;
    const controller = new AbortController();
    setAllowed(false);
    fetch(`${api}/admin/platform/clubs?offset=${offset}`, { headers: { Authorization: `Bearer ${accessToken}` }, signal: controller.signal })
      .then(async r => { const data = await r.json(); if (!r.ok) throw new Error(data.detail || "Unable to load clubs."); setClubs(data.clubs); setNext(data.next_offset); setAllowed(true); })
      .catch(e => { if (e.name !== "AbortError") setMessage(e.message); });
    return () => controller.abort();
  }, [api, accessToken, offset, revision]);
  async function save(path: string, method: string, body: object) {
    if (!api || !allowed || busy) return;
    setBusy(true); setMessage("");
    try {
      const r = await fetch(`${api}${path}`, { method, headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const data = await r.json();
      if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Check the form fields.");
      setMessage(method === "POST" ? "Draft club and administrator saved. No invitation has been sent." : "Onboarding status saved.");
      if (method === "POST") { setName(""); setSlug(""); setEmail(""); setOffset(0); }
      setRevision(n => n + 1);
    } catch(e) { setMessage(e instanceof Error ? e.message : "Unable to save."); }
    finally { setBusy(false); }
  }
  if (loading) return <p>Loading…</p>;
  if (!accessToken) return <p><Link href="/admin/login">Sign in</Link> with your PCS Super Admin account.</p>;
  return <section style={{ maxWidth: 1000, margin: "0 auto", padding: 24 }}>
    <h1>PCS administration</h1>
    <p>Create club accounts and follow their onboarding progress.</p>
    {!api && <p>The administration service is not configured.</p>}
    {message && <p role="status">{message}</p>}
    {allowed && <>
      <form onSubmit={e => { e.preventDefault(); void save("/admin/platform/clubs", "POST", { name, slug, administrator_email: email }); }} style={{ display: "grid", gap: 16, padding: 20, border: "1px solid #cbd5e1", borderRadius: 12 }}>
        <h2>Create a club</h2>
        <label>Club name <input required maxLength={120} value={name} onChange={e => setName(e.target.value)} disabled={busy}/></label>
        <label>Club address <input required minLength={3} maxLength={60} pattern="[a-z0-9]+(-[a-z0-9]+)*" placeholder="la-ribera" value={slug} onChange={e => setSlug(e.target.value)} disabled={busy}/><small> Lowercase letters, numbers, and hyphens. This address is permanent.</small></label>
        <label>First administrator’s email <input type="email" required maxLength={254} value={email} onChange={e => setEmail(e.target.value)} disabled={busy}/></label>
        <p>New clubs start as drafts with one administrator. Invitations, trial activation, and public signup will follow.</p>
        <button disabled={busy} type="submit">{busy ? "Saving…" : "Create draft club"}</button>
      </form>
      <h2>Clubs</h2>
      {!clubs.length && <p>No clubs on this page.</p>}
      <div style={{ overflowX: "auto" }}><table style={{ width: "100%", textAlign: "left" }}>
        <thead><tr><th>Club</th><th>Account</th><th>Onboarding</th></tr></thead>
        <tbody>{clubs.map(c => <tr key={c.id}>
          <td style={{ padding: 12 }}><strong>{c.name}</strong><br/>{c.slug}</td>
          <td>{c.plan_status} · {c.is_active ? "Active" : "Draft / inactive"}</td>
          <td><select aria-label={`Onboarding for ${c.name}`} value={c.onboarding_status} disabled={busy} onChange={e => void save(`/admin/platform/clubs/${encodeURIComponent(c.id)}/onboarding`, "PATCH", { status: e.target.value })}>
            {!["draft", "in_progress", "ready_for_review"].includes(c.onboarding_status) && <option value={c.onboarding_status}>{c.onboarding_status}</option>}
            <option value="draft">Draft</option><option value="in_progress">In progress</option><option value="ready_for_review">Ready for review</option>
          </select></td>
        </tr>)}</tbody>
      </table></div>
      <p><button disabled={busy || offset === 0} onClick={() => setOffset(n => Math.max(0,n-50))}>Previous</button> <button disabled={busy || next === null} onClick={() => next !== null && setOffset(next)}>Next</button></p>
    </>}
    <p><Link href="/admin/staff">Club staff</Link> · <Link href="/admin/interclub">Interclub planning</Link> · <Link href="/admin">Club operations</Link></p>
  </section>;
}
