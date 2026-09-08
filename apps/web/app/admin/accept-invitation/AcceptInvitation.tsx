"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { AdminSession, authorizeAndSaveAdminSession, consumeStaffInvitationSession, getAdminApiBaseUrl, refreshAdminSession, signInWithPassword } from "@/lib/adminAuthClient";
import { AvailableWorkspace, selectAdminWorkspace } from "@/lib/adminWorkspace";
import { StaffInvitation, describeStaffScopes } from "@/lib/staffInvitations";
import styles from "./invitation.module.css";

type Review = { invitation: StaffInvitation; club: { id: string; slug: string; name: string }; organizer?: { name: string } };

export default function AcceptInvitation() {
  const params = useSearchParams(), id = params.get("invitation") || "", clubJoin = params.get("kind") === "club";
  return <InvitationContent key={`${clubJoin}:${id}`} id={id} clubJoin={clubJoin} />;
}

function InvitationContent({ id, clubJoin }: { id: string; clubJoin: boolean }) {
  const api = getAdminApiBaseUrl();
  const invitationRoot = `${api}/${clubJoin ? "club-invitations" : "staff-invitations"}/${encodeURIComponent(id)}`;
  const [session, setSession] = useState<AdminSession | null>(null);
  const [ready, setReady] = useState(false);
  const [review, setReview] = useState<Review | null>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [revision, setRevision] = useState(0);
  const initialization = useRef<Promise<AdminSession | null> | null>(null);
  const pending = useRef(false);
  const mounted = useRef(false);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  useEffect(() => {
    let active = true;
    initialization.current ||= consumeStaffInvitationSession();
    initialization.current.then(value => { if (active) setSession(value); })
      .catch(error => { if (active) setMessage(error.message); })
      .finally(() => { if (active) setReady(true); });
    return () => { active = false; };
  }, []);
  useEffect(() => {
    setReview(null);
    if (!session || !api || !id) return;
    const controller = new AbortController();
    fetch(invitationRoot, {
      headers: { Authorization: `Bearer ${session.access_token}` }, cache: "no-store", signal: controller.signal
    }).then(async response => {
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Unable to load invitation.");
      if (!controller.signal.aborted) { setReview(data); setMessage(""); }
    }).catch(error => { if (!controller.signal.aborted) setMessage(error.message); });
    return () => controller.abort();
  }, [api, id, invitationRoot, session, revision]);

  async function perform(action: () => Promise<void>) {
    if (pending.current) return;
    pending.current = true; setBusy(true); setMessage("");
    try { await action(); }
    catch (error) { if (mounted.current) setMessage(error instanceof Error ? error.message : "Unable to complete this action."); }
    finally { pending.current = false; if (mounted.current) setBusy(false); }
  }
  async function accept() {
    if (!session || !review || !api) return;
    const current = await refreshAdminSession(session);
    if (!current) throw new Error("Sign in again to accept this invitation.");
    const response = await fetch(`${invitationRoot}/accept`, {
      method: "POST", headers: { Authorization: `Bearer ${current.access_token}` }
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Unable to accept invitation.");
    if (!mounted.current) return;
    // Refresh all clubs only after the grant is committed. Opening this invited
    // club is explicit; the account retains its other club assignments.
    const authorized = await authorizeAndSaveAdminSession(current, "", { preserveOnUnavailable: true });
    const workspacesResponse = await fetch(`${api}/admin/auth/workspaces`, {
      headers: { Authorization: `Bearer ${authorized.access_token}` }, cache: "no-store"
    });
    if (!workspacesResponse.ok) throw new Error("Invitation accepted. Sign in to open your club.");
    const workspaces = (await workspacesResponse.json()).workspaces as AvailableWorkspace[];
    const workspace = workspaces.find(club => club.club_id === data.invitation.club_id);
    if (!workspace) throw new Error("Your club access has changed. Ask a club administrator for help.");
    if (mounted.current) selectAdminWorkspace(workspace);
  }

  if (!id || !/^[0-9a-f-]{36}$/i.test(id)) return <section className={styles.card}><h1>Staff invitation</h1><p>Open the full invitation link shared by your club administrator.</p></section>;
  if (!ready) return <p role="status">Checking sign-in…</p>;
  return <section className={styles.card}>
    <p className={styles.eyebrow}>{clubJoin ? "Join your club" : "Club staff"}</p><h1>Review your invitation</h1>
    <p>Sign in with the invited email. You will see the club and permissions before accepting.</p>
    {!api && <p role="alert">The invitation service is unavailable.</p>}
    {!session ? <form className={styles.form} onSubmit={event => {
      event.preventDefault(); void perform(async () => {
        const signedIn = await signInWithPassword(email, password);
        if (mounted.current) { setPassword(""); setSession(signedIn); }
      });
    }}>
      <label>Invited email<input type="email" autoComplete="username" required disabled={busy} value={email} onChange={e => setEmail(e.target.value)} /></label>
      <label>Password<input type="password" autoComplete="current-password" required disabled={busy} value={password} onChange={e => setPassword(e.target.value)} /></label>
      <button disabled={busy || !api}>Sign in to review</button>
      <p>New account or prefer email? Request a sign-in link using your invited email.</p>
      <button type="button" disabled={busy || !api || !email.trim()} onClick={() => void perform(async () => {
        const response = await fetch(`${invitationRoot}/sign-in`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Check the invited email and try again.");
        if (mounted.current) setMessage(data.message);
      })}>Email me a sign-in link</button>
    </form> : <>
      <p>Signed in{session.user?.email ? ` as ${session.user.email}` : " with your account"}.</p>
      <button disabled={busy} onClick={() => { setSession(null); setReview(null); setMessage(""); setPassword(""); }}>Use another account</button>
      {review ? <div className={styles.details}>
        <h2>{review.club.name}</h2>
        {clubJoin && <p>{review.organizer?.name || "An interclub organizer"} invited you to set up this club in PCS. Accept to manage its players, staff and meet rosters. You can review season invitations in your club workspace.</p>}
        <p><strong>{review.invitation.role === "administrator" ? "Administrator" : "Operator"}</strong></p>
        <p>{review.invitation.role === "administrator" ? "Full club control, including staff and all programs." : describeStaffScopes(review.invitation.scopes)}</p>
        <p>Invited email: {review.invitation.email}</p>
        {review.invitation.access_expires_at && <p>Access ends {new Date(review.invitation.access_expires_at).toLocaleString()}.</p>}
        <p>Invitation {review.invitation.status === "pending" ? `expires ${new Date(review.invitation.expires_at).toLocaleString()}` : review.invitation.status}.</p>
        {["pending", "accepted"].includes(review.invitation.status) ? <button disabled={busy} onClick={() => void perform(accept)}>{busy ? "Opening club…" : review.invitation.status === "accepted" ? "Open club" : "Accept invitation"}</button> : <p>Ask a club administrator for a new invitation.</p>}
      </div> : <button disabled={busy} onClick={() => setRevision(n => n + 1)}>Reload invitation</button>}
    </>}
    {message && <p role="status">{message}</p>}
    <p><Link href="/admin/login">Go to staff sign-in</Link></p>
  </section>;
}
