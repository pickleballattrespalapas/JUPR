"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { ADMIN_PASSWORD_MIN_LENGTH, AdminSession, authorizeAndSaveAdminSession, consumeStaffInvitationSession, getAdminApiBaseUrl, refreshAdminSession, setInvitationPassword, signInWithPassword } from "@/lib/adminAuthClient";
import { AvailableWorkspace, selectAdminWorkspace } from "@/lib/adminWorkspace";
import { StaffInvitation, describeStaffScopes } from "@/lib/staffInvitations";
import styles from "./invitation.module.css";

type Review = { invitation: StaffInvitation; club: { id: string; slug: string; name: string }; organizer?: { name: string } };

export default function AcceptInvitation() {
  const params = useSearchParams(), id = params.get("invitation") || "", clubJoin = params.get("kind") === "club";
  return <InvitationContent key={`${clubJoin}:${id}`} id={id} clubJoin={clubJoin} setupPassword={params.get("setup") === "password"} />;
}

function InvitationContent({ id, clubJoin, setupPassword }: { id: string; clubJoin: boolean; setupPassword: boolean }) {
  const api = getAdminApiBaseUrl();
  const invitationRoot = `${api}/${clubJoin ? "club-invitations" : "staff-invitations"}/${encodeURIComponent(id)}`;
  const [session, setSession] = useState<AdminSession | null>(null);
  const [ready, setReady] = useState(false);
  const [review, setReview] = useState<Review | null>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [mode, setMode] = useState<"create" | "signin">("create");
  const [needsPassword, setNeedsPassword] = useState(setupPassword);
  const [emailEnabled, setEmailEnabled] = useState<boolean | null>(null);
  const [optionsError, setOptionsError] = useState(false);
  const [optionsRevision, setOptionsRevision] = useState(0);
  const [emailRequested, setEmailRequested] = useState(false);
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
    if (!api || !id) return;
    const controller = new AbortController();
    setOptionsError(false); setEmailEnabled(null);
    fetch(`${invitationRoot}/sign-in`, { cache: "no-store", signal: controller.signal })
      .then(async response => {
        const data = await response.json();
        if (!response.ok || typeof data.email_enabled !== "boolean") throw new Error("Unavailable");
        if (!controller.signal.aborted) setEmailEnabled(data.email_enabled);
      })
      .catch(() => { if (!controller.signal.aborted) setOptionsError(true); });
    return () => controller.abort();
  }, [api, id, invitationRoot, optionsRevision]);
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
  function changeMode(next: "create" | "signin") {
    setMode(next); setPassword(""); setConfirmPassword(""); setMessage(""); setEmailRequested(false);
  }
  async function requestEmail(createAccount: boolean) {
    if (emailEnabled !== true) return;
    const response = await fetch(`${invitationRoot}/sign-in`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, setup_password: createAccount })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Check the invited email and try again.");
    if (mounted.current) {
      setEmailEnabled(data.email_enabled === true);
      setEmailRequested(data.email_enabled === true);
      setMessage(data.message);
    }
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
  const canAccept = review && ["pending", "accepted"].includes(review.invitation.status);
  const choosingPassword = session && needsPassword && canAccept;
  return <section className={styles.card}>
    <p className={styles.eyebrow}>{clubJoin ? "Join your club" : "Club staff"}</p>
    <h1>{!session ? mode === "create" ? "Create your PCS account" : "Sign in to your invitation" : choosingPassword ? "Set your password" : "Review your invitation"}</h1>
    {!api && <p role="alert">The invitation service is unavailable.</p>}
    {!session ? <>
      <div className={styles.choices} role="group" aria-label="Account options">
        <button type="button" disabled={busy} aria-pressed={mode === "create"} onClick={() => changeMode("create")}>Create account</button>
        <button type="button" disabled={busy} aria-pressed={mode === "signin"} onClick={() => changeMode("signin")}>Already have an account</button>
      </div>
      {mode === "create" ? <>
        <p>New to PCS or never set a password? Start with the email your organizer invited.</p>
        <ol className={styles.steps}><li>Verify your email</li><li>Choose a password</li><li>Review and accept your invitation</li></ol>
        <p>The email link brings you back here to set your password. An invitation does not create a password for you.</p>
      </> : <p>Use your existing PCS email and password. You will review the club and permissions before accepting.</p>}
      {emailEnabled === false && <div className={styles.notice} role="status">
        <strong>New account setup is unavailable in this test environment.</strong>
        <p>Verification and sign-in emails are disabled. To test invitation acceptance, use an existing test account with the invited email. {clubJoin ? "The organizer can update the invitation to that account’s email in the club setup." : "Ask your club administrator for an invitation to that test account’s email."}</p>
      </div>}
      {optionsError && <div className={styles.notice} role="status">
        <p>We could not check email verification availability. You can still sign in with an existing password.</p>
        <button type="button" disabled={busy} onClick={() => setOptionsRevision(n => n + 1)}>Retry email availability</button>
      </div>}
      <form className={styles.form} onSubmit={event => {
        event.preventDefault(); void perform(async () => {
          if (mode === "create") return requestEmail(true);
          const signedIn = await signInWithPassword(email, password);
          if (mounted.current) { setPassword(""); setNeedsPassword(false); setSession(signedIn); }
        });
      }}>
        <label>Invited email<input type="email" autoComplete="username" required disabled={busy} value={email} onChange={e => { setEmail(e.target.value); setEmailRequested(false); setMessage(""); }} /></label>
        {mode === "signin" && <label>Password<input type="password" autoComplete="current-password" required disabled={busy} value={password} onChange={e => setPassword(e.target.value)} /></label>}
        <button className={styles.primary} disabled={busy || !api || (mode === "create" && emailEnabled !== true)}>{busy ? "Please wait…" : mode === "create" ? "Email me a verification link" : "Sign in to review"}</button>
        {mode === "signin" && <>
          <button type="button" disabled={busy || !api || !email.trim() || emailEnabled !== true} onClick={() => void perform(() => requestEmail(false))}>Email me a sign-in link</button>
          <button type="button" disabled={busy} onClick={() => changeMode("create")}>I need to set a password</button>
        </>}
      </form>
      {emailRequested && <div className={styles.notice}><strong>Check your email</strong><p>If this email matches the invitation, follow the newest link to {mode === "create" ? "choose your password" : "review your invitation"}. Check your spam folder too.</p></div>}
    </> : <>
      <p>Signed in{session.user?.email ? ` as ${session.user.email}` : " with your account"}.</p>
      <button disabled={busy} onClick={() => { setSession(null); setReview(null); setMessage(""); setPassword(""); setConfirmPassword(""); setEmailRequested(false); }}>Use another account</button>
      {choosingPassword ? <form className={styles.form} onSubmit={event => {
        event.preventDefault(); void perform(async () => {
          if (password !== confirmPassword) throw new Error("The passwords do not match.");
          const current = await setInvitationPassword(password, session);
          if (mounted.current) { setPassword(""); setConfirmPassword(""); setNeedsPassword(false); setSession(current); setMessage("Password saved. Review your invitation below."); }
        });
      }}>
        <p>Your email is verified. Choose the password you’ll use to sign in to PCS, then review your invitation to {review.club.name}.</p>
        <label>New password<input type="password" autoComplete="new-password" required minLength={ADMIN_PASSWORD_MIN_LENGTH} disabled={busy} value={password} onChange={e => setPassword(e.target.value)} aria-describedby="password-help" /></label>
        <p className={styles.help} id="password-help">Use at least {ADMIN_PASSWORD_MIN_LENGTH} characters. Your password belongs to your PCS account across all your clubs.</p>
        <label>Confirm password<input type="password" autoComplete="new-password" required minLength={ADMIN_PASSWORD_MIN_LENGTH} disabled={busy} value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} /></label>
        <button className={styles.primary} disabled={busy}>{busy ? "Saving password…" : "Save password and continue"}</button>
        <button type="button" disabled={busy} onClick={() => { setPassword(""); setConfirmPassword(""); setNeedsPassword(false); setMessage(""); }}>Continue with my existing password</button>
      </form> : review ? <div className={styles.details}>
        <h2>{review.club.name}</h2>
        {clubJoin && <p>{review.organizer?.name || "An interclub organizer"} invited you to set up this club in PCS. Accept to manage its players, staff and meet rosters. You can review season invitations in your club workspace.</p>}
        <p><strong>{review.invitation.role === "administrator" ? "Administrator" : "Operator"}</strong></p>
        <p>{review.invitation.role === "administrator" ? "Full club control, including staff and all programs." : describeStaffScopes(review.invitation.scopes)}</p>
        <p>Invited email: {review.invitation.email}</p>
        {review.invitation.access_expires_at && <p>Access ends {new Date(review.invitation.access_expires_at).toLocaleString()}.</p>}
        <p>Invitation {review.invitation.status === "pending" ? `expires ${new Date(review.invitation.expires_at).toLocaleString()}` : review.invitation.status}.</p>
        {canAccept ? <>
          <button className={styles.primary} disabled={busy} onClick={() => void perform(accept)}>{busy ? "Opening club…" : review.invitation.status === "accepted" ? "Open club" : "Accept invitation"}</button>
          <button disabled={busy} onClick={() => { setNeedsPassword(true); setMessage(""); }}>Set a password</button>
        </> : <p>Ask a club administrator for a new invitation.</p>}
      </div> : <button disabled={busy} onClick={() => setRevision(n => n + 1)}>Reload invitation</button>}
    </>}
    {message && <p role="status">{message}</p>}
    <p><Link href="/admin/login">Go to staff sign-in</Link></p>
  </section>;
}
