"use client";
import { useEffect, useRef, useState } from "react";
import type { StaffInvitation } from "@/lib/staffInvitations";
import styles from "./setup.module.css";

export type ClubJoinInvitation = StaffInvitation & { club_name: string; revision: number };
export type InviteClubInput = { name: string; email: string; invitation_id: string };

export default function ClubInvitationPanel({ disabled, atClubLimit, invitations, onInvite, onUpdate, allowCreate = true }: {
  disabled: boolean; atClubLimit: boolean; invitations: ClubJoinInvitation[]; allowCreate?: boolean;
  onInvite: (input: InviteClubInput) => Promise<boolean>;
  onUpdate: (invitation: ClubJoinInvitation, action: "cancel" | "renew", email: string) => void;
}) {
  const [showForm, setShowForm] = useState(false), [name, setName] = useState(""), [email, setEmail] = useState("");
  const identity = useRef<{ key: string; id: string } | null>(null);
  useEffect(() => {
    if (identity.current && invitations.some(invitation => invitation.id === identity.current?.id)) {
      setShowForm(false); setName(""); setEmail(""); identity.current = null;
    }
  }, [invitations]);
  return <section className={styles.invitePanel} aria-label={allowCreate ? "Invite a new club" : "Club account invitations"}>
    {allowCreate && <div className={styles.toolbar}><div><h3>Club missing from the list?</h3><p className={styles.muted}>Invite its administrator here. The club will be added to your season.</p></div>
      <button type="button" disabled={disabled || atClubLimit} aria-expanded={showForm} onClick={() => setShowForm(value => !value)}>{showForm ? "Close invitation form" : "Invite a new club"}</button></div>}
    {allowCreate && showForm && <form className={styles.form} onSubmit={async event => {
      event.preventDefault();
      if (disabled || atClubLimit) return;
      const key = JSON.stringify([name.trim(), email.trim().toLowerCase()]);
      if (identity.current?.key !== key) identity.current = { key, id: crypto.randomUUID() };
      if (await onInvite({ name: name.trim(), email: email.trim().toLowerCase(), invitation_id: identity.current.id })) {
        setName(""); setEmail(""); setShowForm(false); identity.current = null;
      }
    }}>
      <fieldset className={styles.form} disabled={disabled || atClubLimit}><div className={styles.grid}>
        <label className={styles.field}>Club name<input aria-label="New club name" required minLength={3} maxLength={120} value={name} onChange={e => setName(e.target.value)} placeholder="e.g. La Ribera Pickleball" /></label>
        <label className={styles.field}>Administrator’s email<input aria-label="New club administrator email" required type="email" maxLength={254} value={email} onChange={e => setEmail(e.target.value)} placeholder="admin@club.com" /></label>
      </div><p className={styles.muted}>Create an invitation link to share with this person. They’ll sign in with this email and accept to manage their club. The link lasts seven days.</p>
      <div><button type="submit" className={styles.primary} disabled={disabled || atClubLimit}>Create club invitation</button></div></fieldset>
    </form>}
    {invitations.length > 0 && <div className={styles.invitationList}><h3>Club account invitations</h3>{invitations.map(invitation => <InvitationRow key={`${invitation.id}:${invitation.revision}`} invitation={invitation} disabled={disabled} onUpdate={onUpdate} />)}</div>}
  </section>;
}

function InvitationRow({ invitation, disabled, onUpdate }: { invitation: ClubJoinInvitation; disabled: boolean;
  onUpdate: (invitation: ClubJoinInvitation, action: "cancel" | "renew", email: string) => void }) {
  const [editing, setEditing] = useState(false), [email, setEmail] = useState(invitation.email), [message, setMessage] = useState("");
  const link = typeof window === "undefined" ? "" : `${window.location.origin}/admin/accept-invitation?invitation=${encodeURIComponent(invitation.id)}&kind=club`;
  return <article className={styles.invitationRow}>
    <strong>{invitation.club_name}</strong><p>{invitation.email} · {invitation.status === "accepted" ? "Administrator joined" : invitation.status === "pending" ? "Waiting for administrator" : `Invitation ${invitation.status}`}</p>
    {invitation.status === "pending" && <><label className={styles.field}>Invitation link<input aria-label={`Invitation link for ${invitation.club_name}`} readOnly value={link} onFocus={e => e.target.select()} /></label>
      <div className={styles.toolbar}><button type="button" disabled={disabled} onClick={async () => { try { await navigator.clipboard.writeText(link); setMessage("Invitation link copied."); } catch { setMessage("Select the invitation link above and copy it."); } }}>Copy invitation link</button>
        <button type="button" disabled={disabled} onClick={() => onUpdate(invitation, "cancel", invitation.email)}>Cancel invitation</button></div>
      <p className={styles.muted}>Share this link with the administrator. No email has been sent. Expires {new Date(invitation.expires_at).toLocaleDateString()}.</p></>}
    {invitation.status !== "accepted" && <button type="button" disabled={disabled} onClick={() => setEditing(value => !value)}>{editing ? "Close invitation update" : "Update or renew invitation"}</button>}
    {editing && <form className={styles.form} onSubmit={event => { event.preventDefault(); if (!disabled) onUpdate(invitation, "renew", email); }}>
      <label className={styles.field}>Administrator’s email<input aria-label={`Update email for ${invitation.club_name}`} type="email" required maxLength={254} disabled={disabled} value={email} onChange={e => setEmail(e.target.value)} /></label>
      <p>Save the recipient and renew the link for seven days. Only this email can accept it.</p><div><button type="submit" disabled={disabled}>Save invitation</button></div>
    </form>}
    {message && <p role="status">{message}</p>}
  </article>;
}
