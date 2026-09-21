"use client";

import { useEffect, useRef, useState } from "react";
import type { PoolMember } from "@/lib/interclubPlayerPool";
import { usePoolResource } from "./usePoolResource";
import { RequestStatus } from "./PoolPanelCommon";
import styles from "./playerPool.module.css";

type Applicant = Pick<PoolMember, "id" | "club_id" | "name" | "player_id" | "revision" | "approval_status" | "late_join" | "approval_reason">;

export default function SeasonEligibilityApprovals({ root, accessToken, clubs }: {
  root: string; accessToken: string; clubs: { id: string; name: string }[];
}) {
  const resource = usePoolResource<{ members: Applicant[] }>(`${root}/pool/approvals`, accessToken);
  const [message, setMessage] = useState("");
  const section = useRef<HTMLElement | null>(null);
  const followedAnchor = useRef(false);
  useEffect(() => setMessage(""), [root]);
  useEffect(() => {
    if (resource.loading || followedAnchor.current || typeof window === "undefined" ||
        window.location.hash !== "#season-eligibility-approvals" || !section.current) return;
    followedAnchor.current = true;
    section.current.scrollIntoView({ block: "start" });
    section.current.focus({ preventScroll: true });
  }, [resource.loading]);
  async function decide(member: Applicant, approve: boolean, reason: string) {
    const response = await resource.perform(json => json(`${root}/pool/approvals`, "POST", {
      member_id: member.id, expected_revision: member.revision, approve, reason,
    }));
    if (response !== null) {
      setMessage(`${member.name}: ${approve ? "approved for the season pool" : "not approved"}.`);
      resource.reload();
    }
  }
  return <section id="season-eligibility-approvals" ref={section} tabIndex={-1} className={styles.panel} aria-label="Season eligibility approvals">
    <h3>Late season signups</h3>
    <p>Players joining after the season begins need the league organizer’s approval before a club can select them. Clubs keep control of player contact details.</p>
    <RequestStatus {...resource} />
    {message && <p role="status" className={styles.notice}>{message}</p>}
    {resource.data && !resource.data.members.some(member => member.approval_status === "pending" && member.late_join) && <p>No late signups are waiting for approval.</p>}
    {resource.data?.members.filter(member => member.approval_status === "pending" && member.late_join).map(member => <Approval key={`${member.id}:${member.revision}`} member={member} clubName={clubs.find(club => club.id === member.club_id)?.name || member.club_id} disabled={resource.disabled} decide={decide} />)}
    <button disabled={resource.busy || resource.loading} onClick={resource.reload}>Refresh eligibility requests</button>
  </section>;
}

function Approval({ member, clubName, disabled, decide }: {
  member: Applicant; clubName: string; disabled: boolean;
  decide: (member: Applicant, approve: boolean, reason: string) => Promise<void>;
}) {
  const [reason, setReason] = useState("");
  return <article className={styles.card}>
    <h4>{member.name}</h4><p>{clubName} · Late addition</p>
    {!member.player_id && <p>The club must link this signup to its player record before the player can enter a roster.</p>}
    <label>Decision reason<input maxLength={500} value={reason} onChange={event => setReason(event.target.value)} /></label>
    <div className={styles.toolbar}>
      <button className={styles.primary} disabled={disabled || !reason.trim() || !member.player_id} onClick={() => void decide(member, true, reason.trim())}>Approve season eligibility</button>
      <button disabled={disabled || !reason.trim()} onClick={() => void decide(member, false, reason.trim())}>Do not approve</button>
    </div>
  </article>;
}
