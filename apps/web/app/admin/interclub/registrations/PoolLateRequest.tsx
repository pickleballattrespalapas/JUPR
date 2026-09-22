"use client";

import { useEffect, useRef, useState } from "react";
import { poolGender, poolRating, sortedPoolDivisions, type LatePlayerRequest, type PoolMember, type PoolPlayerChoice, type PoolPlayerChoices } from "@/lib/interclubPlayerPool";
import { RequestStatus } from "./PoolPanelCommon";
import { usePoolResource } from "./usePoolResource";
import styles from "./playerPool.module.css";

type Props = {
  root: string; accessToken: string; members: PoolMember[]; disabled: boolean;
  onRequest: (request: LatePlayerRequest) => Promise<void>; onClose: () => void;
  onViewMember: (member: PoolMember) => void;
};

function existingSignupStatus(member: PoolMember) {
  if (member.status === "withdrawn") return "This signup was withdrawn. A new late request cannot replace it.";
  if (member.approval_status === "approved") return "Already registered and approved. No late request is needed.";
  if (member.approval_status === "pending") return "Awaiting commissioner approval. A request already exists.";
  if (member.approval_status === "rejected") return "The existing request was rejected. View the signup for the decision.";
  return "A season signup already exists. View it to check its status.";
}

export function PoolLateRequest({ root, accessToken, members, disabled, onRequest, onClose, onViewMember }: Props) {
  const [query, setQuery] = useState(""), [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<PoolPlayerChoice | null>(null), [reason, setReason] = useState("");
  const heading = useRef<HTMLHeadingElement | null>(null);
  const resource = usePoolResource<PoolPlayerChoices>(`${root}/pool/players?q=${encodeURIComponent(query)}&offset=${offset}`, accessToken);
  useEffect(() => { heading.current?.focus(); }, []);
  const existing = new Map(members.filter(member => member.player_id).map(member => [String(member.player_id), member]));
  const selectedIsExisting = !!selected && existing.has(String(selected.id));
  const players = resource.data?.players || [];
  const newPlayers = players.filter(player => !existing.has(String(player.id)));
  const existingPlayers = players.filter(player => existing.has(String(player.id)));
  async function submit() {
    if (disabled || !selected || selectedIsExisting || !reason.trim()) return;
    const playerId = Number(selected.id);
    if (!Number.isSafeInteger(playerId) || playerId <= 0) return;
    await onRequest({ player_id: playerId, reason: reason.trim() });
  }
  return <section className={styles.bulkPanel} aria-label="Request a late player">
    <div className={styles.row}><h4 ref={heading} tabIndex={-1}>Request a late player</h4><button disabled={disabled} onClick={onClose}>Close late player request</button></div>
    <p>Choose a club player who has not joined this season’s pool and explain why they are joining late. The commissioner must approve the request before this player can join a lineup.</p>
    <p className={styles.muted}>Submitting a request sends no email and does not give permission to email the player.</p>
    <RequestStatus {...resource} />
    {resource.error && <button disabled={resource.loading} onClick={resource.reload}>Retry player search</button>}
    <form onSubmit={event => { event.preventDefault(); void submit(); }}><fieldset disabled={disabled}>
      <label>Find a late player in this club<input type="search" value={query} maxLength={80} placeholder="Search by name" onChange={event => { setQuery(event.target.value); setOffset(0); }} /></label>
      {newPlayers.length > 0 && <fieldset className={styles.candidates}><legend>Players available for a late request</legend>{newPlayers.map(player => <label key={player.id} className={styles.choice}><input type="radio" name="late-player" checked={selected != null && String(selected.id) === String(player.id)} onChange={() => setSelected(player)} /><span><strong>{player.name}</strong><span className={styles.playerMeta}>Club {poolRating(player.rating)}{player.league_rating != null ? ` · League ${poolRating(player.league_rating)}` : ""} · {poolGender(player.gender)}</span></span></label>)}</fieldset>}
      {players.length > 0 && newPlayers.length === 0 && <p role="status" className={styles.notice}>Everyone on this page already has a season signup. View their status below, or search for another player.</p>}
      {existingPlayers.length > 0 && <section aria-label="Existing season signups" className={styles.existingSignups}><h5>Already have a season signup</h5>{existingPlayers.map(player => {
        const member = existing.get(String(player.id))!;
        return <div key={player.id} className={styles.existingSignup}><div><strong>{player.name}</strong><p>{existingSignupStatus(member)}</p></div><button type="button" aria-label={`View ${member.name} in player pool`} onClick={() => onViewMember(member)}>View in player pool</button></div>;
      })}</section>}
      {resource.data?.players.length === 0 && <p>No matching club player. Create their player record in Players, then return to request their late entry.</p>}
      <div className={styles.toolbar}>{offset > 0 && <button type="button" disabled={resource.loading} onClick={() => setOffset(0)}>First club players</button>}{resource.data?.next_offset != null && <button type="button" disabled={resource.loading} onClick={() => setOffset(resource.data!.next_offset!)}>More club players</button>}</div>
      {selected && <p className={styles.notice}><strong>Selected: {selected.name}</strong><br />Current rating eligibility: {sortedPoolDivisions(selected.eligible_divisions).join(", ") || "No eligible division"}.{selectedIsExisting && <> This player is already in the season pool.</>}</p>}
      <label>Reason for late entry<textarea required maxLength={500} rows={3} value={reason} onChange={event => setReason(event.target.value)} placeholder="Explain the late arrival or other reason for the request." /></label>
      <button className={styles.primary} type="submit" disabled={disabled || !selected || selectedIsExisting || !reason.trim()}>Submit late player request</button>
    </fieldset></form>
  </section>;
}
