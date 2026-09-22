"use client";

import { useEffect, useRef, useState } from "react";
import { poolGender, poolRating, sortedPoolDivisions, type LatePlayerInput, type PoolMember, type PoolPlayerChoice, type PoolPlayerChoices } from "@/lib/interclubPlayerPool";
import { RequestStatus } from "./PoolPanelCommon";
import { usePoolResource } from "./usePoolResource";
import { emptyNewPlayer, newPlayerDetails, PoolNewPlayerFields } from "./PoolNewPlayerFields";
import styles from "./playerPool.module.css";

type Props = {
  root: string; accessToken: string; members: PoolMember[]; disabled: boolean;
  onRequest: (request: LatePlayerInput) => Promise<void>; onClose: () => void;
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
  const [mode, setMode] = useState<"existing" | "new">("existing"), [draft, setDraft] = useState(emptyNewPlayer);
  const heading = useRef<HTMLHeadingElement | null>(null);
  const resource = usePoolResource<PoolPlayerChoices>(`${root}/pool/players?q=${encodeURIComponent(query)}&offset=${offset}`, accessToken);
  useEffect(() => { heading.current?.focus(); }, []);
  const existing = new Map(members.filter(member => member.player_id).map(member => [String(member.player_id), member]));
  const selectedIsExisting = !!selected && existing.has(String(selected.id));
  const players = resource.data?.players || [];
  const newPlayers = players.filter(player => !existing.has(String(player.id)));
  const existingPlayers = players.filter(player => existing.has(String(player.id)));
  const newPlayer = newPlayerDetails(draft);
  async function submit() {
    if (disabled) return;
    if (mode === "new") { if (newPlayer) await onRequest({ new_player: newPlayer, reason: reason.trim() }); return; }
    if (!selected || selectedIsExisting) return;
    const playerId = Number(selected.id);
    if (!Number.isSafeInteger(playerId) || playerId <= 0) return;
    await onRequest({ player_id: playerId, reason: reason.trim() });
  }
  return <section className={styles.bulkPanel} aria-label="Request a late player">
    <div className={styles.row}><h4 ref={heading} tabIndex={-1}>Request a late player</h4><button disabled={disabled} onClick={onClose}>Close late player request</button></div>
    <p>Choose an existing club player or create a new player here. The commissioner must approve the request before this player can join a lineup.</p>
    <p className={styles.muted}>Submitting a request sends no email and does not give permission to email the player.</p>
    <RequestStatus {...resource} />
    {resource.error && <button disabled={resource.loading} onClick={resource.reload}>Retry player search</button>}
    <form onSubmit={event => { event.preventDefault(); void submit(); }}><fieldset disabled={disabled}>
      <div className={styles.toolbar}><button type="button" aria-pressed={mode === "existing"} onClick={() => setMode("existing")}>Choose existing player</button><button type="button" aria-pressed={mode === "new"} onClick={() => setMode("new")}>Create new player</button></div>
      {mode === "new" ? <><p>The club player record and late entry request will be saved together, pending commissioner approval.</p><PoolNewPlayerFields draft={draft} onChange={setDraft} /></> : <>
      <label>Find a late player in this club<input type="search" value={query} maxLength={80} placeholder="Search by name" onChange={event => { setQuery(event.target.value); setOffset(0); }} /></label>
      {newPlayers.length > 0 && <fieldset className={styles.candidates}><legend>Players available for a late request</legend>{newPlayers.map(player => <label key={player.id} className={styles.choice}><input type="radio" name="late-player" checked={selected != null && String(selected.id) === String(player.id)} onChange={() => setSelected(player)} /><span><strong>{player.name}</strong><span className={styles.playerMeta}>Club {poolRating(player.rating)}{player.league_rating != null ? ` · League ${poolRating(player.league_rating)}` : ""} · {poolGender(player.gender)}</span></span></label>)}</fieldset>}
      {players.length > 0 && newPlayers.length === 0 && <p role="status" className={styles.notice}>Everyone on this page already has a season signup. View their status below, or search for another player.</p>}
      {existingPlayers.length > 0 && <section aria-label="Existing season signups" className={styles.existingSignups}><h5>Already have a season signup</h5>{existingPlayers.map(player => {
        const member = existing.get(String(player.id))!;
        return <div key={player.id} className={styles.existingSignup}><div><strong>{player.name}</strong><p>{existingSignupStatus(member)}</p></div><button type="button" aria-label={`View ${member.name} in player pool`} onClick={() => onViewMember(member)}>View in player pool</button></div>;
      })}</section>}
      {resource.data?.players.length === 0 && <p>No matching club player. Choose Create new player above to add their record and request late entry here.</p>}
      <div className={styles.toolbar}>{offset > 0 && <button type="button" disabled={resource.loading} onClick={() => setOffset(0)}>First club players</button>}{resource.data?.next_offset != null && <button type="button" disabled={resource.loading} onClick={() => setOffset(resource.data!.next_offset!)}>More club players</button>}</div>
      {selected && <p className={styles.notice}><strong>Selected: {selected.name}</strong><br />Current rating eligibility: {sortedPoolDivisions(selected.eligible_divisions).join(", ") || "No eligible division"}.{selectedIsExisting && <> This player is already in the season pool.</>}</p>}
      </>}
      <label>Notes (optional)<textarea maxLength={500} rows={3} value={reason} onChange={event => setReason(event.target.value)} placeholder="Anything the commissioner should know." /></label>
      <button className={styles.primary} type="submit" disabled={disabled || (mode === "new" ? !newPlayer : !selected || selectedIsExisting)}>{mode === "new" ? "Create player and request late entry" : "Submit late player request"}</button>
    </fieldset></form>
  </section>;
}
