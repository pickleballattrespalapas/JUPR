"use client";
import { useEffect, useRef, useState } from "react";
import type { InterclubMeet, RegistrationSeason } from "@/lib/interclubRegistration";
import { availabilityLabel, type AvailabilityResponse, type MeetAvailabilityData, type PoolMember, type SeasonPool } from "@/lib/interclubPlayerPool";
import { usePoolResource } from "./usePoolResource";
import { InvitationEmail } from "./PoolInvitationEmail";
import { RequestStatus, ShareLink } from "./PoolPanelCommon";
import styles from "./playerPool.module.css";

type PoolProps = { root: string; accessToken: string; clubName: string; season: RegistrationSeason };
type MeetProps = { meetRoot: string; accessToken: string; clubName: string; season: RegistrationSeason; meet: InterclubMeet; onResponses?: (responses: AvailabilityResponse[]) => void };
function emailsRoot(root: string) { return root.replace(/\/interclub\/registrations\//, "/interclub/player-pools/") + "/emails"; }
function messageDate(value: string, timezone?: string) { return new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short", ...(timezone ? { timeZone: timezone } : {}) }); }
function localInput(value: string) { const date = new Date(value); return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16); }

export function SeasonPlayerPool(props: PoolProps) { return <SeasonPoolPanel key={props.root} {...props} />; }
function SeasonPoolPanel({ root, accessToken, clubName, season }: PoolProps) {
  const resource = usePoolResource<SeasonPool>(`${root}/pool`, accessToken);
  const [memberFilter, setMemberFilter] = useState("active"), [message, setMessage] = useState("");
  const [invite, setInvite] = useState(false), data = resource.data;
  async function changeOpen() {
    if (!data) return;
    const next = await resource.perform<SeasonPool>(json => json(`${root}/pool`, "PUT", { expected_revision: data.signup.revision, open: !data.signup.open }));
    if (next) { resource.setData(next); setMessage(next.signup.open ? "Season signup is open. Share the link with your players." : "Season signup is closed. Your player pool is saved."); }
  }
  async function updateMember(member: PoolMember, playerId: string | null, status: PoolMember["status"]) {
    const next = await resource.perform<{ member: PoolMember }>(json => json(`${root}/pool/members/${encodeURIComponent(member.id)}`, "PATCH", { expected_revision: member.revision, player_id: playerId, status }));
    if (next) { resource.setData(old => old ? { ...old, members: old.members.map(row => row.id === next.member.id ? next.member : row) } : old); setMessage("Player details updated."); return true; }
    return false;
  }
  return <section className={styles.panel} aria-label="Season player pool">
    <h3>1. Build your season player pool</h3>
    <p>Invite players to register their interest in representing {clubName} in {season.details.name}. One signup adds them to your season pool. They can choose their availability separately for each meet.</p>
    <p className={styles.muted}>Signing up does not reserve a team place. Your club chooses its final lineup for each meet.</p>
    <RequestStatus {...resource} />{message && <p role="status" className={styles.notice}>{message}</p>}
    {data && <>
      <div className={styles.toolbar}><span className={styles.badge}>{data.signup.open ? "Season signup open" : "Season signup closed"}</span><button className={!data.signup.open ? styles.primary : undefined} disabled={resource.disabled} onClick={() => void changeOpen()}>{data.signup.open ? "Close season signup" : "Open season signup"}</button></div>
      {data.signup.url && <ShareLink url={data.signup.url} label="Season signup link" />}
      {data.email_mode === "dry_run" && <p className={styles.notice}>Test environment: no emails are sent. Copy the signup link to try the player form.</p>}
      <button disabled={!data.signup.open || resource.disabled} onClick={() => setInvite(value => !value)}>{invite ? "Close season invitation email" : "Invite club players by email"}</button>
      {invite && data.signup.open && <InvitationEmail root={emailsRoot(root)} accessToken={accessToken} kind="season" />}
      <div className={styles.toolbar}><h4>Season signups ({data.members.filter(member => member.status === "active").length} active)</h4><label>Show players<select value={memberFilter} onChange={event => setMemberFilter(event.target.value)}><option value="active">Active signups</option><option value="withdrawn">Withdrawn signups</option><option value="all">All signups</option></select></label></div>
      {!data.members.some(member => memberFilter === "all" || member.status === memberFilter) && <p>No {memberFilter === "withdrawn" ? "withdrawn " : ""}players have signed up yet.</p>}
      {data.members.filter(member => memberFilter === "all" || member.status === memberFilter).map(member => <MemberCard key={`${member.id}:${member.revision}`} member={member} root={root} accessToken={accessToken} disabled={resource.disabled} onUpdate={updateMember} />)}
    </>}
    <div className={styles.toolbar}><button disabled={resource.busy || resource.loading} onClick={() => { setMessage(""); resource.reload(); }}>Reload player pool</button></div>
  </section>;
}
function MemberCard({ member, root, accessToken, disabled, onUpdate }: { member: PoolMember; root: string; accessToken: string; disabled: boolean; onUpdate: (member: PoolMember, playerId: string | null, status: PoolMember["status"]) => Promise<boolean> }) {
  const [linking, setLinking] = useState(false);
  return <article className={styles.card}>
    <div className={styles.row}><strong>{member.name}</strong><span className={styles.badge}>{member.status === "active" ? "Active signup" : "Withdrawn"}</span></div>
    <p>{member.email} · Divisions: {member.divisions.join(", ") || "None selected"}</p>{member.notes && <p className={styles.notes}>{member.notes}</p>}
    <p className={styles.muted}>{member.player_id ? "Linked to this club’s player record." : "Not linked to a club player record. Link the correct player before choosing them for a roster."}</p>
    <div className={styles.toolbar}><button disabled={disabled} onClick={() => setLinking(value => !value)}>{linking ? "Cancel player link" : member.player_id ? "Change linked player" : "Link club player"}</button><button disabled={disabled} onClick={() => void onUpdate(member, member.player_id, member.status === "active" ? "withdrawn" : "active")}>{member.status === "active" ? "Withdraw from season pool" : "Restore season signup"}</button></div>
    {member.manage_url && <details><summary>Player’s private signup update link</summary><p className={styles.muted}>Share only with {member.name}; this link lets them update or withdraw their season signup.</p><ShareLink url={member.manage_url} label={`Signup update link for ${member.name}`} /></details>}
    {linking && <PlayerLink member={member} root={root} accessToken={accessToken} disabled={disabled} onLink={async playerId => { if (await onUpdate(member, playerId, member.status)) setLinking(false); }} />}
  </article>;
}
function PlayerLink({ member, root, accessToken, disabled, onLink }: { member: PoolMember; root: string; accessToken: string; disabled: boolean; onLink: (playerId: string | null) => Promise<void> }) {
  const [query, setQuery] = useState(member.name.slice(0, 80)), [offset, setOffset] = useState(0);
  const resource = usePoolResource<{ players: { id: string; name: string; starting_rating: number }[]; next_offset: number | null }>(`${root}/players?q=${encodeURIComponent(query)}&offset=${offset}`, accessToken);
  return <div className={styles.card}><label>Find a player in this club<input type="search" value={query} maxLength={80} onChange={event => { setQuery(event.target.value); setOffset(0); }} /></label><RequestStatus {...resource} />
    {resource.data?.players.map(player => <div key={player.id} className={styles.toolbar}><span>{player.name} · Rating {player.starting_rating}</span><button disabled={disabled} onClick={() => void onLink(String(player.id))}>Link {player.name}</button></div>)}
    {resource.data?.players.length === 0 && <p>No matching club player. Add their player record in Players, then return here to link it.</p>}
    <div className={styles.toolbar}>{offset > 0 && <button disabled={resource.loading} onClick={() => setOffset(0)}>First players</button>}{resource.data?.next_offset != null && <button disabled={resource.loading} onClick={() => setOffset(resource.data!.next_offset!)}>More players</button>}{member.player_id && <button disabled={disabled} onClick={() => void onLink(null)}>Unlink player record</button>}</div>
  </div>;
}
export function MeetAvailability(props: MeetProps) { return <MeetAvailabilityPanel key={props.meetRoot} {...props} />; }
function MeetAvailabilityPanel({ meetRoot, accessToken, clubName, season, meet, onResponses }: MeetProps) {
  const root = meetRoot.replace(/\/meets\/[^/]+$/, ""), resource = usePoolResource<MeetAvailabilityData>(`${meetRoot}/availability`, accessToken);
  const [deadline, setDeadline] = useState(""), [invite, setInvite] = useState(false), [filter, setFilter] = useState("all"), [message, setMessage] = useState("");
  const responseCallback = useRef(onResponses); responseCallback.current = onResponses;
  useEffect(() => { if (resource.data) { setDeadline(resource.data.settings.deadline ? localInput(resource.data.settings.deadline) : ""); responseCallback.current?.(resource.data.responses); } }, [resource.data]);
  const data = resource.data, upcoming = new Date(meet.starts_at).getTime() > Date.now();
  const deadlinePassed = !!data?.settings.deadline && new Date(data.settings.deadline).getTime() <= Date.now();
  async function save(open: boolean) {
    const value = open ? deadline : data?.settings.deadline;
    if (!data || !value) return;
    const time = new Date(value); if (!Number.isFinite(time.getTime())) return;
    const next = await resource.perform<MeetAvailabilityData>(json => json(`${meetRoot}/availability`, "PUT", { expected_revision: data.settings.revision, open, deadline: time.toISOString() }));
    if (next) { resource.setData(next); setMessage(open ? "Meet replies are open. Choose players from your season pool to invite." : "Meet replies are closed. Existing responses are saved."); }
  }
  const visible = data?.responses.filter(row => filter === "all" || (filter === "withdrawn" ? row.member_status === "withdrawn" : row.member_status === "active" && row.status === filter)) || [];
  return <section className={styles.panel} aria-label="Meet player availability"><h3>2. Invite players for this meet</h3><p>{clubName} · {messageDate(meet.starts_at, season.details.timezone)}</p>
    <p>A week or two before the meet, invite players from your season pool to say whether they can play. You still choose the final team roster.</p><RequestStatus {...resource} />{message && <p role="status" className={styles.notice}>{message}</p>}
    {data && <>
      <form onSubmit={event => { event.preventDefault(); void save(true); }}><fieldset disabled={resource.disabled || !upcoming}><div className={styles.toolbar}>
        <label>Reply deadline (your device’s local time)<input type="datetime-local" required max={localInput(meet.starts_at)} value={deadline} onChange={event => setDeadline(event.target.value)} /></label><button className={styles.primary} disabled={!deadline} type="submit">{data.settings.open ? "Save reply deadline" : "Open meet replies"}</button>{data.settings.open && <button type="button" onClick={() => void save(false)}>Close meet replies</button>}
      </div></fieldset></form>
      {!upcoming && <p>This meet has started. Availability responses are kept for reference.</p>}<p className={styles.muted}>{data.settings.open && deadlinePassed ? "Reply deadline has passed" : data.settings.open ? "Replies are open" : "Replies are closed"}{data.settings.deadline ? ` · Deadline: ${messageDate(data.settings.deadline, season.details.timezone)}` : ""}.</p>
      {data.email_mode === "dry_run" && <p className={styles.notice}>Test environment: prepare test invitations below, then copy each player’s private response link. No emails are sent.</p>}
      <button disabled={!data.settings.open || resource.disabled || !upcoming || deadlinePassed} onClick={() => setInvite(value => !value)}>{invite ? "Close meet invitation email" : "Choose players to invite"}</button>
      {invite && data.settings.open && !deadlinePassed && <InvitationEmail root={emailsRoot(root)} accessToken={accessToken} kind="meet" meetId={meet.id} onPrepared={resource.reload} />}
      <h4 style={{ marginTop: "1.5rem" }}>Player responses</h4><p>{(["available", "maybe", "unavailable", "invited"] as const).map(status => `${data.responses.filter(row => row.member_status === "active" && row.status === status).length} ${availabilityLabel[status].toLowerCase()}`).join(" · ")}</p>
      <label>Filter responses<select value={filter} onChange={event => setFilter(event.target.value)}><option value="all">All responses</option>{Object.entries(availabilityLabel).map(([value, label]) => <option key={value} value={value}>{label}</option>)}<option value="withdrawn">Withdrawn from season</option></select></label>
      {!data.responses.length && <p>No players have been invited to this meet yet. Open replies, then choose players from your season pool.</p>}{data.responses.length > 0 && !visible.length && <p>No players match this response filter.</p>}
      {visible.map(row => <article key={row.id} className={styles.card}><div className={styles.row}><strong>{row.name}</strong><span className={styles.badge}>{row.member_status === "withdrawn" ? "Withdrawn from season" : availabilityLabel[row.status]}</span></div>
        <p>{row.email} · {row.divisions.join(", ")}</p>{row.notes && <p className={styles.notes}>{row.notes}</p>}{!row.player_id && <p className={styles.muted}>Link this signup to a club player in the season pool before selecting them for a roster.</p>}
        {row.response_url && row.member_status === "active" && upcoming && <details><summary>Player’s private reply link</summary><p className={styles.muted}>Share only with {row.name}; this link lets them change their response.</p><ShareLink url={row.response_url} label={`Reply link for ${row.name}`} /></details>}
      </article>)}
    </>}
    <div className={styles.toolbar}><button disabled={resource.busy || resource.loading} onClick={() => { setMessage(""); resource.reload(); }}>Reload availability</button></div>
  </section>;
}
