"use client";
import { useEffect, useRef, useState } from "react";
import MeetSignupQueues from "@/components/MeetSignupQueues";
import { MeetSignupBoard, MeetSignupEntry, meetSignupOpen } from "@/lib/interclubMeetSignup";
import { SignupPlayer, signupMeetTime } from "@/lib/interclubPlayerSignup";
import { meetLocalTime, meetUtcTime } from "@/lib/interclubSetup";
import { usePoolResource } from "./usePoolResource";
import { RequestStatus, ShareLink } from "./PoolPanelCommon";
import styles from "./playerPool.module.css";

export default function MeetSignupPanel({ root, accessToken, onAutomatic, onChanged }: {
  root: string; accessToken: string; onAutomatic: (active: boolean) => void; onChanged: () => void;
}) {
  const resource = usePoolResource<MeetSignupBoard>(`${root}/signup`, accessToken);
  const { data, disabled } = resource;
  const [deadline, setDeadline] = useState(""), [query, setQuery] = useState("");
  const [players, setPlayers] = useState<SignupPlayer[]>([]), [playerId, setPlayerId] = useState(""), [division, setDivision] = useState("");
  const [lookupError, setLookupError] = useState(""), [lookupBusy, setLookupBusy] = useState(false);
  const [now, setNow] = useState(Date.now());
  const callback = useRef(onAutomatic); callback.current = onAutomatic;
  const token = useRef(accessToken); token.current = accessToken;
  const addRequest = useRef<{ key: string; id: string } | null>(null);
  const open = Boolean(data && meetSignupOpen(data, now));
  useEffect(() => { callback.current(open); }, [open]);
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(timer); }, []);
  useEffect(() => {
    setPlayers([]); setPlayerId(""); setLookupError("");
    if (!open || query.trim().length < 2) { setLookupBusy(false); return; }
    const controller = new AbortController(); setLookupBusy(true);
    const timer = setTimeout(() => {
      fetch(`${root}/players?q=${encodeURIComponent(query.trim())}`, { headers: { Authorization: `Bearer ${token.current}` }, signal: controller.signal, cache: "no-store" })
        .then(async response => { const value = await response.json(); if (!response.ok) throw new Error("Could not find season players. Try again."); if (!controller.signal.aborted) setPlayers(value.players); })
        .catch(() => { if (!controller.signal.aborted) setLookupError("Could not find season players. Try again."); })
        .finally(() => { if (!controller.signal.aborted) setLookupBusy(false); });
    }, 250);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [open, query, root]);
  async function settings(nextOpen: boolean) {
    if (!data) return;
    await resource.perform(async json => {
      const cutoff = deadline ? meetUtcTime(deadline, data.season.timezone) : data.signup.deadline;
      const result = await json<MeetSignupBoard>(`${root}/signup`, "PUT", { open: nextOpen, expected_revision: data.signup.revision, expected_meet_revision: data.meet.revision, deadline: cutoff });
      resource.setData(result); callback.current(meetSignupOpen(result)); onChanged();
    });
  }
  async function action(body: object) {
    await resource.perform(async json => {
      const result = await json<MeetSignupBoard>(`${root}/signup/actions`, "POST", body);
      resource.setData(result); onChanged();
    });
  }
  function entryActions(entry: MeetSignupEntry) {
    const vacancy = data!.entries.filter(row => row.status === "active" && row.division === entry.division && row.gender === entry.gender && row.placement === "confirmed").length < 2;
    return <details><summary>Manage {entry.name}</summary>
      {entry.email && <p>{entry.email}</p>}
      {entry.manage_url && <ShareLink url={entry.manage_url} label="Private player link" />}
      {open && <div className={styles.toolbar}>
        {entry.priority === "play_up" && entry.placement === "waitlist" && vacancy && <button disabled={disabled} onClick={() => void action({ action: "promote", id: entry.id, expected_revision: entry.revision })}>Approve into open spot</button>}
        <button disabled={disabled} onClick={() => void action({ action: "remove", id: entry.id, expected_revision: entry.revision })}>Remove signup</button>
      </div>}
    </details>;
  }
  return <section className={styles.panel} aria-label="Meet signup and substitute pool">
    <h3>Share a link to fill this meet</h3>
    <p>The first two women and two men in each division’s rating band get spots for your club. Everyone after them joins the substitute pool in registration order. Lower-rated players can sign up to play up and wait behind players in the rating band.</p>
    <RequestStatus error={resource.error} blocked={resource.blocked} loading={resource.loading} />
    {data && <>
      {data.signup.schedule_changed && <p className={styles.error}>The schedule changed. Players need to register again for the new date. Existing manual lineup changes are protected; if reopening is blocked, close signup and manage the team in Lineups.</p>}
      {open ? <>
        <p className={styles.notice}><strong>Signup is open.</strong> A team appears in Lineups as soon as all four spots are filled. Withdrawals promote the next eligible substitute automatically until {signupMeetTime(data.signup.deadline, data.season.timezone)}.</p>
        {data.signup.url && <ShareLink url={data.signup.url} label="Share this meet signup link with your players" openLabel="Open player signup page" />}
        <p className={styles.muted}>Players playing up stay waitlisted until you approve them into a vacancy. Rating-band players keep priority. Close signup before making manual lineup changes.</p>
        <div className={styles.toolbar}><button disabled={disabled} onClick={() => void action({ action: "refresh" })}>Refresh signups and lineups</button>
          <button disabled={disabled} onClick={() => void settings(false)}>Close signup and manage lineups</button></div>
        <details className={styles.card}><summary>Add someone who confirmed with you</summary>
          <p>This uses the same rating priority and registration order as the player link.</p>
          <form onSubmit={event => { event.preventDefault(); const selectedDivision = division || data.season.divisions[0]; const key = `${playerId}:${selectedDivision}`; if (addRequest.current?.key !== key) addRequest.current = { key, id: crypto.randomUUID() }; void action({ action: "add", player_id: playerId, division: selectedDivision, request_id: addRequest.current.id }); }}>
            <label>Find an approved season player<input value={query} disabled={disabled} onChange={event => setQuery(event.target.value)} /></label>
            {lookupBusy && <p role="status">Finding players…</p>}{lookupError && <p role="alert">{lookupError}</p>}
            {players.length > 0 && <label>Player<select required disabled={disabled} value={playerId} onChange={event => setPlayerId(event.target.value)}><option value="">Choose player</option>{players.map(player => <option key={player.id} value={player.id}>{player.name} · {player.league_rating ?? player.rating}</option>)}</select></label>}
            <label>Division<select value={division || data.season.divisions[0]} disabled={disabled} onChange={event => setDivision(event.target.value)}>{data.season.divisions.map(value => <option key={value}>{value}</option>)}</select></label>
            <button disabled={disabled || !playerId || lookupBusy} type="submit">Add meet signup</button>
          </form>
        </details>
      </> : <>
        <p>{data.signup.configured ? "Signup is closed. Existing lineups can be managed in Lineups." : "Open signup to get a link for this meet. Your players must already be approved in the season pool."}</p>
        <label>Signup deadline ({data.season.timezone})<input type="datetime-local" disabled={disabled} max={meetLocalTime(data.meet.roster_deadline, data.season.timezone)} value={deadline || meetLocalTime(data.signup.deadline, data.season.timezone)} onChange={event => setDeadline(event.target.value)} /></label>
        <div className={styles.toolbar}><button className={styles.primary} disabled={disabled || Date.parse(data.meet.roster_deadline) <= now} onClick={() => void settings(true)}>{data.signup.configured ? "Reopen meet signup" : "Open meet signup"}</button></div>
        {data.signup.url && <ShareLink url={data.signup.url} label="Meet signup link (closed)" openLabel="View signup page" />}
      </>}
      {data.entries.some(entry => entry.status === "active") && <MeetSignupQueues board={data} actions={entryActions} />}
    </>}
    <button disabled={resource.busy || resource.loading} onClick={resource.reload}>Reload signup section</button>
  </section>;
}
