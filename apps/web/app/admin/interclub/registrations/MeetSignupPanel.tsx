"use client";
import { useEffect, useRef, useState } from "react";
import MeetSignupQueues, { SignupSlotGender } from "@/components/MeetSignupQueues";
import { MeetSignupBoard, MeetSignupEntry, SignupGender, meetSignupOpen, signupGenderLabels, signupPlacement } from "@/lib/interclubMeetSignup";
import { signupMeetTime } from "@/lib/interclubPlayerSignup";
import { meetLocalTime, meetUtcTime } from "@/lib/interclubSetup";
import { usePoolResource } from "./usePoolResource";
import { RequestStatus, ShareLink } from "./PoolPanelCommon";
import { LineupPlayerChoice } from "@/lib/interclubRegistration";
import MeetSignupPlayerPicker from "./MeetSignupPlayerPicker";
import styles from "./playerPool.module.css";

export default function MeetSignupPanel({ root, accessToken, onAutomatic, onChanged }: {
  root: string; accessToken: string; onAutomatic: (active: boolean) => void; onChanged: () => void;
}) {
  const resource = usePoolResource<MeetSignupBoard>(`${root}/signup`, accessToken);
  const { data, disabled } = resource;
  const [deadline, setDeadline] = useState("");
  const [slot, setSlot] = useState<{ root: string; division: string; gender: SignupSlotGender } | null>(null);
  const [notice, setNotice] = useState("");
  const trigger = useRef<HTMLButtonElement | null>(null);
  const [now, setNow] = useState(Date.now());
  const callback = useRef(onAutomatic); callback.current = onAutomatic;
  const addRequest = useRef<{ key: string; id: string } | null>(null);
  const open = Boolean(data && meetSignupOpen(data, now));
  useEffect(() => { callback.current(open); }, [open]);
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(timer); }, []);
  function closePicker() { setSlot(null); trigger.current?.focus(); }
  async function addPlayer(player: LineupPlayerChoice, gender: SignupGender) {
    if (!slot || slot.root !== root || !open) return;
    const key = `${root}:${player.id}:${slot.division}:${gender}`;
    if (addRequest.current?.key !== key) addRequest.current = { key, id: crypto.randomUUID() };
    await resource.perform(async json => {
      const result = await json<MeetSignupBoard>(`${root}/signup/actions`, "POST", { action: "add", player_id: player.id, division: slot.division, gender, request_id: addRequest.current!.id });
      resource.setData(result); closePicker();
      const entry = result.entries.find(row => row.player_id === player.id && row.status === "active");
      setNotice(entry ? `${player.name}: ${signupPlacement(entry)}.` : "Registration saved."); onChanged();
    });
  }
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
      {entry.declared_gender && <p>Gender: {signupGenderLabels[entry.declared_gender]}</p>}
      {entry.reviewed_gender && <p>Reviewed placement: {entry.reviewed_gender === "female" ? "Women’s place" : "Men’s place"}</p>}
      {open && entry.placement === "review" && entry.gender === "unknown" && <GenderReview entry={entry} disabled={disabled} onReview={gender => void action({ action: "review_gender", id: entry.id, expected_revision: entry.revision, lineup_gender: gender })} />}
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
    {notice && <p role="status" className={styles.notice}>{notice}</p>}
    {data && <>
      {data.signup.schedule_changed && <p className={styles.error}>The schedule changed. Players need to register again for the new date. Existing manual lineup changes are protected; if reopening is blocked, close signup and manage the team in Lineups.</p>}
      {open ? <>
        <p className={styles.notice}><strong>Signup is open.</strong> A team appears in Lineups as soon as all four spots are filled. Withdrawals promote the next eligible substitute automatically until {signupMeetTime(data.signup.deadline, data.season.timezone)}.</p>
        {data.signup.url && <ShareLink url={data.signup.url} label="Share this meet signup link with your players" openLabel="Open player signup page" />}
        <p className={styles.muted}>Players playing up stay waitlisted until you approve them into a vacancy. Rating-band players keep priority. Close signup before making manual lineup changes.</p>
        <div className={styles.toolbar}><button disabled={disabled} onClick={() => void action({ action: "refresh" })}>Refresh signups and lineups</button>
          <button disabled={disabled} onClick={() => void settings(false)}>Close signup and manage lineups</button></div>
      </> : <>
        <p>{data.signup.configured ? "Signup is closed. Existing lineups can be managed in Lineups." : "Open signup to get a link for this meet. Your players must already be approved in the season pool."}</p>
        <label>Signup deadline ({data.season.timezone})<input type="datetime-local" disabled={disabled} max={meetLocalTime(data.meet.roster_deadline, data.season.timezone)} value={deadline || meetLocalTime(data.signup.deadline, data.season.timezone)} onChange={event => setDeadline(event.target.value)} /></label>
        <div className={styles.toolbar}><button className={styles.primary} disabled={disabled || Date.parse(data.meet.roster_deadline) <= now} onClick={() => void settings(true)}>{data.signup.configured ? "Reopen meet signup" : "Open meet signup"}</button></div>
        {data.signup.url && <ShareLink url={data.signup.url} label="Meet signup link (closed)" openLabel="View signup page" />}
      </>}
      <MeetSignupQueues board={data} actions={entryActions} addingDisabled={disabled}
        onAdd={open ? (division, gender, button) => { trigger.current = button; setSlot({ root, division, gender }); setNotice(""); } : undefined}
        picker={(division, gender) => open && slot?.root === root && slot.division === division && slot.gender === gender ? <MeetSignupPlayerPicker key={`${root}:${division}:${gender}`} root={root} accessToken={accessToken} board={data} division={division} gender={gender}
          disabled={disabled} mutationError={resource.error} blocked={resource.blocked} onReload={resource.reload} onAdd={(player, selectedGender) => void addPlayer(player, selectedGender)} onClose={closePicker} /> : null} />
    </>}
    <button disabled={resource.busy || resource.loading} onClick={resource.reload}>Reload signup section</button>
  </section>;
}

function GenderReview({ entry, disabled, onReview }: { entry: MeetSignupEntry; disabled: boolean; onReview: (gender: SignupSlotGender) => void }) {
  const [gender, setGender] = useState<SignupSlotGender | "">("");
  return <form aria-label={`Review placement for ${entry.name}`} onSubmit={event => { event.preventDefault(); if (gender && !disabled) onReview(gender); }}>
    <p>Choose a lineup place after reviewing with the player. Their gender selection stays unchanged.</p>
    <label>Lineup place<select aria-label="Lineup place" required disabled={disabled} value={gender} onChange={event => setGender(event.target.value as SignupSlotGender | "")}>
      <option value="">Choose lineup place</option><option value="female">Women’s place</option><option value="male">Men’s place</option>
    </select></label><button type="submit" disabled={disabled || !gender}>Approve placement</button>
  </form>;
}
