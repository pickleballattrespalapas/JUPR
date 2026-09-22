"use client";

import { useEffect, useRef, useState } from "react";
import { readBrowserWorkspace } from "@/lib/adminWorkspace";
import { CompetitionContext, CompetitionMeet, CompetitionPhase, competitionRequest, phaseLabels } from "@/lib/interclubCompetition";
import { meetLocalTime, meetUtcTime } from "@/lib/interclubSetup";
import { registrationMeetPlanning } from "@/lib/interclubRegistrationWindow";
import styles from "./competition.module.css";

export default function ScheduleMeet({ root, clubId, accessToken, context, disabled, meet, defaultOpen = false, onScheduled, onCancel, onReload }: {
  root: string; clubId: string; accessToken: string; context: CompetitionContext; disabled: boolean;
  meet?: CompetitionMeet; defaultOpen?: boolean; onScheduled: (meet: CompetitionMeet, message?: string) => void; onCancel?: () => void; onReload?: () => void;
}) {
  const timezone = context.season.details.timezone;
  const [phase, setPhase] = useState<CompetitionPhase>(meet?.competition_phase || "regular");
  const [host, setHost] = useState(meet?.host_club_id || ""), [clubs, setClubs] = useState<string[]>(meet?.club_ids || []);
  const [startsAt, setStartsAt] = useState(() => meetLocalTime(meet?.starts_at || null, timezone));
  const [deadline, setDeadline] = useState(() => meetLocalTime(meet?.roster_deadline || null, timezone));
  const [courts, setCourts] = useState(meet?.courts || 4), [minutes, setMinutes] = useState(meet?.duration_minutes || 180);
  const [error, setError] = useState(""), [busy, setBusy] = useState(false), [blocked, setBlocked] = useState(false);
  const token = useRef(accessToken); token.current = accessToken;
  const pending = useRef<AbortController | null>(null);
  const expectedRevision = useRef(meet?.revision);
  const meetRevision = meet?.revision;
  const createRequest = useRef<{ signature: string; id: string } | null>(null);
  const permitted = context.is_organizer && registrationMeetPlanning(context.season.registration) && (!meet || meet.schedule_editable === true);
  const deadlineEditable = !meet || meet.schedule_deadline_editable === true;
  const courtsEditable = !meet || meet.courts_editable === true;
  useEffect(() => () => pending.current?.abort(), [root]);
  useEffect(() => {
    if (meetRevision !== undefined && meetRevision !== expectedRevision.current) {
      setBlocked(true); setError("This meet changed while you were editing. Reload the schedule before saving. Your draft is kept below.");
    }
  }, [meetRevision]);
  function changeStart(value: string) {
    if (deadlineEditable && (!deadline || deadline === startsAt)) setDeadline(value);
    setStartsAt(value);
  }
  async function schedule() {
    if (pending.current || disabled || blocked || !permitted) return;
    if (readBrowserWorkspace()?.clubId !== clubId) { setBlocked(true); setError("Your selected club changed. Reopen this workspace."); return; }
    let start: string | null, cutoff: string | null;
    try {
      start = meetUtcTime(startsAt, timezone);
      cutoff = deadlineEditable ? meetUtcTime(deadline, timezone) : meet!.roster_deadline;
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Choose valid meet dates."); return; }
    if (!start || !cutoff) { setError("Choose the meet date and roster deadline."); return; }
    if (Date.parse(start) <= Date.now()) { setError("Choose a future meet date."); return; }
    if (Date.parse(cutoff) > Date.parse(start)) { setError("The roster deadline must be no later than the meet."); return; }
    if (deadlineEditable && Date.parse(cutoff) <= Date.now()) { setError("Choose a future roster deadline."); return; }
    if (!Number.isInteger(courts) || courts < 1 || courts > 100 || !Number.isInteger(minutes) || minutes < 30 || minutes > 180) { setError("Choose 1–100 courts and a duration of 30–180 minutes."); return; }
    if (!meet && (!host || !clubs.includes(host) || clubs.length < 2 || (phase === "regular" && clubs.length > 4))) { setError("Choose a host and two to four participating clubs for a regular meet."); return; }
    const controller = new AbortController(); pending.current = controller; setBusy(true); setError("");
    try {
      const timing = { starts_at: start, roster_deadline: cutoff, courts: courtsEditable ? courts : meet!.courts, duration_minutes: minutes };
      const newMeet = { host_club_id: host, club_ids: clubs, competition_phase: phase, ...timing };
      if (!meet && createRequest.current?.signature !== JSON.stringify(newMeet)) createRequest.current = { signature: JSON.stringify(newMeet), id: crypto.randomUUID() };
      const result = await competitionRequest<{ meet: CompetitionMeet; availability_reset_count?: number; publication_review_required?: boolean }>(meet ? `${root}/meets/${encodeURIComponent(meet.id)}/schedule` : `${root}/meets`, token.current, controller.signal,
        meet ? "PUT" : "POST", meet ? { expected_revision: expectedRevision.current, ...timing } : { ...newMeet, request_id: createRequest.current!.id });
      if (controller.signal.aborted) return;
      if (!result.meet?.id || result.meet.season_id !== context.season.id || (meet && result.meet.id !== meet.id)) {
        setBlocked(true); throw new Error("Could not confirm the saved meet. Reload the schedule before trying again.");
      }
      const dateChanged = meet && (Date.parse(meet.starts_at) !== Date.parse(start) || meet.duration_minutes !== minutes);
      const message = `${meet ? "Meet schedule saved." : "Meet added."}${dateChanged ? " Reopen availability and send fresh invitations for the new date, then confirm each club’s existing lineup." : ""}${result.publication_review_required ? " Review and publish the updated public schedule." : ""}`;
      onScheduled(result.meet, message);
    } catch (cause) {
      if (!controller.signal.aborted) {
        const status = (cause as { status?: number })?.status;
        if (!status || [401, 403, 409].includes(status) || status >= 500) setBlocked(true);
        setError(cause instanceof Error && !(cause instanceof TypeError) ? cause.message : "Could not confirm the saved meet. Reload the schedule before trying again.");
      }
    } finally { pending.current = null; if (!controller.signal.aborted) setBusy(false); }
  }
  const clubName = (id: string) => context.clubs.find(club => club.id === id)?.name || id;
  return <details open={defaultOpen || Boolean(meet) || undefined} className={`${styles.page} ${styles.card}`}><summary>{meet ? "Edit meet date" : "Add meet"}</summary>
    <p>{meet ? `Hosted by ${clubName(meet.host_club_id)} · ${meet.club_ids.map(clubName).join(", ")}.` : "Choose a date, host and participating clubs. Clubs can then choose their lineups for this meet."} All times use {timezone}.</p>
    {error && <p role="alert" className={styles.error}>{error}</p>}
    {!permitted && <p className={styles.warning}>{meet?.schedule_locked_reason || "The commissioner can change the meet schedule after season registration closes."}</p>}
    <form onSubmit={event => { event.preventDefault(); void schedule(); }}><fieldset className={styles.gameFields} disabled={disabled || busy || blocked || !permitted}>
      <div className={styles.twoColumns}>
        {!meet && <><label>Competition<select aria-label="Competition" value={phase} onChange={event => setPhase(event.target.value as CompetitionPhase)}>{Object.entries(phaseLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          <label>Host club<select aria-label="Host club" required value={host} onChange={event => { setHost(event.target.value); setClubs(old => event.target.value ? Array.from(new Set([...old, event.target.value])) : old); }}><option value="">Choose host</option>{context.clubs.map(club => <option key={club.id} value={club.id}>{club.name}</option>)}</select></label></>}
        <label>Meet date and time ({timezone})<input aria-label="Meet date and time" required type="datetime-local" value={startsAt} onChange={event => changeStart(event.target.value)} /></label>
        <label>Roster deadline ({timezone})<input aria-label="Roster deadline" required type="datetime-local" max={startsAt || undefined} disabled={!deadlineEditable} value={deadline} onChange={event => setDeadline(event.target.value)} /></label>
        <label>Courts<input aria-label="Courts" required type="number" min={1} max={100} disabled={!courtsEditable} value={courts} onChange={event => setCourts(Number(event.target.value))} /></label>
        <label>Duration (minutes)<input aria-label="Duration (minutes)" required type="number" min={30} max={180} value={minutes} onChange={event => setMinutes(Number(event.target.value))} /></label>
      </div>
      {!deadlineEditable && <p className={styles.muted}>The roster deadline is locked to preserve submitted lineups and eligibility ratings.</p>}
      {!courtsEditable && <p className={styles.muted}>Court count is locked because pairings have been prepared.</p>}
      {!meet && <fieldset className={styles.players}><legend>Participating clubs{phase === "regular" ? " (two to four)" : " (all clubs playing finals or playoffs)"}</legend>{context.clubs.map(club => <label key={club.id} className={styles.check}><input aria-label={club.name} type="checkbox" checked={clubs.includes(club.id)} disabled={club.id === host} onChange={event => setClubs(old => event.target.checked ? [...old, club.id] : old.filter(id => id !== club.id))} />{club.name}</label>)}</fieldset>}
      {meet && <p className={styles.warning}>Changing the date, time or duration resets availability answers and private reply links. Reopen availability and send fresh invitations afterward. Existing lineups are kept; each club should confirm its players for the new date.</p>}
      <div className={styles.toolbar}><button className={styles.primary} type="submit" disabled={!meet && (clubs.length < 2 || phase === "regular" && clubs.length > 4 || !host)}>{busy ? "Saving…" : meet ? "Save meet date" : "Add meet"}</button></div>
    </fieldset></form>
    <div className={styles.toolbar}>{onCancel && <button disabled={busy} type="button" onClick={onCancel}>Cancel</button>}{blocked && onReload && <button disabled={busy} type="button" onClick={onReload}>Reload schedule</button>}</div>
  </details>;
}
