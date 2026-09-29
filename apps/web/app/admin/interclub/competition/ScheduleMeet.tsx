"use client";

import { useEffect, useRef, useState } from "react";
import { readBrowserWorkspace } from "@/lib/adminWorkspace";
import { championshipQualifications, CompetitionContext, CompetitionMeet, CompetitionPhase, competitionRequest, phaseLabels } from "@/lib/interclubCompetition";
import { meetLocalTime, meetUtcTime, meetSeasonDateIssue, seasonDateLabel } from "@/lib/interclubSetup";
import { registrationMeetPlanning } from "@/lib/interclubRegistrationWindow";
import styles from "./competition.module.css";

export default function ScheduleMeet({ root, clubId, accessToken, context, disabled, meet, fixedPhase, defaultOpen = false, onScheduled, onCancel, onReload }: {
  root: string; clubId: string; accessToken: string; context: CompetitionContext; disabled: boolean;
  meet?: CompetitionMeet; fixedPhase?: Exclude<CompetitionPhase, "regular">; defaultOpen?: boolean; onScheduled: (meet: CompetitionMeet, message?: string) => void; onCancel?: () => void; onReload?: () => void;
}) {
  const timezone = context.season.details.timezone;
  const [phase, setPhase] = useState<CompetitionPhase>(meet?.competition_phase || fixedPhase || "regular");
  const [host, setHost] = useState(meet?.host_club_id || ""), [clubs, setClubs] = useState<string[]>(meet?.club_ids || []);
  const [startsAt, setStartsAt] = useState(() => meetLocalTime(meet?.starts_at || null, timezone));
  const [deadline, setDeadline] = useState(() => meetLocalTime(meet?.roster_deadline || null, timezone));
  const [courts, setCourts] = useState(meet?.courts || 4);
  // Preserve the stored scheduling window without requiring an organizer estimate.
  const minutes = meet?.duration_minutes || 180;
  const qualifications = championshipQualifications(context);
  const qualifiedClubs = Array.from(new Set(qualifications.flatMap(qualification => phase === "final"
    ? qualification.status === "ready" && qualification.qualifiers.length === 2 ? qualification.qualifiers : []
    : qualification.status === "playoff_required" ? qualification.playoff_required : [])));
  const participatingClubs = meet || phase === "regular" ? clubs : qualifiedClubs;
  const availableHosts = context.clubs.filter(club => phase === "regular" || qualifiedClubs.includes(club.id));
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
    if (!Number.isInteger(courts) || courts < 1 || courts > 100) { setError("Choose 1–100 courts."); return; }
    const seasonIssue = meetSeasonDateIssue(context.season.details, start, minutes);
    if (seasonIssue) { setError(seasonIssue); return; }
    if (!meet && (!host || !participatingClubs.includes(host) || participatingClubs.length < 2 || (phase === "regular" && participatingClubs.length > 4))) { setError(phase === "regular" ? "Choose a host and two to four participating clubs for a regular meet." : "Choose a host from the qualified clubs shown below."); return; }
    const controller = new AbortController(); pending.current = controller; setBusy(true); setError("");
    try {
      const timing = { starts_at: start, roster_deadline: cutoff, courts: courtsEditable ? courts : meet!.courts, duration_minutes: minutes };
      const newMeet = { host_club_id: host, club_ids: participatingClubs, competition_phase: phase, ...timing };
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
  let dateHint = "";
  try {
    const start = meetUtcTime(startsAt, timezone);
    if (start && Number.isInteger(minutes) && minutes >= 30 && minutes <= 180) dateHint = meetSeasonDateIssue(context.season.details, start, minutes) || "";
  } catch { /* The date input may still be incomplete. Validate it on submit. */ }
  return <details open={defaultOpen || Boolean(meet) || undefined} className={`${styles.page} ${styles.card}`}><summary>{meet ? "Edit meet date" : fixedPhase === "final" ? "Schedule championship meet" : fixedPhase === "qualifier" ? "Schedule qualifying playoff" : "Add meet"}</summary>
    <p>{meet ? `Hosted by ${clubName(meet.host_club_id)} · ${meet.club_ids.map(clubName).join(", ")}.` : "Choose a date, host and participating clubs. Clubs can then choose their lineups for this meet."} All times use {timezone}.</p>
    <p><strong>Season dates: {seasonDateLabel(context.season.details.start_date)} – {seasonDateLabel(context.season.details.end_date)}.</strong> The meet must start and finish within these dates.</p>
    {dateHint && dateHint !== error && <p role="status" className={styles.warning}>{dateHint}</p>}
    {error && <p role="alert" className={styles.error}>{error}</p>}
    {!permitted && <p className={styles.warning}>{meet?.schedule_locked_reason || "The commissioner can change the meet schedule after season registration closes."}</p>}
    <form onSubmit={event => { event.preventDefault(); void schedule(); }}><fieldset className={styles.gameFields} disabled={disabled || busy || blocked || !permitted}>
      <div className={styles.twoColumns}>
        {!meet && <>{!fixedPhase && <label>Competition<select aria-label="Competition" value={phase} onChange={event => { setPhase(event.target.value as CompetitionPhase); setHost(""); }}>{Object.entries(phaseLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>}
          <label>Host club<select aria-label="Host club" required value={host} onChange={event => { setHost(event.target.value); setClubs(old => event.target.value ? Array.from(new Set([...old, event.target.value])) : old); }}><option value="">Choose host</option>{availableHosts.map(club => <option key={club.id} value={club.id}>{club.name}</option>)}</select></label></>}
        <label>Meet date and time ({timezone})<input aria-label="Meet date and time" required type="datetime-local" min={`${context.season.details.start_date}T00:00`} max={`${context.season.details.end_date}T23:59`} value={startsAt} onChange={event => changeStart(event.target.value)} /></label>
        <label>Roster deadline ({timezone})<input aria-label="Roster deadline" required type="datetime-local" max={startsAt || undefined} disabled={!deadlineEditable} value={deadline} onChange={event => setDeadline(event.target.value)} /></label>
        <label>Courts<input aria-label="Courts" required type="number" min={1} max={100} disabled={!courtsEditable} value={courts} onChange={event => setCourts(Number(event.target.value))} /></label>
      </div>
      {!deadlineEditable && <p className={styles.muted}>The roster deadline is locked to preserve submitted lineups and eligibility ratings.</p>}
      {!courtsEditable && <p className={styles.muted}>Court count is locked because pairings have been prepared.</p>}
      {!meet && phase === "regular" && <fieldset className={styles.players}><legend>Participating clubs (two to four)</legend>{context.clubs.map(club => <label key={club.id} className={styles.check}><input aria-label={club.name} type="checkbox" checked={clubs.includes(club.id)} disabled={club.id === host} onChange={event => setClubs(old => event.target.checked ? [...old, club.id] : old.filter(id => id !== club.id))} />{club.name}</label>)}</fieldset>}
      {!meet && phase !== "regular" && <section aria-label="Qualified clubs" className={styles.notice}>
        <h3>{phase === "final" ? "Finalists from season results" : "Clubs needing a qualifying playoff"}</h3>
        {qualifications.map(qualification => <p key={qualification.division}><strong>{qualification.division}: </strong>{phase === "final"
          ? qualification.status === "ready" && qualification.qualifiers.length === 2 ? qualification.qualifiers.map(clubName).join(" vs ") : qualification.status === "playoff_required" ? "Awaiting qualifying playoff — schedule this final after the tie is resolved." : "Not enough eligible clubs to confirm finalists."
          : qualification.status === "playoff_required" ? qualification.playoff_required.map(clubName).join(", ") : "No playoff needed."}</p>)}
        <p>{participatingClubs.length >= 2 ? "These clubs are included automatically. Choose their players in meet lineups after scheduling, then prepare each skill-level matchup." : "Approve the qualifying results before scheduling this meet."}</p>
      </section>}
      {meet && <p className={styles.warning}>Changing the date or time resets availability answers and private reply links. Reopen availability and send fresh invitations afterward. Existing lineups are kept; each club should confirm its players for the new date.</p>}
      <div className={styles.toolbar}><button className={styles.primary} type="submit" disabled={!meet && (participatingClubs.length < 2 || phase === "regular" && participatingClubs.length > 4 || !participatingClubs.includes(host))}>{busy ? "Saving…" : meet ? "Save meet date" : "Add meet"}</button></div>
    </fieldset></form>
    <div className={styles.toolbar}>{onCancel && <button disabled={busy} type="button" onClick={onCancel}>Cancel</button>}{blocked && onReload && <button disabled={busy} type="button" onClick={onReload}>Reload schedule</button>}</div>
  </details>;
}
