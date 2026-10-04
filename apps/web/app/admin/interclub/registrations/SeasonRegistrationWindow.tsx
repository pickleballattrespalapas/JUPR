"use client";

import { useEffect, useRef, useState } from "react";
import { apiError, type RegistrationSeason } from "@/lib/interclubRegistration";
import { meetLocalTime, meetUtcTime } from "@/lib/interclubSetup";
import { registrationPhase } from "@/lib/interclubRegistrationWindow";
import styles from "./registrations.module.css";

export default function SeasonRegistrationWindow({ root, accessToken, season, commissioner, firstMeetAt, onSaved, onReload }: {
  root: string; accessToken: string; season: RegistrationSeason; commissioner: boolean;
  firstMeetAt?: string | null;
  onSaved: (season: RegistrationSeason) => void; onReload: () => void;
}) {
  const registration = season.registration;
  const timezone = season.details.timezone;
  const [editing, setEditing] = useState(false);
  const [opens, setOpens] = useState(() => meetLocalTime(registration?.opens_at || null, timezone));
  const [closes, setCloses] = useState(() => meetLocalTime(registration?.closes_at || null, timezone));
  const [error, setError] = useState(""), [busy, setBusy] = useState(false), [blocked, setBlocked] = useState(false), [saved, setSaved] = useState(false);
  const token = useRef(accessToken); token.current = accessToken;
  const pending = useRef(false), request = useRef<AbortController | null>(null);
  const editRevision = useRef(registration?.revision || 0);
  useEffect(() => () => request.current?.abort(), [root]);
  useEffect(() => {
    if (editing && editRevision.current !== (registration?.revision || 0)) {
      setBlocked(true); setError("The commissioner’s registration dates changed while you were editing. Your draft is kept below. Reload registration dates before saving."); return;
    }
    setOpens(meetLocalTime(registration?.opens_at || null, timezone));
    setCloses(meetLocalTime(registration?.closes_at || null, timezone));
    setBlocked(false);
  }, [registration?.opens_at, registration?.closes_at, registration?.revision, timezone, editing]);
  async function save() {
    if (pending.current || blocked || !commissioner) return;
    let opensAt: string | null, closesAt: string | null;
    try { opensAt = meetUtcTime(opens, timezone); closesAt = meetUtcTime(closes, timezone); }
    catch { setError(`Choose valid registration times in ${timezone}. A time that repeats or is skipped during a clock change cannot be used.`); return; }
    if (!opensAt || !closesAt) { setError(`Choose valid opening and closing times in ${timezone}. A time that repeats or is skipped during a clock change cannot be used.`); return; }
    if (Date.parse(closesAt) <= Date.parse(opensAt)) { setError("Registration must close after it opens."); return; }
    if (firstMeetAt && Date.parse(closesAt) > Date.parse(firstMeetAt)) { setError("Registration must close by the first scheduled meet. Choose an earlier closing time."); return; }
    pending.current = true; setBusy(true); setError(""); setSaved(false);
    const controller = new AbortController(); request.current = controller;
    try {
      const response = await fetch(`${root}/registration-window`, { method: "PUT", signal: controller.signal,
        headers: { Authorization: `Bearer ${token.current}`, "Content-Type": "application/json" },
        body: JSON.stringify({ expected_revision: registration?.revision || 0, opens_at: opensAt, closes_at: closesAt }) });
      const result = await response.json();
      if (controller.signal.aborted) return;
      if (!response.ok) {
        if ([401, 403, 409, 503].includes(response.status)) setBlocked(true);
        throw new Error(apiError(result, "Unable to save registration dates."));
      }
      if (result.season?.id !== season.id || !result.season?.registration) { setBlocked(true); throw new Error("Could not confirm the registration dates. Reload the season to check them."); }
      onSaved(result.season); setEditing(false); setSaved(true);
    } catch (cause) {
      if (!controller.signal.aborted) {
        if (cause instanceof TypeError || cause instanceof SyntaxError) setBlocked(true);
        setError(cause instanceof Error && !(cause instanceof TypeError) && !(cause instanceof SyntaxError) ? cause.message : "Could not confirm the registration dates. Reload the season to check them.");
      }
    } finally { pending.current = false; if (!controller.signal.aborted) setBusy(false); }
  }
  const format = (iso: string) => new Date(iso).toLocaleString(undefined, { timeZone: timezone, dateStyle: "medium", timeStyle: "short" });
  const status = registrationPhase(registration);
  return <section className={styles.card} aria-labelledby="season-registration-heading">
    <h3 id="season-registration-heading">Season registration</h3>
    <p><strong>{({ unconfigured: "Registration dates not set", scheduled: "Registration has not opened yet", open: "Registration is open", closed: "Registration is closed" })[status]}</strong></p>
    {registration?.opens_at && registration.closes_at ? <p>Opens: {format(registration.opens_at)}<br />Closes: {format(registration.closes_at)}<br /><span className={styles.muted}>All times in {timezone}.</span></p>
      : <p>The league commissioner needs to set the opening and closing dates before players can register.</p>}
    <p className={styles.muted}>The commissioner sets one registration period for every club. Meet signup, lineups, and meet settings open after registration closes.</p>
    {firstMeetAt && <p className={styles.muted}>First scheduled meet: {format(firstMeetAt)} ({timezone}). Registration must close by this time.</p>}
    {saved && <p role="status">Registration dates saved for every club in this league.</p>}
    {commissioner && !editing && <button onClick={() => { editRevision.current = registration?.revision || 0; setEditing(true); setSaved(false); setError(""); }}>{registration?.opens_at ? "Edit registration dates" : "Set registration dates"}</button>}
    {commissioner && editing && <form className={styles.form} onSubmit={event => { event.preventDefault(); void save(); }}>
      <fieldset disabled={busy || blocked} className={styles.form}><legend>Commissioner registration dates · {timezone}</legend>
        <label>Registration opens ({timezone})<input required type="datetime-local" aria-label="Registration opens" value={opens} onChange={event => setOpens(event.target.value)} /></label>
        <label>Registration closes ({timezone})<input required type="datetime-local" aria-label="Registration closes" max={firstMeetAt ? meetLocalTime(firstMeetAt, timezone) : undefined} value={closes} onChange={event => setCloses(event.target.value)} /></label>
        <div className={styles.toolbar}><button className={styles.primary} type="submit">{busy ? "Saving dates…" : "Save registration dates"}</button><button type="button" onClick={() => { setEditing(false); setError(""); }}>Cancel</button></div>
      </fieldset>
    </form>}
    {error && <p className={styles.notice} role="alert">{error}</p>}
    {blocked && <button disabled={busy} onClick={onReload}>Reload registration dates</button>}
  </section>;
}
