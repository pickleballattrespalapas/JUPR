"use client";

import { useEffect, useRef, useState } from "react";
import { readBrowserWorkspace } from "@/lib/adminWorkspace";
import { CompetitionContext, CompetitionMeet, CompetitionPhase, competitionRequest, fromLocalInput, phaseLabels } from "@/lib/interclubCompetition";
import styles from "./competition.module.css";

export default function ScheduleMeet({ root, clubId, accessToken, context, disabled, onScheduled }: { root: string; clubId: string; accessToken: string; context: CompetitionContext; disabled: boolean; onScheduled: (meet: CompetitionMeet) => void }) {
  const [phase, setPhase] = useState<CompetitionPhase>("final"), [host, setHost] = useState(""), [clubs, setClubs] = useState<string[]>([]), [startsAt, setStartsAt] = useState(""), [deadline, setDeadline] = useState("");
  const [courts, setCourts] = useState(4), [minutes, setMinutes] = useState(180), [error, setError] = useState(""), [busy, setBusy] = useState(false), [blocked, setBlocked] = useState(false);
  const token = useRef(accessToken); token.current = accessToken;
  const pending = useRef<AbortController | null>(null);
  useEffect(() => () => pending.current?.abort(), [root]);
  async function schedule() {
    if (pending.current || disabled || blocked) return;
    if (readBrowserWorkspace()?.clubId !== clubId) { setBlocked(true); setError("Your selected club changed. Reopen this workspace."); return; }
    const controller = new AbortController(); pending.current = controller; setBusy(true); setError("");
    try {
      const result = await competitionRequest<{ meet: CompetitionMeet }>(`${root}/meets`, token.current, controller.signal, "POST", { host_club_id: host, club_ids: clubs, starts_at: fromLocalInput(startsAt), roster_deadline: fromLocalInput(deadline), courts, duration_minutes: minutes, competition_phase: phase });
      if (!controller.signal.aborted) onScheduled(result.meet);
    } catch (cause) {
      if (!controller.signal.aborted) {
        const status = (cause as { status?: number })?.status;
        if (!status || status === 409 || status >= 500) setBlocked(true);
        setError(cause instanceof Error && !(cause instanceof TypeError) ? cause.message : "Could not confirm the new meet. Refresh the season before scheduling again.");
      }
    } finally { pending.current = null; if (!controller.signal.aborted) setBusy(false); }
  }
  return <details className={styles.card}><summary>Schedule a championship, qualifying playoff or additional meet</summary>
    <p>Create its date and roster deadline here. Clubs then choose four eligible players at each skill level in Meet rosters, before you generate the pairings.</p>
    {error && <p role="alert" className={styles.error}>{error}</p>}
    {blocked && <p className={styles.warning}>Refresh the season to check whether the meet was created before retrying.</p>}
    <form onSubmit={event => { event.preventDefault(); void schedule(); }}><fieldset className={styles.gameFields} disabled={disabled || busy || blocked}>
      <div className={styles.twoColumns}><label>Competition<select value={phase} onChange={event => setPhase(event.target.value as CompetitionPhase)}>{Object.entries(phaseLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <label>Host club<select required value={host} onChange={event => { setHost(event.target.value); setClubs(old => Array.from(new Set([...old, event.target.value]))); }}><option value="">Choose host</option>{context.clubs.map(club => <option key={club.id} value={club.id}>{club.name}</option>)}</select></label>
        <label>Meet time (your device’s time)<input required type="datetime-local" value={startsAt} onChange={event => setStartsAt(event.target.value)} /></label>
        <label>Roster deadline (your device’s time)<input required type="datetime-local" value={deadline} onChange={event => setDeadline(event.target.value)} /></label>
        <label>Courts<input required type="number" min={1} max={100} value={courts} onChange={event => setCourts(Number(event.target.value))} /></label>
        <label>Minutes<input required type="number" min={30} max={180} value={minutes} onChange={event => setMinutes(Number(event.target.value))} /></label>
      </div>
      <fieldset className={styles.players}><legend>Participating clubs{phase === "regular" ? " (two to four)" : " (all clubs playing finals or playoffs)"}</legend>{context.clubs.map(club => <label key={club.id} className={styles.check}><input type="checkbox" checked={clubs.includes(club.id)} disabled={club.id === host} onChange={event => setClubs(old => event.target.checked ? [...old, club.id] : old.filter(id => id !== club.id))} />{club.name}</label>)}</fieldset>
      <div className={styles.toolbar}><button className={styles.primary} type="submit" disabled={clubs.length < 2 || phase === "regular" && clubs.length > 4 || !host}>{busy ? "Scheduling…" : "Schedule meet"}</button></div>
    </fieldset></form>
  </details>;
}
