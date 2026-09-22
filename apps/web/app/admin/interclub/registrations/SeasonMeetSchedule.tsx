"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { CompetitionContext, CompetitionMeet, competitionRequest, phaseLabels } from "@/lib/interclubCompetition";
import { registrationMeetPlanning } from "@/lib/interclubRegistrationWindow";
import { RegistrationDetail } from "@/lib/interclubRegistration";
import ScheduleMeet from "../competition/ScheduleMeet";
import { workflowHref } from "../InterclubWorkflow";
import styles from "./registrations.module.css";

export default function SeasonMeetSchedule({ root, clubId, accessToken, seasonData, meetPlanningOpen, disabled, onSaved }: {
  root: string; clubId: string; accessToken: string; seasonData: RegistrationDetail; meetPlanningOpen: boolean;
  disabled: boolean; onSaved: (meet: CompetitionMeet) => void;
}) {
  const [context, setContext] = useState<CompetitionContext | null>(null);
  const [loading, setLoading] = useState(false), [error, setError] = useState(""), [message, setMessage] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [savedMeet, setSavedMeet] = useState<CompetitionMeet | null>(null);
  const [editor, setEditor] = useState<{ meet?: CompetitionMeet } | null>(null);
  const token = useRef(accessToken); token.current = accessToken;
  const heading = useRef<HTMLHeadingElement | null>(null);
  useEffect(() => {
    setContext(null); setEditor(null); setError("");
    if (!meetPlanningOpen || !seasonData.is_organizer) { setLoading(false); return; }
    const controller = new AbortController(); setLoading(true);
    competitionRequest<CompetitionContext>(root, token.current, controller.signal)
      .then(next => {
        if (!controller.signal.aborted) {
          if (next.season?.id !== seasonData.season.id || !Array.isArray(next.meets)) throw new Error("Could not confirm this season’s schedule. Reload the schedule.");
          setContext(next);
        }
      })
      .catch(cause => { if (!controller.signal.aborted) setError(cause instanceof Error && !(cause instanceof TypeError) ? cause.message : "Unable to load the meet schedule. Try again."); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [root, meetPlanningOpen, seasonData.is_organizer, seasonData.season.id, seasonData.season.registration?.revision, refresh]);
  const timezone = seasonData.season.details.timezone;
  const when = (iso: string) => new Date(iso).toLocaleString(undefined, { timeZone: timezone, dateStyle: "medium", timeStyle: "short" });
  const clubName = (id: string) => seasonData.clubs.find(club => club.id === id)?.name || context?.clubs.find(club => club.id === id)?.name || id;
  const meets = [...(context?.meets || seasonData.meet_schedule || seasonData.meets)].sort((a, b) => Date.parse(a.starts_at) - Date.parse(b.starts_at));
  const canEdit = Boolean(meetPlanningOpen && context?.is_organizer && registrationMeetPlanning(context.season.registration));
  return <section id="meet-schedule" className={styles.card} aria-labelledby="meet-schedule-heading">
    <h3 id="meet-schedule-heading" ref={heading} tabIndex={-1}>Meet schedule</h3>
    <p>All dates and times use {timezone}.</p>
    {message && <div role="status" className={styles.notice}><p>{message}</p><div className={styles.toolbar}>
      {savedMeet && <Link href={workflowHref("availability", seasonData.season.id, savedMeet.id)}>Meet availability</Link>}
      <Link href={`/admin/interclub/publication?season=${encodeURIComponent(seasonData.season.id)}`}>Review public schedule</Link>
    </div></div>}
    {!meetPlanningOpen && <p>Adding meets and changing dates opens after season registration closes.</p>}
    {loading && <p role="status">Loading meet schedule…</p>}
    {error && <p role="alert" className={styles.notice}>{error}</p>}
    <div className={styles.toolbar}>
      {!editor && <button className={styles.primary} disabled={disabled || loading || !canEdit} onClick={() => { setEditor({}); setMessage(""); }}>Add meet</button>}
      {meetPlanningOpen && !editor && <button disabled={loading || disabled} onClick={() => setRefresh(value => value + 1)}>{error ? "Retry loading schedule" : "Refresh schedule"}</button>}
    </div>
    {editor && context && canEdit && <ScheduleMeet key={editor.meet?.id || "new"} root={root} clubId={clubId} accessToken={accessToken} context={context} meet={editor.meet}
      defaultOpen disabled={disabled || loading} onCancel={() => setEditor(null)} onReload={() => setRefresh(value => value + 1)}
      onScheduled={(meet, notice) => { setEditor(null); setSavedMeet(meet); setMessage(notice || "Meet schedule saved."); setRefresh(value => value + 1); onSaved(meet); heading.current?.focus({ preventScroll: true }); }} />}
    {meets.length ? <div className={styles.scroll}><table className={styles.table}><caption>Season meets</caption>
      <thead><tr><th>Date and time</th><th>Host</th><th>Participating clubs</th><th>Schedule changes</th></tr></thead>
      <tbody>{meets.map(meet => {
        const current = context?.meets.find(value => value.id === meet.id);
        return <tr key={meet.id}><td>{when(meet.starts_at)}{current && <div className={styles.muted}>{phaseLabels[current.competition_phase || "regular"]}</div>}</td><td>{clubName(meet.host_club_id)}</td><td>{meet.club_ids.map(clubName).join(", ")}</td><td>
          {current && canEdit ? <><button disabled={disabled || loading || Boolean(editor) || current.schedule_editable !== true} aria-label={`Edit date for ${when(meet.starts_at)}`} onClick={() => { setEditor({ meet: current }); setMessage(""); }}>Edit date</button>
            {!current.schedule_editable && <p className={styles.muted}>{current.schedule_locked_reason || "This meet’s date is locked."}</p>}</> : <span>{!meetPlanningOpen ? "After registration closes" : loading ? "Loading schedule permissions…" : error ? "Reload schedule to check permissions" : "Only the commissioner can change dates"}</span>}
        </td></tr>;
      })}</tbody>
    </table></div> : !loading && <p>No meets have been scheduled yet.</p>}
  </section>;
}
