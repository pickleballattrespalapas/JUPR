"use client";

import Link from "next/link";
import { useState } from "react";
import { championshipQualifications, CompetitionContext, CompetitionMeet, CompetitionPhase, phaseLabels } from "@/lib/interclubCompetition";
import { workflowHref } from "../InterclubWorkflow";
import ScheduleMeet from "./ScheduleMeet";
import styles from "./competition.module.css";

export default function ChampionshipSetup({ context, root, clubId, accessToken, onScheduled, onSelectMeet }: {
  context: CompetitionContext; root: string; clubId: string; accessToken: string;
  onScheduled: (meet: CompetitionMeet) => void; onSelectMeet: (meetId: string) => void;
}) {
  const [scheduling, setScheduling] = useState<Exclude<CompetitionPhase, "regular"> | null>(null);
  const qualifications = championshipQualifications(context);
  const clubName = (id: string) => context.clubs.find(club => club.id === id)?.name || id;
  const meets = context.meets.filter(meet => meet.competition_phase && meet.competition_phase !== "regular");
  return <section aria-labelledby="championship-setup-heading" className={styles.card}>
    <p className={styles.eyebrow}>Regular season complete</p>
    <h2 id="championship-setup-heading">{context.club_cup.status === "complete" ? "Championship results" : "Set up championships"}</h2>
    <p>All regular-season meets have approved results. Finalists below come from those results and any completed qualifying playoffs.</p>
    {context.club_cup.status === "complete" && context.is_organizer && <div className={styles.toolbar}><Link className={styles.primary} href={`/admin/interclub/awards?season=${encodeURIComponent(context.season.id)}`}>Review season awards & final results →</Link></div>}
    <div className={styles.finalists}>{qualifications.map(qualification => {
      const final = context.batches.find(batch => batch.phase === "final" && batch.document.encounters.some(encounter => encounter.division === qualification.division));
      return <article key={qualification.division} className={styles.finalist} aria-label={`${qualification.division} championship qualification`}>
        <p className={styles.eyebrow}>Skill level {qualification.division}</p>
        {qualification.status === "ready" && qualification.qualifiers.length === 2 ? <>
          <h3>{qualification.qualifiers.map(clubName).join(" vs ")}</h3>
          <p>{final?.state === "approved" ? "Final results approved" : final ? "Final pairings prepared" : "Qualified for the final"}</p>
        </> : qualification.status === "playoff_required" ? <>
          <h3>Qualifying playoff needed</h3>
          {qualification.qualifiers.length > 0 && <p>Already qualified: {qualification.qualifiers.map(clubName).join(", ")}</p>}
          <p>Tied clubs: {qualification.playoff_required.map(clubName).join(", ")}. Approve the playoff results to confirm the remaining finalist{qualification.qualifiers.length ? "" : "s"}.</p>
        </> : <><h3>Finalists not yet confirmed</h3><p>Not enough eligible clubs have qualifying season results for this skill level.</p></>}
      </article>;
    })}</div>
    {meets.length > 0 && <section aria-label="Championship meets"><h3>Championship meets</h3>{meets.map(meet => {
      const batch = context.batches.find(batch => batch.meet_id === meet.id && batch.phase === meet.competition_phase);
      return <div className={styles.championshipMeet} key={meet.id}>
        <div><strong>{phaseLabels[meet.competition_phase!]} · {clubName(meet.host_club_id)}</strong>
          <p>{new Date(meet.starts_at).toLocaleString(undefined, { timeZone: context.season.details.timezone, dateStyle: "medium", timeStyle: "short" })} · {batch?.state === "approved" ? "Results approved" : batch?.state === "submitted" ? "Awaiting approval" : batch ? "Pairings prepared" : "Choose lineups next"}</p></div>
        <div className={styles.toolbar}>{!batch && <Link className={styles.button} href={workflowHref("lineups", context.season.id, meet.id)}>Choose lineups</Link>}<button onClick={() => onSelectMeet(meet.id)}>{batch?.state === "approved" ? "Review results" : "Open meet"}</button></div>
      </div>;
    })}</section>}
    {context.is_organizer && <>
      {!scheduling && <div className={styles.toolbar}>
        {qualifications.some(q => q.status === "ready") && context.club_cup.status !== "complete" && <button className={styles.primary} onClick={() => setScheduling("final")}>Schedule championship meet</button>}
        {qualifications.some(q => q.status === "playoff_required") && <button onClick={() => setScheduling("qualifier")}>Schedule qualifying playoff</button>}
      </div>}
      {scheduling && <ScheduleMeet key={scheduling} root={root} clubId={clubId} accessToken={accessToken} context={context} disabled={false} fixedPhase={scheduling} defaultOpen onCancel={() => setScheduling(null)} onScheduled={onScheduled} />}
      {context.club_cup.status !== "complete" && <p className={styles.muted}>Schedule the meet, choose each club’s players, then prepare the final for each skill level in that meet.</p>}
    </>}
  </section>;
}
