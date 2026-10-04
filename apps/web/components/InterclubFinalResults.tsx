"use client";
import Link from "next/link";
import { useState } from "react";
import type { PublicLeague } from "@/lib/interclubPublic";
import type { AwardRecipient } from "@/lib/interclubAwards";
import { interclubAwardLabels, recipientGroups } from "@/lib/interclubAwards";
import { CompetitionStandings } from "./PublicInterclubCompetition";
import styles from "./InterclubHonors.module.css";

export function SeasonAwardRecipients({ awards, names }: { awards: AwardRecipient[]; names: Record<string, string> }) {
  const [club, setClub] = useState("");
  const [kind, setKind] = useState("");
  const recipients = recipientGroups(awards.filter(row => !kind || row.award_key === kind)).filter(row => !club || row.club_id === club);
  return <section aria-label="Player awards"><h2>Player awards</h2>
    <div className={styles.filters}><label>Club<select aria-label="Filter player awards by club" value={club} onChange={event => setClub(event.target.value)}><option value="">All clubs</option>{Object.entries(names).map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></label>
      <label>Award<select aria-label="Filter player awards by award" value={kind} onChange={event => setKind(event.target.value)}><option value="">All awards and badges</option>{Object.entries(interclubAwardLabels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label></div>
    <ul className={styles.recipients}>{recipients.map(row => <li key={row.id}><strong>{row.name}</strong><small>{names[row.club_id]}</small>{row.titles.map(title => <p key={title}>{title}</p>)}</li>)}</ul>
    {!recipients.length && <p>No player awards to show.</p>}
  </section>;
}

export default function InterclubFinalResults({ league, awards }: { league: PublicLeague; awards?: AwardRecipient[] }) {
  const final = league.final_results;
  const names = Object.fromEntries(league.document.clubs.map(club => [club.id, club.name]));
  if (!final?.complete) return <section className={styles.section}><h2>Final results are not ready yet</h2><p>The season’s final results appear after all scheduled meets and championships have been approved and published.</p>{league.id && <Link href={`/interclub/${league.id}`}>View current standings and match results →</Link>}</section>;
  return <section className={styles.section} aria-label="Final season results">
    <div className={styles.hero}><p className={styles.eyebrow}>Season complete · {league.document.name}</p><p aria-hidden="true" className={styles.medal}>🏆</p>
      <h2>{final.champions.map(id => names[id] || id).join(" & ")}</h2><strong>{final.champions.length > 1 ? "Joint Club Cup champions" : "Club Cup champions"}</strong>
      <p className={styles.muted}>{final.clubs} clubs · {final.players} players · {final.meets} meets</p>
    </div>
    <section aria-label="Skill-level champions"><h2>Champions by skill level</h2><div className={styles.grid}>{final.divisions.map(result => <article key={result.division} className={styles.card}>
      <p className={styles.eyebrow}>{result.division} championship</p><h3>🏆 {names[result.winner] || result.winner}</h3>
      <p>{result.tiebreak ? `Won the singles tiebreak ${result.tiebreak.winner_score}–${result.tiebreak.runner_up_score} after a ${result.games_won}–${result.games_lost} split` : `Won ${result.games_won}–${result.games_lost}`}</p>
      <p className={styles.muted}>Runner-up: {names[result.runner_up] || result.runner_up}</p>
    </article>)}</div></section>
    <CompetitionStandings league={league} />
    {awards && awards.length > 0 ? <SeasonAwardRecipients awards={awards} names={names} /> : <p className={styles.notice}>Player trophies appear here once the organizer awards the season honors.</p>}
    <div className={styles.actions}>{league.id && <Link href={`/interclub/${league.id}?view=results`}>View all match results</Link>}{league.document.clubs.filter(club => club.slug).map(club => <Link key={club.id} href={`/clubs/${encodeURIComponent(club.slug!)}/trophies`}>{club.name} trophy case</Link>)}</div>
  </section>;
}
