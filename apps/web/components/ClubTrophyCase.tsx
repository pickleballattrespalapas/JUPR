"use client";
import Link from "./PublicClubLink";
import { useState } from "react";
import type { SeasonTrophy } from "@/lib/interclubAwards";
import styles from "./InterclubHonors.module.css";

export default function ClubTrophyCase({ clubName, trophies }: { clubName: string; trophies: SeasonTrophy[] }) {
  const [kind, setKind] = useState("");
  const filtered = trophies.filter(trophy => !kind || trophy.award_key === kind);
  return <section className={styles.section} aria-label="Club trophy case">
    <div><p className={styles.eyebrow}>{clubName}</p><h1>Club trophy case</h1><p className={styles.muted}>Championships, Club Cups, and seasons played together.</p></div>
    {!!trophies.length && <div className={styles.filters}><label>Awards<select aria-label="Filter club trophies" value={kind} onChange={event => setKind(event.target.value)}><option value="">All trophies</option><option value="club_cup_champion">Club Cup champions</option><option value="division_champion">Skill-level champions</option><option value="participation">Season participation</option></select></label></div>}
    <div className={styles.grid}>{filtered.map(trophy => <article className={styles.card} key={trophy.id} data-club-trophy="">
      <span className={styles.medal} aria-hidden="true">{trophy.award_key === "participation" ? "🏅" : "🏆"}</span>
      <h2>{trophy.title}</h2><p>{trophy.season_name}</p><p className={styles.muted}>{new Intl.DateTimeFormat("en-US", { month: "long", year: "numeric", timeZone: "America/Mazatlan" }).format(new Date(trophy.earned_at))}</p>
      <Link href={trophy.results_href}>View final season results →</Link>
    </article>)}</div>
    {!trophies.length && <p className={styles.notice}>This club’s trophies will appear here when a season organizer awards them.</p>}
    {!!trophies.length && !filtered.length && <p>No trophies in this category yet.</p>}
  </section>;
}
