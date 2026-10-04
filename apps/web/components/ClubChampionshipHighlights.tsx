import Link from "next/link";
import type { SeasonTrophy } from "@/lib/interclubAwards";
import styles from "./ClubChampionshipHighlights.module.css";

export default function ClubChampionshipHighlights({
  trophies,
  slug,
}: {
  trophies: SeasonTrophy[];
  slug: string;
}) {
  return (
    <aside className={styles.championships} aria-label="Club championships">
      <h2>Club championships</h2>
      <div className={styles.cards}>
        {trophies.slice(0, 3).map((trophy) => {
          const leagueChampion = trophy.award_key === "club_cup_champion";
          const earned = new Intl.DateTimeFormat("en-US", {
            month: "long", year: "numeric", timeZone: "America/Mazatlan",
          }).format(new Date(trophy.earned_at));
          return (
            <Link
              key={trophy.id}
              href={trophy.results_href}
              className={`${styles.card} ${leagueChampion ? styles.leagueChampion : ""}`}
              aria-label={`${trophy.title} — ${earned}. View final season results`}
              data-home-championship=""
            >
              <span className={styles.trophy} aria-hidden="true">🏆</span>
              <div className={styles.details}>
                <span className={styles.season}>{trophy.season_name}</span>
                <h3>{leagueChampion ? "League Champion" : `${trophy.division} Division Champion`}</h3>
                <time className={styles.date} dateTime={trophy.earned_at}>{earned}</time>
              </div>
            </Link>
          );
        })}
      </div>
      <Link className={styles.caseLink} href={`/clubs/${encodeURIComponent(slug)}/trophies`}>
        View trophy case{trophies.length > 3 ? ` (${trophies.length})` : ""} →
      </Link>
    </aside>
  );
}
