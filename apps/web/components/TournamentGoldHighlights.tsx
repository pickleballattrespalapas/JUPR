import Link from "next/link";
import type { TournamentGoldHighlight } from "@/lib/tournamentHighlights";
import styles from "./TournamentGoldHighlights.module.css";

export default function TournamentGoldHighlights({ highlights }: { highlights: TournamentGoldHighlight[] }) {
  if (!highlights.length) return null;
  return (
    <section className={`${styles.championships} ${styles.tournamentHighlights}`} aria-label="Tournament gold medalists">
      <h2>Tournament gold medalists</h2>
      <div className={styles.goldCards}>
        {highlights.map((highlight) => (
          <Link
            key={highlight.id}
            href={highlight.results_href}
            className={styles.card}
            aria-label={`${highlight.recipient} — ${highlight.division}, ${highlight.tournament_name}. View full results`}
            data-home-tournament-gold=""
            prefetch={false}
          >
            <span className={styles.trophy} aria-hidden="true">🥇</span>
            <div className={styles.details}>
              <span className={styles.season}>{highlight.tournament_name}</span>
              <h3>{highlight.recipient}</h3>
              <span className={styles.division}>{highlight.division}</span>
              {highlight.players.length > 0 && <span className={styles.winners}>{highlight.players.join(" · ")}</span>}
              <span className={styles.resultLink}>View full results →</span>
            </div>
          </Link>
        ))}
      </div>
    </section>
  );
}
