import Link from "next/link";
import type { EventHistory } from "@/lib/eventSeasons";
import styles from "./EventHistory.module.css";

export default function EventHistoryView({ history, admin = false }: { history: EventHistory; admin?: boolean }) {
  const seasons = admin ? history.seasons : history.seasons
    .filter(season => season.honors.length > 0)
    .sort((a, b) => b.position - a.position);

  return <section className={styles.history} aria-label="Event history">
    {admin && <div><h2>{history.series_name} history</h2><p className={styles.muted}>Champions, awards and results, season by season.</p></div>}
    {seasons.map(season => <article className={admin ? styles.season : styles.awardSeason} key={season.source_id}>
      <header className={styles.seasonHeader}>
        {admin ? <><div><p className={styles.eyebrow}>{season.label} · {season.status}{season.selected ? " · This season" : ""}</p>
          <h3>{season.name}</h3>
          {season.start_date && <p className={styles.muted}>{season.start_date}{season.end_date ? ` – ${season.end_date}` : ""}</p>}
        </div>
        <div className={styles.actions}>
          {season.results_href && <Link href={season.results_href}>View results →</Link>}
          {season.admin_href && <Link href={season.admin_href}>{season.status === "Draft" ? "Continue setup" : "Open event"} →</Link>}
        </div></> : <h2>{season.name}{season.label && !season.name.toLowerCase().includes(season.label.toLowerCase()) ? ` — ${season.label}` : ""}</h2>}
      </header>
      {!!season.honors.length && <ul className={styles.honors}>
        {season.honors.map(honor => <li key={honor.id}>
          <span className={styles.medal} aria-hidden="true">{honor.placement === 1 ? "🏆" : "🏅"}</span>
          <div><strong>{honor.recipient}</strong><span>{honor.title}{honor.placement > 1 ? ` · ${honor.placement === 2 ? "Runner-up" : `Place ${honor.placement}`}` : ""}</span>
            {honor.record && <small>{honor.record}</small>}
          </div>
        </li>)}
      </ul>}
      {admin && !season.honors.length && <p className={styles.muted}>{season.complete ? "No published awards for this season yet. The saved results remain available above." : "Champions and awards will appear when this season is finished and its honors are published."}</p>}
    </article>)}
    {!seasons.length && <p className={styles.muted}>{admin ? "No published seasons yet." : "No published awards yet."}</p>}
  </section>;
}
