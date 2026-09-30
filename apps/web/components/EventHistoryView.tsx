import Link from "next/link";
import type { EventHistory, EventSeason } from "@/lib/eventSeasons";
import styles from "./EventHistory.module.css";

function placeLabel(place: number) {
  const lastTwo = place % 100;
  const suffix = lastTwo >= 11 && lastTwo <= 13 ? "th" : ["th", "st", "nd", "rd"][place % 10] || "th";
  return `${place}${suffix} place`;
}

function HonorsByPlace({ honors, admin }: { honors: EventSeason["honors"]; admin: boolean }) {
  const places = Array.from(new Set(honors.map(honor => honor.placement))).sort((a, b) => a - b);
  const Heading = admin ? "h4" : "h3";

  return <div className={styles.placementGroups}>
    {places.map(place => <section key={place} aria-label={placeLabel(place)}>
      <Heading className={styles.placementHeading}>
        <span className={styles.medal} aria-hidden="true">{place === 1 ? "🏆" : place === 2 ? "🥈" : place === 3 ? "🥉" : "🏅"}</span>
        {placeLabel(place)}
      </Heading>
      <ul className={styles.honors}>
        {honors.filter(honor => honor.placement === place).map(honor => <li key={honor.id}>
          <div><strong>{honor.recipient}</strong><span>{honor.title}</span>
            {honor.record && <small>{honor.record}</small>}
          </div>
        </li>)}
      </ul>
    </section>)}
  </div>;
}

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
      {!!season.honors.length && <HonorsByPlace honors={season.honors} admin={admin} />}
      {admin && !season.honors.length && <p className={styles.muted}>{season.complete ? "No published awards for this season yet. The saved results remain available above." : "Champions and awards will appear when this season is finished and its honors are published."}</p>}
    </article>)}
    {!seasons.length && <p className={styles.muted}>{admin ? "No published seasons yet." : "No published awards yet."}</p>}
  </section>;
}
