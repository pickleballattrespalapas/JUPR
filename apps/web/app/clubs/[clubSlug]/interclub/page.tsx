import Link from "next/link";
import { publicSiteFetch } from "@/lib/clubSiteServer";
import styles from "@/components/ClubWebsite.module.css";
export default async function ClubInterclubPage({
  params,
}: {
  params: { clubSlug: string };
}) {
  const data = await publicSiteFetch<{
    leagues: {
      id: string;
      name: string;
      start_date: string;
      end_date: string;
    }[];
  }>(`/public/clubs/${encodeURIComponent(params.clubSlug)}/interclub`);
  return (
    <section>
      <h1>Interclub leagues</h1>
      <p>
        Follow the whole league: standings, meet schedules and results from all
        participating clubs.
      </p>
      <div className={styles.grid}>
        {data?.leagues.map((league) => (
          <Link
            className={styles.cardLink}
            href={`/interclub/${league.id}`}
            key={league.id}
          >
            <h2>{league.name}</h2>
            <p>
              {league.start_date} – {league.end_date}
            </p>
            <small>Open league →</small>
          </Link>
        ))}
      </div>
      {!data?.leagues.length && (
        <p>No interclub leagues have been published for this club yet.</p>
      )}
    </section>
  );
}
