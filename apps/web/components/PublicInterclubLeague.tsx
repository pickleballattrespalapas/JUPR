import type { PublicLeague } from "@/lib/interclubPublic";
import { meetTime } from "@/lib/interclubPublic";
import styles from "./ClubWebsite.module.css";
export default function PublicInterclubLeague({
  league,
}: {
  league: PublicLeague;
}) {
  const doc = league.document,
    names = Object.fromEntries(doc.clubs.map((c) => [c.id, c.name]));
  return (
    <section>
      <p className={styles.eyebrow}>Interclub league</p>
      <h1>{doc.name}</h1>
      <p>
        {doc.start_date} – {doc.end_date} · {doc.clubs.length} participating
        clubs
      </p>
      <nav className={styles.actions} aria-label="League sections">
        <a className={styles.button} href="#standings">
          Standings
        </a>
        <a className={styles.button} href="#schedule">
          Schedule
        </a>
        <a className={styles.button} href="#results">
          Results
        </a>
      </nav>
      <section id="standings">
        <h2>Standings</h2>
        <p>
          Ranked by encounter wins, then game difference and point difference.
          Clubs with equal totals are tied.
        </p>
        {!doc.results.length && (
          <p className={styles.notice}>
            No results have been published yet. All clubs start at zero.
          </p>
        )}
        {league.standings.map((group) => (
          <section key={group.division}>
            <h3>{group.division} division</h3>
            <div style={{ overflowX: "auto" }}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    {[
                      "Club",
                      "Played",
                      "Won",
                      "Lost",
                      "Games won",
                      "Games lost",
                      "Point difference",
                    ].map((h) => (
                      <th key={h} scope="col">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {group.rows.map((row) => (
                    <tr key={row.club_id}>
                      <th scope="row">{row.name}</th>
                      <td>{row.played}</td>
                      <td>{row.wins}</td>
                      <td>{row.losses}</td>
                      <td>{row.games_won}</td>
                      <td>{row.games_lost}</td>
                      <td>
                        {row.point_difference > 0 ? "+" : ""}
                        {row.point_difference}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        ))}
      </section>
      <section id="schedule">
        <h2>Meet schedule</h2>
        <p>
          All times use {doc.timezone}. Clubs choose players separately for each
          meet.
        </p>
        <div className={styles.grid}>
          {doc.meets.map((meet, i) => (
            <article key={meet.id} className={styles.card}>
              <p className={styles.eyebrow}>Meet {i + 1}</p>
              <h3>{meetTime(meet.starts_at, doc.timezone)}</h3>
              <p>
                <strong>Host:</strong>{" "}
                {names[meet.host_club_id || ""] || "To be confirmed"}
              </p>
              <p>
                {meet.club_ids
                  .map((id) => names[id])
                  .filter(Boolean)
                  .join(" · ")}
              </p>
              <small>
                {meet.courts} courts · {meet.duration_minutes} minutes
              </small>
            </article>
          ))}
        </div>
        {!doc.meets.length && <p>Meet dates will be announced here.</p>}
      </section>
      <section id="results">
        <h2>Results</h2>
        {!doc.results.length ? (
          <p>Results will appear after the organizer publishes them.</p>
        ) : (
          <div className={styles.grid}>
            {doc.results.map((row) => {
              const meet = doc.meets.find((m) => m.id === row.meet_id);
              const wins = row.games.filter((g) => g.a > g.b).length;
              return (
                <article className={styles.card} key={row.id}>
                  <p className={styles.eyebrow}>
                    {row.division} ·{" "}
                    {meet
                      ? meetTime(meet.starts_at, doc.timezone)
                      : "Meet result"}
                  </p>
                  <h3>
                    {names[row.club_a]} vs {names[row.club_b]}
                  </h3>
                  <p>
                    <strong>
                      {names[wins >= 2 ? row.club_a : row.club_b]}
                    </strong>{" "}
                    won {Math.max(wins, 3 - wins)}–{Math.min(wins, 3 - wins)}{" "}
                    games.
                  </p>
                  <p>
                    {row.games.map((g, i) => (
                      <span key={i}>
                        {i ? " · " : ""}Game {i + 1}: {g.a}–{g.b}
                      </span>
                    ))}
                  </p>
                </article>
              );
            })}
          </div>
        )}
      </section>
    </section>
  );
}
