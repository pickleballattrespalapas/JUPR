import { CompetitionContext, StandingRow } from "@/lib/interclubCompetition";
import styles from "./competition.module.css";

export default function Standings({ data, clubName }: { data: CompetitionContext; clubName: (id: string) => string }) {
  const divisions = data.standings?.divisions || {};
  return <section className={styles.section} aria-labelledby="season-standings-heading">
    <h2 id="season-standings-heading">Standings &amp; Club Cup</h2>
    <p>Official approved results only. Standings points, then pairings won, games won, point differential and head-to-head determine places.</p>
    {Object.keys(divisions).length ? Object.entries(divisions).map(([division, rows]) => {
      const qualification = data.standings.qualification?.[division] || data.qualifying?.[division];
      return <article className={styles.card} key={division}><h3>Skill level {division}</h3><RankingTable rows={rows} clubName={clubName} />
        {qualification?.status === "playoff_required" && <p role="status" className={styles.warning}><strong>Qualifying playoff needed:</strong> {qualification.playoff_required.map(clubName).join(", ")}. The tie crosses the final’s qualifying boundary. Arrange a full MLP-style playoff; no club advances on alphabetical order.</p>}
        {qualification?.qualifiers?.length ? <p><strong>Final qualifiers:</strong> {qualification.qualifiers.map(clubName).join(" · ")}</p> : null}
        {qualification?.status === "insufficient_entries" && <p>At least two clubs must have a regular-season appearance at this skill level before a final can be set.</p>}
      </article>;
    }) : <p>No official standings yet. Submit and approve the first meet’s scores to begin.</p>}
    <article className={styles.card}><h3>Overall Club Cup</h3><p>Points are added across all categories. Division-final winners add 6 points; runners-up add 3.</p>
      {data.club_cup?.standings?.length ? <RankingTable rows={data.club_cup.standings} clubName={clubName} cup /> : <p>No Cup points yet.</p>}
      {data.club_cup?.status === "complete" && data.club_cup.champions?.length > 0 && <p className={styles.success}><strong>{data.club_cup.champions.length > 1 ? "Joint Club Cup champions" : "Club Cup champion"}:</strong> {data.club_cup.champions.map(clubName).join(" · ")}</p>}
      {data.club_cup?.status === "provisional" && <p className={styles.muted}>Cup standings remain provisional until championship finals are official.</p>}
    </article>
  </section>;
}

function RankingTable({ rows, clubName, cup = false }: { rows: StandingRow[]; clubName: (id: string) => string; cup?: boolean }) {
  return <div className={styles.tableScroll}><table className={styles.table}><thead><tr><th>Place</th><th>Club</th>{cup && <><th>Regular season</th><th>Finals bonus</th></>}<th>Points</th><th>Pairings won</th><th>Games won</th><th>Point difference</th>{!cup && <th>Meets played</th>}</tr></thead><tbody>
    {rows.map((row, index) => <tr key={row.club_id}><td>{row.position ?? index + 1}{row.tied ? " =" : ""}</td><th scope="row">{clubName(row.club_id)}</th>{cup && <><td>{row.regular_points ?? 0}</td><td>{row.championship_points ?? 0}</td></>}<td><strong>{row.points}</strong></td><td>{row.pairings_won}</td><td>{row.games_won}</td><td>{row.point_differential}</td>{!cup && <td>{row.meets_played ?? 0}</td>}</tr>)}
  </tbody></table></div>;
}
