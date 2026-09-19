import type { CompetitionResult, PublicLeague } from "@/lib/interclubPublic";
import styles from "./ClubWebsite.module.css";

const pairingNames: Record<string, string> = { women: "Women’s doubles", men: "Men’s doubles", mixed_a: "Mixed doubles A", mixed_b: "Mixed doubles B" };

export function CompetitionStandings({ league }: { league: PublicLeague }) {
  const names = Object.fromEntries(league.document.clubs.map(club => [club.id, club.name]));
  return <>
    <p>Standings points, then pairings won, games won, point differential and head-to-head. A matchup win earns 3 points, a split earns 1 each. Weather draws follow the league’s partial-result rules.</p>
    {!league.document.competition_results?.length && <p className={styles.notice}>No official meet results have been published yet.</p>}
    {league.standings.map(group => <section key={group.division}>
      <h3>{group.division} skill level</h3>
      <div style={{ overflowX: "auto" }} tabIndex={0} role="region" aria-label={`${group.division} standings`}><table className={styles.table} aria-label={`${group.division} standings`}>
        <thead><tr>{["Place", "Club", "Points", "Meets played", "Pairings won", "Games won", "Point differential"].map(label => <th scope="col" key={label}>{label}</th>)}</tr></thead>
        <tbody>{group.rows.map(row => <tr key={row.club_id}>
          <td>{row.tied ? "T" : ""}{row.position}</td><th scope="row">{names[row.club_id] || row.name || row.club_id}</th>
          <td><strong>{row.points ?? 0}</strong></td><td>{row.meets_played ?? 0}</td><td>{row.pairings_won ?? 0}</td><td>{row.games_won}</td><td>{row.point_differential ?? 0}</td>
        </tr>)}</tbody>
      </table></div>
      {!group.rows.length && <p>No results at this skill level yet.</p>}
      {league.qualification?.[group.division]?.playoff_required && (Array.isArray(league.qualification[group.division].playoff_required) ? (league.qualification[group.division].playoff_required as string[]).length > 0 : true) && <p className={styles.notice}>A qualifying playoff is required to decide the championship places.</p>}
    </section>)}
    {league.club_cup && <section id="club-cup"><h2>Overall Club Cup</h2>
      <p>Points from every regular-season category are added together. Each skill-level final adds 6 points for its winner and 3 for its runner-up.</p>
      <div style={{ overflowX: "auto" }} tabIndex={0} role="region" aria-label="Club Cup standings"><table className={styles.table} aria-label="Club Cup standings">
        <thead><tr>{["Club", "Regular season", "Championship", "Total"].map(label => <th scope="col" key={label}>{label}</th>)}</tr></thead>
        <tbody>{league.club_cup.standings.map(row => <tr key={row.club_id}><th scope="row">{names[row.club_id] || row.name || row.club_id}</th><td>{row.regular_points}</td><td>{row.championship_points}</td><td><strong>{row.points}</strong></td></tr>)}</tbody>
      </table></div>
      {league.club_cup.status === "provisional" ? <p>Cup standings are provisional until the championship finals are complete.</p> : league.club_cup.champions.length > 0 && <p><strong>{league.club_cup.champions.length > 1 ? "Joint Club Cup champions" : "Club Cup champion"}:</strong> {league.club_cup.champions.map(id => names[id] || id).join(" · ")}</p>}
    </section>}
  </>;
}

export function CompetitionResults({ results, names }: { results: CompetitionResult[]; names: Record<string, string> }) {
  if (!results.length) return <p>Results will appear after the organizer approves and publishes them.</p>;
  return <div className={styles.grid}>{results.map(row => <article key={`${row.meet_id}:${row.phase}:${row.id}`} className={styles.card}>
    <p className={styles.eyebrow}>{row.division} · {row.phase === "final" ? "Championship final" : row.phase === "qualifier" ? "Qualifying playoff" : "Regular season"}</p>
    <h3>{names[row.club_a] || row.club_a} vs {names[row.club_b] || row.club_b}</h3>
    {row.weather === "finalized_partial" && <p>Finalized using available results; no further play could be scheduled.</p>}
    {row.pairings.map(pairing => <div key={pairing.kind}><h4>{pairingNames[pairing.kind] || pairing.kind}</h4>
      <ul>{pairing.games.map((game, index) => <li key={index}>Game {index + 1}: {game.status === "unplayed" ? "Not played" : game.status === "double_forfeit" ? "Both clubs forfeited — a game loss for each club" : game.status === "forfeit" ? `Forfeit — ${names[game.winner === "a" ? row.club_a : row.club_b] || "Winning club"} wins` : `${game.a}–${game.b}${game.status === "retired" ? ` (injury retirement; ${names[game.winner === "a" ? row.club_a : row.club_b] || "winning club"} awarded the game)` : ""}`}</li>)}</ul>
    </div>)}
    {row.tiebreak?.status === "completed" && <p><strong>Rotating singles:</strong> {row.tiebreak.a}–{row.tiebreak.b}. This tiebreak does not affect individual ratings.</p>}
  </article>)}</div>;
}
