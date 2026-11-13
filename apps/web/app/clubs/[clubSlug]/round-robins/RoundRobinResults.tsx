import { Display } from "@/components/ClubDisplay";
import PlayGeneratorStandingsTable from "@/components/PlayGeneratorStandingsTable";
import { roundRobinDate, roundRobinFormat, type RoundRobinResult } from "@/lib/roundRobinHistory";
import styles from "./roundRobins.module.css";

export default function RoundRobinResults({ session }: { session: RoundRobinResult }) {
  return <>
    <h1>{session.title}</h1>
    <p className={styles.intro}>{roundRobinDate(session.created_at)} · {session.participant_count} players · {roundRobinFormat(session.play_format)} · Complete</p>
    <Display field="result_summary">
      {session.scoring_mode === "unscored" ? <p>This session was played without recorded scores.</p> :
        <PlayGeneratorStandingsTable rows={session.standings} sortMode={session.standings_sort} title="Final standings" />}
    </Display>
    <Display field="result_scores">
      <h2>Round results</h2>
      <div className={styles.rounds}>
        {session.rounds.map(round => <details className={styles.card} key={round.number}>
          <summary>Round {round.number}{round.status === "skipped" ? " · Skipped" : round.status === "saved" ? " · Complete" : " · Not completed"}</summary>
          {round.matches.length ? <div className={styles.scroll}><table className={styles.scores}>
            <thead><tr><th scope="col">Court</th><th scope="col">Team / Player</th><th scope="col">Score</th><th scope="col">Team / Player</th></tr></thead>
            <tbody>{round.matches.map((match, index) => {
              const scored = round.status !== "skipped" && !["not_played", "skipped", "cancelled", "canceled"].includes(match.status) && match.score_a != null && match.score_b != null;
              return <tr key={index}><td>{match.court || "—"}</td><td>{match.side_a.join(" / ")}</td>
                <td>{scored ? `${match.score_a}–${match.score_b}` : "Not scored"}</td><td>{match.side_b.join(" / ")}</td></tr>;
            })}</tbody>
          </table></div> : <p>No games recorded for this round.</p>}
        </details>)}
      </div>
    </Display>
  </>;
}
