import Link from "next/link";
import { getPublicTeamTournamentResults } from "@/lib/tournamentTeamCompetitionApi";
import styles from "@/components/tournaments/TournamentTeamCompetition.module.css";

type Props = {
  params: { clubSlug: string; tournamentId: string; drawId: string };
};

function number(value: unknown): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export default async function TournamentTeamResultsDetail({ params }: Props) {
  const { data, error } = await getPublicTeamTournamentResults(
    params.clubSlug,
    params.tournamentId,
    params.drawId
  );
  const teamNames = new Map(
    (data?.teams || []).map((team) => [team.id, team.name])
  );

  return (
    <section>
      <p style={{ color: "#2563eb", fontWeight: 800, marginBottom: "0.4rem" }}>
        {data?.tournament.name || "Team tournament"}
      </p>
      <h1 style={{ marginTop: 0 }}>{data?.draw.name || "Published results"}</h1>
      {error ? (
        <p role="alert" style={{ color: "#b91c1c" }}>
          These results are unavailable. {error}
        </p>
      ) : null}

      {data ? (
        <>
          <section className={styles.card}>
            <h2>Standings</h2>
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Team</th>
                    <th>Matches</th>
                    <th>Games</th>
                    <th>Difference</th>
                  </tr>
                </thead>
                <tbody>
                  {data.standings.map((row, index) => (
                    <tr key={String(row.team_id || index)}>
                      <td>{number(row.rank) || index + 1}</td>
                      <td>{String(row.team_name || "Team")}</td>
                      <td>
                        {number(row.match_wins)}–{number(row.match_losses)}
                      </td>
                      <td>
                        {number(row.game_wins)}–{number(row.game_losses)}
                      </td>
                      <td>{number(row.game_differential)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className={styles.card} style={{ marginTop: "1rem" }}>
            <h2>Playoff bracket</h2>
            {data.bracket.length ? (
              <div className={styles.grid}>
                {data.bracket.map((matchup) => (
                  <article className={styles.slot} key={matchup.id}>
                    <strong>
                      {matchup.playoff_game_code || `Round ${matchup.round_number}`}
                    </strong>
                    <p>
                      {teamNames.get(String(matchup.team_a_id || "")) || "TBD"}{" "}
                      {matchup.team_a_game_wins ?? "—"}
                      <br />
                      {teamNames.get(String(matchup.team_b_id || "")) || "TBD"}{" "}
                      {matchup.team_b_game_wins ?? "—"}
                    </p>
                    <small>{matchup.status}</small>
                  </article>
                ))}
              </div>
            ) : (
              <p>No playoff bracket is configured for this division.</p>
            )}
          </section>

          <section className={styles.card} style={{ marginTop: "1rem" }}>
            <h2>Podium</h2>
            {data.podium.length ? (
              <ol>
                {data.podium.map((place) => (
                  <li key={place.placement}>
                    <strong>{place.team_name}</strong>
                  </li>
                ))}
              </ol>
            ) : (
              <p>The podium will appear after final results are published.</p>
            )}
          </section>

          <section className={styles.card} style={{ marginTop: "1rem" }}>
            <h2>Team rosters</h2>
            <div className={styles.grid}>
              {data.teams.map((team) => (
                <article className={styles.slot} key={team.id}>
                  <h3>{team.name}</h3>
                  <ul>
                    {team.members.map((member) => (
                      <li key={member.id}>
                        {member.display_name || member.display_name_snapshot || "Player"} ·{" "}
                        {member.slot.replace("_", " ").toLowerCase()}
                      </li>
                    ))}
                  </ul>
                </article>
              ))}
            </div>
          </section>
        </>
      ) : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href={`/clubs/${params.clubSlug}/tournament-team-results`}>
          All team tournaments
        </Link>
      </p>
    </section>
  );
}
