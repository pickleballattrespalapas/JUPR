import PublicTournamentSponsors from "@/components/PublicTournamentSponsors";
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

function matchupStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    COMPLETE: "Final",
    COMPLETED: "Final",
    FINAL: "Final",
    LIVE: "In progress",
    IN_PROGRESS: "In progress",
    READY: "Ready",
    LINEUPS_PENDING: "Awaiting lineups",
    TIEBREAK_REQUIRED: "Tiebreak needed",
    CORRECTION_REQUIRED: "Result under review",
    SCHEDULED: "Upcoming",
    PENDING: "Upcoming",
    BYE: "Bye",
    VOID: "No result"
  };
  return labels[String(status || "").trim().toUpperCase()] || "Status unavailable";
}

function rosterSpotLabel(slot: string): string {
  const labels: Record<string, string> = {
    MAN_1: "Men’s roster spot 1",
    MAN_2: "Men’s roster spot 2",
    WOMAN_1: "Women’s roster spot 1",
    WOMAN_2: "Women’s roster spot 2"
  };
  return labels[String(slot || "").trim().toUpperCase()] || "Roster spot";
}

function matchupTitle(matchup: { playoff_game_code?: string | null; round_number: number; slot_number: number }): string {
  const code = String(matchup.playoff_game_code || "").trim().toUpperCase();
  const labels: Record<string, string> = {
    F: "Final",
    FINAL: "Final",
    B: "Bronze medal match",
    BRONZE: "Bronze medal match",
    SF1: "Semifinal 1",
    SF2: "Semifinal 2",
    QF1: "Quarterfinal 1",
    QF2: "Quarterfinal 2",
    QF3: "Quarterfinal 3",
    QF4: "Quarterfinal 4"
  };
  return labels[code] || `Round ${matchup.round_number} · Match ${matchup.slot_number}`;
}

export default async function TournamentTeamResultsDetail({ params }: Props) {
  const { data, error, status } = await getPublicTeamTournamentResults(
    params.clubSlug,
    params.tournamentId,
    params.drawId
  );
  const missing = status === 404;
  const teamNames = new Map(
    (data?.teams || []).map((team) => [team.id, team.name])
  );

  return (
    <section>
      <p style={{ color: "#2563eb", fontWeight: 800, marginBottom: "0.4rem" }}>
        Four-player team results
      </p>
      <h1 style={{ marginTop: 0, marginBottom: 0 }}>
        {data?.tournament.name || (missing ? "Team results not found" : "Team tournament results")}
      </h1>
      <PublicTournamentSponsors clubSlug={params.clubSlug} tournamentId={data?.tournament.id} placement="header" />
      {data?.draw.name ? <h2>{data.draw.name}</h2> : null}
      {error ? (
        <p role="alert" style={{ color: "#b91c1c" }}>
          {missing
            ? "We couldn’t find these team results. They may no longer be public."
            : "These team results are unavailable right now. Please try again shortly."}
        </p>
      ) : null}

      {data ? (
        <>
          <section className={styles.card}>
            <h2>Standings</h2>
            {data.standings.length ? <div className={styles.tableWrap}>
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
            </div> : <p>Standings will appear after play begins.</p>}
          </section>

          <section className={styles.card} style={{ marginTop: "1rem" }}>
            <h2>Playoff bracket</h2>
            {data.bracket.length ? (
              <div className={styles.grid}>
                {data.bracket.map((matchup) => (
                  <article className={styles.slot} key={matchup.id}>
                    <strong>
                      {matchupTitle(matchup)}
                    </strong>
                    <p>
                      {teamNames.get(String(matchup.team_a_id || "")) || "TBD"}{" "}
                      {matchup.team_a_game_wins ?? "—"}
                      <br />
                      {teamNames.get(String(matchup.team_b_id || "")) || "TBD"}{" "}
                      {matchup.team_b_game_wins ?? "—"}
                    </p>
                    <small>{matchupStatusLabel(matchup.status)}</small>
                  </article>
                ))}
              </div>
            ) : (
              <p>The playoff bracket will appear when it’s ready.</p>
            )}
          </section>

          <section className={styles.card} style={{ marginTop: "1rem" }}>
            <h2>Medalists</h2>
            {data.podium.length ? (
              <ol>
                {data.podium.map((place) => (
                  <li key={place.placement}>
                    <strong>{place.team_name}</strong>
                  </li>
                ))}
              </ol>
            ) : (
              <p>Medal winners will appear after the final matches.</p>
            )}
          </section>

          <section className={styles.card} style={{ marginTop: "1rem" }}>
            <h2>Team rosters</h2>
            {data.teams.length ? <div className={styles.grid}>
              {data.teams.map((team) => (
                <article className={styles.slot} key={team.id}>
                  <h3>{team.name}</h3>
                  <ul>
                    {team.members.map((member) => (
                      <li key={member.id}>
                        {member.display_name || member.display_name_snapshot || "Player"} ·{" "}
                        {rosterSpotLabel(member.slot)}
                      </li>
                    ))}
                  </ul>
                </article>
              ))}
            </div> : <p>No team rosters are available yet.</p>}
          </section>
        </>
      ) : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href={`/clubs/${params.clubSlug}/tournament-team-results`}>
          All team tournaments
        </Link>
      </p>
      <PublicTournamentSponsors clubSlug={params.clubSlug} tournamentId={data?.tournament.id} placement="footer" />
    </section>
  );
}
