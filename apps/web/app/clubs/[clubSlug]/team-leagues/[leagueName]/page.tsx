import Link from "next/link";
import { getPublicTeamLeague, teamLeagueApiBaseUrl } from "@/lib/teamLeagueApi";
import TeamLeagueRegistrationForm from "./TeamLeagueRegistrationForm";

type Props = { params: { clubSlug: string; leagueName: string } };
const card = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white", minWidth: 0 };

function categoryLabel(category: string): string {
  return ({ mens: "Men's", womens: "Women's", mixed: "Mixed", open: "Open" } as Record<string, string>)[category] || "Open";
}

export default async function TeamLeagueDetailPage({ params }: Props) {
  const leagueName = decodeURIComponent(params.leagueName);
  const { data, error } = await getPublicTeamLeague(params.clubSlug, leagueName);
  if (error || !data) {
    return (
      <section>
        <h1>Team league</h1>
        <p style={{ color: "#b91c1c" }}>{error || "League not found."}</p>
        <Link href={`/clubs/${params.clubSlug}/team-leagues`}>Back to team leagues</Link>
      </section>
    );
  }
  const teamNames = new Map(data.teams.map((team) => [team.id, team.team_name]));
  return (
    <section style={{ display: "grid", gap: "1rem", minWidth: 0, maxWidth: "100%" }}>
      <header>
        <p style={{ color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Team league</p>
        <h1>{data.league.league_name}</h1>
        <p style={{ color: "#475569" }}>
          {data.league.venue || "Venue to be announced"} ·{" "}
          {categoryLabel(data.league.team_category)} · {data.league.team_size}-player primary roster ·{" "}
          {data.league.allow_substitutes ? "Substitutes allowed" : "No substitutes"} ·{" "}
          {data.league.playoff_format === "none" ? "Round robin season" : "Round robin plus playoffs"}
        </p>
      </header>
      <article style={card}>
        <TeamLeagueRegistrationForm apiBase={teamLeagueApiBaseUrl()} clubSlug={params.clubSlug} leagueName={leagueName} detail={data} />
      </article>
      <article style={card}>
        <h2 style={{ marginTop: 0 }}>Teams</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 200px), 1fr))", gap: "0.75rem", minWidth: 0 }}>
          {data.teams.map((team) => (
            <div key={team.id} style={{ padding: "0.75rem", borderRadius: "10px", background: "#f8fafc", minWidth: 0, overflowWrap: "anywhere" }}>
              <strong>{team.team_name}</strong>
              <div style={{ color: "#475569" }}>
                {team.players.map((player) => `${player.player_name}${player.role === "alternate" ? " (alternate)" : ""}`).join(", ")}
              </div>
            </div>
          ))}
        </div>
        {!data.teams.length ? <p>No confirmed teams yet.</p> : null}
      </article>
      <article style={{ ...card, background: "#eff6ff", borderColor: "#bfdbfe" }} data-testid="team-league-awards">
        <h2 style={{ marginTop: 0 }}>Awards race</h2>
        {data.award_progress.awards.length ? <>
          <p style={{ color: "#475569" }}>Current leaders who have met each configured minimum criterion.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 210px), 1fr))", gap: "0.75rem" }}>
            {data.award_progress.awards.map((award) => <div key={`${award.category_key}-${award.rank}-${award.team_id || award.player_id}`} style={{ padding: "0.75rem", borderRadius: "10px", background: "white" }}><strong>{award.category_label}{award.rank && award.rank > 1 ? ` · #${award.rank}` : ""}</strong><br />{award.recipient_name || "—"}{award.is_co_winner ? " · co-leader" : ""}<br /><small>{award.metric_display || "—"} · Minimum {award.min_games ?? 0} {String(award.minimum_metric || "games").replace(/_/g, " ")}</small></div>)}
          </div>
        </> : <p style={{ color: "#475569", marginBottom: 0 }}>No team has met the current award qualification criteria yet.</p>}
      </article>
      <article style={card}>
        <h2 style={{ marginTop: 0 }}>Team standings</h2>
        <p style={{ color: "#475569" }}>This record table is separate from the current awards race.</p>
        <div style={{ overflowX: "auto", maxWidth: "100%", minWidth: 0 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "520px" }}>
            <thead><tr><th align="left">Rank</th><th align="left">Team</th><th>Played</th><th>Wins</th><th>Losses</th><th>Point diff.</th></tr></thead>
            <tbody>
              {data.standings.map((row) => <tr key={String(row.team_id)}><td>{String(row.rank)}</td><td>{String(row.team_name)}</td><td align="center">{String(row.games_played)}</td><td align="center">{String(row.wins)}</td><td align="center">{String(row.losses)}</td><td align="center">{String(row.point_differential)}</td></tr>)}
            </tbody>
          </table>
        </div>
      </article>
      <article style={card}>
        <h2 style={{ marginTop: 0 }}>Schedule and results</h2>
        <div style={{ display: "grid", gap: "0.65rem" }}>
          {data.fixtures.filter((fixture) => fixture.status !== "bye").map((fixture) => (
            <div key={fixture.id} style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 120px), 1fr))", gap: "0.75rem", padding: "0.75rem", borderRadius: "10px", background: "#f8fafc", alignItems: "center", minWidth: 0, overflowWrap: "anywhere" }}>
              <span>{fixture.phase === "playoff" ? `Playoff ${fixture.round_number}` : `Week ${fixture.week_number}`}</span>
              <strong>{teamNames.get(String(fixture.team_a_id)) || "TBD"} vs {teamNames.get(String(fixture.team_b_id)) || "TBD"}</strong>
              <span>{fixture.team_a_score == null ? "Scheduled" : `${fixture.team_a_score}–${fixture.team_b_score}`}</span>
            </div>
          ))}
        </div>
        {!data.fixtures.length ? <p>The schedule has not been published.</p> : null}
      </article>
      <p><Link href={`/clubs/${params.clubSlug}/team-leagues`}>All team leagues</Link> · <Link href={`/clubs/${params.clubSlug}`}>Club home</Link></p>
    </section>
  );
}
