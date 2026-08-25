import Link from "next/link";
import PublicTournamentNav from "@/components/PublicTournamentNav";
import {
  getPublicTournamentResults,
  getPublicTournamentResultsIndex,
  type PublicTournamentDrawResult,
  type PublicTournamentGameResult
} from "@/lib/tournamentResultsApi";

type Props = {
  params: { clubSlug: string };
  searchParams?: {
    tournament_id?: string;
    view?: string;
  };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const stateColors: Record<string, { color: string; background: string }> = {
  COMPLETE: { color: "#166534", background: "#dcfce7" },
  LIVE: { color: "#9a3412", background: "#ffedd5" },
  READY: { color: "#1d4ed8", background: "#dbeafe" },
  SCHEDULED: { color: "#475569", background: "#f1f5f9" }
};

function gameTitle(game: PublicTournamentGameResult): string {
  if (game.playoff_round) return game.playoff_round;
  if (game.round_number) return `Round ${game.round_number}`;
  return game.stage === "PLAYOFF" ? "Playoff" : "Match";
}

function scoreText(game: PublicTournamentGameResult): string {
  if (game.outcome_label) {
    return game.winner_name
      ? `${game.outcome_label} · ${game.winner_name}`
      : game.outcome_label;
  }
  if (game.score_a != null && game.score_b != null) {
    return `${game.score_a}–${game.score_b}`;
  }
  return game.state === "FINAL" ? "Final" : game.state.toLowerCase();
}

function DrawResults({ draw }: { draw: PublicTournamentDrawResult }) {
  const badge = stateColors[draw.state] || stateColors.SCHEDULED;
  return (
    <article style={{ ...cardStyle, marginBottom: "1rem" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: "0.75rem",
          flexWrap: "wrap"
        }}
      >
        <div>
          <p style={{ color: "#2563eb", fontWeight: 800, margin: 0 }}>
            {draw.event_family_label} · {draw.division_name}
          </p>
          <h2 style={{ margin: "0.25rem 0" }}>{draw.name}</h2>
          <p style={{ color: "#64748b", margin: 0 }}>
            {draw.scheduled_days
              .map((day) =>
                day.event_date ? `${day.label} · ${day.event_date}` : day.label
              )
              .join(" · ") || "Schedule to be announced"}
          </p>
        </div>
        <span
          style={{
            borderRadius: "999px",
            padding: "0.3rem 0.65rem",
            fontWeight: 800,
            color: badge.color,
            background: badge.background
          }}
        >
          {draw.state}
        </span>
      </div>

      {draw.podium.length ? (
        <section aria-label={`${draw.name} medalists`}>
          <h3>Podium and medals</h3>
          <ol style={{ display: "grid", gap: "0.4rem", paddingLeft: "1.5rem" }}>
            {draw.podium.map((entry) => (
              <li key={`${entry.placement}:${entry.team_name}`}>
                <strong>{entry.medal || `Place ${entry.placement}`}</strong>: {entry.team_name}
              </li>
            ))}
          </ol>
        </section>
      ) : null}

      {draw.standings.length ? (
        <section style={{ overflowX: "auto" }}>
          <h3>Standings</h3>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                {['Rank', 'Team / player', 'W', 'L', 'PF', 'PA', '+/−'].map((label) => (
                  <th key={label} scope="col" style={{ textAlign: label === 'Team / player' ? 'left' : 'right', padding: "0.45rem", borderBottom: "1px solid #cbd5e1" }}>{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {draw.standings.map((row) => (
                <tr key={row.public_team_key}>
                  <td style={{ textAlign: "right", padding: "0.45rem" }}>{row.rank ?? "—"}</td>
                  <th scope="row" style={{ textAlign: "left", padding: "0.45rem" }}>{row.team_name}</th>
                  <td style={{ textAlign: "right", padding: "0.45rem" }}>{row.wins ?? 0}</td>
                  <td style={{ textAlign: "right", padding: "0.45rem" }}>{row.losses ?? 0}</td>
                  <td style={{ textAlign: "right", padding: "0.45rem" }}>{row.points_for ?? 0}</td>
                  <td style={{ textAlign: "right", padding: "0.45rem" }}>{row.points_against ?? 0}</td>
                  <td style={{ textAlign: "right", padding: "0.45rem" }}>{row.differential ?? 0}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}

      {draw.bracket.length ? (
        <section>
          <h3>Playoff bracket</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.65rem" }}>
            {draw.bracket.map((game) => (
              <article key={game.public_game_key} style={{ border: "1px solid #cbd5e1", borderRadius: "10px", padding: "0.75rem", background: "#f8fafc" }}>
                <strong>{gameTitle(game)}</strong>
                <p style={{ margin: "0.4rem 0" }}>{game.team_a_name} vs. {game.team_b_name}</p>
                <p style={{ margin: 0, fontWeight: 800 }}>{scoreText(game)}</p>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      <section>
        <h3>Completed scores</h3>
        {draw.scores.length ? (
          <div style={{ display: "grid", gap: "0.5rem" }}>
            {draw.scores.map((game) => (
              <div key={game.public_game_key} style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap", borderBottom: "1px solid #e2e8f0", paddingBottom: "0.5rem" }}>
                <span><strong>{gameTitle(game)}</strong> · {game.team_a_name} vs. {game.team_b_name}</span>
                <strong>{scoreText(game)}</strong>
              </div>
            ))}
          </div>
        ) : (
          <p style={{ color: "#64748b" }}>No scores have been published yet.</p>
        )}
      </section>
    </article>
  );
}

export default async function TournamentResultsPage({ params, searchParams }: Props) {
  const tournamentId = String(searchParams?.tournament_id || "").trim();
  const view = searchParams?.view === "past" ? "past" : "current";

  if (!tournamentId) {
    const { data, error } = await getPublicTournamentResultsIndex(
      params.clubSlug,
      view
    );
    return (
      <section>
        <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Tournament results</p>
        <h1 style={{ marginTop: 0 }}>Choose tournament results</h1>
        <p style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
          <Link href={`/clubs/${params.clubSlug}/tournament-results`}>Current tournaments</Link>
          <Link href={`/clubs/${params.clubSlug}/tournament-results?view=past`}>Past tournaments</Link>
          <Link href={`/clubs/${params.clubSlug}/tournaments`}>Tournament Home</Link>
        </p>
        {error ? <p role="alert" style={{ color: "#b91c1c" }}>{error}</p> : null}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.75rem" }}>
          {(data?.tournaments || []).map((tournament) => (
            <article key={tournament.id} style={cardStyle}>
              <h2 style={{ marginTop: 0 }}>{tournament.name}</h2>
              <p style={{ color: "#475569" }}>{tournament.start_date || "Date TBD"}{tournament.end_date ? ` – ${tournament.end_date}` : ""}</p>
              <Link href={`/clubs/${params.clubSlug}/tournament-results?tournament_id=${encodeURIComponent(tournament.id)}${view === "past" ? "&view=past" : ""}`} style={{ fontWeight: 800 }}>Open live scores and results</Link>
            </article>
          ))}
        </div>
        {!error && !data?.tournaments.length ? <p>No {view} tournament results are published.</p> : null}
      </section>
    );
  }

  const { data, error } = await getPublicTournamentResults(
    params.clubSlug,
    tournamentId
  );
  if (error || !data) {
    return (
      <section>
        <h1>Tournament results not found</h1>
        <p role="alert" style={{ color: "#b91c1c" }}>{error || "This tournament is not publicly available."}</p>
        <Link href={`/clubs/${params.clubSlug}/tournament-results${view === "past" ? "?view=past" : ""}`}>Choose another tournament</Link>
      </section>
    );
  }

  const registrationSlug = data.tournament.settings.registration_slug || null;
  return (
    <section>
      <p style={{ margin: "0 0 0.75rem" }}>
        <Link href={`/clubs/${params.clubSlug}/tournament-results${data.tournament.status === "COMPLETED" ? "?view=past" : ""}`}>← Choose another tournament</Link>
      </p>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Live & Results</p>
      <h1 style={{ marginTop: 0 }}>{data.tournament.name}</h1>
      <p style={{ color: "#475569", maxWidth: "52rem" }}>Patron-safe live progress, completed scores, round-robin standings, playoff brackets, and official medalists.</p>
      <PublicTournamentNav
        clubSlug={params.clubSlug}
        tournamentName={data.tournament.name}
        tournamentId={data.tournament.id}
        registrationSlug={registrationSlug}
        active="results"
      />
      {data.draws.map((draw) => <DrawResults key={draw.public_draw_key} draw={draw} />)}
      {!data.draws.length ? <article style={cardStyle}><h2 style={{ marginTop: 0 }}>No standard draw results yet</h2><p style={{ color: "#475569", marginBottom: 0 }}>Singles and doubles draws will appear here after tournament staff publish them.</p></article> : null}
      <p><Link href={`/clubs/${params.clubSlug}/tournament-team-results`}>Four-player team results</Link></p>
    </section>
  );
}
