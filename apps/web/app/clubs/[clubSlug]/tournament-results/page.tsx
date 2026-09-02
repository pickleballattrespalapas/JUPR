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
    draw?: string;
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

function seriesGameScoresText(game: PublicTournamentGameResult): string | null {
  const scores = [...(game.game_scores || [])].sort(
    (left, right) => left.game_number - right.game_number
  );
  if (!scores.length) return null;
  return scores
    .map((score) => `Game ${score.game_number}: ${score.score_a}–${score.score_b}`)
    .join(" · ");
}

function tiebreakCriterionLabel(criterion: string): string {
  const labels: Record<string, string> = {
    WINS: "Wins",
    HEAD_TO_HEAD: "Head-to-head",
    POINT_DIFFERENTIAL: "Point differential",
    POINTS_FOR: "Total points scored",
    TEAM_NUMBER: "Original team number"
  };
  const normalized = String(criterion || "").trim().toUpperCase();
  return labels[normalized] || normalized.replaceAll("_", " ").toLowerCase();
}

function tiebreakOutcomeLabel(outcome: string, detail: string): string {
  if (/not applied/i.test(detail)) return "Not applied";
  const labels: Record<string, string> = {
    RESOLVED: "Resolved",
    PARTIAL: "Partially resolved",
    PARTIALLY_RESOLVED: "Partially resolved",
    UNRESOLVED: "Still tied",
    NOT_APPLIED: "Not applied",
    SKIPPED: "Not applied",
    FALLBACK: "Final fallback"
  };
  const normalized = String(outcome || "").trim().toUpperCase();
  return labels[normalized] || normalized.replaceAll("_", " ").toLowerCase();
}

function DrawResults({ draw }: { draw: PublicTournamentDrawResult }) {
  const badge = stateColors[draw.state] || stateColors.SCHEDULED;
  const tiebreakExplanations = draw.tiebreak_explanations || [];
  const rankingPolicyDescription = draw.ranking_policy?.description?.trim() || null;
  const rankingCriteria = draw.ranking_policy?.criteria || [];
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
        <section>
          <h3>Standings</h3>
          <div style={{ overflowX: "auto" }}>
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
                    <th scope="row" style={{ textAlign: "left", padding: "0.45rem" }}>
                      {row.team_name}{row.retired ? " · Retired" : ""}
                    </th>
                    <td style={{ textAlign: "right", padding: "0.45rem" }}>{row.wins ?? 0}</td>
                    <td style={{ textAlign: "right", padding: "0.45rem" }}>{row.losses ?? 0}</td>
                    <td style={{ textAlign: "right", padding: "0.45rem" }}>{row.points_for ?? 0}</td>
                    <td style={{ textAlign: "right", padding: "0.45rem" }}>{row.points_against ?? 0}</td>
                    <td style={{ textAlign: "right", padding: "0.45rem" }}>{row.differential ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {tiebreakExplanations.length ? (
            <details
              data-testid="public-tiebreak-explanation"
              aria-label={`How tied teams were ranked in ${draw.name}`}
              style={{
                marginTop: "0.75rem",
                border: "1px solid #bfdbfe",
                borderRadius: "10px",
                background: "#eff6ff",
                color: "#172554",
                minWidth: 0
              }}
            >
              <summary
                style={{
                  cursor: "pointer",
                  padding: "0.7rem 0.8rem",
                  fontWeight: 800,
                  overflowWrap: "anywhere"
                }}
              >
                How tied teams were ranked
                <span
                  style={{
                    marginLeft: "0.45rem",
                    color: "#1d4ed8",
                    fontSize: "0.78rem",
                    whiteSpace: "nowrap"
                  }}
                >
                  {draw.round_robin_complete ? "Final" : "Provisional"} · {tiebreakExplanations.length} {tiebreakExplanations.length === 1 ? "tie" : "ties"}
                </span>
              </summary>
              <div style={{ padding: "0 0.8rem 0.75rem" }}>
                {rankingPolicyDescription ? (
                  <p style={{ margin: "0 0 0.35rem", color: "#334155", fontSize: "0.82rem" }}>
                    {rankingPolicyDescription}
                  </p>
                ) : null}
                {rankingCriteria.length ? (
                  <p style={{ margin: "0 0 0.35rem", color: "#334155", fontSize: "0.82rem" }}>
                    <strong>Rule order:</strong> {rankingCriteria.map(tiebreakCriterionLabel).join(" → ")}
                  </p>
                ) : null}
                <p style={{ margin: "0 0 0.15rem", color: "#334155", fontSize: "0.82rem", fontWeight: 700 }}>
                  {draw.round_robin_complete
                    ? "Final round-robin order."
                    : "Provisional — this order may change until round-robin play is complete."}
                </p>
                {tiebreakExplanations.map((explanation, explanationIndex) => (
                  <article
                    key={`${explanation.title}:${explanationIndex}`}
                    style={{
                      paddingTop: "0.65rem",
                      borderTop: "1px solid #bfdbfe",
                      overflowWrap: "anywhere"
                    }}
                  >
                    <h4 style={{ margin: 0, color: "#0f172a", fontSize: "0.95rem" }}>{explanation.title}</h4>
                    <p style={{ margin: "0.2rem 0 0", color: "#334155", fontSize: "0.86rem" }}>{explanation.summary}</p>
                    <ol
                      aria-label={`Tie-break steps for ${explanation.title} in ${draw.name}`}
                      style={{ margin: "0.5rem 0 0", paddingLeft: "1.25rem", color: "#334155", fontSize: "0.82rem" }}
                    >
                      {(explanation.steps || []).map((step, stepIndex) => (
                        <li key={`${step.criterion}:${stepIndex}`} style={{ marginTop: stepIndex ? "0.35rem" : 0 }}>
                          <span style={{ display: "flex", alignItems: "center", gap: "0.35rem", flexWrap: "wrap", marginBottom: "0.1rem" }}>
                            <strong>{tiebreakCriterionLabel(step.criterion)}</strong>
                            <span
                              style={{
                                border: "1px solid #bfdbfe",
                                borderRadius: "999px",
                                padding: "0.05rem 0.35rem",
                                background: "#dbeafe",
                                color: "#1e40af",
                                fontSize: "0.68rem",
                                fontWeight: 800,
                                textTransform: "uppercase"
                              }}
                            >
                              {tiebreakOutcomeLabel(step.outcome, step.detail)}
                            </span>
                          </span>
                          <span>{step.detail}</span>
                        </li>
                      ))}
                    </ol>
                  </article>
                ))}
              </div>
            </details>
          ) : null}
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
                {seriesGameScoresText(game) ? (
                  <p style={{ margin: "0.25rem 0 0", color: "#475569", fontSize: "0.85rem" }}>
                    {seriesGameScoresText(game)}
                  </p>
                ) : null}
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
                <span style={{ textAlign: "right" }}>
                  <strong>{scoreText(game)}</strong>
                  {seriesGameScoresText(game) ? (
                    <small style={{ display: "block", marginTop: "0.2rem", color: "#475569" }}>
                      {seriesGameScoresText(game)}
                    </small>
                  ) : null}
                </span>
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

  const currentDraws = data.draws
    .filter((draw) => draw.state === "LIVE" || draw.state === "READY")
    .sort((left, right) => Number(right.state === "LIVE") - Number(left.state === "LIVE"));
  const selectedDrawKey = String(searchParams?.draw || "").trim();
  const selectedCurrentDraw = currentDraws.find((draw) => draw.public_draw_key === selectedDrawKey)
    || currentDraws[0]
    || null;
  const completedDraws = data.draws.filter((draw) => draw.state === "COMPLETE");
  const upcomingDraws = data.draws.filter((draw) => draw.state === "SCHEDULED");
  const drawHref = (publicDrawKey: string) => {
    const query = new URLSearchParams({
      tournament_id: data.tournament.id,
      draw: publicDrawKey
    });
    if (view === "past") query.set("view", "past");
    return `/clubs/${params.clubSlug}/tournament-results?${query.toString()}`;
  };
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
      {currentDraws.length ? (
        <section aria-labelledby="current-draws-title" style={{ marginBottom: "1.25rem" }}>
          <h2 id="current-draws-title" style={{ marginBottom: "0.25rem" }}>Current draws</h2>
          <p style={{ color: "#475569", marginTop: 0 }}>Choose a draw to see its live standings, scores, and bracket.</p>
          <nav
            aria-label="Current tournament draws"
            style={{
              display: "flex",
              gap: "0.6rem",
              overflowX: "auto",
              padding: "0.15rem 0 0.55rem",
              marginBottom: "0.75rem"
            }}
          >
            {currentDraws.map((draw) => {
              const selected = selectedCurrentDraw?.public_draw_key === draw.public_draw_key;
              const badge = stateColors[draw.state] || stateColors.READY;
              return (
                <Link
                  key={draw.public_draw_key}
                  href={drawHref(draw.public_draw_key)}
                  prefetch={false}
                  scroll={false}
                  aria-current={selected ? "page" : undefined}
                  style={{
                    display: "grid",
                    gap: "0.15rem",
                    flex: "0 0 auto",
                    minWidth: "min(18rem, 76vw)",
                    padding: "0.7rem 0.85rem",
                    border: selected ? "2px solid #2563eb" : "1px solid #cbd5e1",
                    borderRadius: "12px",
                    background: selected ? "#eff6ff" : "#ffffff",
                    color: "#0f172a",
                    textDecoration: "none",
                    boxShadow: selected ? "0 0 0 2px rgb(37 99 235 / 10%)" : "none"
                  }}
                >
                  <strong>{draw.event_family_label} · {draw.division_name}</strong>
                  <span style={{ color: "#475569", fontSize: "0.9rem" }}>{draw.name}</span>
                  <span style={{ color: badge.color, fontSize: "0.78rem", fontWeight: 850 }}>{draw.state}</span>
                </Link>
              );
            })}
          </nav>
          {selectedCurrentDraw ? <DrawResults draw={selectedCurrentDraw} /> : null}
        </section>
      ) : null}
      {completedDraws.length ? (
        <section aria-labelledby="completed-draws-title">
          <h2 id="completed-draws-title">Completed draws</h2>
          {completedDraws.map((draw) => <DrawResults key={draw.public_draw_key} draw={draw} />)}
        </section>
      ) : null}
      {upcomingDraws.length ? (
        <section aria-labelledby="upcoming-draws-title">
          <h2 id="upcoming-draws-title">Upcoming draws</h2>
          {upcomingDraws.map((draw) => <DrawResults key={draw.public_draw_key} draw={draw} />)}
        </section>
      ) : null}
      {!data.draws.length ? <article style={cardStyle}><h2 style={{ marginTop: 0 }}>No standard draw results yet</h2><p style={{ color: "#475569", marginBottom: 0 }}>Singles and doubles draws will appear here after tournament staff publish them.</p></article> : null}
      <p><Link href={`/clubs/${params.clubSlug}/tournament-team-results`}>Four-player team results</Link></p>
    </section>
  );
}
