import Link from "next/link";
import PublicLeagueNav from "@/components/PublicLeagueNav";
import { getClubLeagueResults, type LeagueResultsRecentMatch } from "@/lib/api";

type Props = {
  params: { clubSlug: string; leagueName: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white",
  minWidth: 0
};

const inputStyle = {
  width: "100%",
  boxSizing: "border-box" as const,
  padding: "0.65rem",
  border: "1px solid #cbd5e1",
  borderRadius: "10px",
  background: "white",
  font: "inherit"
};

function decodeLeagueName(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function firstParam(searchParams: Props["searchParams"], key: string): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function ratingLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(3);
}

function percentLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${Number(value).toFixed(1)}%`;
}

function deltaLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const amount = Number(value);
  return `${amount >= 0 ? "+" : ""}${amount.toFixed(3)}`;
}

function dateLabel(value?: string | null): string {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value).slice(0, 10);
  return parsed.toISOString().slice(0, 10);
}

function opponentsLabel(match: LeagueResultsRecentMatch): string {
  return match.opponents.map((opponent) => opponent.player_name).join(" / ") || "—";
}

export default async function PublicLeaguePlayerSummariesPage({ params, searchParams }: Props) {
  const leagueName = decodeLeagueName(params.leagueName);
  const requestedPlayer = firstParam(searchParams, "player");
  const { data, error } = await getClubLeagueResults(
    params.clubSlug,
    leagueName,
    null,
    requestedPlayer
  );
  const found = data?.selected_league === leagueName;

  if (error || !data || !found) {
    return (
      <section>
        <p style={{ color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
          Player Summaries
        </p>
        <h1>{leagueName}</h1>
        <article style={{ ...cardStyle, borderColor: "#fecaca", background: "#fef2f2" }}>
          <h2 style={{ marginTop: 0 }}>Player summaries unavailable</h2>
          <p style={{ color: "#7f1d1d" }}>
            {error || "This league is not currently available as an active public league."}
          </p>
          <Link href={`/clubs/${params.clubSlug}/leagues`}>Return to all leagues</Link>
        </article>
      </section>
    );
  }

  const selectedPlayerId = data.selected_player_id ?? null;
  const summary = data.player_summary;
  const selectedPlayerName = summary?.player_name || "Select a player";

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Player Summaries
      </p>
      <h1 style={{ marginTop: 0 }}>{leagueName} player summaries</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Open any player’s league record, weekly trend, rating movement, and recent matches.
      </p>

      <PublicLeagueNav clubSlug={params.clubSlug} leagueName={leagueName} active="player" />

      <form method="get" style={{ ...cardStyle, display: "grid", gridTemplateColumns: "minmax(0, 1fr) auto", gap: "0.65rem", alignItems: "end", marginBottom: "1rem" }}>
        <label>
          <strong>Player</strong><br />
          <select name="player" defaultValue={selectedPlayerId == null ? "" : String(selectedPlayerId)} style={inputStyle}>
            <option value="">Choose a player</option>
            {data.players.map((player) => (
              <option key={String(player.player_id)} value={String(player.player_id)}>
                {player.player_name}
              </option>
            ))}
          </select>
        </label>
        <button type="submit" style={{ border: "1px solid #0f172a", borderRadius: "999px", padding: "0.65rem 0.95rem", background: "#0f172a", color: "white", fontWeight: 800 }}>
          Open summary
        </button>
      </form>

      {summary && selectedPlayerId != null ? (
        <>
          <article style={{ ...cardStyle, marginBottom: "1rem", background: "#eff6ff", borderColor: "#bfdbfe" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap", alignItems: "flex-start" }}>
              <div>
                <h2 style={{ marginTop: 0 }}>{selectedPlayerName}</h2>
                <Link href={`/clubs/${params.clubSlug}/players/${selectedPlayerId}`}>Open full player profile</Link>
              </div>
              <div style={{ textAlign: "right" }}>
                <strong style={{ fontSize: "1.15rem" }}>{ratingLabel(summary.rating_jupr)}</strong>
                <div style={{ color: "#64748b" }}>{deltaLabel(summary.rating_delta_jupr)} season</div>
              </div>
            </div>
          </article>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>League rank</strong><br />#{summary.rank ?? "—"}</article>
            <article style={cardStyle}><strong>Record</strong><br />{summary.wins ?? 0}-{summary.losses ?? 0}</article>
            <article style={cardStyle}><strong>Games</strong><br />{summary.games ?? 0}</article>
            <article style={cardStyle}><strong>Win percentage</strong><br />{percentLabel(summary.win_pct)}</article>
          </div>

          <section>
            <h2>Weekly trend</h2>
            {data.player_weekly.length ? (
              <div style={{ display: "grid", gap: "0.65rem" }}>
                {data.player_weekly.map((row) => (
                  <article
                    key={`${row.week_num}-${row.player_id}`}
                    style={{
                      ...cardStyle,
                      display: "grid",
                      gridTemplateColumns: "minmax(0, 1fr) auto",
                      gap: "0.75rem",
                      alignItems: "center"
                    }}
                  >
                    <div>
                      <strong>{row.week_num ? `Week ${row.week_num}` : "Season"}</strong>
                      <div style={{ color: "#64748b", fontSize: "0.88rem", marginTop: "0.2rem" }}>
                        {row.wins ?? 0}-{row.losses ?? 0} · {row.games ?? 0} games · {percentLabel(row.win_pct)} wins
                      </div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <strong>#{row.rank ?? "—"}</strong>
                      <div style={{ color: "#64748b", fontSize: "0.82rem" }}>{deltaLabel(row.rating_delta_jupr)}</div>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <article style={cardStyle}>No weekly snapshots are available for this player yet.</article>
            )}
          </section>

          <section style={{ marginTop: "1.25rem" }}>
            <h2>Recent league matches</h2>
            {data.recent_matches.length ? (
              <div style={{ display: "grid", gap: "0.65rem" }}>
                {data.recent_matches.map((match) => (
                  <article key={String(match.match_id)} style={cardStyle}>
                    <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap" }}>
                      <div>
                        <Link href={`/clubs/${params.clubSlug}/matches/${match.match_id}`} style={{ fontWeight: 800 }}>
                          {dateLabel(match.date)} · {match.week_label || (match.week_num ? `Week ${match.week_num}` : "League match")}
                        </Link>
                        <div style={{ color: "#64748b", marginTop: "0.25rem" }}>
                          Partner: {match.partner?.player_name || "—"} · Opponents: {opponentsLabel(match)}
                        </div>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <strong>{match.result} · {match.score_for}-{match.score_against}</strong>
                        <div style={{ color: "#64748b" }}>{deltaLabel(match.rating_delta_jupr)}</div>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <article style={cardStyle}>No recent league matches are available for this player.</article>
            )}
          </section>
        </>
      ) : (
        <article style={cardStyle}>Choose a player to open a league summary.</article>
      )}
    </section>
  );
}
