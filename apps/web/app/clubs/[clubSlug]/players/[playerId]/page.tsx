import Link from "next/link";
import { getClubPlayerProfile, type PublicMatch } from "@/lib/api";

type PlayerProfilePageProps = {
  params: { clubSlug: string; playerId: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

type SectionKey = "overview" | "leagues" | "matches";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const thStyle = { textAlign: "left" as const, borderBottom: "1px solid #cbd5e1", padding: "0.6rem", whiteSpace: "nowrap" as const, color: "#475569", fontSize: "0.82rem" };
const tdStyle = { borderBottom: "1px solid #e2e8f0", padding: "0.6rem", whiteSpace: "nowrap" as const };

function firstParam(searchParams: PlayerProfilePageProps["searchParams"], key: string): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function normalizeSection(value: string | null): SectionKey {
  if (value === "leagues" || value === "matches") return value;
  return "overview";
}

function pageHref({ clubSlug, playerId, section, league }: { clubSlug: string; playerId: string; section?: SectionKey | null; league?: string | null }): string {
  const params = new URLSearchParams();
  if (section && section !== "overview") params.set("section", section);
  if (league) params.set("league", league);
  const query = params.toString();
  return `/clubs/${clubSlug}/players/${playerId}${query ? `?${query}` : ""}`;
}

function ratingValue(value?: number | null): number | null {
  if (value == null || Number.isNaN(Number(value))) return null;
  const n = Number(value);
  return n > 20 ? n / 400 : n;
}

function ratingLabel(value?: number | null): string {
  const rating = ratingValue(value);
  return rating == null ? "—" : rating.toFixed(3);
}

function pctLabel(wins?: number | null, losses?: number | null): string {
  const w = wins ?? 0;
  const l = losses ?? 0;
  const total = w + l;
  return total > 0 ? `${((w / total) * 100).toFixed(1)}%` : "—";
}

function formatMatchDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function dateLabel(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10);
  return date.toISOString().slice(0, 10);
}

function teamLabel(clubSlug: string, players: Array<{ id?: string | number | null; name: string }>) {
  return players.length ? (
    <>
      {players.map((player, index) => (
        <span key={`${player.id ?? player.name}-${index}`}>
          {index > 0 ? " / " : ""}
          {player.id != null ? <Link href={`/clubs/${clubSlug}/players/${player.id}`}>{player.name}</Link> : player.name}
        </span>
      ))}
    </>
  ) : "—";
}

function plainTeamLabel(players: Array<{ name: string }>): string {
  return players.map((p) => p.name).filter(Boolean).join(" / ") || "—";
}

function matchLabel(match: PublicMatch): string {
  const scoreA = match.score_t1 ?? null;
  const scoreB = match.score_t2 ?? null;
  return scoreA == null && scoreB == null ? "—" : `${scoreA ?? 0}–${scoreB ?? 0}`;
}

function sectionVisible(active: SectionKey, section: SectionKey): boolean {
  return active === "overview" || active === section;
}

export default async function PlayerProfilePage({ params, searchParams }: PlayerProfilePageProps) {
  const { clubSlug, playerId } = params;
  const section = normalizeSection(firstParam(searchParams, "section"));
  const selectedLeague = firstParam(searchParams, "league");
  const { data, error } = await getClubPlayerProfile(clubSlug, playerId);
  const player = data?.player;

  if (error || !player) {
    return (
      <section>
        <h1>Player unavailable</h1>
        <p style={{ color: "#b91c1c" }}>We could not load this player profile. {error}</p>
        <p><Link href={`/clubs/${clubSlug}/players`}>Back to players</Link></p>
      </section>
    );
  }

  const wins = player.wins ?? 0;
  const losses = player.losses ?? 0;
  const leagueRatings = data?.league_ratings ?? [];
  const matches = data?.recent_matches ?? [];
  const leagues = Array.from(new Set([...leagueRatings.map((row) => row.league_name).filter(Boolean), ...matches.map((match) => match.league).filter(Boolean)] as string[])).sort((a, b) => a.localeCompare(b));
  const filteredMatches = selectedLeague ? matches.filter((match) => match.league === selectedLeague) : matches;
  const lastMatch = [...matches].sort((a, b) => String(b.date ?? "").localeCompare(String(a.date ?? "")))[0];

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        {data?.club?.name ?? clubSlug} · Player profile
      </p>
      <h1 style={{ marginTop: 0 }}>{player.name}</h1>
      <p style={{ color: "#475569", maxWidth: "760px" }}>
        Public rating, league-specific records, recent match history, and direct links into the surrounding club pages.
      </p>

      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
        {(["overview", "leagues", "matches"] as SectionKey[]).map((item) => {
          const active = item === section;
          return (
            <Link key={item} href={pageHref({ clubSlug, playerId, section: item, league: selectedLeague })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
              {item === "overview" ? "Overview" : item === "leagues" ? "League ratings" : "Recent matches"}
            </Link>
          );
        })}
        <Link href={`/clubs/${clubSlug}/leaderboards?player=${encodeURIComponent(String(player.id))}#leaderboard-player-${encodeURIComponent(String(player.id))}`} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: "white", color: "#0f172a", textDecoration: "none", fontWeight: 700 }}>Leaderboard row</Link>
        <Link href={`/clubs/${clubSlug}/badge-codex`} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: "white", color: "#0f172a", textDecoration: "none", fontWeight: 700 }}>Badge codex</Link>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "1rem", marginBottom: "1rem" }}>
        <article style={cardStyle}><strong>Current rating</strong><div style={{ fontSize: "2rem", fontWeight: 800 }}>{ratingLabel(player.rating)}</div></article>
        <article style={cardStyle}><strong>Matches</strong><div style={{ fontSize: "2rem", fontWeight: 800 }}>{player.matches_played ?? wins + losses}</div></article>
        <article style={cardStyle}><strong>Record</strong><div style={{ fontSize: "2rem", fontWeight: 800 }}>{wins}/{losses}</div></article>
        <article style={cardStyle}><strong>Win %</strong><div style={{ fontSize: "2rem", fontWeight: 800 }}>{pctLabel(wins, losses)}</div></article>
        <article style={cardStyle}><strong>Status</strong><div style={{ fontSize: "2rem", fontWeight: 800 }}>{player.is_active === false ? "Inactive" : "Active"}</div></article>
        <article style={cardStyle}><strong>Last played</strong><div style={{ fontSize: "2rem", fontWeight: 800 }}>{dateLabel(player.last_game_at ?? lastMatch?.date)}</div></article>
      </div>

      {sectionVisible(section, "leagues") ? (
        <section style={{ ...cardStyle, marginBottom: "1rem" }}>
          <h2 style={{ marginTop: 0 }}>League ratings</h2>
          {leagueRatings.length === 0 ? <p style={{ color: "#475569" }}>No league-specific ratings yet.</p> : null}
          {leagueRatings.length > 0 ? (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "640px" }}>
                <thead><tr><th style={thStyle}>League</th><th style={thStyle}>Rating</th><th style={thStyle}>Matches</th><th style={thStyle}>W/L</th><th style={thStyle}>Win %</th><th style={thStyle}>Status</th></tr></thead>
                <tbody>
                  {leagueRatings.map((row, index) => (
                    <tr key={`${row.league_name ?? "league"}-${index}`}>
                      <td style={tdStyle}>{row.league_name ?? "Overall"}</td>
                      <td style={tdStyle}>{ratingLabel(row.rating)}</td>
                      <td style={tdStyle}>{row.matches_played ?? (row.wins ?? 0) + (row.losses ?? 0)}</td>
                      <td style={tdStyle}>{row.wins ?? 0}/{row.losses ?? 0}</td>
                      <td style={tdStyle}>{pctLabel(row.wins, row.losses)}</td>
                      <td style={tdStyle}>{row.is_active === false ? "Inactive" : "Active"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </section>
      ) : null}

      {sectionVisible(section, "matches") ? (
        <section style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Recent matches</h2>
          {leagues.length ? (
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
              <Link href={pageHref({ clubSlug, playerId, section: "matches" })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: !selectedLeague ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: !selectedLeague ? 800 : 600 }}>All leagues</Link>
              {leagues.map((league) => {
                const active = league === selectedLeague;
                return (
                  <Link key={league} href={pageHref({ clubSlug, playerId, section: "matches", league })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: active ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                    {league}
                  </Link>
                );
              })}
            </div>
          ) : null}
          {filteredMatches.length === 0 ? <p style={{ color: "#475569" }}>No recent public matches yet.</p> : null}
          {filteredMatches.length > 0 ? (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "760px" }}>
                <thead><tr><th style={thStyle}>Date</th><th style={thStyle}>Team 1</th><th style={thStyle}>Score</th><th style={thStyle}>Team 2</th><th style={thStyle}>League</th></tr></thead>
                <tbody>
                  {filteredMatches.map((match, index) => {
                    const detailHref = match.id ? `/clubs/${clubSlug}/matches/${match.id}` : `/clubs/${clubSlug}/matches`;
                    return (
                      <tr key={`${match.id ?? index}`}>
                        <td style={tdStyle}>{match.id ? <Link href={detailHref}>{formatMatchDate(match.date)}</Link> : formatMatchDate(match.date)}</td>
                        <td style={tdStyle}>{teamLabel(clubSlug, match.team_1)}</td>
                        <td style={tdStyle}>{match.id ? <Link href={detailHref}>{matchLabel(match)}</Link> : matchLabel(match)}</td>
                        <td style={tdStyle}>{teamLabel(clubSlug, match.team_2)}</td>
                        <td style={tdStyle}>{match.league ?? "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : null}
        </section>
      ) : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href={`/clubs/${clubSlug}/players`}>Back to players</Link>
        <span style={{ color: "#64748b" }}> · </span>
        <Link href={`/clubs/${clubSlug}/matches?q=${encodeURIComponent(player.name)}`}>Search this player in match history</Link>
        <span style={{ color: "#64748b" }}> · </span>
        <Link href={`/clubs/${clubSlug}/match-explorer?me=${encodeURIComponent(String(player.id))}`}>Use in Match Explorer</Link>
      </p>
    </section>
  );
}
