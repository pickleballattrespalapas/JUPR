import Link from "next/link";
import { getClubLeagueResults, getClubPlayerProfile, type LeagueAwardProgressRow, type LeagueResultsResponse, type PublicMatch, type PublicRatingHistoryPoint, type PublicRelationship } from "@/lib/api";

type PlayerProfilePageProps = {
  params: { clubSlug: string; playerId: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

type SectionKey = "overview" | "ratings" | "positions" | "trophies" | "social" | "matches" | "badges";
type HistoryKey = "recent" | "all";
type PlayerAwardPlacement = LeagueAwardProgressRow & { eligible_count?: number | null };
type LeaguePosition = {
  leagueName: string;
  result: LeagueResultsResponse | null;
  error: string | null;
  awards: PlayerAwardPlacement[];
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const thStyle = { textAlign: "left" as const, borderBottom: "1px solid #cbd5e1", padding: "0.6rem", whiteSpace: "nowrap" as const, color: "#475569", fontSize: "0.82rem" };
const tdStyle = { borderBottom: "1px solid #e2e8f0", padding: "0.6rem", whiteSpace: "nowrap" as const };
const pillStyle = { border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", color: "#0f172a", textDecoration: "none" };

function firstParam(searchParams: PlayerProfilePageProps["searchParams"], key: string): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function normalizeSection(value: string | null): SectionKey {
  if (value === "awards") return "trophies";
  if (value === "ratings" || value === "positions" || value === "trophies" || value === "social" || value === "matches" || value === "badges") return value;
  return "overview";
}

function normalizeHistory(value: string | null): HistoryKey {
  return value === "all" ? "all" : "recent";
}

function pageHref({ clubSlug, playerId, section, league, history }: { clubSlug: string; playerId: string; section?: SectionKey | null; league?: string | null; history?: HistoryKey | null }): string {
  const params = new URLSearchParams();
  if (section && section !== "overview") params.set("section", section);
  if (league) params.set("league", league);
  if (history === "all") params.set("history", "all");
  const query = params.toString();
  return `/clubs/${clubSlug}/players/${encodeURIComponent(playerId)}${query ? `?${query}` : ""}`;
}

function ratingLabel(value?: number | null): string {
  return value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(3);
}

function signedRating(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(3)}`;
}

function pctLabel(wins?: number | null, losses?: number | null): string {
  const total = (wins ?? 0) + (losses ?? 0);
  return total ? `${(((wins ?? 0) / total) * 100).toFixed(1)}%` : "—";
}

function formatDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value).slice(0, 10) : date.toISOString().slice(0, 10);
}

function teamLabel(clubSlug: string, players: Array<{ id?: string | number | null; name: string }>) {
  return players.length ? (
    <>{players.map((player, index) => <span key={`${player.id ?? player.name}-${index}`}>{index ? " / " : ""}{player.id != null ? <Link href={`/clubs/${clubSlug}/players/${player.id}`}>{player.name}</Link> : player.name}</span>)}</>
  ) : "—";
}

function matchScore(match: PublicMatch): string {
  return match.score_t1 == null && match.score_t2 == null ? "—" : `${match.score_t1 ?? 0}–${match.score_t2 ?? 0}`;
}

function sectionVisible(active: SectionKey, section: SectionKey): boolean {
  return active === section;
}

function gameLabel(value: number): string {
  return `${value} ${value === 1 ? "game" : "games"}`;
}

function isLeagueName(value?: string | null): value is string {
  const normalized = String(value || "").trim().toLowerCase();
  return Boolean(normalized) && normalized !== "popup" && normalized !== "pop up";
}

function playerAwardPlacements(result: LeagueResultsResponse | null, playerId: string | number): PlayerAwardPlacement[] {
  const races = result?.award_progress.races || [];
  if (races.length) {
    return races.flatMap((race) => (race.entries || [])
      .filter((entry) => String(entry.player_id ?? "") === String(playerId))
      .map((entry) => ({ ...entry, eligible_count: race.eligible_count ?? race.entries.length })));
  }
  return (result?.award_progress.awards || [])
    .filter((award) => String(award.player_id ?? "") === String(playerId));
}

function relationshipCard(clubSlug: string, title: string, testId: string, relationship?: PublicRelationship | null) {
  return (
    <article style={cardStyle} data-testid={testId}>
      <strong>{title}</strong>
      {relationship ? (
        <><h3 style={{ marginBottom: "0.25rem" }}><Link href={`/clubs/${clubSlug}/players/${relationship.player_id}`}>{relationship.player_name}</Link></h3><p style={{ margin: 0, color: "#475569" }}>{relationship.matches} matches · {relationship.wins}-{relationship.losses} · {relationship.win_pct?.toFixed(1) ?? "—"}%</p></>
      ) : <p style={{ color: "#475569" }}>Not enough public match history yet.</p>}
    </article>
  );
}

function RatingTrend({ points, clubSlug }: { points: PublicRatingHistoryPoint[]; clubSlug: string }) {
  const known = points.filter((point) => point.rating_after_jupr != null);
  if (!known.length) return <p data-testid="player-rating-trend-empty" style={{ color: "#475569" }}>No authoritative rating snapshots are available yet.</p>;
  const values = known.map((point) => Number(point.rating_after_jupr));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 0.05);
  const width = 720;
  const height = 190;
  const padding = 24;
  const coordinates = known.map((point, index) => {
    const x = known.length === 1 ? width / 2 : padding + (index / (known.length - 1)) * (width - padding * 2);
    const y = height - padding - ((Number(point.rating_after_jupr) - min) / span) * (height - padding * 2);
    return { point, x, y };
  });
  const series = [
    { format: "doubles", label: "Doubles", color: "#2563eb" },
    { format: "singles", label: "Singles", color: "#7c3aed" }
  ].map((definition) => ({
    ...definition,
    coordinates: coordinates.filter(({ point }) => point.match_format === definition.format)
  })).filter((item) => item.coordinates.length > 0);
  return (
    <div data-testid="player-rating-trend" style={{ overflowX: "auto" }}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Separate doubles and singles JUPR trends by rated match" style={{ width: "100%", minWidth: "560px", height: "auto", border: "1px solid #e2e8f0", borderRadius: "10px", background: "#f8fafc" }}>
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#cbd5e1" />
        {series.map((item) => item.coordinates.length > 1 ? <polyline key={item.format} data-rating-series={item.format} points={item.coordinates.map(({ x, y }) => `${x},${y}`).join(" ")} fill="none" stroke={item.color} strokeWidth="4" strokeLinejoin="round" /> : null)}
        {series.flatMap((item) => item.coordinates.map(({ point, x, y }) => <circle key={`${item.format}-${point.match_id ?? point.match_number}-${point.match_number}`} data-rating-point={item.format} cx={x} cy={y} r="5" fill={item.color}><title>{`${formatDate(point.date)} · ${item.label} · ${ratingLabel(point.rating_after_jupr)}`}</title></circle>))}
      </svg>
      <p style={{ margin: "0.45rem 0 0", color: "#475569", fontSize: "0.88rem" }}>{series.map((item, index) => <span key={item.format}>{index ? " · " : ""}<span aria-hidden="true" style={{ color: item.color }}>●</span> {item.label}</span>)}. Each format has its own line so a switch between doubles and singles never looks like a rating jump or drop.</p>
      <ul style={{ display: "none" }}>{known.map((point) => <li key={`trend-${point.match_id ?? point.match_number}`}><Link href={`/clubs/${clubSlug}/matches/${point.match_id}`}>{formatDate(point.date)} {point.match_format_label} {ratingLabel(point.rating_after_jupr)}</Link></li>)}</ul>
    </div>
  );
}

export default async function PlayerProfilePage({ params, searchParams }: PlayerProfilePageProps) {
  const { clubSlug, playerId } = params;
  const section = normalizeSection(firstParam(searchParams, "section"));
  const selectedLeague = firstParam(searchParams, "league");
  const historyView = normalizeHistory(firstParam(searchParams, "history"));
  const { data, error } = await getClubPlayerProfile(clubSlug, playerId, { recent: 12, history: 500 });
  const player = data?.player;

  if (error || !player || !data) {
    return (
      <section data-testid="player-profile-error-state">
        <h1>Player unavailable</h1>
        <p style={{ color: "#b91c1c" }}>We could not load this public player profile. No private player data was exposed. Please try again shortly.</p>
        <p><Link href={`/clubs/${clubSlug}/players`}>Back to players</Link></p>
      </section>
    );
  }

  const leagueRatings = data.league_ratings ?? [];
  const sourceMatches = historyView === "all" ? data.match_history : data.recent_matches;
  const leagues = Array.from(new Set([...leagueRatings.map((row) => row.league_name).filter(Boolean), ...data.match_history.map((match) => match.league).filter(Boolean)] as string[])).sort((a, b) => a.localeCompare(b));
  const leaguePositionNames = leagues.filter(isLeagueName);
  const matches = selectedLeague ? sourceMatches.filter((match) => match.league === selectedLeague) : sourceMatches;
  const awards = data.awards;
  const social = data.social;
  const verifiedLabel = data.verified_updates.status === "enabled" ? "Verified updates enabled" : data.verified_updates.status === "pending" ? "Verified updates pending review" : "Verified updates available";
  const leaguePositions: LeaguePosition[] = section === "positions"
    ? await Promise.all(leaguePositionNames.map(async (leagueName) => {
      const response = await getClubLeagueResults(clubSlug, leagueName, null, player.id);
      const result = response.data;
      return {
        leagueName,
        result,
        error: response.error,
        awards: playerAwardPlacements(result, player.id)
      };
    }))
    : [];

  return (
    <section data-testid="player-profile">
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>{data.club?.name ?? clubSlug} · Player profile</p>
      <h1 style={{ marginTop: 0 }}>{data.identity.display_name}</h1>
      <p style={{ color: "#475569", maxWidth: "820px" }}>Public display-name profile with Python-authoritative ratings, match formats, earned awards, match relationships, and privacy-safe Club Social aggregates.</p>

      <article data-testid="player-public-identity" style={{ ...cardStyle, background: "#eff6ff", marginBottom: "1rem" }}>
        <strong>{verifiedLabel}</strong>
        <p style={{ margin: "0.35rem 0", color: "#334155" }}>This page uses the player&apos;s approved public display name. Contact details, legal names, social identity keys, and subscription records are never included.</p>
        {data.verified_updates.can_request ? <Link href={`/clubs/${clubSlug}/verified-updates?player_id=${encodeURIComponent(String(player.id))}`}>Request verified player updates</Link> : <span>Update access is managed through the private link in the verified email.</span>}
        <span style={{ color: "#64748b" }}> · </span><Link href="/profile-privacy">Request an alias or privacy review</Link>
      </article>

      <nav aria-label="Player profile sections" style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
        {(["overview", "ratings", "positions", "trophies", "social", "matches", "badges"] as SectionKey[]).map((item) => (
          <Link key={item} data-testid={`player-section-${item}`} aria-current={item === section ? "page" : undefined} href={pageHref({ clubSlug, playerId, section: item, league: selectedLeague, history: historyView })} style={{ ...pillStyle, background: item === section ? "#dbeafe" : "white", fontWeight: item === section ? 800 : 600 }}>
            {item === "overview" ? "Overview" : item === "positions" ? "League positions" : item === "trophies" ? "Trophy case" : item === "badges" ? "Badge cabinet" : item[0].toUpperCase() + item.slice(1)}
          </Link>
        ))}
        <Link href={`/clubs/${clubSlug}/verified-updates?player_id=${encodeURIComponent(String(player.id))}`} style={{ ...pillStyle, background: "white", fontWeight: 700 }}>Request verified updates</Link>
      </nav>

      <div data-testid="player-summary-cards" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(165px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
        <article style={cardStyle}><strong>Doubles / overall</strong><div style={{ fontSize: "1.8rem", fontWeight: 800 }}>{ratingLabel(player.rating_jupr)}</div></article>
        <article style={cardStyle}><strong>Singles</strong><div style={{ fontSize: "1.8rem", fontWeight: 800 }}>{ratingLabel(player.singles_rating_jupr)}</div></article>
        <article style={cardStyle}><strong>Doubles record</strong><div style={{ fontSize: "1.8rem", fontWeight: 800 }}>{player.wins ?? 0}-{player.losses ?? 0}</div><small>{pctLabel(player.wins, player.losses)}</small></article>
        <article style={cardStyle}><strong>Singles record</strong><div style={{ fontSize: "1.8rem", fontWeight: 800 }}>{player.singles_wins ?? 0}-{player.singles_losses ?? 0}</div><small>{pctLabel(player.singles_wins, player.singles_losses)}</small></article>
        <article style={cardStyle}><strong>Major honors</strong><div style={{ fontSize: "1.8rem", fontWeight: 800 }}>{awards.trophy_count ?? awards.trophies.length}</div><small>Trophy case</small></article>
        <article style={cardStyle}><strong>Badges earned</strong><div style={{ fontSize: "1.8rem", fontWeight: 800 }}>{awards.badge_award_count}</div><small>{awards.prestige_total} prestige</small></article>
        <article style={cardStyle}><strong>Last played</strong><div style={{ fontSize: "1.35rem", fontWeight: 800 }}>{formatDate(player.last_game_at)}</div><small>{player.is_active === false ? "Inactive" : "Active"}</small></article>
      </div>

      {sectionVisible(section, "overview") ? (
        <section data-testid="player-overview" style={{ display: "grid", gap: "1rem", marginBottom: "1rem" }}>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Player overview</h2>
            <p style={{ color: "#475569", marginBottom: 0 }}>
              Browse ratings, league positions, major honors, social play, match history, and repeatable badges without leaving this player&apos;s profile.
            </p>
          </article>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", gap: "0.75rem" }}>
            <article style={cardStyle}><strong>League positions</strong><p style={{ color: "#475569" }}>Awards-race placement, league rating, record, and qualification status by league.</p><Link href={pageHref({ clubSlug, playerId, section: "positions" })}>Open league positions</Link></article>
            <article style={cardStyle}><strong>Trophy case</strong><p style={{ color: "#475569" }}>End-of-league awards and tournament podium honors only.</p><Link href={pageHref({ clubSlug, playerId, section: "trophies" })}>Open trophy case</Link></article>
            <article style={cardStyle}><strong>Badge cabinet</strong><p style={{ color: "#475569" }}>Repeatable progression and participation achievements.</p><Link href={pageHref({ clubSlug, playerId, section: "badges" })}>Open badge cabinet</Link></article>
          </div>
        </section>
      ) : null}

      {sectionVisible(section, "positions") ? (
        <section id="positions" data-testid="player-league-positions" style={{ display: "grid", gap: "1rem", marginBottom: "1rem" }}>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>My league positions</h2>
            <p style={{ color: "#475569", marginBottom: 0 }}>This is a player-only view of current awards-race placement, league rating, record, and eligibility.</p>
          </article>
          {leaguePositions.length === 0 ? <article style={cardStyle}>No public league positions yet.</article> : leaguePositions.map((position) => {
            const summary = position.result?.player_summary;
            const minimumGames = Number(position.result?.league?.min_games ?? 0);
            const games = Number(summary?.games ?? 0);
            const eligible = minimumGames <= 0 || games >= minimumGames;
            const needed = Math.max(0, minimumGames - games);
            return (
              <article key={position.leagueName} style={cardStyle} data-testid="player-league-position">
                <h3 style={{ marginTop: 0 }}>{position.leagueName}</h3>
                {position.error || !summary ? <p style={{ color: "#92400e" }}>League-position data is not available right now.</p> : <>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(145px, 1fr))", gap: "0.65rem", marginBottom: "0.85rem" }}>
                    <div><strong>League rating</strong><br />{ratingLabel(summary.rating_jupr)}</div>
                    <div><strong>Record</strong><br />{summary.wins ?? 0}-{summary.losses ?? 0}</div>
                    <div><strong>Qualification</strong><br />{eligible ? `Eligible (${gameLabel(games)})` : `${gameLabel(needed)} needed`}</div>
                  </div>
                  <div style={{ borderTop: "1px solid #e2e8f0", paddingTop: "0.75rem" }}>
                    <strong>Awards race</strong>
                    {position.awards.length ? <ul style={{ marginBottom: 0 }}>{position.awards.map((award) => <li key={`${award.category_key}-${award.rank}`}>{award.category_label} — #{award.rank ?? "—"}{award.eligible_count ? ` of ${award.eligible_count} eligible` : ""} · {award.metric_display || "—"}{award.is_co_winner ? " · tied" : ""}</li>)}</ul> : <p style={{ color: "#475569", marginBottom: 0 }}>{eligible ? "No current awards-race placement is available yet." : `Awards become eligible after ${gameLabel(minimumGames)}.`}</p>}
                  </div>
                </>}
              </article>
            );
          })}
        </section>
      ) : null}

      {sectionVisible(section, "ratings") ? (
        <section id="ratings" data-testid="player-ratings" style={{ display: "grid", gap: "1rem", marginBottom: "1rem" }}>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Rating trend and snapshot</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.65rem", marginBottom: "1rem" }}>
              <div><strong>Starting JUPR</strong><br />{ratingLabel(data.rating_summary.starting_rating_jupr)}</div>
              <div><strong>Highest JUPR</strong><br />{ratingLabel(data.rating_summary.highest_rating_jupr)}</div>
              <div><strong>Lowest JUPR</strong><br />{ratingLabel(data.rating_summary.lowest_rating_jupr)}</div>
              <div><strong>Last 10</strong><br />{data.rating_summary.last_10_record}</div>
              <div><strong>Last 10 delta</strong><br />{signedRating(data.rating_summary.last_10_delta_jupr)}</div>
              <div><strong>Current streak</strong><br />{data.rating_summary.current_streak ?? "—"}</div>
            </div>
            <RatingTrend points={data.rating_history} clubSlug={clubSlug} />
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Format breakdown</h2>
            <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", minWidth: "600px" }}><thead><tr><th style={thStyle}>Format</th><th style={thStyle}>Matches</th><th style={thStyle}>Record</th><th style={thStyle}>Win %</th><th style={thStyle}>Rating delta</th></tr></thead><tbody>
              {data.rating_breakdowns.map((row) => <tr key={row.format} data-testid="player-format-row"><td style={tdStyle}>{row.label}</td><td style={tdStyle}>{row.matches}</td><td style={tdStyle}>{row.wins}-{row.losses}</td><td style={tdStyle}>{row.win_pct?.toFixed(1) ?? "—"}%</td><td style={tdStyle}>{signedRating(row.rating_delta_jupr)}</td></tr>)}
            </tbody></table></div>
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>League rating breakdown</h2>
            {leagueRatings.length === 0 ? <p style={{ color: "#475569" }}>No league-specific ratings yet.</p> : (
              <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", minWidth: "700px" }}><thead><tr><th style={thStyle}>League</th><th style={thStyle}>JUPR</th><th style={thStyle}>Gain</th><th style={thStyle}>Matches</th><th style={thStyle}>Record</th><th style={thStyle}>Status</th></tr></thead><tbody>
                {leagueRatings.map((row, index) => <tr key={`${row.league_name ?? "league"}-${index}`}><td style={tdStyle}>{row.league_name ?? "Overall"}</td><td style={tdStyle}>{ratingLabel(row.rating_jupr)}</td><td style={tdStyle}>{signedRating(row.rating_gain_jupr)}</td><td style={tdStyle}>{row.matches_played ?? 0}</td><td style={tdStyle}>{row.wins ?? 0}-{row.losses ?? 0}</td><td style={tdStyle}>{row.is_active === false ? "Inactive" : "Active"}</td></tr>)}
              </tbody></table></div>
            )}
          </article>
        </section>
      ) : null}

      {sectionVisible(section, "trophies") ? (
        <section id="trophies" data-testid="player-trophies" style={{ display: "grid", gap: "1rem", marginBottom: "1rem" }}>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Trophy case</h2>
            <p style={{ color: "#475569" }}>Major honors only: end-of-league awards and tournament podiums. Progression and repeatable achievements live in the Badge Cabinet.</p>
            {awards.trophies.length === 0 ? <p style={{ color: "#475569" }}>No end-of-league awards or tournament podium honors yet.</p> : <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem" }}>{awards.trophies.map((trophy, index) => <div key={`${trophy.badge_id}-${trophy.earned_at ?? index}`} data-testid="player-trophy" style={{ border: "1px solid #f59e0b", borderRadius: "12px", padding: "0.85rem", background: "#fffbeb" }}><strong>🏆 {trophy.title}</strong><p style={{ margin: "0.35rem 0" }}>{trophy.placement ? `Place #${trophy.placement}` : "Major award"}{trophy.context_label ? ` · ${trophy.context_label}` : ""}</p><small>{formatDate(trophy.earned_at)}</small></div>)}</div>}
          </article>
        </section>
      ) : null}

      {sectionVisible(section, "badges") ? (
        <section id="badges" data-testid="player-badges" style={{ display: "grid", gap: "1rem", marginBottom: "1rem" }}>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Badge cabinet</h2>
            <p style={{ color: "#475569" }}>Repeatable progression, momentum, participation, and skill achievements.</p>
            {awards.badges.length === 0 ? <p style={{ color: "#475569" }}>No repeatable badges earned yet.</p> : <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>{awards.badges.map((badge) => <article key={badge.badge_id} data-testid="player-badge" style={{ border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.85rem" }}><strong>🏅 {badge.name}{badge.count > 1 ? ` ×${badge.count}` : ""}</strong><p style={{ margin: "0.35rem 0", color: "#475569" }}>{badge.category} · {badge.prestige} prestige{badge.rarity ? ` · ${badge.rarity}` : ""}</p><small>{badge.description ?? badge.requirements ?? `Last earned ${formatDate(badge.last_earned_at)}`}</small></article>)}</div>}
          </article>
        </section>
      ) : null}

      {sectionVisible(section, "social") ? (
        <section id="social" data-testid="player-social" style={{ ...cardStyle, marginBottom: "1rem" }}>
          <h2 style={{ marginTop: 0 }}>Club Social projection</h2>
          <p><strong>{social.identity.label}</strong></p>
          {!social.available ? <p style={{ color: "#92400e" }}>Club Social aggregates are unavailable; no identity or event details were exposed.</p> : null}
          {social.available && social.summary ? <>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.65rem", marginBottom: "1rem" }}>
              <div><strong>Events</strong><br />{social.summary.events}</div><div><strong>Matches</strong><br />{social.summary.matches}</div><div><strong>Record</strong><br />{social.summary.wins}-{social.summary.losses}</div><div><strong>Score diff</strong><br />{social.summary.score_diff >= 0 ? "+" : ""}{social.summary.score_diff}</div><div><strong>Last appearance</strong><br />{formatDate(social.summary.last_appearance)}</div>
            </div>
            {social.skill_breakdown.length ? <div style={{ overflowX: "auto", marginBottom: "1rem" }}><h3>Skill-level breakdown</h3><table style={{ width: "100%", borderCollapse: "collapse", minWidth: "620px" }}><thead><tr><th style={thStyle}>Skill</th><th style={thStyle}>Events</th><th style={thStyle}>Matches</th><th style={thStyle}>Record</th><th style={thStyle}>Diff</th></tr></thead><tbody>{social.skill_breakdown.map((row) => <tr key={row.label}><td style={tdStyle}>{row.label}</td><td style={tdStyle}>{row.events}</td><td style={tdStyle}>{row.matches}</td><td style={tdStyle}>{row.wins}-{row.losses}</td><td style={tdStyle}>{row.score_diff >= 0 ? "+" : ""}{row.score_diff}</td></tr>)}</tbody></table></div> : null}
            <h3>Recent social events</h3>
            {social.recent_events.length ? <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", minWidth: "700px" }}><thead><tr><th style={thStyle}>Date</th><th style={thStyle}>Event</th><th style={thStyle}>Format</th><th style={thStyle}>Skill</th><th style={thStyle}>Record</th><th style={thStyle}>Diff</th></tr></thead><tbody>{social.recent_events.map((event, index) => <tr key={`${event.date ?? index}-${event.name}`}><td style={tdStyle}>{formatDate(event.date)}</td><td style={tdStyle}>{event.name}</td><td style={tdStyle}>{event.event_type}</td><td style={tdStyle}>{event.skill_labels.join(", ")}</td><td style={tdStyle}>{event.wins}-{event.losses}</td><td style={tdStyle}>{event.score_diff >= 0 ? "+" : ""}{event.score_diff}</td></tr>)}</tbody></table></div> : <p style={{ color: "#475569" }}>No linked public Club Social history yet.</p>}
          </> : null}
          <div style={{ borderTop: "1px solid #e2e8f0", marginTop: "1rem", paddingTop: "1rem" }}>
            <h2>Match relationships</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem" }}>
              {relationshipCard(clubSlug, "Best partner", "player-best-partner", data.relationships.best_partner)}
              {relationshipCard(clubSlug, "Rival", "player-rival", data.relationships.rival)}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem", marginTop: "1rem" }}>
              <div><h3>Frequent partners</h3>{data.relationships.partners.length ? <ul>{data.relationships.partners.map((row) => <li key={`partner-${row.player_id}`}><Link href={`/clubs/${clubSlug}/players/${row.player_id}`}>{row.player_name}</Link> · {row.matches} · {row.wins}-{row.losses}</li>)}</ul> : <p style={{ color: "#475569" }}>No doubles partners yet.</p>}</div>
              <div><h3>Frequent opponents</h3>{data.relationships.rivals.length ? <ul>{data.relationships.rivals.map((row) => <li key={`rival-${row.player_id}`}><Link href={`/clubs/${clubSlug}/players/${row.player_id}`}>{row.player_name}</Link> · {row.matches} · {row.wins}-{row.losses}</li>)}</ul> : <p style={{ color: "#475569" }}>No opponents yet.</p>}</div>
            </div>
          </div>
        </section>
      ) : null}

      {sectionVisible(section, "matches") ? (
        <section id="matches" data-testid="player-match-history" style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>{historyView === "all" ? "Full public match history" : "Recent matches"}</h2>
          <p style={{ color: "#475569" }}>Every row carries an explicit Singles or Doubles format label. Rating changes are shown only when a stored server snapshot exists.</p>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "0.75rem" }}>
            <Link data-testid="player-history-recent" aria-current={historyView === "recent" ? "page" : undefined} href={pageHref({ clubSlug, playerId, section: "matches", league: selectedLeague, history: "recent" })} style={{ ...pillStyle, background: historyView === "recent" ? "#dcfce7" : "white" }}>Recent {data.history.recent_limit}</Link>
            <Link data-testid="player-history-all" aria-current={historyView === "all" ? "page" : undefined} href={pageHref({ clubSlug, playerId, section: "matches", league: selectedLeague, history: "all" })} style={{ ...pillStyle, background: historyView === "all" ? "#dcfce7" : "white" }}>Full history ({data.history.total_matches})</Link>
          </div>
          {leagues.length ? <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}><Link href={pageHref({ clubSlug, playerId, section: "matches", history: historyView })} style={{ ...pillStyle, background: !selectedLeague ? "#dbeafe" : "white" }}>All leagues</Link>{leagues.map((league) => <Link key={league} href={pageHref({ clubSlug, playerId, section: "matches", league, history: historyView })} style={{ ...pillStyle, background: selectedLeague === league ? "#dbeafe" : "white" }}>{league}</Link>)}</div> : null}
          {matches.length === 0 ? <p data-testid="player-match-empty" style={{ color: "#475569" }}>No public matches match this view.</p> : <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", minWidth: "1040px" }}><thead><tr><th style={thStyle}>Date</th><th style={thStyle}>Format</th><th style={thStyle}>Result</th><th style={thStyle}>Team 1</th><th style={thStyle}>Score</th><th style={thStyle}>Team 2</th><th style={thStyle}>JUPR before → after</th><th style={thStyle}>League</th></tr></thead><tbody>
            {matches.map((match, index) => { const detailHref = match.id != null ? `/clubs/${clubSlug}/matches/${match.id}` : `/clubs/${clubSlug}/matches`; return <tr key={`${match.id ?? index}`} data-testid="player-match-row" data-format={match.match_format ?? "doubles"}><td style={tdStyle}><Link href={detailHref}>{formatDate(match.date)}</Link></td><td style={tdStyle}>{match.match_format_label ?? (match.match_format === "singles" ? "Singles" : "Doubles")}</td><td style={tdStyle}>{match.player_result === "win" ? "Win" : match.player_result === "loss" ? "Loss" : "—"}</td><td style={tdStyle}>{teamLabel(clubSlug, match.team_1)}</td><td style={tdStyle}><Link href={detailHref}>{matchScore(match)}</Link></td><td style={tdStyle}>{teamLabel(clubSlug, match.team_2)}</td><td style={tdStyle}>{ratingLabel(match.player_rating_before_jupr)} → {ratingLabel(match.player_rating_after_jupr)} ({signedRating(match.player_rating_delta_jupr)})</td><td style={tdStyle}>{match.league ?? "—"}</td></tr>; })}
          </tbody></table></div>}
          {data.history.has_more && historyView === "all" ? <p style={{ color: "#92400e" }}>Showing the newest {data.history.history_limit} public matches. Search the club match history for earlier archive access.</p> : null}
        </section>
      ) : null}

      <p style={{ marginTop: "1rem" }}><Link href={`/clubs/${clubSlug}/players`}>Back to active players</Link><span style={{ color: "#64748b" }}> · </span><Link href={`/clubs/${clubSlug}/matches?q=${encodeURIComponent(player.name)}`}>Search this player in match history</Link><span style={{ color: "#64748b" }}> · </span><Link href={`/clubs/${clubSlug}/match-explorer?me=${encodeURIComponent(String(player.id))}`}>Use in Match Explorer</Link></p>
    </section>
  );
}
