import Link from "next/link";
import { getClubLeagueResults } from "@/lib/api";
import type {
  LeagueResultsHighlights,
  LeagueResultsRecentMatch,
  LeagueResultsStatRow,
  LeagueResultsStanding
} from "@/lib/api";
import PrintButton from "./PrintButton";

type LeagueResultsPageProps = {
  params: { clubSlug: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

type SectionKey = "overall" | "weekly" | "player";

type BarRow = {
  key: string;
  label: string;
  value: number;
  detail?: string;
  href?: string;
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const sectionStyle = {
  marginTop: "1.25rem"
};

const sectionLabels: Record<SectionKey, string> = {
  overall: "Overall",
  weekly: "Weekly",
  player: "Player"
};

function firstParam(searchParams: LeagueResultsPageProps["searchParams"], key: string): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function decodeParam(value: string | null): string | null {
  if (!value) return null;
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function normalizeSection(value: string | null): SectionKey {
  if (value === "weekly" || value === "player") return value;
  return "overall";
}

function safeNumber(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function parsePositiveInt(value: string | null): number | null {
  if (!value) return null;
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return null;
  return Math.round(n);
}

function boundedInt(value: string | null, fallback: number, min: number, max: number): number {
  const parsed = parsePositiveInt(value);
  if (parsed == null) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function ratingLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(3);
}

function dateLabel(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10);
  return date.toISOString().slice(0, 10);
}

function percentLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${Number(value).toFixed(1)}%`;
}

function deltaLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const n = Number(value);
  return `${n >= 0 ? "+" : ""}${n.toFixed(3)}`;
}

function countLabel(value: number | null | undefined, singular: string, plural = `${singular}s`): string {
  const count = Number(value ?? 0);
  return `${count} ${count === 1 ? singular : plural}`;
}

function playerHref(clubSlug: string, playerId: string | number): string {
  return `/clubs/${clubSlug}/players/${playerId}`;
}

function pageHref({
  clubSlug,
  league,
  section,
  week,
  player,
  weeklyMinGames
}: {
  clubSlug: string;
  league?: string | null;
  section?: SectionKey | null;
  week?: number | null;
  player?: string | number | null;
  weeklyMinGames?: number | null;
}): string {
  const params = new URLSearchParams();
  if (league) params.set("league", league);
  if (section && section !== "overall") params.set("section", section);
  if (week) params.set("week", String(week));
  if (player) params.set("player", String(player));
  if (weeklyMinGames) params.set("weekly_min_games", String(weeklyMinGames));
  const query = params.toString();
  return `/clubs/${clubSlug}/league-results${query ? `?${query}` : ""}`;
}

function HighlightList({ title, rows, clubSlug }: { title: string; rows: LeagueResultsStatRow[]; clubSlug: string }) {
  return (
    <article style={cardStyle}>
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      {rows.length === 0 ? <p style={{ color: "#64748b" }}>No data yet.</p> : null}
      <ol style={{ marginBottom: 0, paddingLeft: "1.25rem" }}>
        {rows.map((row) => (
          <li key={`${row.week_num ?? "all"}-${row.player_id}-${title}`}>
            <Link href={playerHref(clubSlug, row.player_id)}>{row.player_name}</Link>
            <span style={{ color: "#475569" }}> — {row.wins ?? 0}-{row.losses ?? 0}, {countLabel(row.games, "game")}</span>
            {row.rating_delta_jupr != null ? <span style={{ color: "#475569" }}> · Δ {deltaLabel(row.rating_delta_jupr)}</span> : null}
          </li>
        ))}
      </ol>
    </article>
  );
}

function HighlightGrid({ highlights, clubSlug }: { highlights: LeagueResultsHighlights; clubSlug: string }) {
  const qualifier = Math.max(1, Number(highlights.min_games || 1));
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem", marginBottom: "1rem" }}>
      <HighlightList title="Biggest climbers" rows={highlights.biggest_climbers} clubSlug={clubSlug} />
      <HighlightList title={`Best win % (${qualifier}+ ${qualifier === 1 ? "game" : "games"})`} rows={highlights.best_win_pct} clubSlug={clubSlug} />
      <HighlightList title="Most active" rows={highlights.most_active} clubSlug={clubSlug} />
    </div>
  );
}

function RecentMatchesTable({ rows, clubSlug }: { rows: LeagueResultsRecentMatch[]; clubSlug: string }) {
  if (!rows.length) return <p style={{ color: "#64748b" }}>No recent league matches found for this player.</p>;
  return (
    <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "14px", background: "white" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "760px" }}>
        <thead>
          <tr>
            {["Date", "Week", "Partner", "Opponents", "Result", "Score", "Rating Δ"].map((heading) => (
              <th key={heading} style={{ textAlign: "left", padding: "0.6rem", borderBottom: "1px solid #cbd5e1", fontSize: "0.8rem", color: "#475569" }}>{heading}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={String(row.match_id)}>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}><Link href={`/clubs/${clubSlug}/matches/${row.match_id}`}>{dateLabel(row.date)}</Link></td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.week_label || (row.week_num ? `Week ${row.week_num}` : "—")}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.partner ? <Link href={playerHref(clubSlug, row.partner.player_id)}>{row.partner.player_name}</Link> : "—"}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.opponents.map((opponent, index) => <span key={String(opponent.player_id)}>{index ? ", " : ""}<Link href={playerHref(clubSlug, opponent.player_id)}>{opponent.player_name}</Link></span>)}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0", fontWeight: 800 }}>{row.result}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.score_for}-{row.score_against}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{deltaLabel(row.rating_delta_jupr)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BarList({ title, rows, emptyText = "No chart data yet." }: { title: string; rows: BarRow[]; emptyText?: string }) {
  const max = Math.max(...rows.map((row) => Math.abs(row.value)), 0);
  return (
    <article style={cardStyle}>
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      {!rows.length ? <p style={{ color: "#64748b" }}>{emptyText}</p> : null}
      <div style={{ display: "grid", gap: "0.65rem" }}>
        {rows.map((row) => {
          const width = max > 0 ? `${Math.max(4, Math.round((Math.abs(row.value) / max) * 100))}%` : "0%";
          const label = row.href ? <Link href={row.href}>{row.label}</Link> : row.label;
          return (
            <div key={row.key}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", fontSize: "0.88rem", marginBottom: "0.25rem" }}>
                <span style={{ fontWeight: 700 }}>{label}</span>
                <span style={{ color: "#475569" }}>{row.detail ?? row.value}</span>
              </div>
              <div style={{ height: "0.6rem", borderRadius: "999px", background: "#e2e8f0", overflow: "hidden" }}>
                <div style={{ width, height: "100%", borderRadius: "999px", background: row.value < 0 ? "#f97316" : "#2563eb" }} />
              </div>
            </div>
          );
        })}
      </div>
    </article>
  );
}

function StandingsTable({ standings, clubSlug }: { standings: LeagueResultsStanding[]; clubSlug: string }) {
  if (!standings.length) return <p>No public standings are available for this league yet.</p>;
  return (
    <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "14px", background: "white" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "720px" }}>
        <thead>
          <tr>
            {[
              "Rank",
              "Player",
              "Rating",
              "Games",
              "Wins",
              "Losses",
              "Win %",
              "Rating Δ"
            ].map((heading) => (
              <th key={heading} style={{ textAlign: "left", padding: "0.6rem", borderBottom: "1px solid #cbd5e1", fontSize: "0.8rem", color: "#475569" }}>{heading}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {standings.map((row) => (
            <tr key={String(row.player_id)}>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.rank ?? "—"}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}><Link href={playerHref(clubSlug, row.player_id)}>{row.player_name}</Link></td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{ratingLabel(row.rating_jupr)}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.matches_played ?? 0}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.losses ?? 0}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{percentLabel(row.win_pct)}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{deltaLabel(row.rating_delta_jupr)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatTable({ rows, clubSlug, title }: { rows: LeagueResultsStatRow[]; clubSlug: string; title: string }) {
  if (!rows.length) return <p style={{ color: "#64748b" }}>No {title.toLowerCase()} data yet.</p>;
  return (
    <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "14px", background: "white" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "640px" }}>
        <thead>
          <tr>
            {[
              "Player",
              "Week",
              "Rank",
              "Rank Δ",
              "Games",
              "Wins",
              "Losses",
              "Win %",
              "Rating Δ"
            ].map((heading) => (
              <th key={heading} style={{ textAlign: "left", padding: "0.6rem", borderBottom: "1px solid #cbd5e1", fontSize: "0.8rem", color: "#475569" }}>{heading}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.week_num ?? "all"}-${row.player_id}`}>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}><Link href={playerHref(clubSlug, row.player_id)}>{row.player_name}</Link></td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.week_num ? `Week ${row.week_num}` : "Season"}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.rank ?? "—"}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.rank_delta == null ? "—" : `${row.rank_delta >= 0 ? "+" : ""}${row.rank_delta}`}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.games ?? 0}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{row.losses ?? 0}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{percentLabel(row.win_pct)}</td>
              <td style={{ padding: "0.6rem", borderBottom: "1px solid #e2e8f0" }}>{deltaLabel(row.rating_delta_jupr)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StandingsCharts({ standings, clubSlug }: { standings: LeagueResultsStanding[]; clubSlug: string }) {
  const topRatings = [...standings]
    .filter((row) => row.rating_jupr != null)
    .sort((a, b) => safeNumber(b.rating_jupr) - safeNumber(a.rating_jupr))
    .slice(0, 10)
    .map((row) => ({
      key: `rating-${row.player_id}`,
      label: row.player_name,
      value: safeNumber(row.rating_jupr),
      detail: ratingLabel(row.rating_jupr),
      href: playerHref(clubSlug, row.player_id)
    }));
  const topMovers = [...standings]
    .filter((row) => row.rating_delta_jupr != null)
    .sort((a, b) => safeNumber(b.rating_delta_jupr) - safeNumber(a.rating_delta_jupr))
    .slice(0, 10)
    .map((row) => ({
      key: `delta-${row.player_id}`,
      label: row.player_name,
      value: safeNumber(row.rating_delta_jupr),
      detail: deltaLabel(row.rating_delta_jupr),
      href: playerHref(clubSlug, row.player_id)
    }));

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem", marginTop: "1rem" }}>
      <BarList title="Top ratings" rows={topRatings} />
      <BarList title="Biggest season movers" rows={topMovers} />
    </div>
  );
}

function PlayerTrend({ rows, playerName }: { rows: LeagueResultsStatRow[]; playerName: string }) {
  const sortedRows = [...rows].sort((a, b) => safeNumber(a.week_num) - safeNumber(b.week_num));
  const ranked = sortedRows.filter((row) => row.rank != null);
  const maxRank = Math.max(...ranked.map((row) => safeNumber(row.rank)), 0);
  const rankRows = ranked.map((row) => ({
    key: `rank-${row.week_num}`,
    label: row.week_num ? `Week ${row.week_num}` : "Season",
    value: maxRank - safeNumber(row.rank) + 1,
    detail: `#${row.rank}${row.rank_delta == null ? "" : ` (${row.rank_delta >= 0 ? "+" : ""}${row.rank_delta})`}`
  }));
  const gamesRows = sortedRows.map((row) => ({
    key: `games-${row.week_num}`,
    label: row.week_num ? `Week ${row.week_num}` : "Season",
    value: safeNumber(row.games),
    detail: countLabel(row.games, "game")
  }));
  const winRows = sortedRows
    .filter((row) => row.win_pct != null)
    .map((row) => ({
      key: `win-${row.week_num}`,
      label: row.week_num ? `Week ${row.week_num}` : "Season",
      value: safeNumber(row.win_pct),
      detail: percentLabel(row.win_pct)
    }));
  const deltaRows = sortedRows
    .filter((row) => row.rating_delta_jupr != null)
    .map((row) => ({
      key: `delta-${row.week_num}`,
      label: row.week_num ? `Week ${row.week_num}` : "Season",
      value: safeNumber(row.rating_delta_jupr),
      detail: deltaLabel(row.rating_delta_jupr)
    }));

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem", marginTop: "1rem" }}>
      <BarList title={`${playerName} — rating rank by week`} rows={rankRows} emptyText="Weekly rank snapshots are unavailable." />
      <BarList title={`${playerName} — games by week`} rows={gamesRows} />
      <BarList title={`${playerName} — win % by week`} rows={winRows} />
      <BarList title={`${playerName} — weekly rating Δ`} rows={deltaRows} />
    </div>
  );
}

export default async function LeagueResultsPage({ params, searchParams }: LeagueResultsPageProps) {
  const { clubSlug } = params;
  const leagueName = decodeParam(firstParam(searchParams, "league"));
  const activeSection = normalizeSection(firstParam(searchParams, "section"));
  const requestedWeek = parsePositiveInt(firstParam(searchParams, "week"));
  const requestedPlayer = firstParam(searchParams, "player");
  const weeklyMinGames = boundedInt(firstParam(searchParams, "weekly_min_games"), 4, 1, 20);
  const { data, error } = await getClubLeagueResults(
    clubSlug,
    leagueName,
    requestedWeek,
    requestedPlayer,
    weeklyMinGames
  );
  const selectedLeague = data?.selected_league ?? null;
  const selectedWeek = data?.selected_week ?? null;
  const recentWeeklyRows = selectedWeek ? (data?.weekly_results ?? []).filter((row) => row.week_num === selectedWeek) : [];
  const playerCandidates = data?.players ?? [];
  const selectedPlayerId = data?.selected_player_id ?? null;
  const selectedPlayerName = data?.player_summary?.player_name ?? "Player";
  const playerWeeklyRows = data?.player_weekly ?? [];
  const showOverall = activeSection === "overall";
  const showWeekly = activeSection === "weekly";
  const showPlayer = activeSection === "player";

  return (
    <section>
      <style>{`
        @media print {
          @page { margin: 8mm; }
          header, footer, nav, .no-print { display: none !important; }
          body { background: white !important; font-size: 10pt; }
          a { color: inherit !important; text-decoration: none !important; }
          main { max-width: none !important; margin: 0 !important; padding: 0 !important; }
          section { color: #0f172a; }
          article { padding: 3mm !important; }
          h1 { font-size: 18pt; margin-bottom: 2mm !important; }
          h2 { font-size: 13pt; margin: 3mm 0 2mm !important; }
          h3 { font-size: 11pt; margin: 2mm 0 !important; }
          table { font-size: 9pt; }
          th, td { padding: 1mm 1.5mm !important; }
          thead { display: table-header-group; }
          tr, h1, h2, h3 { break-inside: avoid; page-break-inside: avoid; }
          section > section { break-inside: auto; }
        }
      `}</style>

      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        League Results
      </p>
      <h1 style={{ marginTop: 0 }}>{data?.club.name ?? clubSlug} league results</h1>
      <p style={{ color: "#334155", maxWidth: "760px" }}>
        Public league standings, weekly results, and season performance. This page is read-only and uses public-safe FastAPI summaries.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>League Results are temporarily unavailable. {error}</p> : null}
      {!error && !data?.leagues?.length ? <p>No public leagues are available yet.</p> : null}

      {data?.leagues?.length ? (
        <div className="no-print" style={{ display: "grid", gap: "0.75rem", marginBottom: "1rem" }}>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            {data.leagues.map((league) => {
              const active = league.name === selectedLeague;
              return (
                <Link key={league.name} href={pageHref({ clubSlug, league: league.name, section: activeSection, week: selectedWeek, player: selectedPlayerId, weeklyMinGames })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                  {league.name}
                </Link>
              );
            })}
          </div>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
            {(Object.keys(sectionLabels) as SectionKey[]).map((section) => {
              const active = section === activeSection;
              return (
                <Link key={section} href={pageHref({ clubSlug, league: selectedLeague, section, week: selectedWeek, player: selectedPlayerId, weeklyMinGames })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                  {sectionLabels[section]}
                </Link>
              );
            })}
            <PrintButton />
          </div>
        </div>
      ) : null}

      {data && selectedLeague ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>League</strong><br />{selectedLeague}</article>
            <article style={cardStyle}><strong>Minimum games</strong><br />{data.league?.min_games ?? 0}</article>
            <article style={cardStyle}><strong>K-factor</strong><br />{data.league?.k_factor ?? "Default"}</article>
            <article style={cardStyle}><strong>Weeks with results</strong><br />{data.weeks.length}</article>
          </div>

          {showOverall ? (
            <section style={sectionStyle}>
              <h2>Current standings</h2>
              <p style={{ color: "#64748b" }}>
                Current rating and rank with the league&apos;s official season record.
                Players awaiting a league rating appear unranked.
              </p>
              <StandingsTable standings={data.standings} clubSlug={clubSlug} />
              <StandingsCharts standings={data.standings} clubSlug={clubSlug} />

              <h2>Season highlights</h2>
              <p style={{ color: "#64748b" }}>Season totals only; win-percentage leaders must meet the league minimum of {data.season_highlights.min_games ?? 1} games.</p>
              <HighlightGrid highlights={data.season_highlights} clubSlug={clubSlug} />
            </section>
          ) : null}

          {showWeekly ? (
            <section style={sectionStyle}>
              <h2>Weekly results{selectedWeek ? ` — Week ${selectedWeek}` : ""}</h2>
              <p style={{ color: "#64748b" }}>
                Weekly results come from active, non-deleted match records; standings use the league&apos;s official rated record.
              </p>
              {data.weeks.length ? (
                <div className="no-print" style={{ display: "flex", gap: "0.45rem", flexWrap: "wrap", marginBottom: "1rem" }}>
                  {data.weeks.map((week) => {
                    const active = week.week_num === selectedWeek;
                    return (
                      <Link key={week.week_num} href={pageHref({ clubSlug, league: selectedLeague, section: "weekly", week: week.week_num, player: selectedPlayerId, weeklyMinGames })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                        {week.week_label}{week.has_results === false ? " · no results" : ""}
                      </Link>
                    );
                  })}
                </div>
              ) : null}
              <div className="no-print" style={{ display: "flex", gap: "0.45rem", flexWrap: "wrap", alignItems: "center", marginBottom: "1rem" }}>
                <strong>Best win % qualification:</strong>
                {[1, 2, 4, 6, 8].map((minimum) => (
                  <Link key={minimum} href={pageHref({ clubSlug, league: selectedLeague, section: "weekly", week: selectedWeek, player: selectedPlayerId, weeklyMinGames: minimum })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.3rem 0.6rem", background: minimum === weeklyMinGames ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: minimum === weeklyMinGames ? 800 : 600 }}>
                    {minimum}+ {minimum === 1 ? "game" : "games"}
                  </Link>
                ))}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem", marginBottom: "1rem" }}>
                <BarList title="Weekly wins" rows={[...recentWeeklyRows].sort((a, b) => safeNumber(b.wins) - safeNumber(a.wins)).slice(0, 10).map((row) => ({ key: `wins-${row.player_id}`, label: row.player_name, value: safeNumber(row.wins), detail: countLabel(row.wins, "win"), href: playerHref(clubSlug, row.player_id) }))} />
                <BarList title="Weekly games" rows={[...recentWeeklyRows].sort((a, b) => safeNumber(b.games) - safeNumber(a.games)).slice(0, 10).map((row) => ({ key: `games-${row.player_id}`, label: row.player_name, value: safeNumber(row.games), detail: countLabel(row.games, "game"), href: playerHref(clubSlug, row.player_id) }))} />
                <BarList title="Weekly rating Δ" rows={recentWeeklyRows.filter((row) => row.rating_delta_jupr != null).sort((a, b) => safeNumber(b.rating_delta_jupr) - safeNumber(a.rating_delta_jupr)).slice(0, 10).map((row) => ({ key: `delta-${row.player_id}`, label: row.player_name, value: safeNumber(row.rating_delta_jupr), detail: deltaLabel(row.rating_delta_jupr), href: playerHref(clubSlug, row.player_id) }))} />
              </div>
              <StatTable title="Weekly" rows={recentWeeklyRows.slice(0, 40)} clubSlug={clubSlug} />
              <h3>Week {selectedWeek ?? "—"} highlights</h3>
              <p style={{ color: "#64748b" }}>These cards are scoped only to the selected week.</p>
              <HighlightGrid highlights={data.weekly_highlights} clubSlug={clubSlug} />
            </section>
          ) : null}

          {showPlayer && selectedPlayerId ? (
            <section style={sectionStyle}>
              <h2>Player summary — {selectedPlayerName}</h2>
              <p style={{ color: "#64748b" }}>
                The summary uses the official rated season record. Weekly and recent activity comes from active, non-deleted matches.
              </p>
              <div className="no-print" style={{ display: "flex", gap: "0.45rem", flexWrap: "wrap", marginBottom: "1rem" }}>
                {playerCandidates.map((row) => {
                  const active = String(row.player_id) === String(selectedPlayerId);
                  return (
                    <Link key={`player-${row.player_id}`} href={pageHref({ clubSlug, league: selectedLeague, section: "player", week: selectedWeek, player: row.player_id, weeklyMinGames })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                      {row.player_name}
                    </Link>
                  );
                })}
              </div>
              <p>
                <Link href={playerHref(clubSlug, selectedPlayerId)}>Open full public player profile</Link>
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem" }}>
                <article style={cardStyle}><strong>Rank</strong><br />{data.player_summary?.rank ?? "—"}</article>
                <article style={cardStyle}><strong>League JUPR</strong><br />{ratingLabel(data.player_summary?.rating_jupr)}</article>
                <article style={cardStyle}><strong>Games</strong><br />{data.player_summary?.games ?? 0}</article>
                <article style={cardStyle}><strong>Win %</strong><br />{percentLabel(data.player_summary?.win_pct)}</article>
              </div>
              <PlayerTrend rows={playerWeeklyRows} playerName={selectedPlayerName} />
              <h3>Player weekly rows</h3>
              <StatTable title="Player weekly" rows={playerWeeklyRows} clubSlug={clubSlug} />
              <h3>Recent matches</h3>
              <RecentMatchesTable rows={data.recent_matches} clubSlug={clubSlug} />
            </section>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
