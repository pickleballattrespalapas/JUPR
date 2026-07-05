import Link from "next/link";
import { getClubLeagueResults } from "@/lib/api";
import type { LeagueResultsStatRow, LeagueResultsStanding } from "@/lib/api";
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
  marginTop: "1.25rem",
  pageBreakInside: "avoid" as const
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
  const n = Number(value);
  return `${n >= 0 ? "+" : ""}${n.toFixed(3)}`;
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
  print
}: {
  clubSlug: string;
  league?: string | null;
  section?: SectionKey | null;
  week?: number | null;
  player?: string | number | null;
  print?: boolean;
}): string {
  const params = new URLSearchParams();
  if (league) params.set("league", league);
  if (section && section !== "overall") params.set("section", section);
  if (week) params.set("week", String(week));
  if (player) params.set("player", String(player));
  if (print) params.set("print", "1");
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
            <span style={{ color: "#475569" }}> — {row.wins ?? 0}-{row.losses ?? 0}, {row.games ?? 0} games</span>
            {row.rating_delta_jupr != null ? <span style={{ color: "#475569" }}> · Δ {deltaLabel(row.rating_delta_jupr)}</span> : null}
          </li>
        ))}
      </ol>
    </article>
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
  const gamesRows = sortedRows.map((row) => ({
    key: `games-${row.week_num}`,
    label: row.week_num ? `Week ${row.week_num}` : "Season",
    value: safeNumber(row.games),
    detail: `${row.games ?? 0} games`
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
  const printMode = firstParam(searchParams, "print") === "1";
  const { data, error } = await getClubLeagueResults(clubSlug, leagueName);
  const selectedLeague = data?.selected_league ?? null;
  const recentWeek = data?.weeks?.length ? data.weeks[data.weeks.length - 1]?.week_num : null;
  const requestedWeek = parsePositiveInt(firstParam(searchParams, "week"));
  const selectedWeek = requestedWeek ?? recentWeek;
  const recentWeeklyRows = selectedWeek ? (data?.weekly_results ?? []).filter((row) => row.week_num === selectedWeek) : [];
  const requestedPlayer = firstParam(searchParams, "player");
  const playerCandidates = [...(data?.standings ?? []), ...(data?.cumulative ?? [])];
  const selectedPlayerId = requestedPlayer && playerCandidates.some((row) => String(row.player_id) === requestedPlayer)
    ? requestedPlayer
    : playerCandidates[0]?.player_id ?? null;
  const selectedPlayerName = selectedPlayerId
    ? playerCandidates.find((row) => String(row.player_id) === String(selectedPlayerId))?.player_name ?? `#${selectedPlayerId}`
    : "Player";
  const playerWeeklyRows = selectedPlayerId
    ? (data?.weekly_results ?? []).filter((row) => String(row.player_id) === String(selectedPlayerId))
    : [];
  const showOverall = printMode || activeSection === "overall";
  const showWeekly = printMode || activeSection === "weekly";
  const showPlayer = printMode || activeSection === "player";

  return (
    <section>
      <style>{`
        @media print {
          .no-print { display: none !important; }
          body { background: white !important; }
          a { color: inherit !important; text-decoration: none !important; }
          section { color: #0f172a; }
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
                <Link key={league.name} href={pageHref({ clubSlug, league: league.name, section: activeSection, week: selectedWeek, player: selectedPlayerId })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                  {league.name}
                </Link>
              );
            })}
          </div>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
            {(Object.keys(sectionLabels) as SectionKey[]).map((section) => {
              const active = section === activeSection;
              return (
                <Link key={section} href={pageHref({ clubSlug, league: selectedLeague, section, week: selectedWeek, player: selectedPlayerId })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                  {sectionLabels[section]}
                </Link>
              );
            })}
            <Link href={pageHref({ clubSlug, league: selectedLeague, week: selectedWeek, player: selectedPlayerId, print: true })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: printMode ? "#fef9c3" : "white", color: "#0f172a", textDecoration: "none", fontWeight: 800 }}>
              Print view
            </Link>
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
              <StandingsTable standings={data.standings} clubSlug={clubSlug} />
              <StandingsCharts standings={data.standings} clubSlug={clubSlug} />

              <h2>Season highlights</h2>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem", marginBottom: "1rem" }}>
                <HighlightList title="Biggest climbers" rows={data.highlights.biggest_climbers} clubSlug={clubSlug} />
                <HighlightList title="Best win %" rows={data.highlights.best_win_pct} clubSlug={clubSlug} />
                <HighlightList title="Most active" rows={data.highlights.most_active} clubSlug={clubSlug} />
              </div>

              <h2>Season cumulative performance</h2>
              <StatTable title="Season" rows={data.cumulative.slice(0, 25)} clubSlug={clubSlug} />
            </section>
          ) : null}

          {showWeekly ? (
            <section style={sectionStyle}>
              <h2>Weekly results{selectedWeek ? ` — Week ${selectedWeek}` : ""}</h2>
              {data.weeks.length ? (
                <div className="no-print" style={{ display: "flex", gap: "0.45rem", flexWrap: "wrap", marginBottom: "1rem" }}>
                  {data.weeks.map((week) => {
                    const active = week.week_num === selectedWeek;
                    return (
                      <Link key={week.week_num} href={pageHref({ clubSlug, league: selectedLeague, section: "weekly", week: week.week_num, player: selectedPlayerId })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                        {week.week_label}
                      </Link>
                    );
                  })}
                </div>
              ) : null}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem", marginBottom: "1rem" }}>
                <BarList title="Weekly wins" rows={recentWeeklyRows.slice(0, 10).map((row) => ({ key: `wins-${row.player_id}`, label: row.player_name, value: safeNumber(row.wins), detail: `${row.wins ?? 0} wins`, href: playerHref(clubSlug, row.player_id) }))} />
                <BarList title="Weekly games" rows={recentWeeklyRows.slice(0, 10).map((row) => ({ key: `games-${row.player_id}`, label: row.player_name, value: safeNumber(row.games), detail: `${row.games ?? 0} games`, href: playerHref(clubSlug, row.player_id) }))} />
                <BarList title="Weekly rating Δ" rows={recentWeeklyRows.filter((row) => row.rating_delta_jupr != null).slice(0, 10).map((row) => ({ key: `delta-${row.player_id}`, label: row.player_name, value: safeNumber(row.rating_delta_jupr), detail: deltaLabel(row.rating_delta_jupr), href: playerHref(clubSlug, row.player_id) }))} />
              </div>
              <StatTable title="Weekly" rows={recentWeeklyRows.slice(0, 40)} clubSlug={clubSlug} />
            </section>
          ) : null}

          {showPlayer && selectedPlayerId ? (
            <section style={sectionStyle}>
              <h2>Player summary — {selectedPlayerName}</h2>
              <div className="no-print" style={{ display: "flex", gap: "0.45rem", flexWrap: "wrap", marginBottom: "1rem" }}>
                {playerCandidates.slice(0, 24).map((row) => {
                  const active = String(row.player_id) === String(selectedPlayerId);
                  return (
                    <Link key={`player-${row.player_id}`} href={pageHref({ clubSlug, league: selectedLeague, section: "player", week: selectedWeek, player: row.player_id })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                      {row.player_name}
                    </Link>
                  );
                })}
              </div>
              <p>
                <Link href={playerHref(clubSlug, selectedPlayerId)}>Open full public player profile</Link>
              </p>
              <PlayerTrend rows={playerWeeklyRows} playerName={selectedPlayerName} />
              <h3>Player weekly rows</h3>
              <StatTable title="Player weekly" rows={playerWeeklyRows} clubSlug={clubSlug} />
            </section>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
