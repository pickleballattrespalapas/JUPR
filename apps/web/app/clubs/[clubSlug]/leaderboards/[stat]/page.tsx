import Link from "@/components/PublicClubLink";
import { getClubLeaderboard, getClubLeagueResults, type LeaderboardEntry, type LeagueAwardProgress } from "@/lib/api";
import { getPublicTeamLeague } from "@/lib/teamLeagueApi";
import { LEADERBOARD_CARD_DETAILS, LEADERBOARD_CARD_LABELS, type LeaderboardCard } from "@/lib/clubSite";
import { safeStatReturnTo, statLeaderboardHref, type StatLeaderboardContext } from "@/lib/statLeaderboardLinks";

type Props = {
  params: Promise<{ clubSlug: string; stat: string }>;
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
};
type RankingRow = LeaderboardEntry & { team_id?: string | null; tied?: boolean; detail?: string };
const panel = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const cell = { padding: "0.8rem", borderBottom: "1px solid #e2e8f0", textAlign: "left" as const, verticalAlign: "top" as const };
function first(params: Awaited<Props["searchParams"]>, key: string) {
  const value = params?.[key];
  return (Array.isArray(value) ? value[0] : value) || "";
}
function positive(value: string, fallback: number, max = 100000) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 1 ? Math.min(Math.floor(number), max) : fallback;
}
function sampleLabel(metric: string) {
  return ({ games: "games", matches_played: "games", games_played: "games", close_games: "close games", upset_wins: "upset wins", best_partnership_games: "games together", weeks_played: "weeks played", playing_days: "playing days", round_robin_wins: "titles", moneyball_wins: "titles", ladder_court_wins: "court wins" } as Record<string, string>)[metric] || metric.replaceAll("_", " ");
}
function playerLink(clubSlug: string, id: string | number) { return `/clubs/${encodeURIComponent(clubSlug)}/players/${encodeURIComponent(String(id))}`; }
function RowName({ row, clubSlug }: { row: RankingRow; clubSlug: string }) {
  if (row.team_members?.length) return <>{row.team_members.map((member, index) => <span key={member.player_id}>{index > 0 ? " & " : ""}<Link href={playerLink(clubSlug, member.player_id)}>{member.player_name}</Link></span>)}</>;
  return row.player_id == null ? <>{row.player_name}</> : <Link href={playerLink(clubSlug, row.player_id)}>{row.player_name}</Link>;
}
function supporting(row: RankingRow, stat: string, sample: string) {
  if (row.detail) return row.detail;
  if (stat === "most_improved" && row.starting_rating_jupr != null && row.rating_gain_jupr != null) return `${row.starting_rating_jupr.toFixed(3)} → ${(row.starting_rating_jupr + row.rating_gain_jupr).toFixed(3)}`;
  if (stat === "over_performance" && row.expected_wins != null && row.actual_wins != null) return `${row.actual_wins} actual wins · ${row.expected_wins.toFixed(1)} expected`;
  if (stat === "biggest_upset" && row.upset_opponents?.length) return `Beat ${row.upset_opponents.map(player => player.player_name).join(" & ")}${row.upset_score_for != null && row.upset_score_against != null ? ` ${row.upset_score_for}–${row.upset_score_against}` : ""}${row.upset_date ? ` · ${row.upset_date.slice(0, 10)}` : ""}`;
  if (row.metric_sample != null) return `${row.metric_sample} ${sample}`;
  if (row.matches_played != null) return `${row.matches_played} games`;
  return "—";
}
function awardRows(progress: LeagueAwardProgress, stat: string) {
  // Older payloads expose recipients only; never present those as a full field.
  return progress.races?.find(race => race.category_key === stat);
}

export default async function StatLeaderboardPage(props: Props) {
  const { clubSlug, stat } = await props.params;
  const params = await props.searchParams;
  const rawSource = first(params, "source");
  const source = rawSource === "awards" || rawSource === "team-awards" || rawSource === "weekly" || rawSource === "season" ? rawSource : undefined;
  const league = first(params, "league").slice(0, 160);
  const leagueView = first(params, "league_view") === "past" ? "past" : "active";
  const rawStatus = first(params, "status");
  const status = leagueView === "past" ? "all" : rawStatus === "inactive" || rawStatus === "all" ? rawStatus : "active";
  const search = first(params, "q").trim().slice(0, 120);
  const page = positive(first(params, "page"), 1);
  const limit = positive(first(params, "per_page"), 50, 100);
  const weeklyMinGames = first(params, "weekly_min_games") ? positive(first(params, "weekly_min_games"), 1, 1000) : undefined;
  const week = first(params, "week") ? positive(first(params, "week"), 0) : undefined;
  const defaultBack = source === "team-awards" ? `/clubs/${encodeURIComponent(clubSlug)}/team-leagues/${encodeURIComponent(league)}` : source ? `/clubs/${encodeURIComponent(clubSlug)}/league-results?league=${encodeURIComponent(league)}${week ? `&week=${week}` : ""}` : `/clubs/${encodeURIComponent(clubSlug)}/leaderboards`;
  const returnTo = safeStatReturnTo(clubSlug, first(params, "return_to") || defaultBack);
  const context: StatLeaderboardContext = { source, league, leagueView, status, search, season: first(params, "season"), returnTo, week, weeklyMinGames };
  const href = (next: { page?: number; search?: string } = {}) => {
    const url = new URL(statLeaderboardHref(clubSlug, stat, { ...context, search: next.search ?? search }), "https://club.invalid");
    if ((next.page ?? page) > 1) url.searchParams.set("page", String(next.page ?? page));
    if (limit !== 50) url.searchParams.set("per_page", String(limit));
    return `${url.pathname}${url.search}`;
  };
  let title: string = LEADERBOARD_CARD_LABELS[stat as LeaderboardCard] || "Award ranking";
  let description = source ? "Full qualifying field for this award." : LEADERBOARD_CARD_DETAILS[stat as LeaderboardCard]?.description || "Full qualifying field for this statistic.";
  let scope = league || "Overall";
  let clubName = clubSlug.replaceAll("-", " ");
  let qualification = "";
  let sample = "games";
  let rows: RankingRow[] = [];
  let total = 0;
  let qualified = 0;
  let failure = "";
  let historicalNote = "";
  let teams = stat === "best_partnership" || stat === "biggest_upset";
  if (rawSource && !source) failure = "This ranking source is not supported.";
  else if (source) {
    if (!league) failure = "Choose a league for this ranking.";
    else if (source === "weekly" && !week) failure = "Choose an exact week for this ranking.";
    else if (source === "weekly" || source === "season" || source === "awards") {
      const { data, error } = await getClubLeagueResults(clubSlug, league, source === "season" ? undefined : week, null, weeklyMinGames);
      if (error || !data) failure = error || "The ranking is unavailable right now.";
      else if (data.selected_league !== league || (source === "weekly" && data.selected_week !== week)) failure = "The selected league or week is unavailable.";
      else {
        clubName = data.club.name;
        if (source === "weekly" || source === "season") {
          const ranking = (source === "weekly" ? data.weekly_stat_rankings : data.season_stat_rankings)?.[stat];
          if (!ranking) failure = `This statistic is unavailable for the selected ${source === "weekly" ? "week" : "league season"}.`;
          else {
            title = ranking.title; description = source === "weekly" ? "Full qualifying field for this week's results." : "Full qualifying field for this league season."; qualification = ranking.qualification;
            scope = source === "weekly" ? `${league} · ${data.weeks.find(item => item.week_num === week)?.week_label || `Week ${week}`}` : `${league} · Full season`;
            rows = ranking.entries; qualified = ranking.total_qualified;
          }
        } else {
          const result = loadAward(data.award_progress);
          if (result) rows = result;
        }
      }
    } else {
      const { data, error } = await getPublicTeamLeague(clubSlug, league);
      if (error || !data) failure = error || "The ranking is unavailable right now.";
      else if (data.league.league_name !== league) failure = "The selected league is unavailable.";
      else { const result = loadAward(data.award_progress); if (result) rows = result; }
    }
    if (!failure) {
      rows = rows.filter(row => !search || row.player_name.toLocaleLowerCase().includes(search.toLocaleLowerCase()));
      total = rows.length;
      rows = rows.slice((page - 1) * limit, page * limit);
    }
  } else if (!Object.hasOwn(LEADERBOARD_CARD_LABELS, stat)) failure = "This statistic is not available.";
  else {
    const { data, error } = await getClubLeaderboard(clubSlug, { stat, leagueName: league || "OVERALL", leagueView, status, season: context.season || null, search, limit, offset: (page - 1) * limit });
    if (error || !data) failure = error || "The ranking is unavailable right now.";
    else if (!data.stat_ranking || data.stat_ranking.key !== stat) failure = "This statistic is not available yet.";
    else {
      const ranking = data.stat_ranking;
      rows = ranking.entries; total = ranking.pagination.total; qualified = ranking.total_qualified;
      clubName = data.club.name;
      scope = `${data.selected_scope === "OVERALL" ? `Overall · ${data.period?.name || "All time"}` : data.selected_scope} · ${data.filters.status === "all" ? "All players" : data.filters.status === "inactive" ? "Inactive players" : "Active players"}`;
      // Resolve the season once so a shared URL stays attached to this period.
      context.season = data.selected_scope === "OVERALL" ? data.period?.id || "all" : undefined;
      sample = sampleLabel(ranking.minimum_metric);
      qualification = `Minimum ${ranking.minimum} ${sample}.`;
      if ((ranking.min_games || 0) > 0 && !["games", "matches_played", "games_played"].includes(ranking.minimum_metric)) qualification += ` At least ${ranking.min_games} games in this period.`;
    }
  }
  function loadAward(progress: LeagueAwardProgress): RankingRow[] | null {
    const race = awardRows(progress, stat);
    if (progress.unavailable_categories?.includes(stat)) { failure = "The complete ranking was not saved for this historical award."; return null; }
    if (!race) { failure = "This award has no published full ranking."; return null; }
    title = race.category_label; teams = race.recipient_type === "team";
    sample = sampleLabel(race.minimum_metric || "games");
    qualification = `Minimum ${race.min_games || 0} ${sample}.`;
    qualified = race.eligible_count ?? race.entries.length;
    historicalNote = progress.finalized ? "Final award results" : progress.frozen_at ? "Saved award standings" : "Current award standings";
    if (progress.frozen_at) historicalNote += ` · saved ${progress.frozen_at.slice(0, 10)}`;
    return race.entries.map((entry, index) => ({
      player_id: entry.player_id, player_name: entry.recipient_name || "—", team_id: entry.team_id,
      team_members: entry.team_members?.length === 2 ? entry.team_members as RankingRow["team_members"] : undefined,
      rank: entry.rank ?? index + 1, tied: Boolean(entry.is_co_winner),
      metric_value: entry.metric_value, metric_display: entry.metric_display, metric_sample: entry.metric_sample,
      matches_played: entry.games, wins: entry.wins, losses: entry.losses,
    }));
  }
  const pages = Math.max(1, Math.ceil(total / limit));
  const identity = teams ? "teams" : "players";
  return <section data-testid="stat-leaderboard" style={{ display: "grid", gap: "1rem", minWidth: 0 }}>
    <p style={{ margin: 0 }}><Link href={returnTo}>← Back to {source ? "results" : "leaderboards"}</Link></p>
    <header>
      <p style={{ color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: ".08em", fontSize: ".78rem" }}>Full leaderboard</p>
      <h1 style={{ margin: ".3rem 0" }}>{title}</h1>
      <p style={{ color: "#475569" }}>{clubName} · {scope}</p>
      {historicalNote ? <p>{historicalNote}</p> : null}
      <p>{description}</p>
    </header>
    {failure ? <div role="alert" style={{ ...panel, background: "#fff7ed" }}><strong>{failure}</strong><p><Link href={href()}>Try again</Link></p></div> : <>
      <div style={panel}>
        <p style={{ marginTop: 0 }}><strong>{qualified} qualifying {identity}</strong> · {qualification}</p>
        <form method="get" action={`/clubs/${encodeURIComponent(clubSlug)}/leaderboards/${encodeURIComponent(stat)}`} style={{ display: "flex", gap: ".65rem", flexWrap: "wrap", alignItems: "end" }} data-testid="stat-ranking-search">
          {Array.from(new URL(href({ page: 1 }), "https://club.invalid").searchParams.entries()).filter(([key]) => key !== "q" && key !== "page").map(([key, value]) => <input key={key} type="hidden" name={key} value={value} />)}
          <label style={{ display: "grid", gap: ".3rem" }}>Find a player{teams ? " or team" : ""}<input name="q" defaultValue={search} maxLength={120} placeholder={teams ? "Either partner or team name" : "Player name"} style={{ padding: ".6rem", border: "1px solid #94a3b8", borderRadius: "6px", maxWidth: "100%" }} /></label>
          <button type="submit" style={{ padding: ".6rem .9rem" }}>Search</button>
          {search ? <Link href={href({ page: 1, search: "" })}>Clear search</Link> : null}
        </form>
        {search ? <p style={{ marginBottom: 0, color: "#475569" }}>{total} matching {identity}. Ranks show positions in the full qualifying field.</p> : null}
      </div>
      {!rows.length ? <div style={panel}>{qualified === 0 ? "Not enough qualifying data yet." : total === 0 ? "No players or teams match this search." : <><p>This page has no results.</p><Link href={href({ page: 1 })}>Go to the first page</Link></>}</div> : <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", overflowX: "auto", maxWidth: "100%" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", background: "white" }}>
          <caption style={{ textAlign: "left", padding: ".8rem", color: "#475569" }}>{title} · {scope}</caption>
          <thead><tr><th scope="col" style={cell}>Rank</th><th scope="col" style={cell}>{teams ? "Team" : "Player"}</th><th scope="col" style={cell}>{title}</th><th scope="col" style={cell}>Details</th></tr></thead>
          <tbody>{rows.map((row, index) => <tr key={row.team_key || row.team_id || `${row.player_id}-${index}`} data-testid="stat-ranking-row">
            <td style={{ ...cell, whiteSpace: "nowrap" }}>#{row.rank ?? "—"}{row.tied ? " (tied)" : ""}</td>
            <td style={{ ...cell, minWidth: "9rem", overflowWrap: "anywhere" }}><RowName row={row} clubSlug={clubSlug} /></td>
            <td style={{ ...cell, fontWeight: 700 }}>{row.metric_display ?? row.metric_value ?? "—"}</td>
            <td style={{ ...cell, color: "#475569" }}>{supporting(row, stat, sample)}</td>
          </tr>)}</tbody>
        </table>
      </div>}
      <nav aria-label="Ranking pages" style={{ display: "flex", flexWrap: "wrap", gap: "1rem", alignItems: "center" }}>
        {page > 1 ? <Link href={href({ page: Math.min(page - 1, pages) })}>Previous</Link> : null}
        <span>Page {page} of {pages} · {total} {search ? "matching" : "qualifying"} {identity}</span>
        {page < pages ? <Link href={href({ page: page + 1 })}>Next</Link> : null}
      </nav>
      <p style={{ color: "#475569", fontSize: ".9rem" }}><Link href={href()} prefetch={false}>Link to this ranking</Link> · Copy this link to share the same statistic and period.</p>
    </>}
  </section>;
}
