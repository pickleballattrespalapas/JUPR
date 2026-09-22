import PublicPlayerSearch from "@/components/PublicPlayerSearch";
import { Display } from "@/components/ClubDisplay";
import Link from "@/components/PublicClubLink";
import { getClubLeaderboard } from "@/lib/api";
import type { LeaderboardBadge, LeaderboardEntry } from "@/lib/api";
import { publicBadgeRarityLabel } from "@/lib/badgeApi";
import {
  DEFAULT_LEADERBOARD_CARDS, LEADERBOARD_CARD_LABELS, LEADERBOARD_CARD_DETAILS, leaderboardSettings,
  type LeaderboardCard,
} from "@/lib/clubSite";

type LeaderboardPageProps = {
  params: { clubSlug: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

type SortKey = "rank" | "rating" | "matches" | "win_pct" | "gain" | "name";
type StatusKey = "active" | "inactive" | "all";
type LeagueView = "active" | "past";

type ViewState = {
  league: string;
  leagueView: LeagueView;
  status: StatusKey;
  sort: SortKey;
  search: string;
  player: string;
  season: string;
  page: number;
  pageSize: number;
};

const DEFAULT_PAGE_SIZE = 50;
const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};
const thStyle = {
  textAlign: "left" as const,
  borderBottom: "1px solid #cbd5e1",
  padding: "0.65rem",
  whiteSpace: "nowrap" as const,
  color: "#475569",
  fontSize: "0.8rem"
};
const tdStyle = {
  borderBottom: "1px solid #e2e8f0",
  padding: "0.65rem",
  verticalAlign: "top" as const
};

function firstParam(searchParams: LeaderboardPageProps["searchParams"], key: string): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function normalizeSort(value: string | null): SortKey {
  if (value === "rating" || value === "matches" || value === "win_pct" || value === "gain" || value === "name") return value;
  return "rank";
}

function normalizeStatus(value: string | null): StatusKey {
  if (value === "inactive" || value === "all") return value;
  return "active";
}

function normalizeLeagueView(value: string | null): LeagueView {
  return value === "past" ? "past" : "active";
}

function positiveInt(value: string | null, fallback: number, maximum = 100000): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 1) return fallback;
  return Math.min(Math.floor(parsed), maximum);
}

function pageHref(
  clubSlug: string,
  state: ViewState,
  overrides: Partial<ViewState> = {},
  anchor?: string
): string {
  const next = { ...state, ...overrides };
  const params = new URLSearchParams();
  if (next.league && next.league !== "OVERALL") params.set("league", next.league);
  if (next.leagueView === "past") params.set("league_view", "past");
  if (next.status !== "active") params.set("status", next.status);
  if (next.sort !== "rank") params.set("sort", next.sort);
  if (next.search) params.set("q", next.search);
  if (next.player) params.set("player", next.player);
  if (next.season) params.set("season", next.season);
  if (next.page > 1) params.set("page", String(next.page));
  if (next.pageSize !== DEFAULT_PAGE_SIZE) params.set("per_page", String(next.pageSize));
  const query = params.toString();
  return `/clubs/${encodeURIComponent(clubSlug)}/leaderboards${query ? `?${query}` : ""}${anchor ? `#${anchor}` : ""}`;
}

function playerHref(clubSlug: string, playerId?: string | number | null): string {
  return playerId == null
    ? `/clubs/${encodeURIComponent(clubSlug)}/players`
    : `/clubs/${encodeURIComponent(clubSlug)}/players/${encodeURIComponent(String(playerId))}`;
}

function playerAnchor(playerId: string | number): string {
  return `leaderboard-player-${encodeURIComponent(String(playerId))}`;
}

function ratingLabel(value?: number | null): string {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return Number(value).toFixed(3);
}

function signedRatingLabel(value?: number | null): string {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  const amount = Number(value);
  return `${amount >= 0 ? "+" : ""}${amount.toFixed(3)}`;
}

function percentLabel(value?: number | null): string {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return `${Number(value).toFixed(1)}%`;
}

function dateLabel(value: string): string {
  return new Intl.DateTimeFormat("en", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }).format(new Date(`${value}T00:00:00Z`));
}

const legacyHighlightPresentation: Partial<Record<LeaderboardCard, {
  value: (row: LeaderboardEntry) => number;
  detail: (row: LeaderboardEntry) => string;
}>> = {
  highest_rating: { value: row => Number(row.rating_jupr ?? 0), detail: row => ratingLabel(row.rating_jupr) },
  most_improved: { value: row => Number(row.rating_gain_jupr ?? 0), detail: row => signedRatingLabel(row.rating_gain_jupr) },
  best_win_pct: { value: row => Number(row.win_pct ?? 0), detail: row => percentLabel(row.win_pct) },
  most_wins: { value: row => Number(row.wins ?? 0), detail: row => `${Number(row.wins ?? 0)} wins` },
  most_matches: { value: row => Number(row.matches_played ?? 0), detail: row => `${Number(row.matches_played ?? 0)} games` },
};
function highlightValue(key: LeaderboardCard, row: LeaderboardEntry): number {
  const value = row.metric_value !== undefined ? row.metric_value : legacyHighlightPresentation[key]?.value(row);
  return value != null && Number.isFinite(Number(value)) ? Number(value) : 0;
}
function highlightDetail(key: LeaderboardCard, row: LeaderboardEntry): string {
  return row.metric_display ?? legacyHighlightPresentation[key]?.detail(row) ?? "—";
}

function matchesPlayed(entry: LeaderboardEntry): number {
  return Number(entry.matches_played ?? (entry.wins ?? 0) + (entry.losses ?? 0));
}

function badgeHref(clubSlug: string, badge: LeaderboardBadge): string {
  const id = encodeURIComponent(String(badge.badge_id));
  return `/clubs/${encodeURIComponent(clubSlug)}/badge-codex?badge=${id}&limit=all#badge-${id}`;
}

function BadgeStrip({ clubSlug, entry }: { clubSlug: string; entry: LeaderboardEntry }) {
  const badges = entry.badges ?? [];
  if (!badges.length) return <span style={{ color: "#64748b" }}>—</span>;
  const overflow = Math.max(0, Number(entry.badge_count ?? badges.length) - badges.length);
  return (
    <span data-testid="leaderboard-badges" style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem" }}>
      {badges.map((badge) => (
        <Link
          key={badge.badge_id}
          href={badgeHref(clubSlug, badge)}
          title={`${badge.name}${badge.rarity ? ` · ${publicBadgeRarityLabel(badge.rarity)}` : ""}`}
          style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.15rem 0.45rem", textDecoration: "none", whiteSpace: "nowrap" }}
        >
          🏆 {badge.name}
        </Link>
      ))}
      {overflow > 0 ? <span style={{ color: "#475569" }}>+{overflow}</span> : null}
    </span>
  );
}

function BarList({
  title,
  description,
  rows,
  value,
  detail,
  clubSlug
}: {
  title: string;
  description: string;
  rows: LeaderboardEntry[];
  value: (row: LeaderboardEntry) => number;
  detail: (row: LeaderboardEntry) => string;
  clubSlug: string;
}) {
  const max = Math.max(...rows.map((row) => Math.abs(value(row))), 0);
  return (
    <article style={cardStyle} data-testid="leaderboard-highlight-card">
      <h2 style={{ marginTop: 0, fontSize: "1.05rem" }}>{title}</h2>
      <p style={{ color: "#475569", fontSize: ".85rem", marginTop: "-.3rem" }}>{description}</p>
      {!rows.length ? <p style={{ color: "#64748b", marginBottom: 0 }}>Not enough qualifying data yet.</p> : null}
      <div style={{ display: "grid", gap: "0.65rem" }}>
        {rows.map((row, index) => {
          const amount = value(row);
          const width = max > 0 && amount !== 0 ? `${Math.min(100, Math.max(4, Math.round((Math.abs(amount) / max) * 100)))}%` : "0%";
          const team = row.team_members?.length === 2 ? row.team_members : null;
          const place = row.rank ?? index + 1;
          return (
            <div key={row.team_key ?? `${title}-${row.player_id ?? row.player_name}`} data-testid={team ? "leaderboard-team-row" : undefined}>
              <div style={{ display: "flex", justifyContent: "space-between", flexWrap: team ? "wrap" : undefined, gap: "0.75rem", fontSize: "0.86rem", marginBottom: "0.25rem", overflowWrap: "anywhere" }}>
                <span style={{ fontWeight: 700, minWidth: 0, flex: team ? "1 1 145px" : undefined }}>
                  {team ? <>
                    <span aria-label={`Place ${place}`} data-testid="leaderboard-team-place" style={{ color: "#475569", marginRight: ".4rem" }}>#{place}</span>
                    {team.map((member, memberIndex) => <span key={member.player_id}>
                      {memberIndex > 0 ? " & " : ""}<Link href={playerHref(clubSlug, member.player_id)}>{member.player_name}</Link>
                    </span>)}
                  </> : row.player_id != null ? <Link href={playerHref(clubSlug, row.player_id)}>{row.player_name}</Link> : row.player_name}
                </span>
                <span style={{ color: "#475569", textAlign: "right", minWidth: 0, marginLeft: team ? "auto" : undefined }}>{detail(row)}</span>
              </div>
              <div style={{ height: "0.55rem", borderRadius: "999px", background: "#e2e8f0", overflow: "hidden" }}>
                <div style={{ width, height: "100%", borderRadius: "999px", background: amount < 0 ? "#dc2626" : "#2563eb" }} />
              </div>
            </div>
          );
        })}
      </div>
    </article>
  );
}

function qualificationLabel(entry: LeaderboardEntry, overall: boolean): string {
  if (overall || entry.qualified == null) return "—";
  return entry.qualified ? "Qualified" : `${matchesPlayed(entry)}/${Number(entry.min_games ?? 0)} games`;
}

export default async function ClubLeaderboardPage({ params, searchParams }: LeaderboardPageProps) {
  const { clubSlug } = params;
  const leagueView = normalizeLeagueView(firstParam(searchParams, "league_view"));
  const requestedLeague = firstParam(searchParams, "league") || (leagueView === "past" ? "" : "OVERALL");
  const selectedStatus = normalizeStatus(firstParam(searchParams, "status"));
  const selectedSort = normalizeSort(firstParam(searchParams, "sort"));
  const search = (firstParam(searchParams, "q") || "").trim().slice(0, 120);
  const selectedPlayer = (firstParam(searchParams, "player") || "").trim().slice(0, 120);
  const requestedSeason = (firstParam(searchParams, "season") || "").trim().slice(0, 80);
  const page = positiveInt(firstParam(searchParams, "page"), 1);
  const pageSize = positiveInt(firstParam(searchParams, "per_page"), DEFAULT_PAGE_SIZE, 100);
  const offset = (page - 1) * pageSize;
  const { data, error } = await getClubLeaderboard(clubSlug, {
    leagueName: requestedLeague,
    leagueView,
    status: selectedStatus,
    search,
    sort: selectedSort,
    playerId: selectedPlayer || null,
    season: requestedSeason || null,
    limit: pageSize,
    offset
  });

  const clubName = data?.club?.name ?? clubSlug.replace(/[-_]/g, " ");
  if (error || !data) {
    return (
      <section data-testid="leaderboard-error-state">
        <p style={{ color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Leaderboards</p>
        <h1>{clubName} leaderboards</h1>
        <div role="alert" style={{ ...cardStyle, borderColor: "#fecaca", background: "#fef2f2" }}>
          <strong>Leaderboards are unavailable right now.</strong>
          <p style={{ marginBottom: 0, color: "#7f1d1d" }}>Please try again shortly.</p>
        </div>
        <p><Link href={`/clubs/${encodeURIComponent(clubSlug)}/leaderboards`}>Retry leaderboards</Link> · <Link href={`/clubs/${encodeURIComponent(clubSlug)}`}>Return to club home</Link></p>
      </section>
    );
  }

  const selectedLeague = data.selected_scope || (data.filters.league_view === "active" ? "OVERALL" : "");
  const overall = selectedLeague === "OVERALL";
  const settings = leaderboardSettings(data.leaderboard_settings);
  const selectedPeriod = data.period;
  const state: ViewState = {
    league: selectedLeague,
    leagueView: data.filters.league_view,
    status: data.filters.status,
    sort: normalizeSort(data.filters.sort),
    search: data.filters.search,
    player: selectedPlayer,
    season: overall ? selectedPeriod?.id || "all" : requestedSeason,
    page,
    pageSize
  };
  const entries = data.leaderboard ?? [];
  const minGames = Number(data.scope?.min_games ?? 0);
  const totalPages = Math.max(1, Math.ceil(Number(data.pagination.total || 0) / Number(data.pagination.limit || pageSize)));
  const currentPage = Math.min(page, totalPages);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Leaderboards</p>
      <h1 style={{ marginTop: 0 }}>{clubName} leaderboards</h1>
      <p style={{ color: "#475569", maxWidth: "780px" }}>
        {state.leagueView === "past"
          ? "See final standings for finished leagues. Active players are shown by default."
          : "See overall and league standings, qualification progress, and earned badges. Active players are shown by default."}
      </p>
      <p style={{ color: "#475569" }}><Link href={`/clubs/${encodeURIComponent(clubSlug)}/players`}>Browse all player profiles</Link></p>

      <nav aria-label="Leaderboard league collections" data-testid="leaderboard-league-view-toggle" style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
        {(["active", "past"] as LeagueView[]).map((option) => {
          const active = option === state.leagueView;
          return (
            <Link
              key={option}
              href={pageHref(clubSlug, state, {
                leagueView: option,
                league: option === "active" ? "OVERALL" : "",
                player: "",
                page: 1
              })}
              aria-current={active ? "page" : undefined}
              style={{ border: `1px solid ${active ? "#2563eb" : "#cbd5e1"}`, borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: active ? "#1d4ed8" : "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}
            >
              {option === "active" ? "Active leagues" : "Past leagues"}
            </Link>
          );
        })}
      </nav>

      <nav aria-label="Leaderboard groups" data-testid="leaderboard-scope-tabs" style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
        {data.scopes.map((scope) => {
          const active = scope.name === selectedLeague;
          return (
            <Link
              key={scope.name}
              href={pageHref(clubSlug, state, { league: scope.name, page: 1, player: "" })}
              aria-current={active ? "page" : undefined}
              data-testid={`leaderboard-scope-${scope.name === "OVERALL" ? "overall" : "league"}`}
              style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}
            >
              {scope.label}{scope.name !== "OVERALL" && scope.min_games > 0 ? ` · min ${scope.min_games}` : ""}
            </Link>
          );
        })}
      </nav>

      {overall && settings.seasons.length > 0 ? <div style={{ ...cardStyle, marginBottom: "1rem" }} data-testid="leaderboard-period-controls">
        <form method="get" action={`/clubs/${encodeURIComponent(clubSlug)}/leaderboards`} style={{ display: "flex", gap: ".75rem", flexWrap: "wrap", alignItems: "end" }}>
          {state.status !== "active" ? <input type="hidden" name="status" value={state.status} /> : null}
          {state.sort !== "rank" ? <input type="hidden" name="sort" value={state.sort} /> : null}
          {state.search ? <input type="hidden" name="q" value={state.search} /> : null}
          {state.pageSize !== DEFAULT_PAGE_SIZE ? <input type="hidden" name="per_page" value={state.pageSize} /> : null}
          <label style={{ display: "grid", gap: ".35rem", fontWeight: 700 }}>Statistics period
            <select name="season" defaultValue={state.season} style={{ border: "1px solid #94a3b8", borderRadius: 8, padding: ".6rem .7rem", font: "inherit", maxWidth: "100%" }}>
              <option value="all">All time</option>
              {settings.seasons.map(season => <option key={season.id} value={season.id}>{season.name}</option>)}
            </select>
          </label>
          <button type="submit" style={{ border: 0, borderRadius: "999px", padding: ".65rem 1rem", background: "#0f172a", color: "white", fontWeight: 800 }}>Show period</button>
        </form>
        {selectedPeriod?.id && selectedPeriod.start_date ? <p style={{ marginBottom: 0, color: "#475569" }} data-testid="leaderboard-period-description">
          <strong>{selectedPeriod.name}</strong> · {dateLabel(selectedPeriod.start_date)}{selectedPeriod.end_date ? ` – ${dateLabel(selectedPeriod.end_date)}` : " onward"}. Wins, games, win percentage and rating improvement cover this period. Ratings and ranks show current standing.
        </p> : <p style={{ marginBottom: 0, color: "#475569" }}>Showing all-time statistics.</p>}
      </div> : null}

      <div style={{ ...cardStyle, display: "grid", gap: "0.85rem", marginBottom: "1rem" }}>
        <form method="get" action={`/clubs/${encodeURIComponent(clubSlug)}/leaderboards`} data-testid="leaderboard-search-form" style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "end" }}>
          {selectedLeague && selectedLeague !== "OVERALL" ? <input type="hidden" name="league" value={selectedLeague} /> : null}
          {state.leagueView === "past" ? <input type="hidden" name="league_view" value="past" /> : null}
          {state.status !== "active" ? <input type="hidden" name="status" value={state.status} /> : null}
          {state.sort !== "rank" ? <input type="hidden" name="sort" value={state.sort} /> : null}
          {state.pageSize !== DEFAULT_PAGE_SIZE ? <input type="hidden" name="per_page" value={state.pageSize} /> : null}
          {state.season ? <input type="hidden" name="season" value={state.season} /> : null}
          <label style={{ display: "grid", gap: "0.3rem", minWidth: "min(100%, 280px)", flex: "1 1 320px", fontWeight: 700 }}>
            Find player
            <PublicPlayerSearch clubSlug={clubSlug} scope="leaderboards" defaultValue={search} filters={{ league_name: selectedLeague || "OVERALL", league_view: state.leagueView, status: state.status, season: state.season || "all" }} />
          </label>
          <button type="submit" style={{ border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: "#0f172a", color: "white", fontWeight: 800 }}>Search</button>
          {search ? <Link href={pageHref(clubSlug, state, { search: "", player: "", page: 1 })}>Clear search</Link> : null}
        </form>

        <div style={{ display: "flex", gap: "0.45rem", flexWrap: "wrap", alignItems: "center" }}>
          <strong>Status:</strong>
          {(["active", "all", "inactive"] as StatusKey[]).map((status) => {
            const active = status === state.status;
            return (
              <Link
                key={status}
                href={pageHref(clubSlug, state, { status, page: 1, player: "" })}
                aria-current={active ? "page" : undefined}
                data-testid={`leaderboard-status-${status}`}
                style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: active ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}
              >
                {status === "active" ? "Active" : status === "inactive" ? "Inactive" : "See all"}
              </Link>
            );
          })}
        </div>

        <div style={{ display: "flex", gap: "0.45rem", flexWrap: "wrap", alignItems: "center" }}>
          <strong>Sort:</strong>
          {(["rank", "rating", "matches", "win_pct", "gain", "name"] as SortKey[]).map((sort) => {
            const active = sort === state.sort;
            const label = sort === "win_pct" ? "Win %" : sort === "gain" ? "Gain" : sort[0].toUpperCase() + sort.slice(1);
            return <Link key={sort} href={pageHref(clubSlug, state, { sort, page: 1 })} aria-current={active ? "page" : undefined} style={{ color: "#0f172a", fontWeight: active ? 800 : 600 }}>{label}</Link>;
          })}
        </div>
      </div>

      {!overall || settings.show_summary ? <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }} data-testid="leaderboard-summary">
        <article style={cardStyle}><strong>Ranked players</strong><br />{data.summary.ranked_players}</article>
        <article style={cardStyle}><strong>Active players</strong><br />{data.summary.active_players}</article>
        <article style={cardStyle}><strong>Inactive players</strong><br />{data.summary.inactive_players}</article>
        <article style={cardStyle}><strong>Leaderboard groups</strong><br />{data.summary.leaderboard_scopes}</article>
      </div> : null}

      {overall && settings.min_games > 0 ? <p style={{ color: "#475569" }} data-testid="leaderboard-performance-qualification">Performance cards require at least {settings.min_games} recorded game{settings.min_games === 1 ? "" : "s"} in this period. Individual cards may have additional minimums.</p> : null}

      {selectedLeague && !overall ? (
        <p style={{ ...cardStyle, background: "#f8fafc", color: "#475569" }} data-testid="leaderboard-qualification-note">
          Qualification for {selectedLeague}: {minGames > 0 ? `at least ${minGames} recorded games.` : "every ranked player currently qualifies."}
        </p>
      ) : null}

      {data.snapshot ? (
        <article id={data.snapshot.player_id != null ? playerAnchor(data.snapshot.player_id) : "player-snapshot"} style={{ ...cardStyle, borderColor: "#93c5fd", background: "#eff6ff", marginBottom: "1rem" }} data-testid="leaderboard-player-snapshot">
          <span id="player-snapshot" />
          <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap" }}>
            <div>
              <p style={{ margin: 0, color: "#1d4ed8", fontWeight: 800, textTransform: "uppercase", fontSize: "0.75rem", letterSpacing: "0.08em" }}>Player summary</p>
              <h2 style={{ margin: "0.2rem 0" }}>{data.snapshot.player_name}</h2>
              <p style={{ margin: 0 }}><Link href={playerHref(clubSlug, data.snapshot.player_id)}>Open full player profile</Link></p>
            </div>
            <Link href={pageHref(clubSlug, state, { player: String(data.snapshot.player_id ?? "") }, "player-snapshot")}>Share this player</Link>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(110px, 1fr))", gap: "0.65rem", marginTop: "0.85rem" }}>
            <div><strong>Rank</strong><br />#{data.snapshot.rank ?? "—"}</div>
            <div><strong>Rating</strong><br />{ratingLabel(data.snapshot.rating_jupr)}</div>
            <div><strong>Gain</strong><br />{signedRatingLabel(data.snapshot.rating_gain_jupr)}</div>
            <div><strong>Gap</strong><br />{ratingLabel(data.snapshot.gap_jupr)}</div>
            <div><strong>W-L</strong><br />{data.snapshot.wins ?? 0}-{data.snapshot.losses ?? 0}</div>
            <div><strong>Games</strong><br />{matchesPlayed(data.snapshot)}</div>
            <div><strong>Win %</strong><br />{percentLabel(data.snapshot.win_pct)}</div>
            <div><strong>Qualification</strong><br />{qualificationLabel(data.snapshot, overall)}</div>
          </div>
          <div style={{ marginTop: "0.8rem" }}><BadgeStrip clubSlug={clubSlug} entry={data.snapshot} /></div>
        </article>
      ) : null}

      {selectedLeague && (!overall || settings.cards.length > 0) ? <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1rem", marginBottom: "1rem" }}>
        {(overall ? settings.cards : DEFAULT_LEADERBOARD_CARDS).map(key => {
          const card = LEADERBOARD_CARD_DETAILS[key];
          const minimum = overall ? settings.card_options?.[key]?.minimum ?? 0 : 0;
          const description = `${card.description}${minimum > 0 ? ` Minimum ${minimum} ${card.sample}.` : ""}`;
          const highlight = <BarList key={key} title={LEADERBOARD_CARD_LABELS[key]} description={description} rows={data.highlights[key] ?? []} value={row => highlightValue(key, row)} detail={row => highlightDetail(key, row)} clubSlug={clubSlug} />;
          // The Overall selector is the authority for its featured cards.
          // Table columns and default league cards retain display preferences.
          if (overall) return highlight;
          return card.fields.reduceRight((child, field) => <Display key={`${key}-${field}`} field={field}>{child}</Display>,
            highlight);
        })}
      </div> : null}

      {data.summary.ranked_players === 0 ? (
        <div style={cardStyle} data-testid="leaderboard-empty-state">
          <strong>{state.leagueView === "past" && !selectedLeague ? "No past leagues have been published yet." : "No leaderboard results are available yet."}</strong>
          <p style={{ marginBottom: 0, color: "#475569" }}>
            {state.leagueView === "past" && !selectedLeague ? "Finished leagues will appear here." : "Standings will appear after matches are recorded."}
          </p>
        </div>
      ) : entries.length === 0 ? (
        <div style={cardStyle} data-testid="leaderboard-filter-empty-state"><strong>No players match these filters.</strong><p style={{ marginBottom: 0 }}><Link href={pageHref(clubSlug, state, { search: "", status: "active", player: "", page: 1 })}>Reset search and status</Link></p></div>
      ) : (
        <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "12px", background: "white" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.92rem", minWidth: "1080px" }} data-testid="leaderboard-table">
            <caption style={{ position: "absolute", width: "1px", height: "1px", padding: 0, margin: "-1px", overflow: "hidden", clip: "rect(0, 0, 0, 0)", whiteSpace: "nowrap", border: 0 }}>
              {selectedLeague === "OVERALL" ? "Overall" : selectedLeague} leaderboard standings
            </caption>
            <thead><tr><th style={thStyle}>Rank</th><th style={thStyle}>Player</th><Display field="ratings"><th style={thStyle}>Rating</th></Display><Display field="rating_changes"><th style={thStyle}>Gain</th></Display><Display field="rating_changes"><th style={thStyle}>Gap</th></Display><Display field="match_counts"><th style={thStyle}>Games</th></Display><Display field="records"><th style={thStyle}>W-L</th></Display><Display field="win_percentage"><th style={thStyle}>Win %</th></Display><Display field="qualification"><th style={thStyle}>Qualification</th></Display><Display field="badges"><th style={thStyle}>Badges</th></Display><Display field="player_status"><th style={thStyle}>Status</th></Display></tr></thead>
            <tbody>
              {entries.map((entry, index) => {
                const snapshotPlayerId = data.snapshot?.player_id;
                const rowId = entry.player_id != null && String(entry.player_id) !== String(snapshotPlayerId ?? "") ? playerAnchor(entry.player_id) : undefined;
                const selected = entry.player_id != null && String(entry.player_id) === String(selectedPlayer);
                const snapshotLink = entry.player_id == null ? null : pageHref(clubSlug, state, { player: String(entry.player_id) }, "player-snapshot");
                return (
                  <tr key={`${entry.player_id ?? entry.player_name}-${index}`} id={rowId} data-testid="leaderboard-row" data-status={entry.is_active === false ? "inactive" : "active"} style={{ background: selected ? "#eff6ff" : undefined }}>
                    <td style={tdStyle}>#{entry.rank ?? entry.rank_position ?? "—"}</td>
                    <td style={tdStyle}>
                      <strong>{entry.player_id != null ? <Link href={playerHref(clubSlug, entry.player_id)}>{entry.player_name}</Link> : entry.player_name}</strong>
                      {snapshotLink ? <><br /><Link href={snapshotLink} style={{ fontSize: "0.8rem" }}>view summary</Link></> : null}
                    </td>
                    <Display field="ratings"><td style={tdStyle}>{ratingLabel(entry.rating_jupr)}</td></Display>
                    <Display field="rating_changes"><td style={{ ...tdStyle, color: Number(entry.rating_gain_jupr ?? 0) < 0 ? "#b91c1c" : "#166534" }}>{signedRatingLabel(entry.rating_gain_jupr)}</td></Display>
                    <Display field="rating_changes"><td style={tdStyle}>{entry.gap_jupr == null ? "Leader" : ratingLabel(entry.gap_jupr)}</td></Display>
                    <Display field="match_counts"><td style={tdStyle}>{matchesPlayed(entry)}</td></Display>
                    <Display field="records"><td style={tdStyle}>{entry.wins ?? 0}-{entry.losses ?? 0}</td></Display>
                    <Display field="win_percentage"><td style={tdStyle}>{percentLabel(entry.win_pct)}</td></Display>
                    <Display field="qualification"><td style={tdStyle}>{qualificationLabel(entry, overall)}</td></Display>
                    <Display field="badges"><td style={tdStyle}><BadgeStrip clubSlug={clubSlug} entry={entry} /></td></Display>
                    <Display field="player_status"><td style={tdStyle}>{entry.is_active === false ? "Inactive" : "Active"}</td></Display>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {data.pagination.total > 0 ? (
        <nav aria-label="Leaderboard pages" data-testid="leaderboard-pagination" style={{ display: "flex", justifyContent: "space-between", gap: "1rem", alignItems: "center", marginTop: "1rem", flexWrap: "wrap" }}>
          <span>Page {currentPage} of {totalPages} · {data.pagination.total} player{data.pagination.total === 1 ? "" : "s"}</span>
          <span style={{ display: "flex", gap: "0.75rem" }}>
            {page > 1 ? <Link rel="prev" href={pageHref(clubSlug, state, { page: page - 1, player: "" })}>Previous</Link> : <span style={{ color: "#94a3b8" }}>Previous</span>}
            {data.pagination.has_more ? <Link rel="next" href={pageHref(clubSlug, state, { page: page + 1, player: "" })}>Next</Link> : <span style={{ color: "#94a3b8" }}>Next</span>}
          </span>
        </nav>
      ) : null}
    </section>
  );
}
