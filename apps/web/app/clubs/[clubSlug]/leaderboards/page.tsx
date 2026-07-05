import Link from "next/link";
import { getClubLeaderboard } from "@/lib/api";
import type { LeaderboardEntry } from "@/lib/api";

type LeaderboardPageProps = {
  params: { clubSlug: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

type SortKey = "rank" | "rating" | "matches" | "win_pct";
type StatusKey = "all" | "active" | "inactive";

type ChartRow = {
  key: string;
  label: string;
  value: number;
  detail: string;
  href?: string;
};

const thStyle = { textAlign: "left" as const, borderBottom: "1px solid #cbd5e1", padding: "0.6rem", whiteSpace: "nowrap" as const, color: "#475569", fontSize: "0.82rem" };
const tdStyle = { borderBottom: "1px solid #e2e8f0", padding: "0.6rem", whiteSpace: "nowrap" as const };
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

function firstParam(searchParams: LeaderboardPageProps["searchParams"], key: string): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function normalizeSort(value: string | null): SortKey {
  if (value === "rating" || value === "matches" || value === "win_pct") return value;
  return "rank";
}

function normalizeStatus(value: string | null): StatusKey {
  if (value === "active" || value === "inactive") return value;
  return "all";
}

function pageHref({ clubSlug, league, status, sort, player }: { clubSlug: string; league?: string | null; status?: StatusKey | null; sort?: SortKey | null; player?: string | number | null }): string {
  const params = new URLSearchParams();
  if (league) params.set("league", league);
  if (status && status !== "all") params.set("status", status);
  if (sort && sort !== "rank") params.set("sort", sort);
  if (player) params.set("player", String(player));
  const query = params.toString();
  return `/clubs/${clubSlug}/leaderboards${query ? `?${query}` : ""}`;
}

function playerHref(clubSlug: string, playerId?: string | number | null): string {
  return playerId == null ? `/clubs/${clubSlug}/players` : `/clubs/${clubSlug}/players/${playerId}`;
}

function playerAnchor(playerId: string | number): string {
  return `leaderboard-player-${encodeURIComponent(String(playerId))}`;
}

function safeNumber(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function entryRating(entry: LeaderboardEntry): number | null {
  const raw = entry.rating_jupr ?? entry.rating;
  if (raw == null || Number.isNaN(Number(raw))) return null;
  const n = Number(raw);
  return n > 20 ? n / 400 : n;
}

function ratingLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(3);
}

function winPct(entry: LeaderboardEntry): number | null {
  const wins = entry.wins ?? 0;
  const losses = entry.losses ?? 0;
  const total = wins + losses;
  return total > 0 ? (wins / total) * 100 : null;
}

function winPctLabel(entry: LeaderboardEntry): string {
  const pct = winPct(entry);
  return pct == null ? "—" : `${pct.toFixed(1)}%`;
}

function matchesPlayed(entry: LeaderboardEntry): number {
  return entry.matches_played ?? (entry.wins ?? 0) + (entry.losses ?? 0);
}

function leagueLabel(value?: string | null): string {
  return value?.trim() || "Overall";
}

function sortEntries(entries: LeaderboardEntry[], sort: SortKey): LeaderboardEntry[] {
  const sorted = [...entries];
  sorted.sort((a, b) => {
    if (sort === "rating") return safeNumber(entryRating(b)) - safeNumber(entryRating(a));
    if (sort === "matches") return matchesPlayed(b) - matchesPlayed(a);
    if (sort === "win_pct") return safeNumber(winPct(b), -1) - safeNumber(winPct(a), -1);
    return safeNumber(a.rank ?? a.rank_position, 999999) - safeNumber(b.rank ?? b.rank_position, 999999);
  });
  return sorted;
}

function BarList({ title, rows, emptyText = "No chart data yet." }: { title: string; rows: ChartRow[]; emptyText?: string }) {
  const max = Math.max(...rows.map((row) => Math.abs(row.value)), 0);
  return (
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>{title}</h2>
      {!rows.length ? <p style={{ color: "#64748b" }}>{emptyText}</p> : null}
      <div style={{ display: "grid", gap: "0.65rem" }}>
        {rows.map((row) => {
          const width = max > 0 ? `${Math.max(4, Math.round((Math.abs(row.value) / max) * 100))}%` : "0%";
          const label = row.href ? <Link href={row.href}>{row.label}</Link> : row.label;
          return (
            <div key={row.key}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", fontSize: "0.88rem", marginBottom: "0.25rem" }}>
                <span style={{ fontWeight: 700 }}>{label}</span>
                <span style={{ color: "#475569" }}>{row.detail}</span>
              </div>
              <div style={{ height: "0.6rem", borderRadius: "999px", background: "#e2e8f0", overflow: "hidden" }}>
                <div style={{ width, height: "100%", borderRadius: "999px", background: "#2563eb" }} />
              </div>
            </div>
          );
        })}
      </div>
    </article>
  );
}

export default async function ClubLeaderboardPage({ params, searchParams }: LeaderboardPageProps) {
  const { clubSlug } = params;
  const selectedLeague = firstParam(searchParams, "league");
  const selectedStatus = normalizeStatus(firstParam(searchParams, "status"));
  const selectedSort = normalizeSort(firstParam(searchParams, "sort"));
  const selectedPlayer = firstParam(searchParams, "player");
  const { data, error } = await getClubLeaderboard(clubSlug);
  const entries = data?.leaderboard ?? [];
  const clubName = data?.club?.name ?? clubSlug;
  const leagues = Array.from(new Set(entries.map((entry) => leagueLabel(entry.league_name)))).sort((a, b) => a.localeCompare(b));
  const filteredEntries = sortEntries(entries, selectedSort).filter((entry) => {
    const leagueOk = !selectedLeague || leagueLabel(entry.league_name) === selectedLeague;
    const statusOk = selectedStatus === "all" || (selectedStatus === "active" ? entry.is_active !== false : entry.is_active === false);
    return leagueOk && statusOk;
  });
  const activeCount = entries.filter((entry) => entry.is_active !== false).length;
  const inactiveCount = entries.length - activeCount;
  const topRatings = sortEntries(filteredEntries, "rating").slice(0, 8).map((entry) => ({
    key: `rating-${entry.player_id ?? entry.player_name}`,
    label: entry.player_name || "Player",
    value: safeNumber(entryRating(entry)),
    detail: ratingLabel(entryRating(entry)),
    href: entry.player_id ? playerHref(clubSlug, entry.player_id) : undefined
  }));
  const mostActive = sortEntries(filteredEntries, "matches").slice(0, 8).map((entry) => ({
    key: `matches-${entry.player_id ?? entry.player_name}`,
    label: entry.player_name || "Player",
    value: matchesPlayed(entry),
    detail: `${matchesPlayed(entry)} matches`,
    href: entry.player_id ? playerHref(clubSlug, entry.player_id) : undefined
  }));
  const bestWinPct = sortEntries(filteredEntries.filter((entry) => winPct(entry) != null), "win_pct").slice(0, 8).map((entry) => ({
    key: `winpct-${entry.player_id ?? entry.player_name}`,
    label: entry.player_name || "Player",
    value: safeNumber(winPct(entry)),
    detail: winPctLabel(entry),
    href: entry.player_id ? playerHref(clubSlug, entry.player_id) : undefined
  }));

  if (error) {
    return (
      <section>
        <h1>{clubName} Leaderboards</h1>
        <p style={{ color: "#b91c1c" }}>Leaderboard data is temporarily unavailable. {error}</p>
      </section>
    );
  }

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Leaderboards
      </p>
      <h1 style={{ marginTop: 0 }}>{clubName} leaderboards</h1>
      <p style={{ color: "#475569", maxWidth: "760px" }}>
        Public ranking table with league filters, top-performer views, player profile links, and active/inactive visibility.
      </p>
      <p style={{ color: "#475569" }}><Link href={`/clubs/${clubSlug}/players`}>Browse all player profiles</Link></p>

      {entries.length === 0 ? <p>No leaderboard data is currently available.</p> : null}

      {entries.length > 0 ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>Ranked players</strong><br />{entries.length}</article>
            <article style={cardStyle}><strong>Active players</strong><br />{activeCount}</article>
            <article style={cardStyle}><strong>Inactive players</strong><br />{inactiveCount}</article>
            <article style={cardStyle}><strong>Leaderboard scopes</strong><br />{leagues.length}</article>
          </div>

          <div style={{ display: "grid", gap: "0.75rem", marginBottom: "1rem" }}>
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
              <Link href={pageHref({ clubSlug, status: selectedStatus, sort: selectedSort, player: selectedPlayer })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: !selectedLeague ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: !selectedLeague ? 800 : 600 }}>All leagues</Link>
              {leagues.map((league) => {
                const active = league === selectedLeague;
                return (
                  <Link key={league} href={pageHref({ clubSlug, league, status: selectedStatus, sort: selectedSort, player: selectedPlayer })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                    {league}
                  </Link>
                );
              })}
            </div>
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
              {(["all", "active", "inactive"] as StatusKey[]).map((status) => {
                const active = status === selectedStatus;
                return (
                  <Link key={status} href={pageHref({ clubSlug, league: selectedLeague, status, sort: selectedSort, player: selectedPlayer })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                    {status === "all" ? "All statuses" : status === "active" ? "Active only" : "Inactive only"}
                  </Link>
                );
              })}
              {(["rank", "rating", "matches", "win_pct"] as SortKey[]).map((sort) => {
                const active = sort === selectedSort;
                return (
                  <Link key={sort} href={pageHref({ clubSlug, league: selectedLeague, status: selectedStatus, sort, player: selectedPlayer })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#fef9c3" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                    Sort: {sort === "win_pct" ? "Win %" : sort[0].toUpperCase() + sort.slice(1)}
                  </Link>
                );
              })}
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem", marginBottom: "1rem" }}>
            <BarList title="Top ratings" rows={topRatings} />
            <BarList title="Most active" rows={mostActive} />
            <BarList title="Best win %" rows={bestWinPct} />
          </div>

          <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "12px", background: "white" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.95rem", minWidth: "760px" }}>
              <thead>
                <tr>
                  <th style={thStyle}>Rank</th>
                  <th style={thStyle}>Player</th>
                  <th style={thStyle}>League</th>
                  <th style={thStyle}>Rating</th>
                  <th style={thStyle}>Matches</th>
                  <th style={thStyle}>W/L</th>
                  <th style={thStyle}>Win %</th>
                  <th style={thStyle}>Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredEntries.map((entry, index) => {
                  const wins = entry.wins;
                  const losses = entry.losses;
                  const wl = wins == null && losses == null ? "—" : `${wins ?? 0}/${losses ?? 0}`;
                  const playerLabel = entry.player_name || "Player";
                  const rowId = entry.player_id ? playerAnchor(entry.player_id) : undefined;
                  const selected = entry.player_id != null && String(entry.player_id) === String(selectedPlayer);
                  return (
                    <tr key={`${entry.league_name ?? "overall"}-${entry.player_name}-${entry.player_id ?? index}`} id={rowId} style={{ background: selected ? "#eff6ff" : undefined }}>
                      <td style={tdStyle}>{entry.rank ?? entry.rank_position ?? index + 1}</td>
                      <td style={tdStyle}>
                        {entry.player_id ? <Link href={playerHref(clubSlug, entry.player_id)}>{playerLabel}</Link> : playerLabel}
                        {entry.player_id ? <><span style={{ color: "#64748b" }}> · </span><Link href={pageHref({ clubSlug, league: selectedLeague, status: selectedStatus, sort: selectedSort, player: entry.player_id }) + `#${playerAnchor(entry.player_id)}`}>rank link</Link></> : null}
                      </td>
                      <td style={tdStyle}>{leagueLabel(entry.league_name)}</td>
                      <td style={tdStyle}>{ratingLabel(entryRating(entry))}</td>
                      <td style={tdStyle}>{matchesPlayed(entry)}</td>
                      <td style={tdStyle}>{wl}</td>
                      <td style={tdStyle}>{winPctLabel(entry)}</td>
                      <td style={tdStyle}>{entry.is_active === false ? "Inactive" : "Active"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </section>
  );
}
