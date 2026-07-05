import Link from "next/link";
import { getClubPlayers } from "@/lib/api";
import type { PublicPlayer } from "@/lib/api";

type PlayersPageProps = {
  params: { clubSlug: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

type SortKey = "rating" | "matches" | "name" | "win_pct" | "recent";
type StatusKey = "all" | "active" | "inactive";

const thStyle = { textAlign: "left" as const, borderBottom: "1px solid #cbd5e1", padding: "0.6rem", whiteSpace: "nowrap" as const, color: "#475569", fontSize: "0.82rem" };
const tdStyle = { borderBottom: "1px solid #e2e8f0", padding: "0.6rem", whiteSpace: "nowrap" as const };
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

function firstParam(searchParams: PlayersPageProps["searchParams"], key: string): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function normalizeSort(value: string | null): SortKey {
  if (value === "matches" || value === "name" || value === "win_pct" || value === "recent") return value;
  return "rating";
}

function normalizeStatus(value: string | null): StatusKey {
  if (value === "active" || value === "inactive") return value;
  return "all";
}

function pageHref({ clubSlug, q, status, sort, player }: { clubSlug: string; q?: string | null; status?: StatusKey | null; sort?: SortKey | null; player?: string | number | null }): string {
  const params = new URLSearchParams();
  if (q) params.set("q", q);
  if (status && status !== "all") params.set("status", status);
  if (sort && sort !== "rating") params.set("sort", sort);
  if (player) params.set("player", String(player));
  const query = params.toString();
  return `/clubs/${clubSlug}/players${query ? `?${query}` : ""}`;
}

function playerAnchor(playerId: string | number): string {
  return `player-${encodeURIComponent(String(playerId))}`;
}

function ratingValue(player: PublicPlayer): number | null {
  if (player.rating == null || Number.isNaN(Number(player.rating))) return null;
  const n = Number(player.rating);
  return n > 20 ? n / 400 : n;
}

function ratingLabel(value?: number | null): string {
  return value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(3);
}

function winPct(player: PublicPlayer): number | null {
  const wins = player.wins ?? 0;
  const losses = player.losses ?? 0;
  const total = wins + losses;
  return total > 0 ? (wins / total) * 100 : null;
}

function winPctLabel(player: PublicPlayer): string {
  const pct = winPct(player);
  return pct == null ? "—" : `${pct.toFixed(1)}%`;
}

function matchesPlayed(player: PublicPlayer): number {
  return player.matches_played ?? (player.wins ?? 0) + (player.losses ?? 0);
}

function dateLabel(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10);
  return date.toISOString().slice(0, 10);
}

function sortPlayers(players: PublicPlayer[], sort: SortKey): PublicPlayer[] {
  const sorted = [...players];
  sorted.sort((a, b) => {
    if (sort === "matches") return matchesPlayed(b) - matchesPlayed(a);
    if (sort === "name") return a.name.localeCompare(b.name);
    if (sort === "win_pct") return Number(winPct(b) ?? -1) - Number(winPct(a) ?? -1);
    if (sort === "recent") return String(b.last_game_at ?? "").localeCompare(String(a.last_game_at ?? ""));
    return Number(ratingValue(b) ?? -1) - Number(ratingValue(a) ?? -1);
  });
  return sorted;
}

export default async function ClubPlayersPage({ params, searchParams }: PlayersPageProps) {
  const { clubSlug } = params;
  const q = (firstParam(searchParams, "q") ?? "").trim();
  const selectedStatus = normalizeStatus(firstParam(searchParams, "status"));
  const selectedSort = normalizeSort(firstParam(searchParams, "sort"));
  const selectedPlayer = firstParam(searchParams, "player");
  const { data, error } = await getClubPlayers(clubSlug);
  const clubName = data?.club?.name ?? clubSlug;
  const players = data?.players ?? [];
  const activePlayers = players.filter((player) => player.is_active !== false);
  const inactivePlayers = players.length - activePlayers.length;
  const filteredPlayers = sortPlayers(players, selectedSort).filter((player) => {
    const textOk = !q || player.name.toLowerCase().includes(q.toLowerCase());
    const statusOk = selectedStatus === "all" || (selectedStatus === "active" ? player.is_active !== false : player.is_active === false);
    return textOk && statusOk;
  });
  const totalMatches = players.reduce((sum, player) => sum + matchesPlayed(player), 0);
  const topRated = sortPlayers(activePlayers, "rating")[0];
  const mostActive = sortPlayers(players, "matches")[0];

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Player profiles
      </p>
      <h1 style={{ marginTop: 0 }}>{clubName} players</h1>
      <p style={{ color: "#334155", maxWidth: "760px" }}>
        Player profiles connect ratings, match history, league-specific performance, and leaderboards.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Players are temporarily unavailable. {error}</p> : null}
      {!error && players.length === 0 ? <p>No public players are available yet.</p> : null}

      {players.length > 0 ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>Public players</strong><br />{players.length}</article>
            <article style={cardStyle}><strong>Active players</strong><br />{activePlayers.length}</article>
            <article style={cardStyle}><strong>Inactive players</strong><br />{inactivePlayers}</article>
            <article style={cardStyle}><strong>Recorded matches</strong><br />{totalMatches}</article>
            <article style={cardStyle}><strong>Top rating</strong><br />{topRated ? <Link href={`/clubs/${clubSlug}/players/${topRated.id}`}>{topRated.name}</Link> : "—"}</article>
            <article style={cardStyle}><strong>Most active</strong><br />{mostActive ? <Link href={`/clubs/${clubSlug}/players/${mostActive.id}`}>{mostActive.name}</Link> : "—"}</article>
          </div>

          <div style={{ display: "grid", gap: "0.75rem", marginBottom: "1rem" }}>
            {q ? <p style={{ margin: 0, color: "#475569" }}>Search filter: <strong>{q}</strong> · <Link href={pageHref({ clubSlug, status: selectedStatus, sort: selectedSort })}>clear search</Link></p> : null}
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
              {(["all", "active", "inactive"] as StatusKey[]).map((status) => {
                const active = status === selectedStatus;
                return (
                  <Link key={status} href={pageHref({ clubSlug, q, status, sort: selectedSort, player: selectedPlayer })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                    {status === "all" ? "All statuses" : status === "active" ? "Active" : "Inactive"}
                  </Link>
                );
              })}
              {(["rating", "matches", "win_pct", "recent", "name"] as SortKey[]).map((sort) => {
                const active = sort === selectedSort;
                return (
                  <Link key={sort} href={pageHref({ clubSlug, q, status: selectedStatus, sort, player: selectedPlayer })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                    Sort: {sort === "win_pct" ? "Win %" : sort === "recent" ? "Recent" : sort[0].toUpperCase() + sort.slice(1)}
                  </Link>
                );
              })}
            </div>
          </div>

          <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "12px", background: "white" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.95rem", minWidth: "820px" }}>
              <thead>
                <tr>
                  <th style={thStyle}>Player</th>
                  <th style={thStyle}>Rating</th>
                  <th style={thStyle}>Matches</th>
                  <th style={thStyle}>W/L</th>
                  <th style={thStyle}>Win %</th>
                  <th style={thStyle}>Last played</th>
                  <th style={thStyle}>Status</th>
                  <th style={thStyle}>Links</th>
                </tr>
              </thead>
              <tbody>
                {filteredPlayers.map((player) => {
                  const wins = player.wins ?? 0;
                  const losses = player.losses ?? 0;
                  const selected = String(player.id) === String(selectedPlayer);
                  return (
                    <tr key={String(player.id)} id={playerAnchor(player.id)} style={{ background: selected ? "#eff6ff" : undefined }}>
                      <td style={tdStyle}><Link href={`/clubs/${clubSlug}/players/${player.id}`}>{player.name}</Link></td>
                      <td style={tdStyle}>{ratingLabel(ratingValue(player))}</td>
                      <td style={tdStyle}>{matchesPlayed(player)}</td>
                      <td style={tdStyle}>{wins}/{losses}</td>
                      <td style={tdStyle}>{winPctLabel(player)}</td>
                      <td style={tdStyle}>{dateLabel(player.last_game_at)}</td>
                      <td style={tdStyle}>{player.is_active === false ? "Inactive" : "Active"}</td>
                      <td style={tdStyle}>
                        <Link href={`/clubs/${clubSlug}/leaderboards?player=${encodeURIComponent(String(player.id))}#leaderboard-player-${encodeURIComponent(String(player.id))}`}>leaderboard</Link>
                        <span style={{ color: "#64748b" }}> · </span>
                        <Link href={pageHref({ clubSlug, q, status: selectedStatus, sort: selectedSort, player: player.id }) + `#${playerAnchor(player.id)}`}>row link</Link>
                      </td>
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
