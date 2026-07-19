import Link from "next/link";
import { getClubPlayers } from "@/lib/api";
import type { PublicPlayer } from "@/lib/api";

type PlayersPageProps = {
  params: { clubSlug: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

type SortKey = "rating" | "singles" | "matches" | "name" | "win_pct" | "recent";
type StatusKey = "active" | "inactive" | "all";

const thStyle = { textAlign: "left" as const, borderBottom: "1px solid #cbd5e1", padding: "0.6rem", whiteSpace: "nowrap" as const, color: "#475569", fontSize: "0.82rem" };
const tdStyle = { borderBottom: "1px solid #e2e8f0", padding: "0.6rem", whiteSpace: "nowrap" as const };
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const pillStyle = { border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", color: "#0f172a", textDecoration: "none" };

function firstParam(searchParams: PlayersPageProps["searchParams"], key: string): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function normalizeSort(value: string | null): SortKey {
  if (value === "singles" || value === "matches" || value === "name" || value === "win_pct" || value === "recent") return value;
  return "rating";
}

function normalizeStatus(value: string | null): StatusKey {
  if (value === "inactive" || value === "all") return value;
  return "active";
}

function positiveInt(value: string | null, fallback: number): number {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function pageHref({
  clubSlug,
  q,
  status,
  sort,
  player,
  page,
  perPage
}: {
  clubSlug: string;
  q?: string | null;
  status?: StatusKey | null;
  sort?: SortKey | null;
  player?: string | number | null;
  page?: number | null;
  perPage?: number | null;
}): string {
  const params = new URLSearchParams();
  if (q) params.set("q", q);
  if (status && status !== "active") params.set("status", status);
  if (sort && sort !== "rating") params.set("sort", sort);
  if (player != null && String(player)) params.set("player", String(player));
  if (page && page > 1) params.set("page", String(page));
  if (perPage && perPage !== 50) params.set("per_page", String(perPage));
  const query = params.toString();
  return `/clubs/${clubSlug}/players${query ? `?${query}` : ""}`;
}

function playerAnchor(playerId: string | number): string {
  return `player-${encodeURIComponent(String(playerId))}`;
}

function ratingValue(raw?: number | null, normalized?: number | null): number | null {
  if (normalized != null && !Number.isNaN(Number(normalized))) return Number(normalized);
  if (raw == null || Number.isNaN(Number(raw))) return null;
  const value = Number(raw);
  return Math.abs(value) > 20 ? value / 400 : value;
}

function ratingLabel(raw?: number | null, normalized?: number | null): string {
  const value = ratingValue(raw, normalized);
  return value == null ? "—" : value.toFixed(3);
}

function winPctLabel(player: PublicPlayer): string {
  const wins = player.wins ?? 0;
  const losses = player.losses ?? 0;
  return wins + losses > 0 ? `${((wins / (wins + losses)) * 100).toFixed(1)}%` : "—";
}

function dateLabel(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value).slice(0, 10) : date.toISOString().slice(0, 10);
}

export default async function ClubPlayersPage({ params, searchParams }: PlayersPageProps) {
  const { clubSlug } = params;
  const q = (firstParam(searchParams, "q") ?? "").trim().slice(0, 80);
  const status = normalizeStatus(firstParam(searchParams, "status"));
  const sort = normalizeSort(firstParam(searchParams, "sort"));
  const selectedPlayer = firstParam(searchParams, "player");
  const page = positiveInt(firstParam(searchParams, "page"), 1);
  const requestedPerPage = positiveInt(firstParam(searchParams, "per_page"), 50);
  const perPage = [25, 50, 100].includes(requestedPerPage) ? requestedPerPage : 50;
  const offset = (page - 1) * perPage;
  const { data, error } = await getClubPlayers(clubSlug, { q, status, sort, limit: perPage, offset });
  const clubName = data?.club?.name ?? clubSlug;
  const players = data?.players ?? [];
  const summary = data?.summary;
  const pagination = data?.pagination;
  const pageCount = Math.max(1, Math.ceil((pagination?.total ?? 0) / perPage));

  return (
    <section data-testid="players-directory">
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Player profiles</p>
      <h1 style={{ marginTop: 0 }}>{clubName} players</h1>
      <p style={{ color: "#334155", maxWidth: "780px" }}>
        Find a public player profile, then explore doubles and singles ratings, verified match history, awards, and club connections. Active players are shown by default.
      </p>

      {error ? (
        <article data-testid="players-error-state" style={{ ...cardStyle, background: "#fef2f2", color: "#991b1b", marginBottom: "1rem" }}>
          Players are temporarily unavailable. No private player data was exposed. Please try again shortly.
        </article>
      ) : null}

      {!error ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>Public players</strong><br />{summary?.public_players ?? 0}</article>
            <article style={cardStyle}><strong>Active players</strong><br />{summary?.active_players ?? 0}</article>
            <article style={cardStyle}><strong>Inactive players</strong><br />{summary?.inactive_players ?? 0}</article>
            <article style={cardStyle}><strong>Matching this view</strong><br />{summary?.filtered_players ?? 0}</article>
          </div>

          <form method="get" action={`/clubs/${clubSlug}/players`} data-testid="players-search-form" style={{ ...cardStyle, marginBottom: "1rem", display: "grid", gridTemplateColumns: "minmax(220px, 1fr) auto", gap: "0.65rem", alignItems: "end" }}>
            <label htmlFor="players-search"><strong>Find player</strong><br />
              <input id="players-search" name="q" defaultValue={q} type="search" placeholder="Search public display name" style={{ width: "100%", padding: "0.6rem", border: "1px solid #94a3b8", borderRadius: "8px", font: "inherit" }} />
            </label>
            <input type="hidden" name="status" value={status} />
            <input type="hidden" name="sort" value={sort} />
            <input type="hidden" name="per_page" value={perPage} />
            <button type="submit" style={{ padding: "0.65rem 1rem", border: "1px solid #0f172a", borderRadius: "999px", background: "#0f172a", color: "white", fontWeight: 800 }}>Search</button>
          </form>

          <div data-testid="players-status-tabs" style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "0.65rem" }}>
            {(["active", "inactive", "all"] as StatusKey[]).map((item) => (
              <Link
                key={item}
                data-testid={`players-status-${item}`}
                aria-current={item === status ? "page" : undefined}
                href={pageHref({ clubSlug, q, status: item, sort, perPage })}
                style={{ ...pillStyle, background: item === status ? "#dbeafe" : "white", fontWeight: item === status ? 800 : 600 }}
              >
                {item === "active" ? "Active" : item === "inactive" ? "Inactive" : "All statuses"}
              </Link>
            ))}
          </div>

          <div data-testid="players-sort-tabs" style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
            {(["rating", "singles", "matches", "win_pct", "recent", "name"] as SortKey[]).map((item) => (
              <Link
                key={item}
                aria-current={item === sort ? "page" : undefined}
                href={pageHref({ clubSlug, q, status, sort: item, perPage })}
                style={{ ...pillStyle, background: item === sort ? "#dcfce7" : "white", fontWeight: item === sort ? 800 : 600 }}
              >
                Sort: {item === "win_pct" ? "Win %" : item === "recent" ? "Recent" : item === "singles" ? "Singles" : item[0].toUpperCase() + item.slice(1)}
              </Link>
            ))}
          </div>

          {players.length === 0 ? (
            <article data-testid="players-filter-empty-state" style={cardStyle}>
              <h2 style={{ marginTop: 0 }}>No players match this view</h2>
              <p>Try a different public display name or status.</p>
              <Link href={pageHref({ clubSlug, status: "active", sort: "rating" })}>Reset to active players</Link>
            </article>
          ) : (
            <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "12px", background: "white" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.95rem", minWidth: "1020px" }}>
                <thead><tr><th style={thStyle}>Player</th><th style={thStyle}>Doubles / overall</th><th style={thStyle}>Singles</th><th style={thStyle}>Doubles MP</th><th style={thStyle}>Singles MP</th><th style={thStyle}>Doubles W/L</th><th style={thStyle}>Singles W/L</th><th style={thStyle}>Win %</th><th style={thStyle}>Last played</th><th style={thStyle}>Status</th><th style={thStyle}>Stable links</th></tr></thead>
                <tbody>
                  {players.map((player) => {
                    const selected = String(player.id) === String(selectedPlayer);
                    const profileHref = `/clubs/${clubSlug}/players/${encodeURIComponent(String(player.id))}`;
                    return (
                      <tr key={String(player.id)} id={playerAnchor(player.id)} data-testid="players-row" data-status={player.is_active === false ? "inactive" : "active"} style={{ background: selected ? "#eff6ff" : undefined }}>
                        <td style={tdStyle}><Link aria-label={`Open ${player.name} profile`} href={profileHref}><strong>{player.name}</strong></Link></td>
                        <td style={tdStyle}>{ratingLabel(player.rating, player.rating_jupr)}</td>
                        <td style={tdStyle}>{ratingLabel(player.singles_rating, player.singles_rating_jupr)}</td>
                        <td style={tdStyle}>{player.matches_played ?? (player.wins ?? 0) + (player.losses ?? 0)}</td>
                        <td style={tdStyle}>{player.singles_matches_played ?? (player.singles_wins ?? 0) + (player.singles_losses ?? 0)}</td>
                        <td style={tdStyle}>{player.wins ?? 0}/{player.losses ?? 0}</td>
                        <td style={tdStyle}>{player.singles_wins ?? 0}/{player.singles_losses ?? 0}</td>
                        <td style={tdStyle}>{winPctLabel(player)}</td>
                        <td style={tdStyle}>{dateLabel(player.last_game_at)}</td>
                        <td style={tdStyle}>{player.is_active === false ? "Inactive" : "Active"}</td>
                        <td style={tdStyle}>
                          <Link href={profileHref}>profile</Link><span style={{ color: "#64748b" }}> · </span>
                          <Link aria-label={`${player.name} stable row link`} href={`${pageHref({ clubSlug, q, status, sort, player: player.id, page, perPage })}#${playerAnchor(player.id)}`}>row link</Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {pageCount > 1 ? (
            <nav data-testid="players-pagination" aria-label="Player directory pages" style={{ display: "flex", gap: "0.75rem", alignItems: "center", marginTop: "1rem" }}>
              {page > 1 ? <Link href={pageHref({ clubSlug, q, status, sort, page: page - 1, perPage })}>Previous</Link> : <span style={{ color: "#94a3b8" }}>Previous</span>}
              <span>Page {Math.min(page, pageCount)} of {pageCount}</span>
              {pagination?.has_more ? <Link href={pageHref({ clubSlug, q, status, sort, page: page + 1, perPage })}>Next</Link> : <span style={{ color: "#94a3b8" }}>Next</span>}
            </nav>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
