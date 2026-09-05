import Link from "next/link";
import { getClubMatches, type PublicMatch } from "@/lib/api";
import { publicMatchTypeLabel } from "@/lib/publicMatchLabels";

type MatchesPageProps = {
  params: { clubSlug: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

type SortKey = "date" | "league" | "score";

const thStyle = { textAlign: "left" as const, borderBottom: "1px solid #cbd5e1", padding: "0.6rem", whiteSpace: "nowrap" as const, color: "#475569", fontSize: "0.82rem" };
const tdStyle = { borderBottom: "1px solid #e2e8f0", padding: "0.6rem", whiteSpace: "nowrap" as const };
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

function firstParam(searchParams: MatchesPageProps["searchParams"], key: string): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function normalizeSort(value: string | null): SortKey {
  if (value === "league" || value === "score") return value;
  return "date";
}

function pageHref({ clubSlug, league, q, sort, match }: { clubSlug: string; league?: string | null; q?: string | null; sort?: SortKey | null; match?: string | number | null }): string {
  const params = new URLSearchParams();
  if (league) params.set("league", league);
  if (q) params.set("q", q);
  if (sort && sort !== "date") params.set("sort", sort);
  if (match) params.set("match", String(match));
  const query = params.toString();
  return `/clubs/${clubSlug}/matches${query ? `?${query}` : ""}`;
}

function matchAnchor(matchId: string | number): string {
  return `match-${encodeURIComponent(String(matchId))}`;
}

function formatMatchDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function teamLabel(clubSlug: string, players: PublicMatch["team_1"]): JSX.Element {
  return (
    <>
      {players.map((player, index) => (
        <span key={String(player.id)}>
          {index > 0 ? " / " : ""}
          <Link href={`/clubs/${clubSlug}/players/${player.id}`}>{player.name}</Link>
        </span>
      ))}
    </>
  );
}

function plainTeamLabel(players: PublicMatch["team_1"]): string {
  return players.map((player) => player.name).filter(Boolean).join(" / ");
}

function scoreLabel(match: PublicMatch): string {
  const scoreA = match.score_t1 ?? null;
  const scoreB = match.score_t2 ?? null;
  return scoreA == null && scoreB == null ? "—" : `${scoreA ?? 0}–${scoreB ?? 0}`;
}

function scoreTotal(match: PublicMatch): number {
  return (match.score_t1 ?? 0) + (match.score_t2 ?? 0);
}

function sortMatches(matches: PublicMatch[], sort: SortKey): PublicMatch[] {
  const sorted = [...matches];
  sorted.sort((a, b) => {
    if (sort === "league") return String(a.league ?? "").localeCompare(String(b.league ?? ""));
    if (sort === "score") return scoreTotal(b) - scoreTotal(a);
    return String(b.date ?? "").localeCompare(String(a.date ?? ""));
  });
  return sorted;
}

export default async function MatchesPage({ params, searchParams }: MatchesPageProps) {
  const { clubSlug } = params;
  const selectedLeague = firstParam(searchParams, "league");
  const q = (firstParam(searchParams, "q") ?? "").trim();
  const selectedSort = normalizeSort(firstParam(searchParams, "sort"));
  const selectedMatch = firstParam(searchParams, "match");
  const { data, error } = await getClubMatches(clubSlug);
  const clubName = data?.club?.name ?? clubSlug;
  const matches = data?.matches ?? [];
  const leagues = Array.from(new Set(matches.map((match) => match.league).filter(Boolean) as string[])).sort((a, b) => a.localeCompare(b));
  const filteredMatches = sortMatches(matches, selectedSort).filter((match) => {
    const leagueOk = !selectedLeague || match.league === selectedLeague;
    const text = `${plainTeamLabel(match.team_1)} ${plainTeamLabel(match.team_2)} ${match.league ?? ""} ${match.week_tag ?? ""}`.toLowerCase();
    const searchOk = !q || text.includes(q.toLowerCase());
    return leagueOk && searchOk;
  });
  const scoredCount = matches.filter((match) => match.score_t1 != null || match.score_t2 != null).length;
  const latestMatch = sortMatches(matches, "date")[0];

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Match history
      </p>
      <h1 style={{ marginTop: 0 }}>{clubName} matches</h1>
      <p style={{ color: "#475569", maxWidth: "760px" }}>Browse recorded scores and see how each match affected player ratings.</p>

      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Match history is unavailable right now. Please try again shortly.</p> : null}
      {!error && matches.length === 0 ? <p>No matches have been recorded yet.</p> : null}

      {matches.length > 0 ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>Recorded matches</strong><br />{matches.length}</article>
            <article style={cardStyle}><strong>Scored matches</strong><br />{scoredCount}</article>
            <article style={cardStyle}><strong>Leagues</strong><br />{leagues.length}</article>
            <article style={cardStyle}><strong>Latest match</strong><br />{latestMatch ? formatMatchDate(latestMatch.date) : "—"}</article>
          </div>

          <div style={{ display: "grid", gap: "0.75rem", marginBottom: "1rem" }}>
            {q ? <p style={{ margin: 0, color: "#475569" }}>Search filter: <strong>{q}</strong> · <Link href={pageHref({ clubSlug, league: selectedLeague, sort: selectedSort })}>clear search</Link></p> : null}
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
              <Link href={pageHref({ clubSlug, q, sort: selectedSort, match: selectedMatch })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: !selectedLeague ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: !selectedLeague ? 800 : 600 }}>All leagues</Link>
              {leagues.map((league) => {
                const active = league === selectedLeague;
                return (
                  <Link key={league} href={pageHref({ clubSlug, league, q, sort: selectedSort, match: selectedMatch })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                    {league}
                  </Link>
                );
              })}
              {(["date", "league", "score"] as SortKey[]).map((sort) => {
                const active = sort === selectedSort;
                return (
                  <Link key={sort} href={pageHref({ clubSlug, league: selectedLeague, q, sort, match: selectedMatch })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                    Sort: {sort[0].toUpperCase() + sort.slice(1)}
                  </Link>
                );
              })}
            </div>
          </div>

          <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "12px", background: "white" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.95rem", minWidth: "820px" }}>
              <thead>
                <tr>
                  <th style={thStyle}>Date</th>
                  <th style={thStyle}>Team 1</th>
                  <th style={thStyle}>Score</th>
                  <th style={thStyle}>Team 2</th>
                  <th style={thStyle}>League</th>
                  <th style={thStyle}>Type</th>
                  <th style={thStyle}>Share</th>
                </tr>
              </thead>
              <tbody>
                {filteredMatches.map((match, index) => {
                  const detailHref = match.id ? `/clubs/${clubSlug}/matches/${match.id}` : `/clubs/${clubSlug}/matches`;
                  const selected = match.id != null && String(match.id) === String(selectedMatch);
                  return (
                    <tr key={`${match.id ?? index}`} id={match.id ? matchAnchor(match.id) : undefined} style={{ background: selected ? "#eff6ff" : undefined }}>
                      <td style={tdStyle}>{match.id ? <Link href={detailHref}>{formatMatchDate(match.date)}</Link> : formatMatchDate(match.date)}</td>
                      <td style={tdStyle}>{teamLabel(clubSlug, match.team_1)}</td>
                      <td style={tdStyle}>{match.id ? <Link href={detailHref}>{scoreLabel(match)}</Link> : scoreLabel(match)}</td>
                      <td style={tdStyle}>{teamLabel(clubSlug, match.team_2)}</td>
                      <td style={tdStyle}>{match.league ?? "—"}</td>
                      <td style={tdStyle}>{publicMatchTypeLabel(match.match_type)}</td>
                      <td style={tdStyle}>{match.id ? <Link aria-label={`Share match from ${formatMatchDate(match.date)}`} href={pageHref({ clubSlug, league: selectedLeague, q, sort: selectedSort, match: match.id }) + `#${matchAnchor(match.id)}`}>share</Link> : "—"}</td>
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
