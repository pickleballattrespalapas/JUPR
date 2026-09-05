import Link from "next/link";
import PublicLeagueNav from "@/components/PublicLeagueNav";
import { getClubLeagueResults, type LeagueResultsStatRow } from "@/lib/api";

type Props = {
  params: { clubSlug: string; leagueName: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white",
  minWidth: 0
};

function decodeLeagueName(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function firstParam(searchParams: Props["searchParams"], key: string): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function parsePositiveInt(value: string | null): number | null {
  if (!value) return null;
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function percentLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${Number(value).toFixed(1)}%`;
}

function deltaLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const amount = Number(value);
  return `${amount >= 0 ? "+" : ""}${amount.toFixed(3)}`;
}

function weeklyHref(clubSlug: string, leagueName: string, week: number): string {
  const query = new URLSearchParams({ week: String(week) });
  return `/clubs/${clubSlug}/leagues/${encodeURIComponent(leagueName)}/weekly-history?${query.toString()}`;
}

function highlightRows(rows: LeagueResultsStatRow[]): string {
  if (!rows.length) return "No qualifying players yet";
  return rows
    .slice(0, 3)
    .map((row) => `${row.player_name} (${row.wins ?? 0}-${row.losses ?? 0})`)
    .join(", ");
}

export default async function PublicLeagueWeeklyHistoryPage({ params, searchParams }: Props) {
  const leagueName = decodeLeagueName(params.leagueName);
  const requestedWeek = parsePositiveInt(firstParam(searchParams, "week"));
  const { data, error } = await getClubLeagueResults(
    params.clubSlug,
    leagueName,
    requestedWeek
  );
  const found = data?.selected_league === leagueName;
  const leagueView = data?.past_leagues.some((league) => league.name === leagueName)
    ? "past"
    : "active";

  if (error || !data || !found) {
    return (
      <section>
        <p style={{ color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
          Weekly History
        </p>
        <h1>{leagueName}</h1>
        <article style={{ ...cardStyle, borderColor: "#fecaca", background: "#fef2f2" }}>
          <h2 style={{ marginTop: 0 }}>Weekly history unavailable</h2>
          <p style={{ color: "#7f1d1d" }}>
            {error || "We couldn't find this league."}
          </p>
          <Link href={`/clubs/${params.clubSlug}/leagues`}>Return to all leagues</Link>
        </article>
      </section>
    );
  }

  const selectedWeek = data.selected_week ?? null;
  const weeklyRows = selectedWeek
    ? data.weekly_results.filter((row) => row.week_num === selectedWeek)
    : [];
  const totalGames = weeklyRows.reduce((sum, row) => sum + Number(row.games || 0), 0);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Weekly History
      </p>
      <h1 style={{ marginTop: 0 }}>{leagueName} weekly history</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Week-by-week activity, records, rating movement, and highlights for {leagueName}.
      </p>

      <PublicLeagueNav clubSlug={params.clubSlug} leagueName={leagueName} active="weekly" leagueView={leagueView} />

      {data.weeks.length ? (
        <nav aria-label="League weeks" style={{ display: "flex", gap: "0.45rem", flexWrap: "wrap", marginBottom: "1rem" }}>
          {data.weeks.map((week) => {
            const active = week.week_num === selectedWeek;
            return (
              <Link
                key={week.week_num}
                href={weeklyHref(params.clubSlug, leagueName, week.week_num)}
                aria-current={active ? "page" : undefined}
                style={{
                  border: `1px solid ${active ? "#2563eb" : "#cbd5e1"}`,
                  borderRadius: "999px",
                  padding: "0.4rem 0.7rem",
                  background: active ? "#dbeafe" : "white",
                  color: active ? "#1d4ed8" : "#0f172a",
                  textDecoration: "none",
                  fontWeight: active ? 800 : 650
                }}
              >
                {week.week_label}{week.has_results === false ? " · no results" : ""}
              </Link>
            );
          })}
        </nav>
      ) : null}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
        <article style={cardStyle}><strong>Selected week</strong><br />{selectedWeek ? `Week ${selectedWeek}` : "No results"}</article>
        <article style={cardStyle}><strong>Players active</strong><br />{weeklyRows.length}</article>
        <article style={cardStyle}><strong>Player appearances</strong><br />{totalGames}</article>
        <article style={cardStyle}><strong>Weeks available</strong><br />{data.weeks.length}</article>
      </div>

      <section>
        <h2>{selectedWeek ? `Week ${selectedWeek} results` : "Weekly results"}</h2>
        {weeklyRows.length ? (
          <div style={{ display: "grid", gap: "0.65rem" }}>
            {[...weeklyRows]
              .sort((a, b) => Number(a.rank ?? 9999) - Number(b.rank ?? 9999))
              .map((row) => (
                <article
                  key={`${row.week_num}-${row.player_id}`}
                  style={{
                    ...cardStyle,
                    display: "grid",
                    gridTemplateColumns: "48px minmax(0, 1fr) auto",
                    gap: "0.75rem",
                    alignItems: "center"
                  }}
                >
                  <strong>#{row.rank ?? "—"}</strong>
                  <div>
                    <Link href={`/clubs/${params.clubSlug}/players/${row.player_id}`} style={{ fontWeight: 800 }}>
                      {row.player_name}
                    </Link>
                    <div style={{ color: "#64748b", fontSize: "0.88rem", marginTop: "0.2rem" }}>
                      {row.wins ?? 0}-{row.losses ?? 0} · {row.games ?? 0} games · {percentLabel(row.win_pct)} wins
                    </div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <strong>{deltaLabel(row.rating_delta_jupr)}</strong>
                    <div style={{ color: "#64748b", fontSize: "0.82rem" }}>
                      {row.rank_delta == null ? "rank unchanged" : `${row.rank_delta >= 0 ? "+" : ""}${row.rank_delta} rank`}
                    </div>
                  </div>
                </article>
              ))}
          </div>
        ) : (
          <article style={cardStyle}>No results are available for this week.</article>
        )}
      </section>

      <section style={{ marginTop: "1.25rem" }}>
        <h2>Weekly highlights</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
          <article style={cardStyle}>
            <h3 style={{ marginTop: 0 }}>Biggest climbers</h3>
            <p style={{ marginBottom: 0 }}>{highlightRows(data.weekly_highlights.biggest_climbers)}</p>
          </article>
          <article style={cardStyle}>
            <h3 style={{ marginTop: 0 }}>Best win percentage</h3>
            <p style={{ marginBottom: 0 }}>{highlightRows(data.weekly_highlights.best_win_pct)}</p>
          </article>
          <article style={cardStyle}>
            <h3 style={{ marginTop: 0 }}>Most active</h3>
            <p style={{ marginBottom: 0 }}>{highlightRows(data.weekly_highlights.most_active)}</p>
          </article>
        </div>
      </section>
    </section>
  );
}
