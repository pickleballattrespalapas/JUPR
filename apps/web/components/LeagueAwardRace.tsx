import Link from "@/components/PublicClubLink";
import { statLeaderboardHref } from "@/lib/statLeaderboardLinks";
import type { LeagueAwardProgress, LeagueAwardProgressRow, LeagueAwardRace } from "@/lib/api";

const cardStyle = {
  border: "1px solid #bfdbfe",
  borderRadius: "14px",
  padding: "1rem",
  background: "#eff6ff"
};

function normalizedRank(entry: LeagueAwardProgressRow, index: number): number {
  const rank = Number(entry.rank);
  return Number.isFinite(rank) && rank > 0 ? rank : index + 1;
}

function fallbackRaces(progress: LeagueAwardProgress): LeagueAwardRace[] {
  const grouped = new Map<string, LeagueAwardProgressRow[]>();
  for (const award of progress.awards || []) {
    const key = award.category_key || award.category_label || "award";
    grouped.set(key, [...(grouped.get(key) || []), award]);
  }
  return Array.from(grouped.entries()).map(([categoryKey, entries]) => ({
    category_key: categoryKey,
    category_label: entries[0]?.category_label || "Award",
    recipient_type: entries[0]?.recipient_type,
    min_games: entries[0]?.min_games,
    minimum_metric: entries[0]?.minimum_metric,
    eligible_count: entries.length,
    entries
  }));
}

export function awardRaces(progress: LeagueAwardProgress): LeagueAwardRace[] {
  const unavailable = new Set(progress.unavailable_categories || []);
  return (progress.races?.filter((race) => race.entries?.length) || fallbackRaces(progress)).filter(race => !unavailable.has(race.category_key));
}

function previewEntries(entries: LeagueAwardProgressRow[]): LeagueAwardProgressRow[] {
  if (entries.length <= 5) return entries;
  const fifthRank = normalizedRank(entries[4], 4);
  return entries.filter((entry, index) => normalizedRank(entry, index) <= fifthRank);
}

function minimumLabel(race: LeagueAwardRace): string {
  const count = race.min_games ?? 0;
  const singular = count === 1;
  const metric = String(race.minimum_metric || "games");
  const labels: Record<string, string> = {
    games: singular ? "game" : "games",
    games_played: singular ? "game" : "games",
    close_games: singular ? "close game" : "close games",
    upset_wins: singular ? "upset win" : "upset wins",
    best_partnership_games: singular ? "partnership game" : "partnership games",
    weeks_played: singular ? "week played" : "weeks played"
  };
  return `At least ${count} ${labels[metric] || (singular ? "qualifying result" : "qualifying results")}`;
}

function playerName(clubSlug: string, entry: LeagueAwardProgressRow) {
  if (entry.team_members?.length) return <>{entry.team_members.map((member, index) => <span key={member.player_id}>{index ? " & " : ""}<Link href={`/clubs/${encodeURIComponent(clubSlug)}/players/${encodeURIComponent(String(member.player_id))}`}>{member.player_name}</Link></span>)}</>;
  if (entry.player_id == null) return entry.recipient_name || "—";
  return <Link href={`/clubs/${clubSlug}/players/${entry.player_id}`}>{entry.recipient_name || "—"}</Link>;
}

function placementRow(clubSlug: string, entry: LeagueAwardProgressRow, index: number) {
  return (
    <li key={`${entry.player_id ?? entry.team_id ?? entry.recipient_name}-${normalizedRank(entry, index)}`} style={{ display: "flex", gap: "0.45rem", justifyContent: "space-between", alignItems: "baseline" }}>
      <span><strong>#{normalizedRank(entry, index)}</strong> {playerName(clubSlug, entry)}{entry.is_co_winner ? " · tied" : ""}</span>
      <span style={{ color: "#334155", whiteSpace: "nowrap" }}>{entry.metric_display || "—"}</span>
    </li>
  );
}

export function LeagueAwardRaceGrid({ progress, clubSlug, leagueName, returnTo, teamLeague = false }: { progress: LeagueAwardProgress; clubSlug: string; leagueName?: string; returnTo?: string; teamLeague?: boolean }) {
  const races = awardRaces(progress);
  if (!races.length && !progress.unavailable_categories?.length) return null;
  return (
    <div data-testid="league-award-races" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "0.75rem" }}>
      {progress.unavailable_categories?.length ? <p role="status" style={{ color: "#475569" }}>Complete historical rankings are unavailable for {progress.unavailable_categories.map(key => key.replaceAll("_", " ")).join(", ")}; the original award recipients remain recorded.</p> : null}
      {races.map((race) => {
        const entries = race.entries || [];
        const preview = previewEntries(entries);
        const eligibleCount = race.eligible_count ?? entries.length;
        const teamAward = (race.recipient_type || entries[0]?.recipient_type) === "team";
        const qualifierLabel = teamAward
          ? `${eligibleCount} ${eligibleCount === 1 ? "team qualifies" : "teams qualify"}`
          : `${eligibleCount} ${eligibleCount === 1 ? "player qualifies" : "players qualify"}`;
        const minimum = minimumLabel(race);
        return (
          <article key={race.category_key} data-testid={`league-award-race-${race.category_key}`} style={cardStyle}>
            <h3 style={{ margin: "0 0 0.25rem" }}>{leagueName ? <Link prefetch={false} href={statLeaderboardHref(clubSlug, race.category_key, { source: teamLeague ? "team-awards" : "awards", league: leagueName, returnTo })} style={{ color: "#1d4ed8", textUnderlineOffset: "3px" }}>{race.category_label}</Link> : race.category_label}</h3>
            <p style={{ margin: "0 0 0.65rem", color: "#475569", fontSize: "0.88rem" }}>
              {qualifierLabel} · {minimum}
            </p>
            <ol style={{ margin: 0, paddingLeft: "1.5rem", display: "grid", gap: "0.3rem" }}>
              {preview.map((entry, index) => placementRow(clubSlug, entry, index))}
            </ol>
            {preview.length < entries.length ? (
              <details style={{ marginTop: "0.75rem" }}>
                <summary style={{ cursor: "pointer", fontWeight: 700 }}>
                  {teamAward ? "View every team that qualifies" : "View everyone who qualifies"} ({entries.length})
                </summary>
                <ol style={{ margin: "0.65rem 0 0", paddingLeft: "1.5rem", display: "grid", gap: "0.3rem" }}>
                  {entries.slice(preview.length).map((entry, index) => placementRow(clubSlug, entry, preview.length + index))}
                </ol>
              </details>
            ) : null}
          </article>
        );
      })}
    </div>
  );
}
