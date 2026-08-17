export const LEAGUE_STATUS_FILTERS = [
  { key: "active", label: "Active" },
  { key: "draft", label: "Draft / inactive" },
  { key: "paused", label: "Paused" },
  { key: "ended", label: "Ended" },
  { key: "archived", label: "Archived" },
  { key: "all", label: "All" }
] as const;

export type LeagueStatusFilter = (typeof LEAGUE_STATUS_FILTERS)[number]["key"];

type LeagueWithStatus = {
  status?: string | null;
};

export function leagueStatusCategory(status: string | null | undefined): Exclude<LeagueStatusFilter, "all"> | "other" {
  const normalized = String(status || "").trim().toLowerCase();
  if (normalized === "inactive") return "draft";
  if (normalized === "draft" || normalized === "active" || normalized === "paused" || normalized === "ended" || normalized === "archived") {
    return normalized;
  }
  return "other";
}

export function leagueStatusCounts(leagues: LeagueWithStatus[]): Record<LeagueStatusFilter, number> {
  const counts: Record<LeagueStatusFilter, number> = {
    active: 0,
    draft: 0,
    paused: 0,
    ended: 0,
    archived: 0,
    all: leagues.length
  };
  for (const league of leagues) {
    const category = leagueStatusCategory(league.status);
    if (category !== "other") counts[category] += 1;
  }
  return counts;
}

export function filterLeaguesByStatus<T extends LeagueWithStatus>(leagues: T[], filter: LeagueStatusFilter): T[] {
  if (filter === "all") return leagues;
  return leagues.filter((league) => leagueStatusCategory(league.status) === filter);
}

export function leagueStatusLabel(filter: LeagueStatusFilter): string {
  return LEAGUE_STATUS_FILTERS.find((item) => item.key === filter)?.label || "All";
}
