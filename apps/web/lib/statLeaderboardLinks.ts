export type StatLeaderboardContext = {
  source?: "awards" | "team-awards" | "weekly" | "season";
  league?: string;
  leagueView?: "active" | "past";
  status?: "active" | "inactive" | "all";
  season?: string;
  search?: string;
  returnTo?: string;
  week?: number;
  weeklyMinGames?: number;
};

/** Keep return navigation inside this club, including its original filters/anchor. */
export function safeStatReturnTo(clubSlug: string, value?: string | null): string {
  const base = `/clubs/${encodeURIComponent(clubSlug)}`;
  const fallback = `${base}/leaderboards`;
  if (!value || /[\\\u0000-\u001f]/.test(value)) return fallback;
  try {
    const url = new URL(value, "https://club.invalid");
    if (!value.startsWith(`${base}/`) || url.origin !== "https://club.invalid" || !url.pathname.startsWith(`${base}/`)) return fallback;
    return `${url.pathname}${url.search}${url.hash}`;
  } catch { return fallback; }
}

export function statLeaderboardHref(clubSlug: string, stat: string, context: StatLeaderboardContext = {}): string {
  const params = new URLSearchParams();
  if (context.source) params.set("source", context.source);
  if (context.league) params.set("league", context.league);
  if (context.leagueView) params.set("league_view", context.leagueView);
  if (context.status) params.set("status", context.status);
  if (context.season) params.set("season", context.season);
  if (context.search) params.set("q", context.search);
  if (context.week) params.set("week", String(context.week));
  if (context.weeklyMinGames) params.set("weekly_min_games", String(context.weeklyMinGames));
  if (context.returnTo) params.set("return_to", safeStatReturnTo(clubSlug, context.returnTo));
  const query = params.toString();
  return `/clubs/${encodeURIComponent(clubSlug)}/leaderboards/${encodeURIComponent(stat)}${query ? `?${query}` : ""}`;
}
