export type LeagueRouteContext = {
  leagueId: string;
  leagueName: string;
  leagueType: string;
};

type SearchValue = string | string[] | undefined;
type SearchRecord = Record<string, SearchValue>;
type SearchReader = { get(name: string): string | null };

function first(value: SearchValue | string | null): string {
  if (Array.isArray(value)) return String(value[0] || "").trim();
  return String(value || "").trim();
}

function readValue(searchParams: SearchRecord | SearchReader | undefined, key: string): string {
  if (!searchParams) return "";
  if ("get" in searchParams && typeof searchParams.get === "function") {
    return first(searchParams.get(key));
  }
  return first((searchParams as SearchRecord)[key]);
}

export function normalizeLeagueType(value: unknown): string {
  const cleaned = String(value || "").trim();
  if (cleaned.toLowerCase() === "team") return "Team";
  if (cleaned.toLowerCase() === "individual") return "Individual";
  return cleaned;
}

export function isTeamLeagueType(value: unknown): boolean {
  return normalizeLeagueType(value) === "Team";
}

/**
 * Keep the selected league's stable identifier separate from its display name.
 * Existing League Manager links used `league` for both values, so legacy deep
 * links remain readable while every newly generated link writes `league_id`.
 */
export function readLeagueRouteContext(
  searchParams: SearchRecord | SearchReader | undefined
): LeagueRouteContext {
  const legacyLeague = readValue(searchParams, "league");
  const leagueId = readValue(searchParams, "league_id") || legacyLeague;
  const leagueName = readValue(searchParams, "league_name") || legacyLeague || leagueId;
  const leagueType = normalizeLeagueType(
    readValue(searchParams, "mode") || readValue(searchParams, "league_type")
  );
  return { leagueId, leagueName, leagueType };
}

export function leagueRouteHref(
  path: string,
  context: LeagueRouteContext,
  extra?: Record<string, string | number | boolean | null | undefined>
): string {
  const params = new URLSearchParams();
  if (context.leagueId) params.set("league_id", context.leagueId);
  if (context.leagueName) {
    params.set("league", context.leagueName);
    params.set("league_name", context.leagueName);
  }
  const leagueType = normalizeLeagueType(context.leagueType);
  if (leagueType) params.set("mode", leagueType);
  Object.entries(extra || {}).forEach(([key, value]) => {
    if (value != null && value !== "") params.set(key, String(value));
  });
  const query = params.toString();
  return query ? `${path}?${query}` : path;
}

export function isExactLeagueResult(
  selectedLeague: string | null | undefined,
  expectedLeagueName: string
): boolean {
  return Boolean(expectedLeagueName && String(selectedLeague || "").trim() === expectedLeagueName.trim());
}
