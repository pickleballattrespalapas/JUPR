export type TournamentRouteContext = {
  tournamentId: string;
  tournamentName: string;
  drawId: string;
  dayId: string;
};

type TournamentRouteContextInput = Omit<TournamentRouteContext, "dayId"> & {
  dayId?: string;
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

/**
 * Accept the current Tournament Manager query names and the legacy handoff
 * names. All tournament routes use this helper so a selected draw cannot be
 * silently dropped when an operator changes modules.
 */
export function readTournamentRouteContext(
  searchParams: SearchRecord | SearchReader | undefined
): TournamentRouteContext {
  let legacyTournamentId = "";
  let legacyDrawId = "";
  if (searchParams && "get" in searchParams && typeof searchParams.get === "function") {
    legacyTournamentId = first(searchParams.get("tournament_id"));
    legacyDrawId = first(searchParams.get("draw_id"));
  } else {
    legacyTournamentId = readValue(searchParams, "tournament_id");
    legacyDrawId = readValue(searchParams, "draw_id");
  }
  const tournamentId = readValue(searchParams, "tournament") || legacyTournamentId;
  const tournamentName = readValue(searchParams, "tournament_name") || readValue(searchParams, "name");
  const drawId = readValue(searchParams, "draw") || legacyDrawId;
  const dayId = readValue(searchParams, "day") || readValue(searchParams, "day_id");
  return { tournamentId, tournamentName, drawId, dayId };
}

export function tournamentRouteHref(
  path: string,
  context: TournamentRouteContextInput,
  extra?: Record<string, string | number | boolean | null | undefined>
): string {
  const params = new URLSearchParams();
  if (context.tournamentId) params.set("tournament", context.tournamentId);
  if (context.tournamentName) {
    params.set("tournament_name", context.tournamentName);
    params.set("name", context.tournamentName);
  }
  if (context.drawId) params.set("draw", context.drawId);
  if (context.dayId) params.set("day", context.dayId);
  Object.entries(extra || {}).forEach(([key, value]) => {
    if (value != null && value !== "") params.set(key, String(value));
  });
  const query = params.toString();
  return query ? `${path}?${query}` : path;
}
