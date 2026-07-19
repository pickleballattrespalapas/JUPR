export type WeeklyRecapNumbers = Record<string, string | number | null>;

export type WeeklyRecapSpotlight = {
  key?: string | null;
  label: string;
  players: string[];
  description?: string | null;
  order?: number | null;
  include?: boolean | null;
};

export type WeeklyRecapHighlight = { display: string };

export type WeeklyRecapAroundClub = {
  leagues: Array<{ league_name: string; highlights: WeeklyRecapHighlight[] }>;
  round_robins: Array<{ event_name: string; highlights: WeeklyRecapHighlight[] }>;
  community_events: Array<{ event_name: string; event_type_label?: string | null; skill_level?: string | null; highlights: WeeklyRecapHighlight[] }>;
};

export type WeeklyRecapTournament = {
  tournament_name: string;
  podium: Array<{ placement: number; display_name: string }>;
};

export type WeeklyRecapJson = {
  week_start?: string | null;
  week_end?: string | null;
  start_date?: string | null;
  end_date?: string | null;
  numbers: WeeklyRecapNumbers;
  numbers_cards: Array<{ key?: string | null; label: string; value: string | number | null }>;
  spotlight: WeeklyRecapSpotlight[];
  around_club: WeeklyRecapAroundClub;
  tournaments: WeeklyRecapTournament[];
  looking_ahead: string[];
  highlights: string[];
};

export type WeeklyRecapSummary = {
  week_start: string;
  week_end: string;
  updated_at?: string | null;
  summary: {
    matches?: string | number | null;
    players?: string | number | null;
    leagues?: string | number | null;
    headline?: string | null;
  };
};

export type WeeklyRecapSelected = WeeklyRecapSummary & {
  recap: WeeklyRecapJson;
};

export type WeeklyRecapsResponse = {
  club: { id: string; slug: string; name: string };
  recaps: WeeklyRecapSummary[];
  selected_recap?: WeeklyRecapSelected | null;
  pagination: {
    page: number;
    page_size: number;
    has_previous: boolean;
    has_next: boolean;
  };
};

type ApiResult<T> = { data: T | null; error: string | null };

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

async function apiErrorMessage(response: Response): Promise<string> {
  const fallback = `API error (${response.status}).`;
  let bodyText = "";
  try {
    bodyText = await response.text();
  } catch {
    return fallback;
  }
  if (!bodyText) return fallback;
  try {
    const payload = JSON.parse(bodyText) as { detail?: unknown; message?: unknown; error?: unknown };
    const detail = payload.detail ?? payload.message ?? payload.error;
    if (Array.isArray(detail)) return `${fallback} ${detail.map((item) => JSON.stringify(item)).join("; ")}`;
    if (detail) return `${fallback} ${String(detail)}`;
  } catch {
    // Fall through to a short text excerpt below.
  }
  return `${fallback} ${bodyText.slice(0, 240)}`;
}

async function fetchJson<T>(path: string): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) return { data: null, error: "Missing JUPR API base URL environment variable." };
  const url = `${apiBase.replace(/\/$/, "")}${path}`;
  try {
    const response = await fetch(url, { next: { revalidate: 60 } });
    if (!response.ok) return { data: null, error: await apiErrorMessage(response) };
    return { data: (await response.json()) as T, error: null };
  } catch (error) {
    return { data: null, error: `Unable to reach API: ${error instanceof Error ? error.message : "Unknown error"}` };
  }
}

export async function getClubWeeklyRecaps(clubSlug: string, weekStart?: string | null, page = 1): Promise<ApiResult<WeeklyRecapsResponse>> {
  const query = new URLSearchParams({ page: String(Math.max(1, Math.round(page))), page_size: "8" });
  if (weekStart) query.set("week_start", weekStart);
  return fetchJson<WeeklyRecapsResponse>(`/clubs/${clubSlug}/weekly-recaps?${query.toString()}`);
}

export function getWeeklyRecapPdfUrl(clubSlug: string, weekStart: string): string | null {
  if (!clubSlug || !weekStart) return null;
  return `/api/clubs/${encodeURIComponent(clubSlug)}/weekly-recaps/${encodeURIComponent(weekStart)}/pdf`;
}
