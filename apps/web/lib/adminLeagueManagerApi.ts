export type AdminLeagueManagerStatusResponse = {
  enabled: boolean;
  status: string;
  leagues_endpoint?: string | null;
  league_detail_endpoint?: string | null;
  league_count?: number | null;
  active_count?: number | null;
  warnings: string[];
};

export type AdminLeagueManagerLeague = {
  league_name: string;
  status: string;
  is_active?: boolean | null;
  started_at?: string | null;
  ended_at?: string | null;
  ended_by?: string | null;
  k_factor?: number | null;
  min_games?: number | null;
  schedule_config?: Record<string, unknown>;
  court_board_defaults?: Record<string, unknown>;
  rules_config?: Record<string, unknown>;
  awards_config?: Record<string, unknown>;
  event_tags?: Record<string, unknown>;
};

export type AdminLeagueManagerSchedulePreviewRow = {
  session: number;
  date: string;
  start?: string | null;
  end?: string | null;
};

export type AdminLeagueManagerStanding = {
  rank: number;
  player_id: number;
  player_name: string;
  rating?: number | null;
  rating_jupr?: number | null;
  starting_rating?: number | null;
  wins?: number | null;
  losses?: number | null;
  matches_played?: number | null;
  is_active?: boolean | null;
  inactive_at?: string | null;
};

export type AdminLeagueManagerListResponse = {
  ok: boolean;
  mode?: string;
  leagues: AdminLeagueManagerLeague[];
  count: number;
};

export type AdminLeagueManagerDetailResponse = {
  ok: boolean;
  mode?: string;
  league: AdminLeagueManagerLeague;
  schedule_preview: AdminLeagueManagerSchedulePreviewRow[];
  standings: AdminLeagueManagerStanding[];
  standings_count: number;
};

type ApiResult<T> = { data: T | null; error: string | null };

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export function getAdminLeagueManagerApiBaseUrl(): string | null {
  return baseUrl();
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
    // Fall through to short text excerpt below.
  }
  return `${fallback} ${bodyText.slice(0, 240)}`;
}

async function fetchJson<T>(path: string): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) return { data: null, error: "Missing JUPR API base URL environment variable." };
  const url = `${apiBase.replace(/\/$/, "")}${path}`;
  try {
    const response = await fetch(url, { next: { revalidate: 30 } });
    if (!response.ok) return { data: null, error: await apiErrorMessage(response) };
    return { data: (await response.json()) as T, error: null };
  } catch (error) {
    return { data: null, error: `Unable to reach API: ${error instanceof Error ? error.message : "Unknown error"}` };
  }
}

export async function getAdminLeagueManagerStatus(clubId = "tres_palapas"): Promise<ApiResult<AdminLeagueManagerStatusResponse>> {
  return fetchJson<AdminLeagueManagerStatusResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/status`);
}
