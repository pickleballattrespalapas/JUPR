export type AdminPlayerEditorStatusResponse = {
  enabled: boolean;
  status: string;
  players_endpoint?: string | null;
  player_detail_endpoint?: string | null;
  player_count?: number | null;
  warnings: string[];
};

export type AdminPlayerEditorPlayer = {
  id: number;
  club_id?: string;
  name: string;
  rating?: number | null;
  rating_jupr?: number | null;
  starting_rating?: number | null;
  starting_jupr?: number | null;
  wins?: number | null;
  losses?: number | null;
  matches_played?: number | null;
  active?: boolean | null;
  inactive_at?: string | null;
  last_game_at?: string | null;
};

export type AdminPlayerEditorLeagueRating = {
  id: number;
  league_name: string;
  rating?: number | null;
  rating_jupr?: number | null;
  starting_rating?: number | null;
  starting_jupr?: number | null;
  wins?: number | null;
  losses?: number | null;
  matches_played?: number | null;
  is_active?: boolean | null;
  inactive_at?: string | null;
};

export type AdminPlayerEditorListResponse = {
  ok: boolean;
  mode?: string;
  players: AdminPlayerEditorPlayer[];
  count: number;
};

export type AdminPlayerEditorDetailResponse = {
  ok: boolean;
  mode?: string;
  player: AdminPlayerEditorPlayer;
  league_ratings: AdminPlayerEditorLeagueRating[];
  match_reference_counts?: Record<string, number>;
};

export type AdminPlayerEditorWriteResponse = {
  ok: boolean;
  mode?: string;
  player?: AdminPlayerEditorPlayer | null;
  warnings?: string[];
};

type ApiResult<T> = { data: T | null; error: string | null };

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export function getAdminPlayerEditorApiBaseUrl(): string | null {
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

export async function getAdminPlayerEditorStatus(clubId = "tres_palapas"): Promise<ApiResult<AdminPlayerEditorStatusResponse>> {
  return fetchJson<AdminPlayerEditorStatusResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/status`);
}
