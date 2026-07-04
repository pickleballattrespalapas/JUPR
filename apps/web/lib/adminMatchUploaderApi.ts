export type AdminMatchUploaderStatusResponse = {
  enabled: boolean;
  status: string;
  submit_endpoint?: string | null;
  round_robin_preview_endpoint?: string | null;
  player_create_endpoint?: string | null;
  max_batch_rows: number;
  league_options: string[];
  week_tag_options: string[];
  round_robin_format_options?: string[];
  round_robin_expected_games?: Record<string, number>;
  warnings: string[];
};

export type AdminMatchUploaderWriteResult = {
  ok: boolean;
  mode?: string;
  submitted_count?: number;
  result?: {
    inserted?: number;
    skipped_incomplete?: number;
    skipped_empty?: number;
    skipped_unrated?: number;
    badge_summary?: Record<string, unknown>;
    player_update_queue?: Record<string, unknown>;
  };
  feedback?: {
    ratings_updated?: boolean;
    latest_match_id?: string | number | null;
    affected_players?: Array<{
      id: number;
      name: string;
      rating_before?: number | null;
      rating_after?: number | null;
      rating_delta?: number | null;
      matches_played_before?: number | null;
      matches_played_after?: number | null;
    }>;
  };
  warnings?: string[];
};

export type AdminMatchUploaderGeneratedPlayer = {
  id: number;
  name: string;
  rating?: number | null;
};

export type AdminMatchUploaderRoundRobinMatch = {
  row_id: string;
  court: number;
  match_index: number;
  label: string;
  t1: AdminMatchUploaderGeneratedPlayer[];
  t2: AdminMatchUploaderGeneratedPlayer[];
  t1_p1: number;
  t1_p2: number;
  t2_p1: number;
  t2_p2: number;
};

export type AdminMatchUploaderRoundRobinCourt = {
  court: number;
  format_type: string;
  expected_games?: number | null;
  player_names: string[];
  matches: AdminMatchUploaderRoundRobinMatch[];
};

export type AdminMatchUploaderRoundRobinPreview = {
  ok: boolean;
  mode?: string;
  source?: string;
  missing_players?: string[];
  courts?: AdminMatchUploaderRoundRobinCourt[];
  match_count?: number;
};

export type AdminMatchUploaderCreatePlayersResult = {
  ok: boolean;
  mode?: string;
  requested_count?: number;
  accepted_count?: number;
  players?: Array<{
    id: number;
    club_id?: string;
    name: string;
    rating?: number | null;
    wins?: number | null;
    losses?: number | null;
    matches_played?: number | null;
    is_active?: boolean | null;
  }>;
  warnings?: string[];
};

type ApiResult<T> = { data: T | null; error: string | null };

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export function getAdminMatchUploaderApiBaseUrl(): string | null {
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

export async function getAdminMatchUploaderStatus(clubId = "tres_palapas"): Promise<ApiResult<AdminMatchUploaderStatusResponse>> {
  return fetchJson<AdminMatchUploaderStatusResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/status`);
}
