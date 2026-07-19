export type AdminLeagueManagerStatusResponse = {
  enabled: boolean;
  status: string;
  leagues_endpoint?: string | null;
  league_create_endpoint?: string | null;
  league_duplicate_endpoint?: string | null;
  league_lifecycle_endpoint?: string | null;
  league_detail_endpoint?: string | null;
  league_settings_update_endpoint?: string | null;
  league_schedule_preview_endpoint?: string | null;
  league_roster_update_endpoint?: string | null;
  league_printout_endpoint?: string | null;
  top_players_printable_endpoint?: string | null;
  league_live_sessions_endpoint?: string | null;
  league_awards_endpoint?: string | null;
  awards_write_enabled?: boolean;
  league_count?: number | null;
  active_count?: number | null;
  warnings: string[];
};

export type AdminLeagueLiveStatusResponse = {
  enabled: boolean;
  status: string;
  sessions_endpoint?: string | null;
  roster_suggestion_endpoint?: string | null;
  round_plan_endpoint?: string | null;
  movement_authority?: string | null;
  service_role_configured?: boolean;
  streamlit_fallback?: string | null;
  session_count?: number | null;
  warnings: string[];
};

export type AdminLeagueManagerLeague = {
  league_name: string;
  league_type?: string | null;
  description?: string | null;
  min_weeks?: number | null;
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

export type AdminLeagueManagerRosterRow = {
  player_id: number;
  player_name: string;
  in_league: boolean;
  league_name: string;
  rating?: number | null;
  rating_jupr?: number | null;
  wins?: number | null;
  losses?: number | null;
  matches_played?: number | null;
  player_active?: boolean | null;
  league_active?: boolean | null;
  last_game_at?: string | null;
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
  schedule_ics?: string;
  schedule_ics_filename?: string;
  standings: AdminLeagueManagerStanding[];
  standings_count: number;
  roster?: AdminLeagueManagerRosterRow[];
  roster_count?: number | null;
  league_roster_count?: number | null;
  validation?: {
    valid: boolean;
    errors: string[];
    warnings: string[];
    capabilities: AdminLeagueManagerCapabilities;
  };
  capabilities?: AdminLeagueManagerCapabilities;
};

export type AdminLeagueManagerCapabilities = {
  settings_mode: "full" | "description_only" | "read_only";
  roster_mutable: boolean;
  lifecycle_actions: Array<"start" | "pause" | "resume" | "end" | "archive">;
  printable: boolean;
};

export type AdminLeaguePrintLeader = {
  player_id: number;
  player_name: string;
  games: number;
  wins: number;
  losses: number;
  rating_delta_elo: number;
  rating_delta_jupr: number;
  win_pct?: number | null;
};

export type AdminLeagueTopPerformer = {
  category_key: string;
  category_label: string;
  player_id: number;
  player_name: string;
  metric_value?: number | null;
  metric_display: string;
  rank: number;
  min_games: number;
};

export type AdminLeaguePrintoutResponse = {
  ok: boolean;
  mode: string;
  league_name: string;
  available_weeks: number[];
  selected_week?: number | null;
  detail: AdminLeagueManagerDetailResponse;
  weekly_rating_leaders: AdminLeaguePrintLeader[];
  weekly_win_leaders: AdminLeaguePrintLeader[];
  season_top_performers: AdminLeagueTopPerformer[];
  season_top_performer_count: number;
  rating_source: "stored_snapshots" | "stored_snapshots_with_python_replay";
  warnings: string[];
};

export type AdminTopPlayersPrintableRow = {
  rank: number;
  player_id: number;
  player_name: string;
  rating: number;
  rating_jupr: number;
  wins: number;
  losses: number;
  games: number;
  record: string;
};

export type AdminTopPlayersPrintableResponse = {
  ok: boolean;
  mode: string;
  period: { label: string; start: string; end_exclusive: string; timezone: "UTC" };
  minimum_games: number;
  limit: number;
  rankings: AdminTopPlayersPrintableRow[];
  ranking_count: number;
  empty_message?: string | null;
};

export type AdminLeagueManagerSchedulePreviewResponse = {
  ok: boolean;
  mode?: string;
  league_name: string;
  schedule_config: Record<string, unknown>;
  schedule_preview: AdminLeagueManagerSchedulePreviewRow[];
  schedule_ics?: string;
  schedule_ics_filename?: string;
};

export type AdminLeagueManagerWriteResponse = {
  ok: boolean;
  mode?: string;
  league?: AdminLeagueManagerLeague | null;
  detail?: AdminLeagueManagerDetailResponse | null;
  created?: boolean | null;
  league_name?: string | null;
  source_league_name?: string | null;
  roster_copied?: boolean | null;
  previous_status?: string | null;
  new_status?: string | null;
  player_id?: number | null;
  action?: string | null;
  league_rating?: Record<string, unknown> | null;
  warnings?: string[];
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

export async function getAdminLeagueLiveStatus(clubId = "tres_palapas"): Promise<ApiResult<AdminLeagueLiveStatusResponse>> {
  return fetchJson<AdminLeagueLiveStatusResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live/status`);
}
