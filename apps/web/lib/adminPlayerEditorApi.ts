export type AdminPlayerEditorStatusResponse = {
  enabled: boolean;
  status: string;
  players_endpoint?: string | null;
  player_detail_endpoint?: string | null;
  social_identities_endpoint?: string | null;
  player_merge_endpoint?: string | null;
  merge_operation_endpoint?: string | null;
  operation_endpoint?: string | null;
  transactional_merge_ready?: boolean;
  player_count?: number | null;
  warnings: string[];
};

export type AdminPlayerEditorOperationResponse = {
  ok: boolean;
  operation_key: string;
  workflow?: "player_editor_create" | "player_editor_update" | "player_editor_league_rating_update" | string;
  status: string;
  result?: AdminPlayerEditorWriteResponse | null;
  error?: string | null;
  recovery_required?: boolean;
  updated_at?: string | null;
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
  state_fingerprint?: string;
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
  state_fingerprint?: string;
};

export type AdminPlayerSocialIdentity = {
  id: string;
  club_id?: string;
  display_name: string;
  normalized_name?: string | null;
  linked_player_id?: number | null;
  linked_player_name?: string | null;
  first_seen_on?: string | null;
  last_seen_on?: string | null;
};

export type AdminPlayerSocialIdentityListResponse = {
  ok: boolean;
  mode?: string;
  people: AdminPlayerSocialIdentity[];
  players: Array<{ id: number; name: string; active?: boolean | null }>;
  summary?: { people?: number; linked?: number; unlinked?: number };
};

export type AdminPlayerMergePreview = {
  ok: boolean;
  mode?: string;
  can_merge?: boolean;
  preview_fingerprint?: string;
  source_player?: { id: number; name: string };
  target_player?: { id: number; name: string };
  match_reference_counts?: Record<string, number>;
  collision_match_ids?: number[];
  official_tournament_match_ids?: number[];
  league_rating_plan?: {
    source_rows?: Array<Record<string, unknown>>;
    target_rows?: Array<Record<string, unknown>>;
    move_ids?: number[];
    delete_ids?: number[];
    conflicts?: string[];
  };
  social_identity_counts?: Record<string, number>;
  warnings?: string[];
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
  league_rating?: AdminPlayerEditorLeagueRating | null;
  league_ratings?: AdminPlayerEditorLeagueRating[];
  club_person?: AdminPlayerSocialIdentity | null;
  linked_count?: number;
  skipped_count?: number;
  source_player_id?: number;
  target_player_id?: number;
  match_updates?: Record<string, number>;
  league_rating_plan?: AdminPlayerMergePreview["league_rating_plan"];
  moved_league_rating_count?: number;
  deleted_conflicting_league_rating_count?: number;
  social_identity_rows_updated?: number;
  transaction_mode?: string;
  operation_id?: string;
  operation_key?: string;
  operation_status?: "merged_pending_replay" | "replay_verified" | "compensated" | string;
  status?: string;
  recovery_required?: boolean;
  reconciled?: boolean;
  preview_fingerprint?: string;
  requires_replay?: boolean;
  recovery?: {
    operation_id: string;
    status: string;
    replay_required: boolean;
    required_replay_scope?: string;
    replay_route?: string;
    tracked_replay_fallback_url?: string;
    compensation_endpoint?: string;
    replay_evidence_endpoint?: string;
    operator_rule?: string;
  };
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
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) return { data: null, error: await apiErrorMessage(response) };
    return { data: (await response.json()) as T, error: null };
  } catch (error) {
    return { data: null, error: `Unable to reach API: ${error instanceof Error ? error.message : "Unknown error"}` };
  }
}

export async function getAdminPlayerEditorStatus(clubId = "tres_palapas"): Promise<ApiResult<AdminPlayerEditorStatusResponse>> {
  return fetchJson<AdminPlayerEditorStatusResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/status`);
}
