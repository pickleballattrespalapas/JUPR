export type AdminMatchUploaderStatusResponse = {
  enabled: boolean;
  singles_write_enabled?: boolean;
  status: string;
  submit_endpoint?: string | null;
  singles_submit_endpoint?: string | null;
  round_robin_preview_endpoint?: string | null;
  player_create_endpoint?: string | null;
  player_operation_endpoint?: string | null;
  max_batch_rows: number;
  league_options: string[];
  doubles_league_options?: string[];
  singles_league_options?: string[];
  week_tag_options: string[];
  round_robin_format_options?: string[];
  round_robin_expected_games?: Record<string, number>;
  warnings: string[];
};

export type AdminMatchUploaderWriteResult = {
  ok: boolean;
  mode?: string;
  submitted_count?: number;
  match_write_committed?: boolean;
  operation?: {
    operation_id?: string;
    idempotency_key?: string;
    request_fingerprint?: string;
    match_format?: "doubles" | "singles" | string;
    committed?: boolean;
    idempotent?: boolean;
    duplicate_request?: boolean;
    match_ids?: Array<string | number>;
  };
  result?: {
    inserted?: number;
    match_format?: string;
    skipped_incomplete?: number;
    skipped_empty?: number;
    skipped_unrated?: number;
    winner_bonus_summary?: Record<string, unknown>;
    badge_summary?: Record<string, unknown>;
    player_update_queue?: Record<string, unknown>;
  };
  feedback?: {
    ratings_updated?: boolean;
    rating_type?: string;
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
  auto_player_updates?: {
    mode?: "auto_sent" | "disabled" | "skipped" | "error" | string;
    reason?: string;
    error_code?: string;
    attempted?: number;
    sent?: number;
    skipped?: number;
    errors?: number;
    email_mode?: string;
    windows?: Array<Record<string, unknown>>;
  };
  recovery?: {
    match_log_route?: string;
    player_updates_route?: string;
    replay_history_route?: string;
    operator_rule?: string;
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
  operation_key?: string;
  status?: string;
  recovery_required?: boolean;
  reconciled?: boolean;
  created_count?: number;
  unchanged_count?: number;
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

export type AdminMatchUploaderPlayerBatchOperation = {
  ok?: boolean;
  operation_key?: string;
  idempotency_key?: string;
  status: string;
  result?: AdminMatchUploaderCreatePlayersResult | null;
  result_json?: AdminMatchUploaderCreatePlayersResult | null;
  error?: string | null;
  error_text?: string | null;
  recovery_required?: boolean;
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
    if (detail && typeof detail === "object") {
      const detailMessage = (detail as { message?: unknown }).message;
      if (typeof detailMessage === "string") return `${fallback} ${detailMessage}`;
      return `${fallback} ${JSON.stringify(detail)}`;
    }
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

export async function getAdminMatchUploaderStatus(clubId = "tres_palapas"): Promise<ApiResult<AdminMatchUploaderStatusResponse>> {
  return fetchJson<AdminMatchUploaderStatusResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/status`);
}
