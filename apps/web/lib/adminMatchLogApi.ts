export type AdminMatchLogPlayer = { id: number | null; name: string };

export type AdminDuplicateResolution = {
  resolution?: string | null;
  reason?: string | null;
  actor_email?: string | null;
  actor_role?: string | null;
  source_page?: string | null;
  resolved_at?: string | null;
};

export type AdminMatchLogMatch = {
  id: number | null;
  row_version?: number | null;
  date?: string | null;
  league?: string | null;
  week_tag?: string | null;
  match_type?: string | null;
  notes?: string | null;
  score: { team1: number; team2: number; display: string };
  team1: AdminMatchLogPlayer[];
  team2: AdminMatchLogPlayer[];
  is_active?: boolean | null;
  context_type?: string | null;
  context_id?: string | number | null;
  created_at?: string | null;
  updated_at?: string | null;
  dup_key?: string | null;
  dup_rank?: number | null;
  dup_count?: number | null;
  would_keep?: boolean | null;
};

export type AdminDuplicateGroup = {
  dup_key: string;
  dup_count: number;
  keep_id: number;
  delete_ids: number[];
  delete_targets?: AdminMatchExclusionTarget[];
  ids: number[];
  league?: string | null;
  week_tag?: string | null;
  match_type?: string | null;
  score: { team1: number; team2: number; display: string };
  team1: AdminMatchLogPlayer[];
  team2: AdminMatchLogPlayer[];
  resolution?: AdminDuplicateResolution | null;
};

export type AdminDuplicateDeletePreview = {
  mode: string;
  keep_ids: number[];
  delete_ids: number[];
  targets?: AdminMatchExclusionTarget[];
  delete_targets?: AdminMatchExclusionTarget[];
  delete_count: number;
  affected_leagues: string[];
  affected_player_ids: number[];
  recompute_scope: { standings: boolean; ratings: boolean };
  recommended_replay_scope: string;
  confirmation_text: string;
};

export type AdminCorrectionPlan = {
  mode: string;
  apply_endpoint?: string | null;
  duplicate_cleanup_endpoint?: string | null;
  exclude_endpoint?: string | null;
  duplicate_no_issue_endpoint?: string | null;
  future_apply_endpoint?: string | null;
  editable_fields_planned: string[];
  required_confirmation_text: string;
  duplicate_cleanup_confirmation_text?: string | null;
  duplicate_no_issue_confirmation_text?: string | null;
  recompute_scope_for_sample_edit: { standings: boolean; ratings: boolean };
  safety_rules: string[];
};

export type AdminMatchEditOperation = {
  id: string;
  status: string;
  recompute_scope?: { standings?: boolean; ratings?: boolean } | null;
  replay_target?: string | null;
  replay_job_id?: string | null;
  error_text?: string | null;
  actor_email?: string | null;
  source?: string | null;
  created_at?: string | null;
  finished_at?: string | null;
};

export type AdminMatchExclusionTarget = {
  match_id: number;
  expected_row_version: number;
};

export type AdminMatchExclusionOperation = {
  id: string;
  status: string;
  recovery_stage?: "replay" | "badge_reconcile" | "finalize" | string | null;
  replay_job_id?: string | null;
  badge_eval_run_id?: string | null;
  affected_player_ids?: number[];
  excluded_ids?: number[];
  targets?: AdminMatchExclusionTarget[];
  result_json?: Record<string, unknown> | null;
  error_text?: string | null;
  actor_email?: string | null;
  source?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  finished_at?: string | null;
};

export type AdminMatchLogResponse = {
  enabled: boolean;
  apply_enabled?: boolean | null;
  status: string;
  filters: {
    filter: string;
    match_id?: number | null;
    match_ids?: number[];
    league?: string | null;
    week_tag?: string | null;
    context_type?: string | null;
    context_id?: string | null;
    context_ids?: string[];
    start_date?: string | null;
    end_date?: string | null;
    limit: number;
  };
  filter_options?: {
    leagues: string[];
    week_tags: string[];
  };
  summary: {
    scanned_matches: number;
    filtered_matches?: number | null;
    returned_matches: number;
    duplicate_groups: number;
    duplicate_delete_count: number;
    resolved_duplicate_groups?: number | null;
  };
  matches: AdminMatchLogMatch[];
  duplicate_groups: AdminDuplicateGroup[];
  duplicate_rows: AdminMatchLogMatch[];
  duplicate_delete_preview?: AdminDuplicateDeletePreview | null;
  resolved_duplicate_groups?: AdminDuplicateGroup[];
  recent_edit_operations?: AdminMatchEditOperation[];
  recent_exclusion_operations?: AdminMatchExclusionOperation[];
  correction_plan: AdminCorrectionPlan;
  warnings: string[];
};

export type AdminSocialMatchLogRow = {
  source_type?: string | null;
  source_label?: string | null;
  id?: string | number | null;
  social_match_id?: string | number | null;
  event_id?: string | number | null;
  event_name?: string | null;
  date?: string | null;
  played_on?: string | null;
  round_number?: number | null;
  court_number?: number | null;
  mini_round_number?: number | null;
  status?: string | null;
  submission_mode?: string | null;
  match_key?: string | null;
  t1_p1?: string | null;
  t1_p2?: string | null;
  t2_p1?: string | null;
  t2_p2?: string | null;
  score_t1?: number | null;
  score_t2?: number | null;
};

export type AdminSocialMatchLogResponse = {
  ok: boolean;
  mode: string;
  rows: AdminSocialMatchLogRow[];
  count: number;
  warnings?: string[];
};

export type AdminMatchLogWriteResult = {
  ok: boolean;
  mode?: string;
  updated_count?: number;
  updated_ids?: number[];
  deleted_count?: number;
  deleted_ids?: number[];
  excluded_count?: number;
  excluded_ids?: number[];
  affected_leagues?: string[];
  affected_player_ids?: number[];
  recompute_scope?: { standings: boolean; ratings: boolean };
  recommended_replay_scope?: string;
  resolution?: string;
  dup_key?: string;
  match_ids?: number[];
  reason?: string;
  social_match_id?: string;
  requested_ids?: string[];
  warnings?: string[];
  badge_summary?: Record<string, unknown>;
  replay_error?: string | null;
  atomic?: boolean | null;
  operation_id?: string | null;
  operation_status?: string | null;
  status?: string | null;
  recovery_stage?: string | null;
  idempotent?: boolean | null;
  replay_job_id?: string | null;
  replay_status?: string | null;
  replay_result?: Record<string, unknown> | null;
  operation?: AdminMatchExclusionOperation | null;
};

type ApiResult<T> = { data: T | null; error: string | null };

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export function getAdminApiBaseUrl(): string | null {
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

async function fetchJson<T>(path: string, accessToken?: string): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) return { data: null, error: "Missing JUPR API base URL environment variable." };
  const url = `${apiBase.replace(/\/$/, "")}${path}`;
  try {
    const response = await fetch(url, {
      cache: "no-store",
      headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : undefined
    });
    if (!response.ok) return { data: null, error: await apiErrorMessage(response) };
    return { data: (await response.json()) as T, error: null };
  } catch (error) {
    return { data: null, error: `Unable to reach API: ${error instanceof Error ? error.message : "Unknown error"}` };
  }
}

export async function getAdminMatchLog(params?: {
  clubId?: string;
  filter?: string | null;
  matchId?: string | number | null;
  matchIds?: string | Array<string | number> | null;
  league?: string | null;
  weekTag?: string | null;
  contextType?: string | null;
  contextId?: string | number | null;
  contextIds?: string | Array<string | number> | null;
  startDate?: string | null;
  endDate?: string | null;
  limit?: string | number | null;
}, accessToken?: string): Promise<ApiResult<AdminMatchLogResponse>> {
  const clubId = params?.clubId || "tres_palapas";
  const query = new URLSearchParams();
  if (params?.filter) query.set("filter", String(params.filter));
  if (params?.matchIds) {
    const matchIds = Array.isArray(params.matchIds)
      ? params.matchIds.map((value) => String(value).trim()).filter(Boolean).join(",")
      : String(params.matchIds).trim();
    if (matchIds) query.set("match_ids", matchIds);
  } else if (params?.matchId) {
    query.set("match_id", String(params.matchId));
  }
  if (params?.league) query.set("league", String(params.league));
  if (params?.weekTag) query.set("week_tag", String(params.weekTag));
  if (params?.contextType) query.set("context_type", String(params.contextType));
  if (params?.contextId) query.set("context_id", String(params.contextId));
  if (params?.contextIds) {
    const contextIds = Array.isArray(params.contextIds)
      ? params.contextIds.map((value) => String(value).trim()).filter(Boolean).join(",")
      : String(params.contextIds).trim();
    if (contextIds) query.set("context_ids", contextIds);
  }
  if (params?.startDate) query.set("start_date", String(params.startDate));
  if (params?.endDate) query.set("end_date", String(params.endDate));
  if (params?.limit) query.set("limit", String(params.limit));
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return fetchJson<AdminMatchLogResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/match-log${suffix}`, accessToken);
}
