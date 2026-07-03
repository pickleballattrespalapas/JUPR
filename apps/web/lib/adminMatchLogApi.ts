export type AdminMatchLogPlayer = { id: number | null; name: string };

export type AdminMatchLogMatch = {
  id: number | null;
  date?: string | null;
  league?: string | null;
  week_tag?: string | null;
  match_type?: string | null;
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
  ids: number[];
  league?: string | null;
  week_tag?: string | null;
  match_type?: string | null;
  score: { team1: number; team2: number; display: string };
  team1: AdminMatchLogPlayer[];
  team2: AdminMatchLogPlayer[];
};

export type AdminDuplicateDeletePreview = {
  mode: string;
  keep_ids: number[];
  delete_ids: number[];
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
  future_apply_endpoint?: string | null;
  editable_fields_planned: string[];
  required_confirmation_text: string;
  recompute_scope_for_sample_edit: { standings: boolean; ratings: boolean };
  safety_rules: string[];
};

export type AdminMatchLogResponse = {
  enabled: boolean;
  status: string;
  filters: {
    filter: string;
    match_id?: number | null;
    league?: string | null;
    week_tag?: string | null;
    start_date?: string | null;
    end_date?: string | null;
    limit: number;
  };
  summary: {
    scanned_matches: number;
    filtered_matches?: number | null;
    returned_matches: number;
    duplicate_groups: number;
    duplicate_delete_count: number;
  };
  matches: AdminMatchLogMatch[];
  duplicate_groups: AdminDuplicateGroup[];
  duplicate_rows: AdminMatchLogMatch[];
  duplicate_delete_preview?: AdminDuplicateDeletePreview | null;
  correction_plan: AdminCorrectionPlan;
  warnings: string[];
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

export async function getAdminMatchLog(params?: {
  clubId?: string;
  filter?: string | null;
  matchId?: string | number | null;
  league?: string | null;
  weekTag?: string | null;
  startDate?: string | null;
  endDate?: string | null;
  limit?: string | number | null;
}): Promise<ApiResult<AdminMatchLogResponse>> {
  const clubId = params?.clubId || "tres_palapas";
  const query = new URLSearchParams();
  if (params?.filter) query.set("filter", String(params.filter));
  if (params?.matchId) query.set("match_id", String(params.matchId));
  if (params?.league) query.set("league", String(params.league));
  if (params?.weekTag) query.set("week_tag", String(params.weekTag));
  if (params?.startDate) query.set("start_date", String(params.startDate));
  if (params?.endDate) query.set("end_date", String(params.endDate));
  if (params?.limit) query.set("limit", String(params.limit));
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return fetchJson<AdminMatchLogResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/match-log${suffix}`);
}
