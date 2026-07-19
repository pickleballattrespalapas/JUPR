export type AdminReplayStatusResponse = {
  enabled: boolean;
  status: string;
  apply_endpoint?: string | null;
  options: string[];
  default_target_reset: string;
  confirmation_text: string;
  warnings: string[];
  safety_rules: string[];
  recent_jobs: AdminReplayJob[];
};

export type AdminReplayJob = {
  id: string;
  target_reset?: string | null;
  status: string;
  actor_email?: string | null;
  source?: string | null;
  created_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  error_text?: string | null;
};

export type AdminReplayResultResponse = {
  ok: boolean;
  mode: string;
  target_reset: string;
  job_id?: string | null;
  job_status?: string | null;
  idempotent_replay?: boolean | null;
  result: {
    target_reset: string;
    players_updated: boolean;
    skipped_incomplete: number;
    matches_rewritten: number;
    matches_snapshots_updated_rows: number;
    league_ratings_rows: number;
    matches_scanned_total: number;
  } | Record<string, never>;
  warnings: string[];
};

type ApiResult<T> = { data: T | null; error: string | null };

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export function getAdminReplayApiBaseUrl(): string | null {
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

export async function getAdminReplayStatus(clubId = "tres_palapas"): Promise<ApiResult<AdminReplayStatusResponse>> {
  return fetchJson<AdminReplayStatusResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/replay-history`);
}
