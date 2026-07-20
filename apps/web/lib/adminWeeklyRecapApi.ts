export type AdminWeeklyRecapStatusResponse = {
  enabled: boolean;
  status: string;
  list_endpoint?: string | null;
  generate_endpoint?: string | null;
  warnings: string[];
};

export type AdminWeeklyRecapRow = {
  id?: string | null;
  club_id?: string | null;
  week_start: string;
  week_end: string;
  status: string;
  generated_json?: Record<string, unknown>;
  edits_json?: Record<string, unknown>;
  final_json?: Record<string, unknown>;
  published_at?: string | null;
  published_by?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  row_version: number;
};

export type AdminWeeklyRecapCandidate = {
  candidate_id: string;
  key: string;
  label: string;
  display: string;
  player_ids?: number[];
  value_json?: Record<string, unknown>;
  band?: string | null;
};

export type AdminWeeklyRecapListResponse = {
  ok: boolean;
  mode?: string;
  recaps: AdminWeeklyRecapRow[];
  count: number;
};

export type AdminWeeklyRecapDetailResponse = {
  ok: boolean;
  mode?: string;
  recap: AdminWeeklyRecapRow;
  candidates?: Record<string, AdminWeeklyRecapCandidate[]>;
  warnings?: string[];
};

export type AdminWeeklyRecapWriteResponse = AdminWeeklyRecapDetailResponse;

type ApiResult<T> = { data: T | null; error: string | null };

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export function getAdminWeeklyRecapApiBaseUrl(): string | null {
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
    // Fall through to text excerpt.
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

export async function getAdminWeeklyRecapStatus(clubId = "tres_palapas"): Promise<ApiResult<AdminWeeklyRecapStatusResponse>> {
  return fetchJson<AdminWeeklyRecapStatusResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/weekly-recap/status`);
}
