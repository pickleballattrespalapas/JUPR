export type AdminSupportRequestsStatus = {
  enabled: boolean;
  status: string;
  requests_endpoint?: string | null;
  update_endpoint?: string | null;
  request_count?: number | null;
  warnings: string[];
};

export type AdminSupportRequest = {
  id: string;
  club_id: string;
  club_slug?: string | null;
  request_type: string;
  status: string;
  requester_name: string;
  requester_email: string;
  player_name?: string | null;
  player_id?: string | number | null;
  match_id?: string | null;
  tournament_id?: string | null;
  subject: string;
  description: string;
  requested_action?: string | null;
  evidence_url?: string | null;
  identity_status: string;
  fulfillment_status: string;
  resolution_action: string;
  resolution_evidence?: string | null;
  source?: string | null;
  admin_note?: string | null;
  reviewed_by?: string | null;
  reviewed_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type AdminSupportRequestsListResponse = {
  ok: boolean;
  mode?: string | null;
  requests: AdminSupportRequest[];
  summary: { total: number; by_status: Record<string, number>; by_type: Record<string, number> };
  warnings?: string[];
};

export type AdminSupportRequestUpdateResponse = {
  ok: boolean;
  mode?: string | null;
  request: AdminSupportRequest;
  warnings?: string[];
};

type ApiResult<T> = { data: T | null; error: string | null };

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export function getAdminSupportRequestsApiBaseUrl(): string | null {
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

export async function getAdminSupportRequestsStatus(clubId = "tres_palapas"): Promise<ApiResult<AdminSupportRequestsStatus>> {
  return fetchJson<AdminSupportRequestsStatus>(`/admin/clubs/${encodeURIComponent(clubId)}/support-requests/status`);
}
