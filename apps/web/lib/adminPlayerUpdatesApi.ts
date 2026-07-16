export type AdminPlayerUpdatesStatusResponse = {
  enabled: boolean;
  status: string;
  send_range_endpoint?: string | null;
  auto_send_enabled?: boolean | null;
  email_mode?: string | null;
  smtp_status?: Record<string, unknown> | null;
  active_subscription_count?: number | null;
  warnings?: string[];
};

export type AdminPlayerUpdatesRangeResponse = {
  ok: boolean;
  mode?: string;
  start_date?: string;
  end_date?: string;
  generation_result?: Record<string, unknown>;
  send_result?: Record<string, unknown>;
  warnings?: string[];
};

type ApiResult<T> = { data: T | null; error: string | null };

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export function getAdminPlayerUpdatesApiBaseUrl(): string | null {
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

export async function getAdminPlayerUpdatesStatus(clubId = "tres_palapas"): Promise<ApiResult<AdminPlayerUpdatesStatusResponse>> {
  return fetchJson<AdminPlayerUpdatesStatusResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/player-updates/status`);
}
