export type AdminBadgeDiagnosticsStatusResponse = {
  enabled: boolean;
  status: string;
  options_endpoint?: string | null;
  debug_endpoint?: string | null;
  audit_endpoint?: string | null;
  state_endpoint?: string | null;
  recompute_endpoint?: string | null;
  revoke_endpoint?: string | null;
  operation_status_endpoint?: string | null;
  confirmation_text?: Record<string, string>;
  required_permissions?: Record<string, string>;
  write_environment?: string;
  service_role_required?: boolean;
  streamlit_fallback?: string;
  badge_count?: number | null;
  player_badge_count?: number | null;
  warnings: string[];
};

export type AdminBadgeOption = {
  badge_id: string;
  name: string;
  status?: string | null;
  state?: "live" | "frozen" | "deprecated";
  state_changed_at?: string | null;
  state_change_reason?: string | null;
  definition_found?: boolean;
  scope?: string | null;
  award_timing?: string | null;
};

export type AdminBadgePlayerOption = {
  id: number;
  name: string;
  rating?: number | null;
  wins?: number | null;
  losses?: number | null;
  matches_played?: number | null;
  active?: boolean | null;
};

export type AdminBadgeOptionsResponse = {
  ok: boolean;
  mode?: string;
  players: AdminBadgePlayerOption[];
  badges: AdminBadgeOption[];
  player_count: number;
  badge_count: number;
};

export type AdminBadgeDebugResponse = {
  ok: boolean;
  mode?: string;
  report: Record<string, unknown>;
};

export type AdminBadgeAuditResponse = {
  ok: boolean;
  mode?: string;
  report: Record<string, unknown>;
};

type ApiResult<T> = { data: T | null; error: string | null };

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export function getAdminBadgeDiagnosticsApiBaseUrl(): string | null {
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
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) return { data: null, error: await apiErrorMessage(response) };
    return { data: (await response.json()) as T, error: null };
  } catch (error) {
    return { data: null, error: `Unable to reach API: ${error instanceof Error ? error.message : "Unknown error"}` };
  }
}

export async function getAdminBadgeDiagnosticsStatus(clubId = "tres_palapas"): Promise<ApiResult<AdminBadgeDiagnosticsStatusResponse>> {
  return fetchJson<AdminBadgeDiagnosticsStatusResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/badges/status`);
}
