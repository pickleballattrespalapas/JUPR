export type AdminWorkflowStatus = {
  key: string;
  label: string;
  streamlit_page_key: string;
  next_route?: string | null;
  api_scope: string;
  access: string;
  risk: "low" | "medium" | "high" | "critical" | string;
  env_flag: string;
  status_when_disabled: string;
  next_action: string;
  enabled: boolean;
  pilot_enabled: boolean;
  effective_status: string;
  can_enable_for_pilot: boolean;
  requires_review_before_enablement: boolean;
  safety_notes: string[];
};

export type AdminOperationsStatusResponse = {
  service: string;
  environment: string;
  mode: string;
  write_pilot_enabled: boolean;
  streamlit_fallback_url: string;
  strict_audit_required: boolean;
  service_role_configured: boolean;
  jwt_verification_configured?: boolean;
  jwt_verification_mode?: string;
  enabled_workflows: string[];
  recommended_sequence: string[];
  pilot_gates: string[];
  permanent_guardrails: string[];
  workflows: AdminWorkflowStatus[];
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

export async function getAdminOperationsStatus(): Promise<ApiResult<AdminOperationsStatusResponse>> {
  return fetchJson<AdminOperationsStatusResponse>("/admin/operations/status");
}
