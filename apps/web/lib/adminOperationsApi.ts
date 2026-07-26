import { getAdminApiBaseUrl } from "@/lib/adminAuthClient";

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

export type AdminOperationsApiResult<T> = {
  data: T | null;
  error: string | null;
  status: number | null;
};

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

async function fetchJson<T>(
  path: string,
  accessToken: string
): Promise<AdminOperationsApiResult<T>> {
  const apiBase = getAdminApiBaseUrl();
  if (!apiBase) {
    return {
      data: null,
      error: "JUPR admin API configuration is missing.",
      status: null
    };
  }
  if (!accessToken) {
    return {
      data: null,
      error: "Admin sign-in is required.",
      status: 401
    };
  }
  const url = `${apiBase.replace(/\/$/, "")}${path}`;
  try {
    const response = await fetch(url, {
      cache: "no-store",
      headers: {
        accept: "application/json",
        Authorization: `Bearer ${accessToken}`
      }
    });
    if (!response.ok) {
      return {
        data: null,
        error: await apiErrorMessage(response),
        status: response.status
      };
    }
    return {
      data: (await response.json()) as T,
      error: null,
      status: response.status
    };
  } catch (error) {
    return {
      data: null,
      error: `Unable to reach API: ${error instanceof Error ? error.message : "Unknown error"}`,
      status: null
    };
  }
}

export async function getAdminOperationsStatus(
  accessToken: string,
  clubId: string
): Promise<AdminOperationsApiResult<AdminOperationsStatusResponse>> {
  const params = new URLSearchParams({ club_id: clubId });
  return fetchJson<AdminOperationsStatusResponse>(
    `/admin/operations/status?${params.toString()}`,
    accessToken
  );
}
