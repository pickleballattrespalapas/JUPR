export type PublicEmailPreferenceSubscription = {
  id?: string | null;
  club_id?: string | null;
  player_id?: string | number | null;
  email_masked?: string | null;
  request_status?: string | null;
  preferences_json?: Record<string, unknown> | null;
  verified_at?: string | null;
  unsubscribed_at?: string | null;
};

export type PublicEmailPreferencesResponse = {
  ok: boolean;
  mode?: string | null;
  found: boolean;
  subscription?: PublicEmailPreferenceSubscription | null;
  status_options?: string[];
  scope_options?: string[];
  message?: string | null;
};

export type PublicEmailUnsubscribeResponse = {
  ok: boolean;
  mode?: string | null;
  scope?: string | null;
  subscription?: PublicEmailPreferenceSubscription | null;
  message?: string | null;
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

async function fetchJson<T>(path: string, init?: RequestInit): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) return { data: null, error: "Missing JUPR API base URL environment variable." };
  const url = `${apiBase.replace(/\/$/, "")}${path}`;
  try {
    const response = await fetch(url, init ?? { next: { revalidate: 0 } });
    if (!response.ok) return { data: null, error: await apiErrorMessage(response) };
    return { data: (await response.json()) as T, error: null };
  } catch (error) {
    return { data: null, error: `Unable to reach API: ${error instanceof Error ? error.message : "Unknown error"}` };
  }
}

export async function getEmailPreferences(params: { token?: string | null; ut?: string | null; sid?: string | null; subscriptionId?: string | null }): Promise<ApiResult<PublicEmailPreferencesResponse>> {
  const query = new URLSearchParams();
  if (params.token) query.set("token", params.token);
  if (params.ut) query.set("ut", params.ut);
  if (params.sid) query.set("sid", params.sid);
  if (params.subscriptionId) query.set("subscription_id", params.subscriptionId);
  return fetchJson<PublicEmailPreferencesResponse>(`/email-preferences${query.toString() ? `?${query.toString()}` : ""}`);
}

export async function unsubscribeEmailPreferences(payload: { token?: string | null; ut?: string | null; sid?: string | null; subscription_id?: string | null; scope?: string | null }): Promise<ApiResult<PublicEmailUnsubscribeResponse>> {
  return fetchJson<PublicEmailUnsubscribeResponse>("/email-preferences/unsubscribe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}
