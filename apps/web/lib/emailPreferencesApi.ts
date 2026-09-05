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
  effective_scope?: string | null;
  changed?: boolean | null;
  already_unsubscribed?: boolean | null;
  subscription?: PublicEmailPreferenceSubscription | null;
  message?: string | null;
};

type ApiResult<T> = {
  data: T | null;
  error: string | null;
  status?: number | null;
};

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

async function responseDetail(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as {
      detail?: unknown;
      message?: unknown;
      error?: unknown;
    };
    const detail = payload.detail ?? payload.message ?? payload.error;
    if (typeof detail === "string") return detail.trim();
    if (
      detail &&
      typeof detail === "object" &&
      typeof (detail as Record<string, unknown>).message === "string"
    ) {
      return String((detail as Record<string, unknown>).message).trim();
    }
  } catch {
    // Public callers never receive an unparsed response body.
  }
  return "";
}

function emailPreferenceError(
  status: number,
  detail: string,
  fallback: string
): string {
  const normalized = detail.toLowerCase();
  if (status === 400 || status === 404) {
    if (normalized.includes("legacy") || normalized.includes("subscription-id")) {
      return "This link is out of date. Open the preferences link in a recent player update email.";
    }
    if (normalized.includes("token") && normalized.includes("required")) {
      return "This link is incomplete. Open the preferences link in your player update email.";
    }
    if (normalized.includes("unsupported") && normalized.includes("scope")) {
      return "Choose which emails you want to stop, then try again.";
    }
    if (normalized.includes("not found") || normalized.includes("unsubscribe link")) {
      return "This email preferences link is no longer valid. Open a newer player update email or contact support.";
    }
    return "This email preferences link isn’t valid. Open the link from your player update email.";
  }
  if (status === 429) return "Too many attempts were made. Wait a moment and try again.";
  if (status === 401) {
    return "This email preferences link can’t be used. Open the link from your latest player update email.";
  }
  if (status === 403) {
    return "Email preference changes aren’t available right now. Please try again later.";
  }
  return fallback;
}

async function fetchJson<T>(path: string, publicError: string, init?: RequestInit): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) return { data: null, error: publicError, status: null };
  const url = `${apiBase.replace(/\/$/, "")}${path}`;
  try {
    const response = await fetch(url, init ?? { next: { revalidate: 0 } });
    if (!response.ok) {
      return {
        data: null,
        error: emailPreferenceError(
          response.status,
          await responseDetail(response),
          publicError
        ),
        status: response.status
      };
    }
    return {
      data: (await response.json()) as T,
      error: null,
      status: response.status
    };
  } catch {
    return { data: null, error: publicError, status: null };
  }
}

export async function getEmailPreferences(params: { token?: string | null; ut?: string | null; sid?: string | null; subscriptionId?: string | null }): Promise<ApiResult<PublicEmailPreferencesResponse>> {
  const query = new URLSearchParams();
  if (params.token) query.set("token", params.token);
  if (params.ut) query.set("ut", params.ut);
  if (params.sid) query.set("sid", params.sid);
  if (params.subscriptionId) query.set("subscription_id", params.subscriptionId);
  return fetchJson<PublicEmailPreferencesResponse>(
    `/email-preferences${query.toString() ? `?${query.toString()}` : ""}`,
    "We couldn’t load your email preferences. Please try again or contact support."
  );
}

export async function unsubscribeEmailPreferences(payload: { token?: string | null; ut?: string | null; scope?: string | null }): Promise<ApiResult<PublicEmailUnsubscribeResponse>> {
  return fetchJson<PublicEmailUnsubscribeResponse>(
    "/email-preferences/unsubscribe",
    "We couldn’t update your email preferences. Please try again.",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  );
}
