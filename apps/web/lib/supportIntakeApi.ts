export type PublicSupportRequestPayload = {
  request_type: "data_correction" | "profile_privacy" | "general_support";
  requester_name: string;
  requester_email: string;
  player_name?: string | null;
  player_id?: string | number | null;
  match_id?: string | number | null;
  tournament_id?: string | null;
  subject: string;
  description: string;
  requested_action?: string | null;
  evidence_url?: string | null;
  consent_to_contact: boolean;
  website?: string | null;
  source?: string | null;
};

export type PublicSupportRequestResponse = {
  club: { id: string; slug: string; name: string };
  ok: boolean;
  mode?: string | null;
  accepted?: boolean | null;
  deduplicated?: boolean | null;
  request?: {
    id: string;
    request_type: string;
    status: string;
    created_at?: string | null;
  } | null;
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

export async function submitPublicSupportRequest(
  clubSlug: string,
  payload: PublicSupportRequestPayload
): Promise<ApiResult<PublicSupportRequestResponse>> {
  const apiBase = baseUrl();
  if (!apiBase) return { data: null, error: "Missing JUPR API base URL environment variable." };
  try {
    const response = await fetch(`${apiBase.replace(/\/$/, "")}/clubs/${encodeURIComponent(clubSlug)}/support/intake`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!response.ok) return { data: null, error: await apiErrorMessage(response) };
    return { data: (await response.json()) as PublicSupportRequestResponse, error: null };
  } catch (error) {
    return { data: null, error: `Unable to reach API: ${error instanceof Error ? error.message : "Unknown error"}` };
  }
}
