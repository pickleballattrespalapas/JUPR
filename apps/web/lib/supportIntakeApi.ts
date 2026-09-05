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

type ApiResult<T> = {
  data: T | null;
  error: string | null;
  status?: number | null;
};

const PUBLIC_SUPPORT_ERROR = "We couldn't submit your request. Please try again. If the problem continues, contact support.";

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

function publicSupportError(status: number, detail: string): string {
  const normalized = detail.toLowerCase();
  if (status === 429) {
    return "You’ve sent several requests recently. Please wait an hour before trying again.";
  }
  if (status === 400) {
    if (normalized.includes("valid email")) return "Enter a valid email address.";
    if (normalized.includes("your name") && normalized.includes("required")) return "Enter your name.";
    if (normalized.includes("consent to contact")) return "Please confirm that club staff may contact you.";
    if (normalized.includes("short subject")) return "Enter a short subject.";
    if (normalized.includes("request details")) return "Describe how we can help.";
    if (normalized.includes("player id") && normalized.includes("number")) return "Player ID must be a number.";
    if (normalized.includes("selected player") && normalized.includes("club")) return "That player isn’t part of this club.";
    if (normalized.includes("unsupported request type")) return "Choose a valid request type.";
    if (normalized.includes("control characters")) return "Remove unusual characters from the evidence link.";
    if (normalized.includes("complete http") || normalized.includes("complete https")) {
      return "Enter a complete link, such as https://example.com.";
    }
    if (normalized.includes("embedded credentials")) {
      return "Use a link that doesn’t include a username or password.";
    }
    return "Check the information you entered and try again.";
  }
  if (status === 403) return "Support requests aren’t available right now. Please try again later.";
  return PUBLIC_SUPPORT_ERROR;
}

function safeServerSuccessMessage(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const message = value.trim();
  if (!message || message.length > 240 || /[<>\u0000-\u001f]/.test(message)) return null;
  if (/\b(?:database|supabase|credential|runtime|stack|traceback|exception|review queue)\b|JUPR_/i.test(message)) {
    return null;
  }
  return message;
}

function publicSuccessMessage(
  requestType: PublicSupportRequestPayload["request_type"],
  deduplicated?: boolean | null
): string {
  if (deduplicated) {
    return "We already received this request. Club staff will review it soon.";
  }
  if (requestType === "data_correction") {
    return "Correction request received. Club staff will review it before making changes.";
  }
  if (requestType === "profile_privacy") {
    return "Privacy request received. Club staff will review it before changing public profile information.";
  }
  return "Request received. Club staff will follow up by email.";
}

export async function submitPublicSupportRequest(
  clubSlug: string,
  payload: PublicSupportRequestPayload
): Promise<ApiResult<PublicSupportRequestResponse>> {
  const apiBase = baseUrl();
  if (!apiBase) return { data: null, error: PUBLIC_SUPPORT_ERROR, status: null };
  try {
    const response = await fetch(`${apiBase.replace(/\/$/, "")}/clubs/${encodeURIComponent(clubSlug)}/support/intake`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!response.ok) {
      return {
        data: null,
        error: publicSupportError(response.status, await responseDetail(response)),
        status: response.status
      };
    }
    const data = (await response.json()) as PublicSupportRequestResponse;
    return {
      data: {
        ...data,
        message:
          safeServerSuccessMessage(data.message) ||
          publicSuccessMessage(payload.request_type, data.deduplicated)
      },
      error: null,
      status: response.status
    };
  } catch {
    return { data: null, error: PUBLIC_SUPPORT_ERROR, status: null };
  }
}
