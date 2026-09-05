import { forwardPublicLiveJson, publicLiveErrorResponse } from "@/lib/publicLiveProxy";

type CreatePayload = {
  event_name?: string;
  event_type?: string;
  participant_names?: string[];
  live_mode?: string;
  total_rounds?: number;
  court_sizes?: number[];
  host_name?: string;
  skill_levels?: string[];
  participant_player_ids?: Record<string, number>;
  idempotency_key?: string;
};

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export async function GET(_request: Request, { params }: { params: { clubSlug: string } }) {
  const base = baseUrl();
  if (!base) {
    return publicLiveErrorResponse(500);
  }

  try {
    const response = await fetch(`${base.replace(/\/$/, "")}/clubs/${encodeURIComponent(params.clubSlug)}/live-sessions`, {
      method: "GET",
      cache: "no-store",
      headers: { accept: "application/json" }
    });
    return forwardPublicLiveJson(response);
  } catch {
    return publicLiveErrorResponse();
  }
}

export async function POST(request: Request, { params }: { params: { clubSlug: string } }) {
  const base = baseUrl();
  if (!base) {
    return publicLiveErrorResponse(500);
  }

  let payload: CreatePayload;
  try {
    payload = (await request.json()) as CreatePayload;
  } catch {
    return publicLiveErrorResponse(400);
  }

  try {
    const forwardedFor = request.headers.get("x-vercel-forwarded-for") || request.headers.get("x-forwarded-for") || "";
    const response = await fetch(`${base.replace(/\/$/, "")}/clubs/${encodeURIComponent(params.clubSlug)}/live-sessions`, {
      method: "POST",
      cache: "no-store",
      headers: {
        accept: "application/json",
        "content-type": "application/json",
        ...(forwardedFor ? { "x-vercel-forwarded-for": forwardedFor } : {})
      },
      body: JSON.stringify({
        event_name: payload.event_name || "JUPR Live Round Robin",
        event_type: payload.event_type || "round_robin",
        participant_names: Array.isArray(payload.participant_names) ? payload.participant_names : [],
        live_mode: payload.live_mode || "quick",
        total_rounds: Number(payload.total_rounds || 3),
        court_sizes: Array.isArray(payload.court_sizes) ? payload.court_sizes : [],
        host_name: payload.host_name || null,
        skill_levels: Array.isArray(payload.skill_levels) ? payload.skill_levels : [],
        participant_player_ids: payload.participant_player_ids && typeof payload.participant_player_ids === "object" ? payload.participant_player_ids : {},
        idempotency_key: payload.idempotency_key || ""
      })
    });
    return forwardPublicLiveJson(response);
  } catch {
    return publicLiveErrorResponse();
  }
}
