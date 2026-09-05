import { forwardPublicLiveJson, publicLiveErrorResponse } from "@/lib/publicLiveProxy";

type ScorePayload = {
  edit_token?: string;
  expected_version?: number;
  idempotency_key?: string;
  scores?: Array<{ match_id?: string; score_a?: number | null; score_b?: number | null }>;
};

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export async function PATCH(request: Request, { params }: { params: { clubSlug: string; sessionKey: string } }) {
  const base = baseUrl();
  if (!base) {
    return publicLiveErrorResponse(500);
  }

  let payload: ScorePayload;
  try {
    payload = (await request.json()) as ScorePayload;
  } catch {
    return publicLiveErrorResponse(400);
  }

  try {
    const forwardedFor = request.headers.get("x-vercel-forwarded-for") || request.headers.get("x-forwarded-for") || "";
    const response = await fetch(`${base.replace(/\/$/, "")}/clubs/${encodeURIComponent(params.clubSlug)}/live-sessions/${encodeURIComponent(params.sessionKey)}/scores`, {
      method: "PATCH",
      cache: "no-store",
      headers: {
        accept: "application/json",
        "content-type": "application/json",
        ...(forwardedFor ? { "x-vercel-forwarded-for": forwardedFor } : {})
      },
      body: JSON.stringify({
        edit_token: payload.edit_token || "",
        expected_version: Number(payload.expected_version || 0),
        idempotency_key: payload.idempotency_key || "",
        scores: Array.isArray(payload.scores) ? payload.scores : []
      })
    });
    return forwardPublicLiveJson(response);
  } catch {
    return publicLiveErrorResponse();
  }
}
