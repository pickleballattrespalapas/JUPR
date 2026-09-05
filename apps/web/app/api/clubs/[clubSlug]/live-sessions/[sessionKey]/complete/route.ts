import { forwardPublicLiveJson, publicLiveErrorResponse } from "@/lib/publicLiveProxy";

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export async function POST(request: Request, { params }: { params: { clubSlug: string; sessionKey: string } }) {
  const base = baseUrl();
  if (!base) return publicLiveErrorResponse(500);
  const payload = await request.json().catch(() => null);
  if (!payload) return publicLiveErrorResponse(400);
  const forwardedFor = request.headers.get("x-vercel-forwarded-for") || request.headers.get("x-forwarded-for") || "";
  try {
    const response = await fetch(
      `${base.replace(/\/$/, "")}/clubs/${encodeURIComponent(params.clubSlug)}/live-sessions/${encodeURIComponent(params.sessionKey)}/complete`,
      {
        method: "POST",
        cache: "no-store",
        headers: {
          accept: "application/json",
          "content-type": "application/json",
          ...(forwardedFor ? { "x-vercel-forwarded-for": forwardedFor } : {})
        },
        body: JSON.stringify(payload)
      }
    );
    return forwardPublicLiveJson(response);
  } catch {
    return publicLiveErrorResponse();
  }
}
