import { forwardPublicLiveJson, publicLiveErrorResponse } from "@/lib/publicLiveProxy";

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export async function GET(_request: Request, { params }: { params: { clubSlug: string; sessionKey: string } }) {
  const base = baseUrl();
  if (!base) return publicLiveErrorResponse(500);
  try {
    const response = await fetch(
      `${base.replace(/\/$/, "")}/clubs/${encodeURIComponent(params.clubSlug)}/live-sessions/${encodeURIComponent(params.sessionKey)}`,
      { method: "GET", cache: "no-store", headers: { accept: "application/json" } }
    );
    return forwardPublicLiveJson(response);
  } catch {
    return publicLiveErrorResponse();
  }
}
