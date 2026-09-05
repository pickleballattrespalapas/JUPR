import { NextResponse } from "next/server";
import { publicLiveErrorResponse } from "@/lib/publicLiveProxy";

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export async function GET(request: Request, { params }: { params: { clubSlug: string; sessionKey: string } }) {
  const base = baseUrl();
  if (!base) return publicLiveErrorResponse(500);
  const format = new URL(request.url).searchParams.get("format") === "json" ? "json" : "csv";
  try {
    const response = await fetch(
      `${base.replace(/\/$/, "")}/clubs/${encodeURIComponent(params.clubSlug)}/live-sessions/${encodeURIComponent(params.sessionKey)}/export?format=${format}`,
      { method: "GET", cache: "no-store" }
    );
    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as { detail?: unknown } | null;
      return publicLiveErrorResponse(response.status, payload?.detail);
    }
    const body = await response.arrayBuffer();
    return new NextResponse(body, {
      status: response.status,
      headers: {
        "content-type": response.headers.get("content-type") || "application/octet-stream",
        "content-disposition": response.headers.get("content-disposition") || `attachment; filename="jupr-live.${format}"`,
        "cache-control": "no-store"
      }
    });
  } catch {
    return publicLiveErrorResponse();
  }
}
