import { NextResponse } from "next/server";

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export async function GET(request: Request, { params }: { params: { clubSlug: string; sessionKey: string } }) {
  const base = baseUrl();
  if (!base) return NextResponse.json({ detail: "JUPR API base URL is not configured." }, { status: 500 });
  const format = new URL(request.url).searchParams.get("format") === "json" ? "json" : "csv";
  try {
    const response = await fetch(
      `${base.replace(/\/$/, "")}/clubs/${encodeURIComponent(params.clubSlug)}/live-sessions/${encodeURIComponent(params.sessionKey)}/export?format=${format}`,
      { method: "GET", cache: "no-store" }
    );
    const body = await response.arrayBuffer();
    return new NextResponse(body, {
      status: response.status,
      headers: {
        "content-type": response.headers.get("content-type") || "application/octet-stream",
        "content-disposition": response.headers.get("content-disposition") || `attachment; filename="jupr-live.${format}"`,
        "cache-control": "no-store"
      }
    });
  } catch (error) {
    return NextResponse.json({ detail: error instanceof Error ? error.message : "Unable to export JUPR Live." }, { status: 502 });
  }
}
