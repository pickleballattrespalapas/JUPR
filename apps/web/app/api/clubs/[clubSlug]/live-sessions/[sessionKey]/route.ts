import { NextResponse } from "next/server";

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export async function GET(_request: Request, { params }: { params: { clubSlug: string; sessionKey: string } }) {
  const base = baseUrl();
  if (!base) return NextResponse.json({ detail: "JUPR API base URL is not configured." }, { status: 500 });
  try {
    const response = await fetch(
      `${base.replace(/\/$/, "")}/clubs/${encodeURIComponent(params.clubSlug)}/live-sessions/${encodeURIComponent(params.sessionKey)}`,
      { method: "GET", cache: "no-store", headers: { accept: "application/json" } }
    );
    const text = await response.text();
    const payload = text ? JSON.parse(text) : {};
    return NextResponse.json(payload, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Unable to refresh JUPR Live session." },
      { status: 502 }
    );
  }
}
