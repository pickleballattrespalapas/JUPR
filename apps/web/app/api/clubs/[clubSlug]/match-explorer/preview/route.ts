import { NextResponse } from "next/server";

const QUERY_KEYS = new Set(["me", "partner", "opp1", "opp2", "context", "score_you", "score_opp"]);

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

function upstreamUrl(base: string, clubSlug: string, searchParams: URLSearchParams): string {
  const params = new URLSearchParams();
  for (const [key, value] of searchParams.entries()) {
    if (QUERY_KEYS.has(key)) params.append(key, value);
  }
  return `${base.replace(/\/$/, "")}/clubs/${encodeURIComponent(clubSlug)}/match-explorer/preview?${params.toString()}`;
}

export async function GET(request: Request, { params }: { params: { clubSlug: string } }) {
  const base = baseUrl();
  if (!base) {
    return NextResponse.json({ detail: "JUPR API base URL is not configured." }, { status: 500 });
  }

  try {
    const url = new URL(request.url);
    const response = await fetch(upstreamUrl(base, params.clubSlug, url.searchParams), {
      method: "GET",
      cache: "no-store",
      headers: { accept: "application/json" }
    });
    const text = await response.text();
    let payload: unknown;
    try {
      payload = text ? JSON.parse(text) : {};
    } catch {
      payload = { detail: text.slice(0, 500) || "FastAPI returned a non-JSON Match Explorer response." };
    }
    return NextResponse.json(payload, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Unable to reach Match Explorer preview service." },
      { status: 502 }
    );
  }
}
