import { NextResponse } from "next/server";

const QUERY_KEYS = new Set(["me", "partner", "opp1", "opp2", "context", "score_you", "score_opp"]);
const PUBLIC_ERROR = "We couldn’t build this preview. Please try again.";

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
    return NextResponse.json({ detail: PUBLIC_ERROR }, { status: 500 });
  }

  try {
    const url = new URL(request.url);
    const response = await fetch(upstreamUrl(base, params.clubSlug, url.searchParams), {
      method: "GET",
      cache: "no-store",
      headers: { accept: "application/json" }
    });
    if (!response.ok) {
      console.error("Match Explorer preview request failed", { status: response.status });
      return NextResponse.json({ detail: PUBLIC_ERROR }, { status: response.status });
    }
    const payload = await response.json().catch(() => null);
    if (!payload) {
      console.error("Match Explorer preview returned an invalid success response");
      return NextResponse.json({ detail: PUBLIC_ERROR }, { status: 502 });
    }
    return NextResponse.json(payload, { status: response.status });
  } catch (error) {
    console.error("Match Explorer preview request failed", error);
    return NextResponse.json(
      { detail: PUBLIC_ERROR },
      { status: 502 }
    );
  }
}
