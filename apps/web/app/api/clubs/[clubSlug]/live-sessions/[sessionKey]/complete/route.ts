import { NextResponse } from "next/server";

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export async function POST(request: Request, { params }: { params: { clubSlug: string; sessionKey: string } }) {
  const base = baseUrl();
  if (!base) return NextResponse.json({ detail: "JUPR API base URL is not configured." }, { status: 500 });
  const payload = await request.json().catch(() => null);
  if (!payload) return NextResponse.json({ detail: "Invalid JSON body." }, { status: 400 });
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
    const result = await response.json().catch(() => ({ detail: "FastAPI returned an invalid JUPR Live response." }));
    return NextResponse.json(result, { status: response.status });
  } catch (error) {
    return NextResponse.json({ detail: error instanceof Error ? error.message : "Unable to complete JUPR Live." }, { status: 502 });
  }
}
