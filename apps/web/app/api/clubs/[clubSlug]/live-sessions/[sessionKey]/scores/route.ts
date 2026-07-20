import { NextResponse } from "next/server";

type ScorePayload = {
  edit_token?: string;
  expected_version?: number;
  idempotency_key?: string;
  scores?: Array<{ match_id?: string; score_a?: number | null; score_b?: number | null }>;
};

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: "FastAPI returned a non-JSON JUPR Live score response." };
  }
}

export async function PATCH(request: Request, { params }: { params: { clubSlug: string; sessionKey: string } }) {
  const base = baseUrl();
  if (!base) {
    return NextResponse.json({ detail: "JUPR API base URL is not configured." }, { status: 500 });
  }

  let payload: ScorePayload;
  try {
    payload = (await request.json()) as ScorePayload;
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body." }, { status: 400 });
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
    const result = await readJson(response);
    return NextResponse.json(result, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Unable to reach JUPR Live score service." },
      { status: 502 }
    );
  }
}
