import { NextResponse } from "next/server";

type CreatePayload = {
  event_name?: string;
  event_type?: string;
  participant_names?: string[];
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
    return { detail: "FastAPI returned a non-JSON JUPR Live response." };
  }
}

export async function POST(request: Request, { params }: { params: { clubSlug: string } }) {
  const base = baseUrl();
  if (!base) {
    return NextResponse.json({ detail: "JUPR API base URL is not configured." }, { status: 500 });
  }

  let payload: CreatePayload;
  try {
    payload = (await request.json()) as CreatePayload;
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body." }, { status: 400 });
  }

  try {
    const response = await fetch(`${base.replace(/\/$/, "")}/clubs/${encodeURIComponent(params.clubSlug)}/live-sessions`, {
      method: "POST",
      cache: "no-store",
      headers: { accept: "application/json", "content-type": "application/json" },
      body: JSON.stringify({
        event_name: payload.event_name || "JUPR Live Round Robin",
        event_type: payload.event_type || "round_robin",
        participant_names: Array.isArray(payload.participant_names) ? payload.participant_names : []
      })
    });
    const result = await readJson(response);
    return NextResponse.json(result, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Unable to reach JUPR Live create service." },
      { status: 502 }
    );
  }
}
