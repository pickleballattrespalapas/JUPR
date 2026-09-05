import { NextResponse } from "next/server";
import { publicLiveErrorText } from "@/lib/publicLiveErrorText";

function upstreamDetail(payload: unknown): unknown {
  return payload && typeof payload === "object" && "detail" in payload
    ? (payload as { detail?: unknown }).detail
    : null;
}

export function publicLiveErrorDetail(status: number, detail?: unknown): string {
  return publicLiveErrorText(status, detail);
}

export function publicLiveErrorResponse(status = 502, detail?: unknown): NextResponse {
  return NextResponse.json({ detail: publicLiveErrorDetail(status, detail) }, { status });
}

export async function forwardPublicLiveJson(response: Response): Promise<NextResponse> {
  const body = await response.text();
  if (!body) return new NextResponse(null, { status: response.status });
  let payload: unknown;
  try {
    payload = JSON.parse(body);
  } catch {
    return publicLiveErrorResponse(response.ok ? 502 : response.status);
  }
  if (!response.ok) return publicLiveErrorResponse(response.status, upstreamDetail(payload));
  return NextResponse.json(payload, { status: response.status });
}
