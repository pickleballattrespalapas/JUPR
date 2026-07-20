import { NextResponse } from "next/server";

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export async function GET(
  _request: Request,
  { params }: { params: { clubSlug: string; weekStart: string } }
) {
  const base = baseUrl();
  if (!base) {
    return NextResponse.json({ detail: "JUPR API base URL is not configured." }, { status: 500 });
  }

  const upstream = `${base.replace(/\/$/, "")}/clubs/${encodeURIComponent(params.clubSlug)}/weekly-recaps/${encodeURIComponent(params.weekStart)}/pdf`;
  try {
    const response = await fetch(upstream, {
      method: "GET",
      cache: "no-store",
      headers: { accept: "application/pdf" }
    });
    const bytes = await response.arrayBuffer();
    if (!response.ok) {
      let detail = `Weekly Recap PDF service returned HTTP ${response.status}.`;
      try {
        const payload = JSON.parse(new TextDecoder().decode(bytes)) as { detail?: unknown };
        if (payload.detail) detail = String(payload.detail);
      } catch {
        // Keep the bounded status message; never reflect an arbitrary upstream body.
      }
      return NextResponse.json({ detail }, { status: response.status });
    }
    const headers = new Headers();
    headers.set("Content-Type", response.headers.get("content-type") || "application/pdf");
    headers.set("Cache-Control", "public, max-age=60, s-maxage=300");
    const disposition = response.headers.get("content-disposition");
    if (disposition) headers.set("Content-Disposition", disposition);
    return new NextResponse(bytes, { status: response.status, headers });
  } catch (error) {
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "Unable to reach Weekly Recap PDF service." },
      { status: 502 }
    );
  }
}
