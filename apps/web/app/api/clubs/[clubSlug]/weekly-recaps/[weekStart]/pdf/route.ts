import { NextResponse } from "next/server";

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

function publicPdfError(status: number): string {
  if (status === 404) return "We couldn’t find that weekly recap.";
  if (status === 429) return "Please wait a moment and try again.";
  if (status === 400 || status === 422) return "We couldn’t create that PDF. Check the week and try again.";
  return "Weekly recap PDFs are temporarily unavailable. Please try again later.";
}

export async function GET(
  _request: Request,
  { params }: { params: { clubSlug: string; weekStart: string } }
) {
  const base = baseUrl();
  if (!base) {
    return NextResponse.json({ detail: publicPdfError(500) }, { status: 500 });
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
      return NextResponse.json(
        { detail: publicPdfError(response.status) },
        { status: response.status }
      );
    }
    const headers = new Headers();
    headers.set("Content-Type", response.headers.get("content-type") || "application/pdf");
    headers.set("Cache-Control", "public, max-age=60, s-maxage=300");
    const disposition = response.headers.get("content-disposition");
    if (disposition) headers.set("Content-Disposition", disposition);
    return new NextResponse(bytes, { status: response.status, headers });
  } catch {
    return NextResponse.json(
      { detail: publicPdfError(502) },
      { status: 502 }
    );
  }
}
