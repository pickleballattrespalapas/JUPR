import { NextResponse } from "next/server";

type Context = { params: Promise<{ clubSlug: string; weekStart: string }> };
async function forward(request: Request, context: Context) {
  const { clubSlug, weekStart } = await context.params;
  const base = process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL;
  if (!base || !/^\d{4}-\d{2}-\d{2}$/.test(weekStart)) return NextResponse.json({ detail: "Edition unavailable." }, { status: 404 });
  const headers = new Headers({ Accept: "application/json" });
  let body: string | undefined;
  if (request.method === "POST") {
    if (!request.headers.get("content-type")?.startsWith("application/json")) return new Response(null, { status: 415 });
    body = await request.text();
    if (body.length > 4096) return new Response(null, { status: 413 });
    headers.set("content-type", "application/json");
    const address = request.headers.get("x-vercel-forwarded-for") || request.headers.get("x-forwarded-for");
    if (address) headers.set("x-vercel-forwarded-for", address);
  }
  try {
    const url = base.replace(/\/$/, "") + "/clubs/" + encodeURIComponent(clubSlug) + "/weekly-recaps/" + weekStart + "/engagement";
    const upstream = await fetch(url, { method: request.method, cache: "no-store", headers, body, signal: AbortSignal.timeout(12000) });
    if (!upstream.ok) {
      const detail = upstream.status === 409 ? "The activity has changed or closed. Refresh the recap."
        : upstream.status === 429 ? "Please try again later." : "Could not load or save this recap activity.";
      return NextResponse.json({ detail }, { status: upstream.status });
    }
    return NextResponse.json(await upstream.json(), { headers: { "Cache-Control": "no-store" } });
  } catch { return NextResponse.json({ detail: "Recap activity is temporarily unavailable." }, { status: 502 }); }
}
export const GET = forward;
export const POST = forward;
