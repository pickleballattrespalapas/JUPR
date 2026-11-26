const privateHeaders = { "Cache-Control": "private, no-store", "Referrer-Policy": "no-referrer" };

export async function POST(request: Request, props: { params: Promise<{ action: string }> }) {
  const { action } = await props.params;
  if (action !== "read" && action !== "save") return new Response(null, { status: 404, headers: privateHeaders });
  const base = process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL;
  if (!base) return new Response(null, { status: 503, headers: privateHeaders });
  if (!request.headers.get("content-type")?.startsWith("application/json")) {
    return new Response(null, { status: 415, headers: privateHeaders });
  }

  try {
    const upstream = await fetch(base.replace(/\/$/, "") + "/public/interclub-club-entry/" + action, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: await request.text(),
      cache: "no-store",
      redirect: "error",
      signal: AbortSignal.any([request.signal, AbortSignal.timeout(15_000)]),
    });
    if (!upstream.ok) return new Response(null, { status: upstream.status, headers: privateHeaders });
    return Response.json(await upstream.json(), { headers: privateHeaders });
  } catch {
    return new Response(null, { status: 502, headers: privateHeaders });
  }
}
