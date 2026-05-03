import { NextRequest, NextResponse } from "next/server";

function getApiBaseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export async function POST(request: NextRequest, { params }: { params: { clubId: string } }) {
  const apiBase = getApiBaseUrl();
  const adminToken = process.env.JUPR_ADMIN_API_TOKEN;

  if (!apiBase) {
    return NextResponse.json({ error: "Missing JUPR_API_BASE_URL." }, { status: 500 });
  }
  if (!adminToken) {
    return NextResponse.json(
      { error: "Missing JUPR_ADMIN_API_TOKEN for score-entry proxy." },
      { status: 500 }
    );
  }

  const payload = await request.json();
  const response = await fetch(`${apiBase.replace(/\/$/, "")}/admin/clubs/${params.clubId}/matches/batch`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-admin-token": adminToken,
      "x-admin-permission": "enter_scores",
    },
    body: JSON.stringify(payload),
    cache: "no-store",
  });

  const body = await response.json();
  return NextResponse.json(body, { status: response.status });
}
