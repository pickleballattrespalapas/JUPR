import { NextResponse } from "next/server";

type CheckResult = {
  name: string;
  ok: boolean;
  status?: number;
  detail: string;
};

type PilotRequest = {
  club_id?: string;
  access_token?: string;
};

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

function apiUrl(base: string, path: string): string {
  return `${base.replace(/\/$/, "")}${path}`;
}

async function readText(response: Response): Promise<string> {
  const text = await response.text().catch(() => "");
  if (!text) return "";
  try {
    const payload = JSON.parse(text) as { detail?: unknown; error?: unknown; message?: unknown };
    return String(payload.detail || payload.error || payload.message || text).slice(0, 240);
  } catch {
    return text.slice(0, 240);
  }
}

async function readJson(response: Response): Promise<Record<string, unknown> | null> {
  const text = await response.text().catch(() => "");
  if (!text) return null;
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return null;
  }
}

async function getJsonCheck(base: string, name: string, path: string, predicate: (payload: Record<string, unknown>) => string | null): Promise<CheckResult> {
  const response = await fetch(apiUrl(base, path), { cache: "no-store", headers: { accept: "application/json" } });
  const payload = await readJson(response);
  if (!response.ok || !payload) {
    return { name, ok: false, status: response.status, detail: (payload ? String(payload.detail || payload.error || payload.message || "") : "") || `HTTP ${response.status}` };
  }
  const problem = predicate(payload);
  return { name, ok: !problem, status: response.status, detail: problem || "Ready." };
}

async function authValidationCheck(base: string, token: string, name: string, method: "PATCH" | "POST", path: string, body: unknown, expectedText: string): Promise<CheckResult> {
  const response = await fetch(apiUrl(base, path), {
    method,
    cache: "no-store",
    headers: {
      accept: "application/json",
      "content-type": "application/json",
      Authorization: `Bearer ${token}`
    },
    body: JSON.stringify(body)
  });
  const detail = await readText(response);
  const ok = response.status === 400 && detail.includes(expectedText);
  return { name, ok, status: response.status, detail: ok ? "Authorized validation reached the expected API guard." : detail || `HTTP ${response.status}` };
}

async function safeCheck(name: string, fn: () => Promise<CheckResult>): Promise<CheckResult> {
  try {
    return await fn();
  } catch (error) {
    return { name, ok: false, detail: error instanceof Error ? error.message : "Check failed." };
  }
}

export async function POST(request: Request) {
  const base = baseUrl();
  if (!base) {
    return NextResponse.json({ detail: "JUPR API base URL is not configured." }, { status: 500 });
  }

  let payload: PilotRequest;
  try {
    payload = (await request.json()) as PilotRequest;
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body." }, { status: 400 });
  }

  const clubId = String(payload.club_id || "tres_palapas");
  const token = String(payload.access_token || "").trim();
  if (!token) {
    return NextResponse.json({ detail: "Missing admin access token." }, { status: 401 });
  }

  const results: CheckResult[] = [];
  results.push(await safeCheck("Operations pilot mode", () => getJsonCheck(base, "Operations pilot mode", "/admin/operations/status", (apiPayload) => apiPayload.write_pilot_enabled === true ? null : "Write pilot flag is not enabled.")));
  results.push(await safeCheck("Match Log flags", () => getJsonCheck(base, "Match Log flags", `/admin/clubs/${encodeURIComponent(clubId)}/match-log?limit=25`, (apiPayload) => apiPayload.enabled === true && apiPayload.apply_enabled === true ? null : "Match Log read/apply flags are not both enabled.")));
  results.push(await safeCheck("Replay flag", () => getJsonCheck(base, "Replay flag", `/admin/clubs/${encodeURIComponent(clubId)}/replay-history`, (apiPayload) => apiPayload.enabled === true ? null : "Replay flag is not enabled.")));
  results.push(await safeCheck("Match Log auth", () => authValidationCheck(base, token, "Match Log auth", "PATCH", `/admin/clubs/${encodeURIComponent(clubId)}/match-log/edits`, { patches: [], confirmation_text: "APPLY", correction_note: "pilot browser validation", source: "next_admin_pilot_browser_validation" }, "No patches provided")));
  results.push(await safeCheck("Replay auth", () => authValidationCheck(base, token, "Replay auth", "POST", `/admin/clubs/${encodeURIComponent(clubId)}/replay-history`, { target_reset: "ALL (Full System Reset)", confirmation_text: "NOT_REPLAY", source: "next_admin_pilot_browser_validation" }, "Type REPLAY")));

  return NextResponse.json({ ok: results.every((result) => result.ok), results });
}
