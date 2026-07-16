import Link from "next/link";
import AdminToolsPanel from "./AdminToolsPanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

type StatusResponse = { enabled: boolean; status: string; roles?: string[]; retention_days?: number; retention_cutoff?: string };

function apiBase(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

async function loadStatus(clubId: string): Promise<{ data: StatusResponse | null; error: string | null }> {
  const base = apiBase();
  if (!base) return { data: null, error: "Missing JUPR API base URL." };
  try {
    const response = await fetch(`${base.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/tools/status`, { next: { revalidate: 30 } });
    if (!response.ok) return { data: null, error: `API error (${response.status}).` };
    return { data: (await response.json()) as StatusResponse, error: null };
  } catch (error) {
    return { data: null, error: error instanceof Error ? error.message : "Unable to reach API." };
  }
}

export default async function AdminToolsPage() {
  const clubId = "tres_palapas";
  const { data: status, error } = await loadStatus(clubId);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Admin Tools</p>
      <h1 style={{ marginTop: 0 }}>Admin Tools</h1>
      <p style={{ color: "#334155", maxWidth: "880px" }}>Review admin activity, inspect system health, and manage staff role assignments through guarded FastAPI routes. Worker/backfill jobs remain Streamlit-only until background job contracts are hardened.</p>
      {error ? <article style={{ ...cardStyle, background: "#fff7ed", color: "#9a3412" }}>Admin Tools status unavailable. {error}</article> : null}
      <AdminToolsPanel apiBase={apiBase()} clubId={clubId} status={status} />
      <p style={{ marginTop: "1rem" }}><Link href="/admin/replay-history">Replay History</Link> · <Link href="/admin/badges">Badge Diagnostics</Link> · <Link href="/admin/match-canonical-audit">Match Canonical Audit</Link> · <Link href="/admin">Operations cockpit</Link></p>
    </section>
  );
}
