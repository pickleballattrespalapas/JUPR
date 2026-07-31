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
    const response = await fetch(`${base.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/tools/status`, { cache: "no-store" });
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
      <p style={{ color: "#334155", maxWidth: "880px" }}>Review admin activity and Club Social submissions, inspect system health, download server-generated safe rating CSVs, manage staff roles, run badge workers/recompute, and preview/apply selected tournament-match backfills through guarded FastAPI/Python services. Every applying action is staging-only, operation-keyed, strictly audited, and paired with a visible stop/recovery path.</p>
      {error ? <article style={{ ...cardStyle, background: "#fff7ed", color: "#9a3412" }}>Admin Tools status unavailable. {error}</article> : null}
      <article style={{ ...cardStyle, marginBottom: "1rem", background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Reports</h2>
        <p><Link href="/admin/top-players-printable">Previous-month Top 50</Link></p>
      </article>
      <AdminToolsPanel apiBase={apiBase()} clubId={clubId} status={status} />
      <p style={{ marginTop: "1rem" }}><Link href="/admin/top-players-printable">Previous-month Top 50</Link> · <Link href="/admin/replay-history">Replay History</Link> · <Link href="/admin/badges">Badge Diagnostics</Link> · <Link href="/admin/match-canonical-audit">Match Canonical Audit</Link> · <Link href="/admin">Operations cockpit</Link></p>
    </section>
  );
}
