import Link from "next/link";
import MatchCanonicalAuditPanel from "./MatchCanonicalAuditPanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

type StatusResponse = { enabled: boolean; status: string; confirmation_text?: string };

function apiBase(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

async function loadStatus(clubId: string): Promise<{ data: StatusResponse | null; error: string | null }> {
  const base = apiBase();
  if (!base) return { data: null, error: "Missing JUPR API base URL." };
  try {
    const response = await fetch(`${base.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/match-canonical-audit/status`, { cache: "no-store" });
    if (!response.ok) return { data: null, error: `API error (${response.status}).` };
    return { data: (await response.json()) as StatusResponse, error: null };
  } catch (error) {
    return { data: null, error: error instanceof Error ? error.message : "Unable to reach API." };
  }
}

export default async function MatchCanonicalAuditPage() {
  const clubId = "tres_palapas";
  const { data: status, error } = await loadStatus(clubId);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Admin Match Canonical Audit</p>
      <h1 style={{ marginTop: 0 }}>Match Canonical Audit</h1>
      <p style={{ color: "#334155", maxWidth: "880px" }}>Compare profile-visible matches against canonical player-match facts without writes. Applying normalization requires <code>manage_matches</code>, the current dry-run fingerprint, the exact proposed IDs, a durable operation key, and one atomic FastAPI/Postgres write with readback and recovery links.</p>
      {error ? <article style={{ ...cardStyle, background: "#fff7ed", color: "#9a3412" }}>Match Canonical Audit status unavailable. {error}</article> : null}
      <MatchCanonicalAuditPanel apiBase={apiBase()} clubId={clubId} status={status} />
      <p style={{ marginTop: "1rem" }}><Link href="/admin/badges">Badge Diagnostics</Link> · <Link href="/admin/match-log">Match Log</Link> · <Link href="/admin/replay-history">Replay History</Link> · <Link href="/admin">Operations cockpit</Link></p>
    </section>
  );
}
