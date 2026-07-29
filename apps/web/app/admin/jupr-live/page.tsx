import Link from "next/link";
import JuprLiveAdminPanel from "./JuprLiveAdminPanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

type StatusResponse = { enabled: boolean; status: string; counts?: Record<string, number>; warnings?: string[] };

function apiBase(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

async function loadStatus(clubId: string): Promise<{ data: StatusResponse | null; error: string | null }> {
  const base = apiBase();
  if (!base) return { data: null, error: "Missing JUPR API base URL." };
  try {
    const response = await fetch(`${base.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/jupr-live/status`, { cache: "no-store" });
    if (!response.ok) return { data: null, error: `API error (${response.status}).` };
    return { data: (await response.json()) as StatusResponse, error: null };
  } catch (error) {
    return { data: null, error: error instanceof Error ? error.message : "Unable to reach API." };
  }
}

export default async function JuprLiveAdminPage() {
  const clubId = "tres_palapas";
  const clubSlug = "tres-palapas";
  const { data: status, error } = await loadStatus(clubId);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Admin JUPR Live</p>
      <h1 style={{ marginTop: 0 }}>JUPR Live Admin</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>Create and manage durable JUPR Live sessions. This is the one-off event surface; Tournament Live remains the tournament-specific draw runner.</p>
      {error ? <article style={{ ...cardStyle, background: "#fff7ed", color: "#9a3412" }}>JUPR Live Admin status unavailable. {error}</article> : null}
      <JuprLiveAdminPanel apiBase={apiBase()} clubId={clubId} clubSlug={clubSlug} status={status} />
      <p style={{ marginTop: "1rem" }}><Link href="/clubs/tres-palapas/live">Public JUPR Live</Link> · <Link href="/admin/match-uploader">Match Uploader</Link> · <Link href="/admin">Operations cockpit</Link></p>
    </section>
  );
}
