import Link from "next/link";
import MoneyballPanel from "./MoneyballPanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

type StatusResponse = { enabled: boolean; status: string; players?: Array<{ id: number; name: string; rating?: number | null; is_active?: boolean | null }>; league_options?: string[]; warnings?: string[] };

function apiBase(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

async function loadStatus(clubId: string): Promise<{ data: StatusResponse | null; error: string | null }> {
  const base = apiBase();
  if (!base) return { data: null, error: "Missing JUPR API base URL." };
  try {
    const response = await fetch(`${base.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/moneyball/status`, { next: { revalidate: 30 } });
    if (!response.ok) return { data: null, error: `API error (${response.status}).` };
    return { data: (await response.json()) as StatusResponse, error: null };
  } catch (error) {
    return { data: null, error: error instanceof Error ? error.message : "Unable to reach API." };
  }
}

export default async function MoneyballPage() {
  const clubId = "tres_palapas";
  const { data: status, error } = await loadStatus(clubId);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Admin Moneyball</p>
      <h1 style={{ marginTop: 0 }}>Moneyball</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>Run an 8-player Moneyball round robin, score results, compute settlement, and save official rated matches through the Python Match Uploader pipeline.</p>
      {error ? <article style={{ ...cardStyle, background: "#fff7ed", color: "#9a3412" }}>Moneyball status unavailable. {error}</article> : null}
      <MoneyballPanel apiBase={apiBase()} clubId={clubId} status={status} />
      <p style={{ marginTop: "1rem" }}><Link href="/admin/match-uploader">Match Uploader</Link> · <Link href="/admin/match-log">Match Log</Link> · <Link href="/admin">Operations cockpit</Link></p>
    </section>
  );
}
