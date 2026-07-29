import Link from "next/link";
import TournamentAdminNav from "@/components/TournamentAdminNav";
import TournamentSetupPanel from "./TournamentSetupPanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

type StatusResponse = { enabled: boolean; status: string; tournament_count?: number | null; warnings?: string[]; confirmation_text?: Record<string, string>; streamlit_fallback_url?: string };

function apiBase(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

async function loadStatus(clubId: string): Promise<{ data: StatusResponse | null; error: string | null }> {
  const base = apiBase();
  if (!base) return { data: null, error: "Missing JUPR API base URL." };
  try {
    const response = await fetch(`${base.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/status`, { cache: "no-store" });
    if (!response.ok) return { data: null, error: `API error (${response.status}).` };
    return { data: (await response.json()) as StatusResponse, error: null };
  } catch (error) {
    return { data: null, error: error instanceof Error ? error.message : "Unable to reach API." };
  }
}

export default async function TournamentSetupPage() {
  const clubId = "tres_palapas";
  const { data: status, error } = await loadStatus(clubId);
  return (
    <>
      <TournamentAdminNav />
      <section>
        <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Tournament Setup</p>
        <h1 style={{ marginTop: 0 }}>Tournament Setup Manager</h1>
        <p style={{ color: "#334155", maxWidth: "860px" }}>
          Dedicated setup workspace for registration settings, registration days, event/division options, builder drafts, publish-impact review, and guarded publishing. Tournament Ops remains the draw/scoring workspace after setup is published.
        </p>
        {error ? <article style={{ ...cardStyle, background: "#fff7ed", color: "#9a3412" }}>Tournament Setup status unavailable. {error}</article> : null}
        <TournamentSetupPanel apiBase={apiBase()} clubId={clubId} status={status} />
        <p style={{ marginTop: "1rem" }}><Link href="/admin">Operations cockpit</Link>{status?.streamlit_fallback_url ? <> · <a href={status.streamlit_fallback_url} target="_blank" rel="noreferrer">Open Streamlit Tournament Setup fallback</a></> : null}</p>
      </section>
    </>
  );
}
