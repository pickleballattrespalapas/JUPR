import { Suspense } from "react";
import { redirect } from "next/navigation";
import TournamentAdminNav from "@/components/TournamentAdminNav";
import SelectedTournamentPanelScope from "../tournaments/SelectedTournamentPanelScope";
import TournamentSetupPanel from "./TournamentSetupPanel";

type StatusResponse = { enabled: boolean; status: string; tournament_count?: number | null; warnings?: string[]; confirmation_text?: Record<string, string>; streamlit_fallback_url?: string };
type Props = { searchParams?: Record<string, string | string[] | undefined> };

function first(value: string | string[] | undefined): string {
  return Array.isArray(value) ? String(value[0] || "") : String(value || "");
}

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

export default async function TournamentSetupPage({ searchParams }: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments/create");

  const clubId = "tres_palapas";
  const { data: status, error } = await loadStatus(clubId);
  return (
    <>
      <Suspense fallback={<div aria-hidden="true" style={{ minHeight: "42px", marginBottom: "1rem" }} />}>
        <TournamentAdminNav />
      </Suspense>
      <section>
        <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Tournament Manager</p>
        <h1 style={{ marginTop: 0 }}>{tournamentName || "Tournament"} setup</h1>
        <p style={{ color: "#334155", maxWidth: "860px" }}>
          Configure registration, days, events, divisions, publish-impact review, and setup publishing for this tournament.
        </p>
        {error ? <p role="alert" style={{ color: "#b91c1c" }}>Tournament Setup is unavailable. {error}</p> : null}
        <SelectedTournamentPanelScope tournamentId={tournamentId} tournamentName={tournamentName || null}>
          <TournamentSetupPanel apiBase={apiBase()} clubId={clubId} status={status} />
        </SelectedTournamentPanelScope>
      </section>
    </>
  );
}
