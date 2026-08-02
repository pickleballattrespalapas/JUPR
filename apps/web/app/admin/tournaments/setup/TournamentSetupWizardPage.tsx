import { redirect } from "next/navigation";
import type { TournamentSetupStep } from "@/components/TournamentSetupWizardNav";
import TournamentSetupWizardPanel from "./TournamentSetupWizardPanel";

type StatusResponse = {
  enabled: boolean;
  status: string;
  tournament_count?: number | null;
  warnings?: string[];
  confirmation_text?: Record<string, string>;
  streamlit_fallback_url?: string;
};

type Props = {
  step: TournamentSetupStep;
  searchParams?: Record<string, string | string[] | undefined>;
};

function first(value: string | string[] | undefined): string {
  return Array.isArray(value) ? String(value[0] || "") : String(value || "");
}

function apiBase(): string | null {
  return (
    process.env.JUPR_API_BASE_URL ||
    process.env.NEXT_PUBLIC_JUPR_API_BASE_URL ||
    null
  );
}

async function loadStatus(
  clubId: string
): Promise<{ data: StatusResponse | null; error: string | null }> {
  const base = apiBase();
  if (!base) return { data: null, error: "Missing JUPR API base URL." };
  try {
    const response = await fetch(
      `${base.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(
        clubId
      )}/tournaments/setup/status`,
      { cache: "no-store" }
    );
    if (!response.ok) {
      return { data: null, error: `API error (${response.status}).` };
    }
    return { data: (await response.json()) as StatusResponse, error: null };
  } catch (error) {
    return {
      data: null,
      error: error instanceof Error ? error.message : "Unable to reach API."
    };
  }
}

export default async function TournamentSetupWizardPage({
  step,
  searchParams
}: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments");

  const clubId = "tres_palapas";
  const { data: status, error } = await loadStatus(clubId);

  return (
    <section>
      <p
        style={{
          margin: "0 0 0.5rem",
          color: "#2563eb",
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          fontSize: "0.78rem"
        }}
      >
        Tournament Manager / Guided Setup
      </p>
      <h1 style={{ marginTop: 0 }}>{tournamentName || "Tournament"} setup</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>
        Complete the six setup steps in order. Each save keeps the current
        tournament context and moves you directly to the next task.
      </p>
      {error ? (
        <p role="alert" style={{ color: "#b91c1c" }}>
          Tournament Setup is unavailable. {error}
        </p>
      ) : null}
      {status ? (
        <TournamentSetupWizardPanel
          apiBase={apiBase()}
          clubId={clubId}
          status={status}
          tournamentId={tournamentId}
          tournamentName={tournamentName || tournamentId}
          step={step}
        />
      ) : null}
    </section>
  );
}
