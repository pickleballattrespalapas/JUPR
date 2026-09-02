import { redirect } from "next/navigation";
import type { TournamentSetupStep } from "@/components/TournamentSetupWizardNav";
import TournamentSetupWizardPanel from "./TournamentSetupWizardPanel";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";

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
  const context = readTournamentRouteContext(searchParams);
  const resolveDivisionId = first(searchParams?.resolveDivision).trim();
  if (!context.tournamentId) redirect("/admin/tournaments");

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
        Tournament Manager / Tournament Builder
      </p>
      <h1 style={{ marginTop: 0 }}>{context.tournamentName || "Tournament"} builder</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>
        Build the tournament across four logical domains: Tournament, Competition,
        Commerce, and Review. Draft saves never change public tournament pages;
        publication remains a separate, guarded action in Review.
      </p>
      {error ? (
        <p role="alert" style={{ color: "#b91c1c" }}>
          Tournament Setup is unavailable. {error}
        </p>
      ) : null}
      {status ? (
        <TournamentSetupWizardPanel
          key={context.tournamentId}
          apiBase={apiBase()}
          clubId={clubId}
          status={status}
          tournamentId={context.tournamentId}
          tournamentName={context.tournamentName || context.tournamentId}
          drawId={context.drawId}
          step={step}
          resolveDivisionId={resolveDivisionId}
        />
      ) : null}
    </section>
  );
}
