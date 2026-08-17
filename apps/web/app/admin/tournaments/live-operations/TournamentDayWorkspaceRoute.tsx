import { redirect } from "next/navigation";
import TournamentPhaseNav from "@/components/TournamentPhaseNav";
import {
  getAdminTournamentApiBaseUrl,
  getAdminTournamentLiveStatus
} from "@/lib/adminTournamentApi";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";
import TournamentDayWorkspacePanel, {
  type TournamentDayWorkspacePanelFocus
} from "./TournamentDayWorkspacePanel";

type SearchParams = Record<string, string | string[] | undefined>;

type Props = {
  searchParams?: SearchParams;
};

function first(searchParams: SearchParams | undefined, key: string): string {
  const value = searchParams?.[key];
  return String(Array.isArray(value) ? value[0] || "" : value || "").trim();
}

function panelFocus(value: string): TournamentDayWorkspacePanelFocus {
  if (value === "queue" || value === "draws" || value === "corrections") return value;
  return "board";
}

export default async function TournamentDayWorkspaceRoute({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");

  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminTournamentLiveStatus(clubId);

  return (
    <section style={{ minWidth: 0, maxWidth: "100%", overflowX: "clip" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Manager / Live Operations
      </p>
      <h1 style={{ marginTop: 0 }}>
        {context.tournamentName ? `${context.tournamentName} day workspace` : "Tournament day workspace"}
      </h1>
      <p style={{ color: "#334155", maxWidth: "850px" }}>
        Run one selected tournament day from authoritative courts, one server-ordered eligible queue,
        and day-fenced draw progression.
      </p>
      <TournamentPhaseNav phase="live" />
      {error ? (
        <p role="alert" style={{ color: "#b91c1c" }}>
          Tournament-day operations are unavailable. {error}
        </p>
      ) : null}
      {status ? (
        <TournamentDayWorkspacePanel
          apiBase={getAdminTournamentApiBaseUrl()}
          clubId={clubId}
          status={status}
          tournamentId={context.tournamentId}
          tournamentName={context.tournamentName}
          initialDayId={context.dayId}
          initialDrawId={context.drawId}
          initialCourtId={first(searchParams, "court")}
          initialGameId={first(searchParams, "game")}
          initialPanel={panelFocus(first(searchParams, "panel"))}
        />
      ) : null}
    </section>
  );
}
