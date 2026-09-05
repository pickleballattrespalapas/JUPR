import Link from "next/link";
import { redirect } from "next/navigation";
import TournamentPhaseNav from "@/components/TournamentPhaseNav";
import { readTournamentRouteContext, tournamentRouteHref } from "@/lib/tournamentRouteContext";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

export default function TournamentPartnersPage({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");
  const partnerBoardParams = new URLSearchParams({
    tournament_id: context.tournamentId
  });
  if (context.drawId) partnerBoardParams.set("draw", context.drawId);
  const partnerBoardHref =
    `/clubs/tres-palapas/tournament-partner-board?${partnerBoardParams.toString()}`;

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
        Tournament Manager / Registration
      </p>
      <h1 style={{ marginTop: 0 }}>
        {context.tournamentName || "Tournament"} partners and teams
      </h1>
      <TournamentPhaseNav phase="registration" />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: "0.85rem"
        }}
      >
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Players Needing Partners</h2>
          <p style={{ color: "#475569" }}>
            Review open partner requests, automatic pairing, and incomplete
            doubles entries.
          </p>
          <Link href={partnerBoardHref}>Open Players Needing Partners</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Registration identities</h2>
          <p style={{ color: "#475569" }}>
            Review events, partner state, team, payment, and extras for each
            registrant.
          </p>
          <Link
            href={tournamentRouteHref("/admin/tournaments/registration/registrants", context)}
          >
            Open registrants
          </Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Four-player teams</h2>
          <p style={{ color: "#475569" }}>
            Manage team composition, invitations, substitutes, and incomplete
            rosters.
          </p>
          <Link
            href={tournamentRouteHref("/admin/tournaments/team-competition", context)}
          >
            Open team play
          </Link>
        </article>
      </div>
    </section>
  );
}
