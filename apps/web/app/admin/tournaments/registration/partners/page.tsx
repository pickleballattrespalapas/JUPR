import Link from "next/link";
import { redirect } from "next/navigation";
import TournamentPhaseNav from "@/components/TournamentPhaseNav";

type Props = { searchParams?: Record<string, string | string[] | undefined> };
function first(value: string | string[] | undefined): string { return Array.isArray(value) ? String(value[0] || "") : String(value || ""); }
function href(path: string, tournament: string, name: string): string { const params = new URLSearchParams({ tournament }); if (name) params.set("name", name); return `${path}?${params.toString()}`; }

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default function TournamentPartnersPage({ searchParams }: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments");
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Tournament Manager / Registration</p>
      <h1 style={{ marginTop: 0 }}>{tournamentName || "Tournament"} partners and teams</h1>
      <TournamentPhaseNav phase="registration" />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.85rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Partner Board</h2>
          <p style={{ color: "#475569" }}>Review open partner requests, automatic pairing, and incomplete doubles entries.</p>
          <Link href="/partner-board">Open Partner Board</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Registration identities</h2>
          <p style={{ color: "#475569" }}>Review each registrant's events, partner state, team, payment, and extras.</p>
          <Link href={href("/admin/tournaments/registration/registrants", tournamentId, tournamentName)}>Open registrants</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Four-player teams</h2>
          <p style={{ color: "#475569" }}>Manage team composition, invitations, substitutes, and incomplete rosters.</p>
          <Link href={href("/admin/tournaments/team-competition", tournamentId, tournamentName)}>Open team play</Link>
        </article>
      </div>
    </section>
  );
}
