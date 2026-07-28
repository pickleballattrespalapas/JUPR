import TeamTournamentAdminPanel from "./TeamTournamentAdminPanel";

export default function TeamTournamentAdminPage() {
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
        Tournament Admin
      </p>
      <h1 style={{ marginTop: 0 }}>Combined ratings and four-player teams</h1>
      <p style={{ color: "#334155", maxWidth: "920px" }}>
        Configure eligibility and team formats, review registrations, run the
        fixed four-game match order, and publish calculated results.
      </p>
      <TeamTournamentAdminPanel clubId="tres_palapas" />
    </section>
  );
}
