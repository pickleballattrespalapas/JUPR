import TournamentCommercePanel from "./TournamentCommercePanel";

export default function AdminTournamentCommercePage() {
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
      <h1 style={{ marginTop: 0 }}>Extras, bundles, and fulfillment</h1>
      <p style={{ color: "#334155", maxWidth: "900px" }}>
        Configure optional tournament purchases and automatic savings, then
        track offline payments and pickup without changing prior order records.
      </p>
      <TournamentCommercePanel clubId="tres_palapas" />
    </section>
  );
}
