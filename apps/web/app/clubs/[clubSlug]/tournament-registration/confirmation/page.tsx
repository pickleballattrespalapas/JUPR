import Link from "next/link";
import { getClubTournamentRegistrationConfirmation } from "@/lib/tournamentRegistrationApi";

type ConfirmationPageProps = {
  params: { clubSlug: string };
  searchParams?: { registration_id?: string; tournament?: string; tournament_id?: string };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

function dateLabel(value?: string | null): string | null {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10);
  return date.toISOString().slice(0, 10);
}

export default async function TournamentRegistrationConfirmationPage({ params, searchParams }: ConfirmationPageProps) {
  const { clubSlug } = params;
  const registrationId = searchParams?.registration_id || "";
  const { data, error } = registrationId
    ? await getClubTournamentRegistrationConfirmation(clubSlug, registrationId, {
        registrationSlug: searchParams?.tournament ?? null,
        tournamentId: searchParams?.tournament_id ?? null
      })
    : { data: null, error: "Missing registration id." };

  return (
    <section style={{ maxWidth: "900px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Registration Confirmation
      </p>
      <h1 style={{ marginTop: 0 }}>{data?.tournament.name ?? "Tournament registration"}</h1>

      {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
      {data ? (
        <>
          <article style={{ ...cardStyle, marginBottom: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Thanks, {data.registration.display_name}</h2>
            <p style={{ color: "#475569" }}>
              Your registration has been received. Save this page for your records; staff will manage payment status, draw import, seeding, and tournament operations.
            </p>
            <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", margin: 0 }}>
              <div><dt style={{ fontWeight: 700 }}>Registration ID</dt><dd style={{ margin: 0 }}>{data.registration.id}</dd></div>
              <div><dt style={{ fontWeight: 700 }}>Submitted</dt><dd style={{ margin: 0 }}>{dateLabel(data.registration.submitted_at) ?? "—"}</dd></div>
              <div><dt style={{ fontWeight: 700 }}>Status</dt><dd style={{ margin: 0 }}>{data.registration.status ?? "confirmed"}</dd></div>
              <div><dt style={{ fontWeight: 700 }}>Estimated total</dt><dd style={{ margin: 0 }}>${Number(data.total_price_usd || 0).toFixed(2)}</dd></div>
            </dl>
          </article>

          <h2>Selected events</h2>
          <div style={{ display: "grid", gap: "0.75rem" }}>
            {data.selections.map((selection) => (
              <article key={selection.selection_id} style={cardStyle}>
                <strong>{selection.event_family_label} — {selection.event_label}</strong>
                <p style={{ margin: "0.35rem 0 0", color: "#64748b" }}>{selection.day_label}</p>
                {selection.partner_mode ? <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>Partner status: {selection.partner_mode}{selection.partner_name ? ` · ${selection.partner_name}` : ""}</p> : null}
              </article>
            ))}
          </div>

          <p style={{ marginTop: "1rem" }}>
            <Link href={`/clubs/${clubSlug}/tournament-registration`}>Back to tournament registration</Link>
          </p>
        </>
      ) : null}
    </section>
  );
}
