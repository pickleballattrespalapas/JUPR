import Link from "next/link";
import { getClubTournamentRegistrationConfirmation } from "@/lib/tournamentRegistrationApi";
import { formatCommerceMoney } from "@/lib/tournamentCommerceApi";
import { recoverPublicFourPlayerTeamSetup } from "@/lib/tournamentTeamCompetitionApi";
import FourPlayerTeamSetupRecovery from "./FourPlayerTeamSetupRecovery";

type ConfirmationPageProps = {
  params: { clubSlug: string };
  searchParams?: {
    confirmation_token?: string;
    email_status?: string;
    team_setup?: string;
  };
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

function deliveryMessage(status?: string): { text: string; color: string; background: string } | null {
  if (status === "failed") return { text: "Your registration was saved, but the confirmation email could not be sent. Tournament staff can still see it.", color: "#991b1b", background: "#fef2f2" };
  if (status === "dry_run") return { text: "Your registration was saved. Email delivery was safely dry-run in this environment.", color: "#92400e", background: "#fffbeb" };
  if (status === "staging_redirect") return { text: "Your registration was saved. The email was sent to the staging redirect address.", color: "#166534", background: "#f0fdf4" };
  if (status === "sent") return { text: "Your registration was saved and the confirmation email was sent.", color: "#166534", background: "#f0fdf4" };
  return null;
}

export default async function TournamentRegistrationConfirmationPage({ params, searchParams }: ConfirmationPageProps) {
  const { clubSlug } = params;
  const confirmationToken = searchParams?.confirmation_token || "";
  const [confirmationResult, teamRecoveryResult] = confirmationToken
    ? await Promise.all([
        getClubTournamentRegistrationConfirmation(clubSlug, confirmationToken),
        recoverPublicFourPlayerTeamSetup(clubSlug, confirmationToken)
      ])
    : [
        { data: null, error: "Missing secure confirmation token." },
        { data: null, error: null }
      ];
  const { data, error } = confirmationResult;
  const teamRecovery = teamRecoveryResult.data;
  const teamSetupNeedsAttention = searchParams?.team_setup === "attention";
  const delivery = deliveryMessage(searchParams?.email_status);
  const commerceQuote = data?.commerce_order?.quote || null;
  const commerceLines =
    commerceQuote?.lines.filter((line) => line.line_type !== "EVENT") || [];
  const rosterQuery = new URLSearchParams();
  if (data?.settings?.registration_slug) rosterQuery.set("tournament", data.settings.registration_slug);
  else if (data?.tournament.id) rosterQuery.set("tournament_id", data.tournament.id);

  return (
    <section style={{ maxWidth: "900px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Registration Confirmation
      </p>
      <h1 style={{ marginTop: 0 }}>{data?.tournament.name ?? "Tournament registration"}</h1>

      {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
      {data ? (
        <>
          {delivery ? <p role="status" style={{ color: delivery.color, background: delivery.background, borderRadius: "10px", padding: "0.75rem" }}>{delivery.text}</p> : null}
          <article style={{ ...cardStyle, marginBottom: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Thanks, {data.registration.display_name}</h2>
            <p style={{ color: "#475569" }}>
              Your registration has been received. Save this page for your records; staff will manage payment status, draw import, seeding, and tournament operations.
            </p>
            <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", margin: 0 }}>
              <div><dt style={{ fontWeight: 700 }}>Submitted</dt><dd style={{ margin: 0 }}>{dateLabel(data.registration.submitted_at) ?? "—"}</dd></div>
              <div><dt style={{ fontWeight: 700 }}>Status</dt><dd style={{ margin: 0 }}>{data.registration.status ?? "confirmed"}</dd></div>
              <div><dt style={{ fontWeight: 700 }}>Payment</dt><dd style={{ margin: 0 }}>{data.registration.payment_status ?? "unpaid"}</dd></div>
              <div><dt style={{ fontWeight: 700 }}>Total due offline</dt><dd style={{ margin: 0 }}>{commerceQuote ? formatCommerceMoney(commerceQuote.total_minor) : `$${Number(data.total_price_usd || 0).toFixed(2)}`}</dd></div>
            </dl>
          </article>

          <h2>Selected events</h2>
          <div style={{ display: "grid", gap: "0.75rem" }}>
            {data.selections.map((selection, index) => (
              <article key={`${selection.day_label}-${selection.event_family_label}-${selection.event_label}-${index}`} style={cardStyle}>
                <strong>{selection.event_family_label} — {selection.event_label}</strong>
                <p style={{ margin: "0.35rem 0 0", color: "#64748b" }}>{selection.day_label}{selection.event_date ? ` · ${dateLabel(selection.event_date)}` : ""}</p>
                <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>
                  {[selection.skill_label, selection.age_label].filter(Boolean).join(" · ") || "Open division"} · ${Number(selection.price_usd || 0).toFixed(2)}
                </p>
                {selection.partner_mode ? <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>Partner status: {selection.partner_mode}{selection.partner_name ? ` · ${selection.partner_name}` : ""}</p> : null}
              </article>
            ))}
            {!data.selections.length ? <p style={{ color: "#92400e", background: "#fffbeb", borderRadius: "10px", padding: "0.75rem" }}>No event selections were found. Contact tournament staff if this is unexpected.</p> : null}
          </div>

          {commerceLines.length ? (
            <>
              <h2>Extras and bundle savings</h2>
              <div style={{ display: "grid", gap: "0.75rem" }}>
                {commerceLines.map((line) => (
                  <article key={line.line_key} style={cardStyle}>
                    <strong>
                      {line.quantity} × {line.label}
                      {line.option_label ? ` — ${line.option_label}` : ""}
                    </strong>
                    <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>
                      {formatCommerceMoney(line.final_total_minor)}
                      {line.savings_minor > 0
                        ? ` · ${formatCommerceMoney(line.savings_minor)} saved`
                        : ""}
                    </p>
                    {line.component_snapshot?.length ? (
                      <ul style={{ marginBottom: 0 }}>
                        {line.component_snapshot.map((component, index) => (
                          <li
                            key={
                              component.id ||
                              `${line.line_key}-${component.component_type}-${index}`
                            }
                          >
                            {component.total_quantity ||
                              component.quantity ||
                              1}{" "}
                            ×{" "}
                            {component.label ||
                              component.option_label ||
                              (component.component_type === "EVENT_OPTION"
                                ? "event entry"
                                : "extra")}
                          </li>
                        ))}
                      </ul>
                    ) : null}
                  </article>
                ))}
              </div>
              {commerceQuote && commerceQuote.discount_minor > 0 ? (
                <p style={{ color: "#166534" }}>
                  Total bundle and giveaway savings:{" "}
                  <strong>
                    {formatCommerceMoney(commerceQuote.discount_minor)}
                  </strong>
                </p>
              ) : null}
            </>
          ) : null}

          {teamRecovery?.events.length ? (
            <FourPlayerTeamSetupRecovery
              clubSlug={clubSlug}
              confirmationToken={confirmationToken}
              initialRecovery={teamRecovery}
              needsAttention={teamSetupNeedsAttention}
            />
          ) : teamSetupNeedsAttention ? (
            <p
              role="alert"
              style={{
                color: "#991b1b",
                background: "#fef2f2",
                borderRadius: "10px",
                padding: "0.75rem"
              }}
            >
              Your registration is saved, but team setup recovery is
              temporarily unavailable. Contact tournament staff and do not
              submit a second registration.
            </p>
          ) : null}

          <article style={{ ...cardStyle, marginTop: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Payment and email</h2>
            <p>
              Payment is handled offline by tournament staff. {data.payment_note}
            </p>
            {data.notification_sender?.from_email ? (
              <p style={{ color: "#475569" }}>Your confirmation email comes from {data.notification_sender.from_name || "JUPR Notifications"} &lt;{data.notification_sender.from_email}&gt;. Check spam or junk if it is not in your inbox.</p>
            ) : (
              <p style={{ color: "#475569" }}>Your confirmation email comes from the tournament registration address. Check spam or junk if it is not in your inbox.</p>
            )}
            <p><Link href={`/clubs/${clubSlug}/tournament-roster${rosterQuery.toString() ? `?${rosterQuery.toString()}` : ""}`}>View the public tournament roster</Link></p>
          </article>

          <p style={{ marginTop: "1rem" }}>
            <Link href={`/clubs/${clubSlug}/tournament-registration`}>Back to tournament registration</Link>
          </p>
        </>
      ) : null}
    </section>
  );
}
