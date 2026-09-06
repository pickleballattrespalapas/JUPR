import PublicTournamentSponsors from "@/components/PublicTournamentSponsors";
import Link from "next/link";
import { getClubTournamentRegistrationConfirmation } from "@/lib/tournamentRegistrationApi";
import { formatCommerceMoney } from "@/lib/tournamentCommerceApi";
import {
  publicTournamentDayLabel,
  publicTournamentEventLabel
} from "@/lib/tournamentRegistrationEligibility";
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
  if (Number.isNaN(date.getTime())) return null;
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeZone: "UTC"
  }).format(date);
}

function scheduleLabel(label: string, value?: string | null): string {
  return publicTournamentDayLabel(label, value);
}

function registrationStatusLabel(status?: string | null): string {
  switch (String(status || "").toUpperCase()) {
    case "CONFIRMED":
      return "Confirmed";
    case "WAITLIST":
    case "WAITLISTED":
      return "On the waitlist";
    case "CANCELLED":
    case "CANCELED":
      return "Canceled";
    case "WITHDRAWN":
      return "Withdrawn";
    default:
      return "Received";
  }
}

function registrationSummary(status?: string | null): string {
  switch (String(status || "").toUpperCase()) {
    case "CONFIRMED":
      return "You’re registered. Save this page, and the organizer will follow up about payment and scheduling.";
    case "WAITLIST":
    case "WAITLISTED":
      return "You’re on the waitlist. Save this page, and the organizer will follow up if a spot opens.";
    case "CANCELLED":
    case "CANCELED":
      return "This registration is canceled. Contact the organizer if you have questions.";
    case "WITHDRAWN":
      return "This registration has been withdrawn. Contact the organizer if you have questions.";
    default:
      return "We received your registration. Save this page, and the organizer will follow up.";
  }
}

function paymentStatusLabel(status?: string | null): string {
  switch (String(status || "").toUpperCase()) {
    case "PAID":
      return "Paid";
    case "PARTIALLY_PAID":
      return "Partially paid";
    case "REFUNDED":
      return "Refunded";
    case "WAIVED":
      return "No payment needed";
    default:
      return "Payment due";
  }
}

function partnerStatusLabel(status?: string | null): string | null {
  switch (String(status || "").toUpperCase()) {
    case "HAS_PARTNER":
      return "Has a partner";
    case "NEEDS_PARTNER":
      return "Looking for a partner";
    case "NONE":
      return null;
    default:
      return null;
  }
}

function deliveryMessage(status?: string): { text: string; color: string; background: string } | null {
  if (status === "failed") return { text: "Your registration was saved, but we couldn’t send the confirmation email. The organizer can still see your registration.", color: "#991b1b", background: "#fef2f2" };
  if (status === "dry_run" || status === "staging_redirect") return { text: "Your registration was saved.", color: "#166534", background: "#f0fdf4" };
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
        { data: null, error: "missing_confirmation_link" },
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
  const manageRegistrationHref = `/clubs/${clubSlug}/tournament-registration${
    rosterQuery.toString() ? `?${rosterQuery.toString()}` : ""
  }#manage-registration`;

  return (
    <section style={{ maxWidth: "900px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Registration Confirmation
      </p>
      <h1 style={{ marginTop: 0, marginBottom: 0 }}>{data?.tournament.name ?? "Tournament registration"}</h1>
      <PublicTournamentSponsors clubSlug={clubSlug} tournamentId={data?.tournament.id} placement="header" />

      {error ? (
        <p style={{ color: "#b91c1c" }}>
          {confirmationToken
            ? "We couldn’t open this confirmation. Try the link in your email again or contact the organizer."
            : "This confirmation link is incomplete. Open the link in your email or contact the organizer."}
        </p>
      ) : null}
      {data ? (
        <>
          {delivery ? <p role="status" style={{ color: delivery.color, background: delivery.background, borderRadius: "10px", padding: "0.75rem" }}>{delivery.text}</p> : null}
          <article style={{ ...cardStyle, marginBottom: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Thanks, {data.registration.display_name}</h2>
            <p style={{ color: "#475569" }}>
              {registrationSummary(data.registration.status)}
            </p>
            <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", margin: 0 }}>
              <div><dt style={{ fontWeight: 700 }}>Submitted</dt><dd style={{ margin: 0 }}>{dateLabel(data.registration.submitted_at) ?? "—"}</dd></div>
              <div><dt style={{ fontWeight: 700 }}>Status</dt><dd style={{ margin: 0 }}>{registrationStatusLabel(data.registration.status)}</dd></div>
              <div><dt style={{ fontWeight: 700 }}>Payment</dt><dd style={{ margin: 0 }}>{paymentStatusLabel(data.registration.payment_status)}</dd></div>
              <div><dt style={{ fontWeight: 700 }}>Amount due to organizer</dt><dd style={{ margin: 0 }}>{commerceQuote ? formatCommerceMoney(commerceQuote.total_minor) : `$${Number(data.total_price_usd || 0).toFixed(2)}`}</dd></div>
            </dl>
          </article>

          <h2>Selected events</h2>
          <div style={{ display: "grid", gap: "0.75rem" }}>
            {data.selections.map((selection, index) => (
              <article key={`${selection.day_label}-${selection.event_family_label}-${selection.event_label}-${index}`} style={cardStyle}>
                <strong>{publicTournamentEventLabel(selection.event_family_label, selection.event_label)}</strong>
                <p style={{ margin: "0.35rem 0 0", color: "#64748b" }}>
                  {(selection.scheduled_days?.length
                    ? selection.scheduled_days
                        .map((day) => scheduleLabel(day.label, day.event_date))
                        .join(" · ")
                    : scheduleLabel(selection.day_label, selection.event_date))}
                </p>
                <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>
                  {[selection.skill_label, selection.age_label].filter(Boolean).join(" · ") || "Open division"} · ${Number(selection.price_usd || 0).toFixed(2)}
                </p>
                {partnerStatusLabel(selection.partner_mode) ? <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>Partner: {partnerStatusLabel(selection.partner_mode)}{selection.partner_name ? ` · ${selection.partner_name}` : ""}</p> : null}
                <p style={{ margin: "0.75rem 0 0" }}>
                  <Link href={manageRegistrationHref} style={{ fontWeight: 800 }}>
                    Edit this event
                  </Link>
                </p>
              </article>
            ))}
            {!data.selections.length ? <p style={{ color: "#92400e", background: "#fffbeb", borderRadius: "10px", padding: "0.75rem" }}>We couldn’t load your events. Please contact the organizer.</p> : null}
          </div>
          <p>
            <Link href={manageRegistrationHref} style={{ fontWeight: 800 }}>
              Add an event
            </Link>
          </p>

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
                  Total savings:{" "}
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
              Your registration is saved, but we couldn’t finish your team.
              Contact the organizer instead of registering again.
            </p>
          ) : null}

          <article style={{ ...cardStyle, marginTop: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>What happens next</h2>
            <p>
              You’ll pay the tournament organizer separately; we won’t charge
              you here. {data.payment_note}
            </p>
            {data.notification_sender?.from_email ? (
              <p style={{ color: "#475569" }}>Look for a confirmation email from {data.notification_sender.from_name || "JUPR Notifications"}. Check spam or junk if you don’t see it.</p>
            ) : (
              <p style={{ color: "#475569" }}>Look for a confirmation email from the tournament organizer. Check spam or junk if you don’t see it.</p>
            )}
            <p><Link href={`/clubs/${clubSlug}/tournament-roster${rosterQuery.toString() ? `?${rosterQuery.toString()}` : ""}`}>View tournament roster</Link></p>
          </article>

          <p style={{ marginTop: "1rem" }}>
            <Link href={`/clubs/${clubSlug}/tournament-registration`}>Back to tournament registration</Link>
          </p>
        </>
      ) : null}
      <PublicTournamentSponsors clubSlug={clubSlug} tournamentId={data?.tournament.id} placement="footer" />
    </section>
  );
}
