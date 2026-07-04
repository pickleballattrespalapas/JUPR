import Link from "next/link";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const supportEmail = "joe@juprleagues.com";

export default function PrivacyPage() {
  return (
    <section style={{ maxWidth: "900px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Privacy Policy
      </p>
      <h1 style={{ marginTop: 0 }}>Pickleball Club Sandwich Privacy Policy</h1>
      <p style={{ color: "#334155" }}>
        Effective date: July 4, 2026. Pickleball Club Sandwich provides club websites, live scoring, ratings, leaderboards, player profiles, tournament registration, roster, and event-management tools for pickleball clubs.
      </p>

      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Information we collect</h2>
          <p style={{ color: "#475569" }}>
            We collect information submitted by players, organizers, and club staff, including names, email addresses, phone numbers, player identifiers, tournament registration details, partner-board preferences, match results, ratings, event participation, support requests, and correction requests. We may also collect basic technical information such as browser, device, log, and usage data needed to operate and secure the service.
          </p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>How we use information</h2>
          <p style={{ color: "#475569" }}>
            We use information to run club and tournament workflows, display public player and event pages, calculate and maintain ratings and standings, send registration and account-related messages, support organizers, investigate data corrections, prevent abuse, and improve the service.
          </p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Public information</h2>
          <p style={{ color: "#475569" }}>
            Public pages may display player names, ratings, match history, badges, league results, tournament divisions, public roster entries, partner-board entries, challenge-ladder status, weekly recaps, and event results. Public tournament roster and board views are designed not to expose private contact details such as phone numbers or email addresses.
          </p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Information sharing</h2>
          <p style={{ color: "#475569" }}>
            We share information with club organizers and service providers as needed to operate the service, process registrations, deliver email, host the website, provide database infrastructure, troubleshoot issues, and comply with legal obligations. We do not sell player contact information.
          </p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Security and retention</h2>
          <p style={{ color: "#475569" }}>
            We use administrative access controls, server-side write paths, audit logging for sensitive staff operations, and hosted infrastructure controls to protect operational data. We retain information for as long as reasonably needed for club operations, rating history, dispute resolution, legal compliance, and service improvement.
          </p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Your choices</h2>
          <p style={{ color: "#475569" }}>
            You may request correction, review, or removal of information by contacting support. Some rating, match, tournament, and audit records may need to be retained for operational integrity, dispute resolution, and legal compliance.
          </p>
          <p><Link href="/data-corrections">Open data correction instructions</Link></p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Contact</h2>
          <p style={{ color: "#475569" }}>
            Privacy questions and requests can be sent to <a href={`mailto:${supportEmail}`}>{supportEmail}</a>.
          </p>
        </article>
      </div>
    </section>
  );
}
