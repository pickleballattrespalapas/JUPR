import Link from "next/link";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const supportEmail = "joe@juprleagues.com";

export default function TermsPage() {
  return (
    <section style={{ maxWidth: "900px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Terms of Use
      </p>
      <h1 style={{ marginTop: 0 }}>Pickleball Club Sandwich Terms of Use</h1>
      <p style={{ color: "#334155" }}>
        Effective date: July 4, 2026. These terms govern use of Pickleball Club Sandwich websites, live scoring, ratings, leaderboards, tournament registration, roster, and event-management tools.
      </p>

      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Use of the service</h2>
          <p style={{ color: "#475569" }}>
            You may use the service for lawful pickleball club, league, tournament, and player-information purposes. You agree not to misuse the service, interfere with its operation, submit false or misleading information, scrape public pages at abusive rates, or attempt to access admin-only systems without authorization.
          </p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Ratings, standings, and records</h2>
          <p style={{ color: "#475569" }}>
            Ratings, rankings, badges, standings, match history, and tournament information are provided to support club operations, fair play, event organization, and player progress tracking. They may be updated, corrected, recalculated, or removed when organizers identify errors or operational needs.
          </p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Public pages are informational</h2>
          <p style={{ color: "#475569" }}>
            Public pages are informational and do not replace organizer decisions. Public pages do not directly create official match records, edit scores, change ratings, or finalize tournament outcomes unless the workflow explicitly says so and the action is accepted by the server.
          </p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Tournament registration</h2>
          <p style={{ color: "#475569" }}>
            Tournament registrations and edits must be accurate and submitted by the registering player or an authorized representative. Event availability, partner-board status, pricing, waitlist state, and eligibility may change. Organizers may review, reject, adjust, waitlist, or correct registrations according to event rules.
          </p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Corrections and disputes</h2>
          <p style={{ color: "#475569" }}>
            Score, player-record, tournament, and profile disputes should be submitted through the correction process or organizer support. Staff review is required before correction, replay, recomputation, or administrative changes are applied.
          </p>
          <p><Link href="/data-corrections">Request a correction</Link></p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Availability and changes</h2>
          <p style={{ color: "#475569" }}>
            The service is provided as available. Features may be changed, paused, or removed. We are not responsible for delays, outages, data-entry mistakes, third-party service failures, or damages beyond the amounts paid directly for use of the service where limitation is permitted by law.
          </p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Contact</h2>
          <p style={{ color: "#475569" }}>
            Questions about these terms can be sent to <a href={`mailto:${supportEmail}`}>{supportEmail}</a>.
          </p>
        </article>
      </div>
    </section>
  );
}
