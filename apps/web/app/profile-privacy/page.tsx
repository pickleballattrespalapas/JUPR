import Link from "next/link";
import ProfilePrivacyRequestForm from "./ProfilePrivacyRequestForm";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

export default function ProfilePrivacyPage() {
  return (
    <section style={{ maxWidth: "920px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Profile privacy
      </p>
      <h1 style={{ marginTop: 0 }}>Request a profile privacy review</h1>
      <p style={{ color: "#334155" }}>
        Ask us to review your display name or other information shown on your public profile. Submitting a request does not immediately hide ratings, matches, tournament results, or leaderboard records; staff will review it first.
      </p>

      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>How privacy review works</h2>
          <ul style={{ color: "#475569", paddingLeft: "1.25rem" }}>
            <li>Staff verify that the requester is connected to the affected player profile.</li>
            <li>Match and tournament records may need to remain in history because they affect ratings and event results or may be needed to resolve disputes.</li>
            <li>Approved privacy changes will not alter official match results.</li>
            <li>Contact details such as email and phone are not shown on public roster or partner-board pages by default.</li>
            <li>Staff close a request only after verifying your identity, making the approved change, and checking that it appears correctly on public pages.</li>
            <li>Do not attach identity documents to the form; staff will arrange an appropriate verification method.</li>
          </ul>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Privacy request form</h2>
          <ProfilePrivacyRequestForm />
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Need a data correction instead?</h2>
          <p style={{ color: "#475569" }}>
            Use the correction form for wrong scores, duplicate matches, player names, tournament entries, badges, or league records.
          </p>
          <Link href="/data-corrections">Open data correction form</Link>
        </article>
      </div>
    </section>
  );
}
