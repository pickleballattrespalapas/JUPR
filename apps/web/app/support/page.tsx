import Link from "next/link";
import { SERVICE_LOCATION, SERVICE_OPERATOR, SUPPORT_EMAIL } from "@/lib/publicSupportContent";
import SupportRequestForm from "./SupportRequestForm";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

export default function SupportPage() {
  const subject = encodeURIComponent("Pickleball Club Sandwich support request");
  const body = encodeURIComponent("Name:\nClub:\nPage or issue:\nDetails:\n");
  return (
    <section style={{ maxWidth: "860px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Support
      </p>
      <h1 style={{ marginTop: 0 }}>Contact Pickleball Club Sandwich support</h1>
      <p style={{ color: "#334155" }}>
        Get help with player profiles, corrections, or anything else on the website.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>General support</h2>
          <p style={{ color: "#475569" }}>Questions about ratings, pages, access, tournament registration, or club setup can start here.</p>
          <a href={`mailto:${SUPPORT_EMAIL}?subject=${subject}&body=${body}`}>Email {SUPPORT_EMAIL}</a>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Data corrections</h2>
          <p style={{ color: "#475569" }}>Wrong score, teammate, opponent, league, tournament entry, or player profile details should use the correction checklist.</p>
          <Link href="/data-corrections">Open correction checklist</Link>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>FAQ</h2>
          <p style={{ color: "#475569" }}>Review how ratings, doubles movement, and eligible match types work.</p>
          <Link href="/faq">Read FAQ</Link>
        </article>
      </div>
      <article id="general-support-form" style={{ ...cardStyle, marginTop: "1rem" }}>
        <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Send a support request</h2>
        <p style={{ color: "#475569" }}>Send your question to the club&apos;s support team. Submitting this form will not change player, match, rating, or tournament information.</p>
        <SupportRequestForm />
      </article>
      <article id="operator" style={{ ...cardStyle, marginTop: "1rem", background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Operator and policy links</h2>
        <p style={{ color: "#475569" }}>{SERVICE_OPERATOR} operates this service for {SERVICE_LOCATION}.</p>
        <p><Link href="/privacy">Privacy policy</Link> · <Link href="/profile-privacy">Profile privacy request</Link> · <Link href="/terms">Terms of use</Link></p>
      </article>
    </section>
  );
}
