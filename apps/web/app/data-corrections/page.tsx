import DataCorrectionForm from "./DataCorrectionForm";
import { SUPPORT_EMAIL } from "@/lib/publicSupportContent";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

export default function DataCorrectionsPage() {
  return (
    <section style={{ maxWidth: "920px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Data corrections
      </p>
      <h1 style={{ marginTop: 0 }}>Request a data correction</h1>
      <p style={{ color: "#334155" }}>
        Tell us about an incorrect score, player, league, tournament, badge, duplicate, or profile. Club staff will review your request before changing anything on the site.
      </p>

      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>What to include</h2>
          <ul style={{ color: "#475569", paddingLeft: "1.25rem" }}>
            <li>Your name and email so staff can follow up.</li>
            <li>The player profile, match, league, tournament, or event involved.</li>
            <li>The exact score, teammate, opponent, date, badge, or rating issue.</li>
            <li>What the corrected value should be.</li>
            <li>Any sheet, screenshot, organizer note, or other context that supports the correction.</li>
          </ul>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Correction request form</h2>
          <p style={{ color: "#475569" }}>
            Staff will review the original information and correct the relevant player, match, league, tournament, badge, or rating details.
          </p>
          <DataCorrectionForm />
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Email fallback</h2>
          <p style={{ color: "#475569" }}>
            If the form is unavailable, send the same details by email.
          </p>
          <a href={`mailto:${SUPPORT_EMAIL}?subject=${encodeURIComponent("Pickleball Club Sandwich data correction request")}`}>Email correction request</a>
        </article>
      </div>
    </section>
  );
}
