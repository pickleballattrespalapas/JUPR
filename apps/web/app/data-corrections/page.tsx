const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const supportEmail = "joe@juprleagues.com";

export default function DataCorrectionsPage() {
  const subject = encodeURIComponent("Pickleball Club Sandwich data correction request");
  const body = encodeURIComponent(
    "Name:\nClub:\nPlayer profile link or player name:\nMatch/date/league if known:\nWhat looks wrong:\nWhat should it be:\nAny screenshots or context:\n"
  );

  return (
    <section style={{ maxWidth: "860px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Data corrections
      </p>
      <h1 style={{ marginTop: 0 }}>Request a data correction</h1>
      <p style={{ color: "#334155" }}>
        This public page is an intake shell only. It helps players send complete correction details to staff; it does not directly edit scores, ratings, badges, or player records.
      </p>

      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>What to include</h2>
          <ul style={{ color: "#475569", paddingLeft: "1.25rem" }}>
            <li>Your name and club.</li>
            <li>The player profile, match, league, tournament, or event involved.</li>
            <li>The exact score, teammate, opponent, date, badge, or rating issue.</li>
            <li>What the corrected value should be.</li>
            <li>Any sheet, screenshot, or organizer context that supports the correction.</li>
          </ul>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>What happens next</h2>
          <p style={{ color: "#475569" }}>
            Staff review the request, verify the source of truth, then apply corrections through admin tools when appropriate. Rating recomputation and audit flows remain staff-controlled.
          </p>
        </article>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Send the request</h2>
          <p style={{ color: "#475569" }}>
            The email link opens a pre-filled checklist so the request has the information needed for review.
          </p>
          <a href={`mailto:${supportEmail}?subject=${subject}&body=${body}`}>Email correction request</a>
        </article>
      </div>
    </section>
  );
}
