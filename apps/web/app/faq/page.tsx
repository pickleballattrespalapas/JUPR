import Link from "next/link";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const faqItems = [
  {
    question: "What is a club rating?",
    answer: "Pickleball Club Sandwich supports club-specific pickleball ratings used to create better matchups, seed events, and keep leveled play fair. Ratings are displayed on a 1.000–7.000 scale."
  },
  {
    question: "How do I get a rating?",
    answer: "You get a rating after recorded eligible matches. Your first recorded results may move your rating more quickly while the system learns your level."
  },
  {
    question: "What matches count?",
    answer: "Official ladders, round robins, league matches, and tournaments with official score entry can count. Open play, drills, clinics, and social play do not count unless explicitly recorded as eligible."
  },
  {
    question: "What affects rating movement?",
    answer: "Movement depends on opponent strength, expected outcome, scoreline, and consistency over time. A close loss to a stronger team can move differently than a one-sided expected win."
  },
  {
    question: "Can my rating go up after a loss?",
    answer: "Yes. If you perform better than expected against a stronger team, your rating can still increase after a loss."
  },
  {
    question: "Can my rating go down after a win?",
    answer: "No. A win is rewarded, though a win that is far below expectation may result in only minimal movement."
  },
  {
    question: "How does the system work for doubles?",
    answer: "Doubles results update each individual player. The system evaluates the strength of both teams, then adjusts each player based on the outcome and score."
  },
  {
    question: "What is the difference between overall and league ratings?",
    answer: "Overall rating reflects recorded eligible play across the club. League rating reflects a specific league or series when that context is shown."
  },
  {
    question: "What if a score was entered wrong?",
    answer: "Report the issue to the organizer or use the data-correction page. Corrections are reviewed by staff; the public site does not directly mutate rating data."
  }
];

export default function FaqPage() {
  return (
    <section style={{ maxWidth: "900px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        FAQ
      </p>
      <h1 style={{ marginTop: 0 }}>Rating FAQs</h1>
      <p style={{ color: "#334155", maxWidth: "760px" }}>
        Answers about ratings, recorded play, player movement, and how to request a correction.
      </p>

      <div style={{ display: "grid", gap: "1rem" }}>
        {faqItems.map((item) => (
          <article key={item.question} style={cardStyle}>
            <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>{item.question}</h2>
            <p style={{ color: "#475569", marginBottom: 0 }}>{item.answer}</p>
          </article>
        ))}
      </div>

      <article style={{ ...cardStyle, marginTop: "1rem", background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Need a data correction?</h2>
        <p style={{ color: "#475569" }}>
          Wrong score, partner, opponent, or event context should be reviewed by staff before ratings are recomputed.
        </p>
        <Link href="/data-corrections">Open data-correction instructions</Link>
      </article>
    </section>
  );
}
