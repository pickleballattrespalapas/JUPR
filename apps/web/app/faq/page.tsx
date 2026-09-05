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
    answer: "You get a rating after your first club-approved matches. Your rating may move more quickly at first while JUPR gets a clearer picture of your level."
  },
  {
    question: "What matches count?",
    answer: "Ladders, round robins, league matches, and tournaments can count when club staff mark the results as rated. Open play, drills, clinics, and social play do not count unless the club marks them as rated."
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
    answer: "Your overall rating reflects rated matches across the club. A league rating reflects only matches from that league or series."
  },
  {
    question: "What if a score was entered wrong?",
    answer: "Report the issue to the organizer or use the data-correction page. Staff will review your request before making any changes, so submitting it does not change your rating immediately."
  },
  {
    question: "How should I use my JUPR rating?",
    answer: "Use it to choose leveled sessions, seed ladders and tournaments fairly, follow progress, and create competitive matches. It reflects your official club results, not a guarantee of any single result."
  },
  {
    question: "Why can JUPR differ from DUPR or my bracket level?",
    answer: "JUPR uses official results from this club. Other ratings and brackets may use different matches, rules, and scales."
  }
];

function anchorFor(question: string): string {
  return question.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

export default function FaqPage() {
  return (
    <section style={{ maxWidth: "900px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        FAQ
      </p>
      <h1 style={{ marginTop: 0 }}>Rating FAQs</h1>
      <p style={{ color: "#334155", maxWidth: "760px" }}>
        Answers about ratings, matches, rating changes, and how to request a correction.
      </p>
      <p><Link href="/how-ratings-work">Read the full rating guide</Link></p>

      <div style={{ display: "grid", gap: "1rem" }}>
        {faqItems.map((item) => (
          <article id={anchorFor(item.question)} key={item.question} style={cardStyle}>
            <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>{item.question}</h2>
            <p style={{ color: "#475569", marginBottom: 0 }}>{item.answer}</p>
          </article>
        ))}
      </div>

      <article style={{ ...cardStyle, marginTop: "1rem", background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Need a data correction?</h2>
        <p style={{ color: "#475569" }}>
          If a score, partner, opponent, or event is wrong, staff will review it and update any affected ratings.
        </p>
        <Link href="/data-corrections">Open data-correction instructions</Link>
      </article>
    </section>
  );
}
