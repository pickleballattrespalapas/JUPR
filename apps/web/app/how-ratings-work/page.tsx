import Link from "next/link";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default function HowRatingsWorkPage() {
  return (
    <section style={{ maxWidth: "820px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Rating system
      </p>
      <h1 style={{ marginTop: 0 }}>How club ratings work</h1>
      <p style={{ color: "#334155" }}>
        Pickleball Club Sandwich uses club match results to update ratings and show players how they change over time. Follow your current rating, match history, and event results in one place.
      </p>
      <nav aria-label="Rating rules sections" style={{ ...cardStyle, marginBottom: "1rem", background: "#f8fafc" }}>
        <Link href="#rating-movement">Rating movement</Link> · <Link href="#rated-matches">Rated matches</Link> · <Link href="#corrections">Corrections</Link> · <Link href="#badges">Badges</Link> · <Link href="/faq">Rating FAQ</Link>
      </nav>
      <div style={{ display: "grid", gap: "1rem" }}>
        <article id="what-jupr-tracks" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>1. What JUPR tracks</h2>
          <p style={{ color: "#475569" }}>JUPR brings together club ratings, leagues, ladders, round robins, tournaments, player profiles, badges, and weekly updates. Ratings come from club-approved match results—not reputation, self-rating, or one strong day.</p>
        </article>
        <article id="rating-movement" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>2. How ratings move</h2>
          <p style={{ color: "#475569" }}>JUPR uses an Elo-style formula. It compares the result with what the teams’ ratings predicted, including the margin of victory.</p>
          <ul style={{ color: "#475569" }}><li>Your rating goes up when you win.</li><li>Underdogs gain more for beating or strongly outperforming higher-rated opponents.</li><li>A close loss to much stronger opponents can still produce positive movement.</li><li>Your rating never goes down after a win.</li></ul>
        </article>
        <article id="rated-matches" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>3. What counts as rated</h2>
          <p style={{ color: "#475569" }}>Only club-approved results marked as rated affect official ratings: verified round robins, official rated leagues and ladders, rated tournament results, and authorized rated events.</p>
        </article>
        <article id="unrated-matches" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>4. What stays unrated</h2>
          <p style={{ color: "#475569" }}>Open play, practice games, warmups, clinics, unfinished or trial events, and social events marked unrated do not affect your rating.</p>
        </article>
        <article id="verified-versus-casual" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>5. Rated and unrated play</h2>
          <p style={{ color: "#475569" }}>Club-approved rated results affect ratings, standings, profiles, and event history. Club Social and other unrated results may still appear on the site, but they do not change ratings.</p>
        </article>
        <article id="corrections" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>6. Correcting a result</h2>
          <p style={{ color: "#475569" }}>Club staff review corrections to official results. If a result changes, JUPR recalculates any affected ratings.</p>
          <Link href="/data-corrections">Request a data correction</Link>
        </article>
        <article id="badges" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>7. How badges are awarded</h2>
          <p style={{ color: "#475569" }}>Badges are based on match results, league and tournament play, attendance, and other club activity. They’re awarded when a player meets the listed requirements.</p>
          <Link href="/clubs/tres-palapas/badge-codex">Open Badges & Trophies</Link>
        </article>
        <article id="other-ratings" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>8. Why JUPR can differ from other ratings</h2>
          <p style={{ color: "#475569" }}>JUPR uses official results from this club, so it may differ from DUPR, bracket levels, or self-ratings. JUPR uses a 1.000–7.000 scale.</p>
        </article>
        <article id="integrity" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>9. How results are protected</h2>
          <p style={{ color: "#475569" }}>Only authorized club staff can publish or correct results. Each change is recorded so it can be reviewed later.</p>
        </article>
      </div>
      <p style={{ marginTop: "1.25rem" }}>
        <Link href="/faq">Read rating FAQs</Link> · <Link href="/clubs/tres-palapas/leaderboards">View Tres Palapas leaderboards</Link>
      </p>
    </section>
  );
}
