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
        Pickleball Club Sandwich turns club match results into durable ratings and leaderboard experiences. The public web app exposes the pieces players care about most: current rating, match history, event results, and movement over time.
      </p>
      <nav aria-label="Rating rules sections" style={{ ...cardStyle, marginBottom: "1rem", background: "#f8fafc" }}>
        <Link href="#rating-movement">Rating movement</Link> · <Link href="#rated-matches">Rated matches</Link> · <Link href="#corrections">Corrections</Link> · <Link href="#badges">Badges</Link> · <Link href="/faq">Rating FAQ</Link>
      </nav>
      <div style={{ display: "grid", gap: "1rem" }}>
        <article id="what-jupr-tracks" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>1. What JUPR tracks</h2>
          <p style={{ color: "#475569" }}>JUPR is the player rating and event record for club leagues, ladders, verified round robins, tournaments, profiles, badges, and weekly updates. Ratings come from official recorded results—not reputation, self-rating, or one strong day.</p>
        </article>
        <article id="rating-movement" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>2. How ratings move</h2>
          <p style={{ color: "#475569" }}>The Python rating service uses an Elo-style model. It estimates an expected result from current player ratings, then compares the final score with that expectation. Score margin matters: 15–13 and 15–3 communicate different performance.</p>
          <ul style={{ color: "#475569" }}><li>Winners receive positive movement.</li><li>Underdogs gain more for beating or strongly outperforming higher-rated opponents.</li><li>A close loss to much stronger opponents can still produce positive movement.</li><li>A win never produces negative movement.</li></ul>
        </article>
        <article id="rated-matches" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>3. What counts as rated</h2>
          <p style={{ color: "#475569" }}>Only club-approved results marked as rated affect official ratings: verified round robins, official rated leagues and ladders, rated tournament results, and authorized rated events.</p>
        </article>
        <article id="unrated-matches" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>4. What stays unrated</h2>
          <p style={{ color: "#475569" }}>Open play, practice games, warmups, clinics, incomplete/test events, and social events marked unrated remain separate from official rating history.</p>
        </article>
        <article id="verified-versus-casual" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>5. Verified versus casual</h2>
          <p style={{ color: "#475569" }}>Verified rated results affect ratings, standings, profiles, and official event history. Club Social and other unrated results may appear in their own experiences but do not change official ratings.</p>
        </article>
        <article id="corrections" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>6. Corrections, exclusions, and replay</h2>
          <p style={{ color: "#475569" }}>Authorized staff correct or exclude official rows through guarded, audited workflows. Rating-impacting corrections are rebuilt by the Python replay service. Public pages can request a review but cannot mutate rated history.</p>
          <Link href="/data-corrections">Request a data correction</Link>
        </article>
        <article id="badges" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>7. How badges are awarded</h2>
          <p style={{ color: "#475569" }}>Badges are calculated from recorded player, match, league, tournament, attendance, and event data. Public descriptions explain the achievement; the authoritative award comes from the recorded data.</p>
          <Link href="/clubs/tres-palapas/badge-codex">Open Badge Codex</Link>
        </article>
        <article id="other-ratings" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>8. Why JUPR can differ from other ratings</h2>
          <p style={{ color: "#475569" }}>JUPR uses matches recorded inside this club system. It can differ from DUPR, bracket level, self-rating, or opinion because each system has different data and rules. Ratings display on the 1.000–7.000 JUPR scale rather than raw Elo points.</p>
        </article>
        <article id="integrity" style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>9. Administrative integrity</h2>
          <p style={{ color: "#475569" }}>Only authorized, club-scoped staff workflows can publish or correct official results. Sensitive operations are permission-checked and audit-attributed, with Streamlit retained as an operational fallback during migration.</p>
        </article>
      </div>
      <p style={{ marginTop: "1.25rem" }}>
        <Link href="/faq">Read rating FAQs</Link> · <Link href="/clubs/tres-palapas/leaderboards">View Tres Palapas leaderboards</Link>
      </p>
    </section>
  );
}
