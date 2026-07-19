const skeleton = {
  borderRadius: "10px",
  background: "#e2e8f0"
};

export default function LeaderboardsLoading() {
  return (
    <section aria-busy="true" aria-live="polite" data-testid="leaderboard-loading-state">
      <p style={{ color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Leaderboards</p>
      <h1>Loading leaderboards…</h1>
      <p style={{ color: "#475569" }}>Loading public-safe standings, qualification, and badge context.</p>
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", margin: "1rem 0" }}>
        {[1, 2, 3, 4].map((item) => <span key={item} style={{ ...skeleton, width: "7rem", height: "2.1rem" }} />)}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
        {[1, 2, 3, 4].map((item) => <span key={item} style={{ ...skeleton, height: "5.5rem" }} />)}
      </div>
      <span style={{ position: "absolute", width: "1px", height: "1px", padding: 0, margin: "-1px", overflow: "hidden", clip: "rect(0, 0, 0, 0)", whiteSpace: "nowrap", border: 0 }}>Loading leaderboard rows</span>
    </section>
  );
}
