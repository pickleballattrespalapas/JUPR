export default function PlayerProfileLoading() {
  return (
    <section data-testid="player-profile-loading-state" aria-live="polite">
      <p style={{ color: "#2563eb", fontWeight: 700 }}>Player profile</p>
      <h1>Loading public profile…</h1>
      <p>Preparing ratings, awards, match formats, and Club Social aggregates.</p>
    </section>
  );
}
