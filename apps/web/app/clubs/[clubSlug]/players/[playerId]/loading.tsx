export default function PlayerProfileLoading() {
  return (
    <section data-testid="player-profile-loading-state" aria-live="polite">
      <p style={{ color: "#2563eb", fontWeight: 700 }}>Player profile</p>
      <h1>Loading player profile…</h1>
      <p>Getting ratings, awards, matches, and Club Social results.</p>
    </section>
  );
}
