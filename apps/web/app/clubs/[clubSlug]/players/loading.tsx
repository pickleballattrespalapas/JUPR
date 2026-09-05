export default function PlayersLoading() {
  return (
    <section data-testid="players-loading-state" aria-live="polite">
      <p style={{ color: "#2563eb", fontWeight: 700 }}>Player profiles</p>
      <h1>Loading players…</h1>
      <p>Getting player ratings and profile links.</p>
    </section>
  );
}
