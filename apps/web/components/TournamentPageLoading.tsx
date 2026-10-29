export default function TournamentPageLoading() {
  return (
    <section role="status" aria-live="polite" aria-busy="true"
      style={{ padding: "1.25rem", border: "1px solid #bfdbfe", borderRadius: "14px", background: "#eff6ff", color: "#1e3a8a" }}>
      <h2 style={{ marginTop: 0 }}>Opening tournament page…</h2>
      <p>Your page is loading. You can choose another page while you wait.</p>
    </section>
  );
}
