export default function AdminEntryPage() {
  return (
    <section style={{ maxWidth: "760px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Organizer entry
      </p>
      <h1 style={{ marginTop: 0 }}>JUPR admin is moving to the web</h1>
      <p style={{ color: "#334155" }}>
        The public website is becoming the main JUPR product surface. Streamlit admin remains the safe production console while organizer workflows are ported one at a time.
      </p>
      <div style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" }}>
        <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Migration order</h2>
        <ol style={{ color: "#475569" }}>
          <li>Public JUPR Live viewer</li>
          <li>Player profiles and match history</li>
          <li>Authenticated organizer score entry</li>
          <li>Full Streamlit admin replacement</li>
        </ol>
      </div>
    </section>
  );
}
