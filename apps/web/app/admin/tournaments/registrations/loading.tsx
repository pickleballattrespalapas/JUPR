import TournamentPhaseNav from "@/components/TournamentPhaseNav";

export default function TournamentRegistrationReportsLoading() {
  return (
    <section aria-busy="true" aria-live="polite">
      <TournamentPhaseNav phase="registration" />
      <div
        role="status"
        style={{
          marginTop: "1rem",
          padding: "1rem 1.25rem",
          border: "1px solid #bfdbfe",
          borderRadius: "14px",
          background: "#eff6ff",
          color: "#1e3a8a"
        }}
      >
        <strong style={{ display: "block", fontSize: "1.05rem" }}>Loading communications and reports…</strong>
        <span style={{ display: "block", marginTop: "0.35rem" }}>
          Building registration filters, recipient previews, and export summaries. The selected tournament and Registration navigation will remain available.
        </span>
      </div>
    </section>
  );
}
