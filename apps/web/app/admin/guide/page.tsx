import Link from "next/link";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const calloutStyle = { ...cardStyle, background: "#f8fafc" };

const validationPhases = [
  "Public read-only smoke: leaderboards, players, match explorer, league results, badge codex, challenge ladder, weekly recap, tournament registration/read pages.",
  "Auth smoke: sign in, sign out, expired token, reset password, wrong club, insufficient permission, and stored-session recovery.",
  "Score and correction smoke: Match Uploader, singles input, Moneyball, Match Log edits, duplicate no-issue suppression, bulk exclude, Match Canonical Audit, and Replay History.",
  "Tournament smoke: registration, edit links, partner board, Tournament Admin, Tournament Ops setup/import/scoring/podium/awards/publish, Tournament Live runner.",
  "League smoke: settings, roster membership, live sessions, court movement, official round submission, printouts, and top-performer closeout.",
  "Communications smoke: player update report dry-run/staging redirect, automatic post-batch send gate, verified request review, and unsubscribe links.",
];

const emergencyPaths = [
  "Use Streamlit fallback for any workflow whose Next flag is disabled or failing.",
  "For score/rating-impacting mistakes, stop further writes, correct through Match Log, then run Replay History using the smallest safe scope.",
  "For Player Editor merges, run Replay History ALL after the merge and validate affected profiles before further public communication.",
  "For tournament publish mistakes, verify tournament_game_id/match rows, use Match Log corrections, then replay affected ratings before sending player updates.",
  "For email issues, switch JUPR_EMAIL_MODE to dry_run or staging_redirect before testing again.",
];

const workflowNotes = [
  ["Match Log", "Corrections, duplicate no-issue suppression, duplicate cleanup, soft exclude, Quick Replay handoff."],
  ["Match Uploader", "Manual/batch scoring, singles input, round-robin preview, new-player create-and-continue, post-batch player updates."],
  ["Tournament Admin/Ops/Live", "Registration management, draw setup, team import, scoring, podiums, awards, official publish, tournament-specific live runner."],
  ["League Manager", "Settings, roster membership, live sessions, court movement, score submission, awards, printouts."],
  ["Player Editor", "Roster/detail edits, league rating edits, social linking, guarded merge."],
  ["Admin Tools", "Role assignment review/update/revoke, activity log, system health, Replay History, Badge Diagnostics, Match Canonical Audit."],
  ["Communications", "Player update date ranges, verified update requests, unsubscribe/preference controls."],
];

export default function AdminGuidePage() {
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Admin guide</p>
      <h1 style={{ marginTop: 0 }}>JUPR Next operations playbook</h1>
      <p style={{ color: "#334155", maxWidth: "900px" }}>This guide is the staff-facing operating sequence for validating and running the Next/Vercel + FastAPI admin stack while Streamlit remains the reference and emergency fallback.</p>

      <article style={calloutStyle}>
        <h2 style={{ marginTop: 0 }}>Permanent guardrails</h2>
        <ul style={{ color: "#334155" }}>
          <li>No privileged Supabase credentials in browser/Vercel code.</li>
          <li>No JavaScript rewrite of rating, replay, badge, or match-processing logic.</li>
          <li>Every production write must go through guarded FastAPI/Python domain services.</li>
          <li>Every destructive or rating-adjacent workflow needs audit attribution and a correction/replay path.</li>
          <li>Streamlit remains available until the specific replacement workflow is proven.</li>
        </ul>
      </article>

      <h2>Validation phases</h2>
      <div style={{ display: "grid", gap: "0.75rem" }}>
        {validationPhases.map((phase, index) => <article key={phase} style={{ ...cardStyle, display: "grid", gridTemplateColumns: "auto 1fr", gap: "0.75rem" }}><strong style={{ color: "#2563eb" }}>{index + 1}</strong><span>{phase}</span></article>)}
      </div>

      <h2>Workflow map</h2>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem" }}>
        {workflowNotes.map(([label, note]) => <article key={label} style={cardStyle}><h3 style={{ marginTop: 0 }}>{label}</h3><p style={{ color: "#475569" }}>{note}</p></article>)}
      </div>

      <h2>Emergency fallback paths</h2>
      <article style={cardStyle}>
        <ol style={{ color: "#334155", margin: 0, paddingLeft: "1.25rem" }}>
          {emergencyPaths.map((item) => <li key={item} style={{ marginBottom: "0.5rem" }}>{item}</li>)}
        </ol>
      </article>

      <h2>Operator links</h2>
      <article style={cardStyle}>
        <p><Link href="/admin">Operations cockpit</Link> · <Link href="/admin/tools">Admin Tools</Link> · <Link href="/admin/replay-history">Replay History</Link> · <Link href="/admin/match-log">Match Log</Link> · <Link href="/admin/tournaments">Tournament Admin</Link> · <Link href="/admin/league-manager">League Manager</Link></p>
      </article>
    </section>
  );
}
