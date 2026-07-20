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

type Runbook = {
  label: string;
  useWhen: string;
  href: string;
  linkLabel: string;
  steps: string[];
  completeWhen: string;
  stopWhen: string;
};

const dayOfRunbooks: Runbook[] = [
  {
    label: "League night",
    useWhen: "A scheduled ladder or league session needs courts, rounds, movement, and official results.",
    href: "/admin/league-manager",
    linkLabel: "Open League Manager",
    steps: [
      "Confirm the club, league, week, total rounds, settings, and active roster before starting a live session.",
      "Review or print the roster/schedule, then create or resume the persisted live session for the correct league week.",
      "Confirm every court assignment before entering scores; submit one reviewed round at a time.",
      "Review automatic movement before advancing. Use an override only when the intended next court order is explicit.",
      "After the final round, verify Match Log rows, end the league when appropriate, and complete awards/closeout separately.",
    ],
    completeWhen: "Every expected round is submitted once, public results agree with Match Log, and the live session has the intended final state.",
    stopWhen: "The selected league/week is wrong, a player or court assignment is missing, a round looks duplicated, or the submitted match count differs from the preview.",
  },
  {
    label: "Quick, paper-sheet, or pop-up scoring",
    useWhen: "Staff need to enter manual, batch, singles, or round-robin results outside a persisted league-night session.",
    href: "/admin/match-uploader",
    linkLabel: "Open Match Uploader",
    steps: [
      "Choose the smallest matching input mode: single match, batch, singles, or round-robin preview.",
      "Confirm all player identities and create a missing player only when the identity and starting rating are known.",
      "Review scores, league/event attribution, rated status, and projected row count before submission.",
      "Submit once, record returned IDs/counts, and verify the new rows in Match Log before sending player updates.",
    ],
    completeWhen: "The expected matches appear once in Match Log with correct players, scores, format, attribution, and snapshots.",
    stopWhen: "A player is ambiguous, a preview count is unexpected, the event should not affect ratings, or similar matches may already exist.",
  },
  {
    label: "Match correction and replay",
    useWhen: "A saved match is wrong, duplicated, or should be excluded from ratings/history.",
    href: "/admin/match-log",
    linkLabel: "Open Match Log",
    steps: [
      "Filter to the exact match and capture its ID, current values, and affected player IDs before changing anything.",
      "Use the narrow correction, duplicate-resolution, cleanup, or soft-exclude action and review its exact draft/preview.",
      "Confirm the activity entry and the remaining Match Log rows before running replay.",
      "Run Replay History with the smallest safe scope; use ALL when the operation or runbook explicitly requires it.",
      "Recheck affected player profiles, leaderboards, league results, and snapshot fields after replay.",
    ],
    completeWhen: "The canonical Match Log is correct, replay succeeds, and all affected read models agree.",
    stopWhen: "The target cannot be uniquely identified, the change affects more rows than previewed, replay scope is unclear, or public results diverge after replay.",
  },
  {
    label: "Player maintenance or merge",
    useWhen: "Staff need to correct a player record, league rating, social identity, or duplicate-player relationship.",
    href: "/admin/players",
    linkLabel: "Open Player Editor",
    steps: [
      "Confirm club scope and player identity using IDs plus match/rating context, not name alone.",
      "For ordinary edits, change only the required fields and verify the saved detail before continuing.",
      "For merges, review the survivor/loser direction, reference counts, and every affected table in the merge preview.",
      "Apply the reviewed merge once, capture the activity result, then run Replay History ALL.",
      "Verify the survivor profile, ratings, match history, and that the duplicate no longer appears in public search.",
    ],
    completeWhen: "The intended player is canonical, references and ratings are consistent, and post-merge replay/public checks pass.",
    stopWhen: "Identity is uncertain, the merge direction is wrong, affected references are unexpected, or Replay History is unavailable.",
  },
  {
    label: "Challenge Ladder administration",
    useWhen: "Staff need to create/resolve challenges, administer passes/holds, or change ladder tiers and ranks.",
    href: "/admin/challenge-ladder",
    linkLabel: "Open Challenge Ladder Admin",
    steps: [
      "Load the dashboard and confirm player status, tier, rank, and any open challenge before preparing a change.",
      "For a new challenge, review eligibility, copy/send the generated notice, then start the acceptance clock based on the sent message.",
      "For a played result, review partners, games, winner, official-match count, and proposed rank movement before publishing.",
      "For whole-tier replacement, load/paste exact names, review removals/moves/recompression, resolve blockers, and apply only the current fingerprinted preview.",
      "Verify the updated ladder, official matches when applicable, and the centralized Admin Tools activity record.",
    ],
    completeWhen: "Challenge state, notices/deadlines, rank order, official matches, and activity records all agree.",
    stopWhen: "Eligibility requires an unexplained override, an open challenge blocks a roster change, a preview is stale, or the final persisted order differs from review.",
  },
  {
    label: "Tournament setup through publish",
    useWhen: "A tournament needs registration management, draws, live scoring, podiums, awards, or official match publication.",
    href: "/admin/tournaments",
    linkLabel: "Open Tournament Admin",
    steps: [
      "Confirm tournament, registration status, selections/divisions, and partner/team integrity before importing into a draw.",
      "Create/review the draw, import teams once, generate games, and verify expected team/game counts.",
      "Use Tournament Live for in-play scoring; confirm round-robin and playoff progression before advancing stages.",
      "Review podiums, awards, singles/doubles format, publish preview, and official match count before publishing.",
      "Verify Match Log/tournament_game_id links and affected player profiles before enabling player-update email handoff.",
    ],
    completeWhen: "Draw state, scores, podiums, awards, official matches, and public tournament pages are mutually consistent.",
    stopWhen: "Teams are duplicated, draw counts differ from preview, a playoff/podium is unresolved, or official matches may already have been published.",
  },
  {
    label: "Player communications",
    useWhen: "Staff are ready to send player updates or process verified update requests/preferences.",
    href: "/admin/player-updates",
    linkLabel: "Open Player Updates",
    steps: [
      "Finish and verify all underlying score, rating, tournament, or roster work before generating a report.",
      "Choose the exact date range and recipients, then inspect the report and exclusion/reason counts.",
      "Test using dry_run or staging_redirect; confirm destination rewriting and message content.",
      "Send only the reviewed range/selection, record delivery results, and honor unsubscribe/preference status.",
    ],
    completeWhen: "The reviewed recipients receive the intended content once and delivery/preference records reflect the run.",
    stopWhen: "Data verification is incomplete, live email mode is unexpected, recipients differ from preview, or staging redirects are not active during testing.",
  },
];

const globalStopConditions = [
  "The browser session, club, environment, or selected entity does not match the intended staging task.",
  "A feature flag, JWT verification, service role, strict audit, or required recovery route is unavailable.",
  "A preview is stale, row/count scope grows unexpectedly, or the response cannot be reconciled with the reviewed draft.",
  "A write succeeds partially or the public/read model disagrees afterward—capture IDs and stop additional writes.",
  "Email mode or recipients are not visibly safe for the current test.",
];

const recoverySequence = [
  "Stop additional writes and capture the route, club/entity IDs, timestamp, response, and operator account.",
  "Review Admin Tools activity and the affected source table/read model before attempting another action.",
  "Correct match data through Match Log when applicable; do not patch ratings or snapshots directly.",
  "Run Replay History using the smallest safe scope, then verify affected public and admin views.",
  "Use Streamlit production/fallback only for the established recovery workflow; do not copy staging-only data into production.",
];

type MigratedSafetyContract = {
  area: string;
  route: string;
  action: string;
  permission: string;
  phrase: string;
  stop: string;
  fallback: string;
};

const migratedSafetyContracts: MigratedSafetyContract[] = [
  { area: "Badge Diagnostics", route: "GET /admin/clubs/{club_id}/badges/options|debug|audit", action: "Read options, evaluator trace, or expected-vs-actual audit", permission: "view_audit_log", phrase: "None — read-only", stop: "Any response attempts a domain write or club scope is wrong.", fallback: "Streamlit Badge Debug / Badge Audit" },
  { area: "Badge Diagnostics", route: "PATCH /admin/clubs/{club_id}/badges/{badge_id}/state", action: "Change one reviewed badge definition state", permission: "run_replay", phrase: "UPDATE BADGE STATE", stop: "Expected state is stale, audit intent fails, or readback differs.", fallback: "Inspect Badge Audit; recover in Streamlit only after state is known." },
  { area: "Badge Diagnostics", route: "POST /admin/clubs/{club_id}/badges/recompute", action: "Dry-run or apply badge recompute", permission: "view_audit_log (dry-run); run_replay (apply)", phrase: "None for dry-run; RECOMPUTE BADGES for apply", stop: "Dry-run writes anything, scope is broader than reviewed, or operation is incomplete.", fallback: "Streamlit Badge Audit plus Replay History" },
  { area: "Badge Diagnostics", route: "PATCH /admin/clubs/{club_id}/badges/revoke", action: "Revoke one exact player badge row", permission: "run_replay", phrase: "REVOKE BADGE", stop: "Exact row/version cannot be proved or compensation fails.", fallback: "Inspect guarded operation and Badge Audit before Streamlit recovery." },
  { area: "Badge Diagnostics", route: "GET /admin/clubs/{club_id}/badges/operations/{operation_key}", action: "Inspect a durable badge write result", permission: "view_audit_log", phrase: "None — read-only", stop: "Status is incomplete or recovery_required; do not retry.", fallback: "Badge Audit / Replay History / Streamlit" },
  { area: "Match Canonical Audit", route: "GET options; POST /admin/clubs/{club_id}/match-canonical-audit/run", action: "Load players or compare canonical facts", permission: "view_audit_log", phrase: "None — read-only", stop: "Read scope is wrong or canonical facts cannot be loaded.", fallback: "Streamlit Match Canonical Audit" },
  { area: "Match Canonical Audit", route: "POST /admin/clubs/{club_id}/match-canonical-audit/normalize", action: "Dry-run exact proposals or atomically apply the same proposal set", permission: "view_audit_log (dry-run); manage_matches (apply)", phrase: "None for dry-run; APPLY NORMALIZE for apply", stop: "Fingerprint, exact IDs, expected values, or readback differ.", fallback: "Match Log then Replay History; Streamlit only after reconciliation." },
  { area: "Match Canonical Audit", route: "GET /admin/clubs/{club_id}/match-canonical-audit/operations/{operation_key}", action: "Inspect canonical normalize completion", permission: "view_audit_log", phrase: "None — read-only", stop: "Status is not completed; do not rerun with a new key.", fallback: "Match Log / Replay History / Streamlit" },
  { area: "Admin Tools", route: "GET overview|reports/ratings|social-submissions|workers/status|backfills/tournament-matches/preview", action: "Read health, roles/activity, safe CSV, queues, and previews", permission: "view_audit_log", phrase: "None — read-only", stop: "Any read route mutates rows or returns the wrong club.", fallback: "Streamlit Admin Tools" },
  { area: "Admin Tools", route: "POST /admin/clubs/{club_id}/tools/social-submissions/{event_id}/moderate", action: "Approve or reject one current Club Social submission", permission: "manage_matches", phrase: "APPROVE SOCIAL SUBMISSION or REJECT SOCIAL SUBMISSION", stop: "Expected status is stale or required audit cannot be completed/compensated.", fallback: "Inspect submission/activity, then Streamlit Club Social review." },
  { area: "Admin Tools", route: "PATCH /admin/clubs/{club_id}/tools/roles", action: "Save or revoke one staff assignment", permission: "manage_roles", phrase: "SAVE ROLE or REVOKE ROLE", stop: "Final super_admin support, version, readback, or audit is uncertain.", fallback: "Preserve the operation key and inspect role/activity state before recovery." },
  { area: "Admin Tools", route: "POST /admin/clubs/{club_id}/tools/workers/badge-queue", action: "Process or drain the badge evaluation queue", permission: "run_replay", phrase: "PROCESS BADGE QUEUE or DRAIN BADGE QUEUE", stop: "Partial work or completion audit is uncertain.", fallback: "Inspect worker status and Badge Audit; do not blindly retry." },
  { area: "Admin Tools", route: "POST /admin/clubs/{club_id}/tools/workers/badge-recompute", action: "Dry-run or apply a badge recompute", permission: "view_audit_log (dry-run); run_replay (apply)", phrase: "None for dry-run; RUN BADGE RECOMPUTE for apply", stop: "Dry-run writes anything or applying result is incomplete.", fallback: "Badge Audit / Replay History / Streamlit" },
  { area: "Admin Tools", route: "POST /admin/clubs/{club_id}/tools/backfills/tournament-matches/apply", action: "Backfill only exact reviewed ready tournament games", permission: "run_replay", phrase: "BACKFILL TOURNAMENT MATCHES", stop: "Preview is stale, duplicate state changes, or inserted IDs/count cannot be proved.", fallback: "Inspect operation, Match Log, and Replay History." },
  { area: "Admin Tools", route: "GET /tools/operations/{operation_key}; POST .../operations/{operation_key}/recover", action: "Inspect any Tools operation; reconcile an uncertain tournament backfill", permission: "view_audit_log (inspect); run_replay (recover)", phrase: "None to inspect; RECOVER TOURNAMENT BACKFILL to reconcile", stop: "Each selected game does not have exactly one official match.", fallback: "Stop writes; use Match Log / Replay History / Streamlit with the key." },
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

      <h2>Day-of operations runbooks</h2>
      <p style={{ color: "#475569", maxWidth: "900px" }}>Choose the runbook that matches the source of truth for the task. A green completion check never overrides a red stop condition.</p>
      <div style={{ display: "grid", gap: "1rem" }}>
        {dayOfRunbooks.map((runbook) => (
          <article key={runbook.label} style={cardStyle}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "1rem", flexWrap: "wrap" }}>
              <div style={{ minWidth: 0 }}><h3 style={{ margin: "0 0 0.35rem" }}>{runbook.label}</h3><p style={{ color: "#475569", margin: 0 }}>{runbook.useWhen}</p></div>
              <Link href={runbook.href}>{runbook.linkLabel}</Link>
            </div>
            <ol style={{ color: "#334155", paddingLeft: "1.25rem" }}>{runbook.steps.map((step) => <li key={step} style={{ marginBottom: "0.45rem" }}>{step}</li>)}</ol>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "0.75rem" }}>
              <div style={{ border: "1px solid #bbf7d0", background: "#f0fdf4", borderRadius: "10px", padding: "0.75rem" }}><strong style={{ color: "#166534" }}>Complete when</strong><p style={{ color: "#166534", marginBottom: 0 }}>{runbook.completeWhen}</p></div>
              <div style={{ border: "1px solid #fecaca", background: "#fef2f2", borderRadius: "10px", padding: "0.75rem" }}><strong style={{ color: "#991b1b" }}>Stop when</strong><p style={{ color: "#991b1b", marginBottom: 0 }}>{runbook.stopWhen}</p></div>
            </div>
          </article>
        ))}
      </div>

      <h2>Global stop conditions</h2>
      <article style={{ ...cardStyle, borderColor: "#fecaca", background: "#fef2f2" }}>
        <ul style={{ color: "#991b1b", margin: 0, paddingLeft: "1.25rem" }}>{globalStopConditions.map((condition) => <li key={condition} style={{ marginBottom: "0.5rem" }}>{condition}</li>)}</ul>
      </article>

      <h2>Migrated route safety contracts</h2>
      <p style={{ color: "#475569", maxWidth: "900px" }}>These are the exact FastAPI gates for Badge Debug/Audit, Match Canonical Audit, and Admin Tools. Status and dry-run routes are read-only. Every applying route is staging-only, service-role server mediated, intent-audited, operation-keyed, and completion-audited.</p>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "1100px", fontSize: "0.88rem" }}>
          <thead><tr>{["Area", "Route", "Action", "Permission", "Exact phrase", "Stop condition", "Fallback/recovery"].map((heading) => <th key={heading} align="left" style={{ borderBottom: "2px solid #cbd5e1", padding: "0.55rem", verticalAlign: "bottom" }}>{heading}</th>)}</tr></thead>
          <tbody>{migratedSafetyContracts.map((contract) => <tr key={`${contract.area}:${contract.route}`}><td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem", verticalAlign: "top" }}><strong>{contract.area}</strong></td><td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem", verticalAlign: "top" }}><code>{contract.route}</code></td><td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem", verticalAlign: "top" }}>{contract.action}</td><td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem", verticalAlign: "top" }}><code>{contract.permission}</code></td><td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem", verticalAlign: "top" }}>{contract.phrase}</td><td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem", verticalAlign: "top", color: "#991b1b" }}>{contract.stop}</td><td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem", verticalAlign: "top" }}>{contract.fallback}</td></tr>)}</tbody>
        </table>
      </div>

      <h2>Recovery sequence</h2>
      <article style={cardStyle}>
        <ol style={{ color: "#334155", margin: 0, paddingLeft: "1.25rem" }}>{recoverySequence.map((step) => <li key={step} style={{ marginBottom: "0.5rem" }}>{step}</li>)}</ol>
      </article>

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
