import Link from "next/link";
import { getAdminReplayApiBaseUrl, getAdminReplayStatus } from "@/lib/adminReplayApi";
import ReplayHistoryForm from "./ReplayHistoryForm";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function AdminReplayHistoryPage() {
  const clubId = "tres_palapas";
  const { data, error } = await getAdminReplayStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin Replay History
      </p>
      <h1 style={{ marginTop: 0 }}>Replay History</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>
        Server-side replay for rebuilding match snapshots, league ratings, and overall player stats after Match Log corrections. This is the Next/FastAPI replacement for the Streamlit Admin Tools replay path.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Replay status is temporarily unavailable. {error}</p> : null}

      {data ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>Status</strong><br />{data.status.replace(/_/g, " ")}</article>
            <article style={cardStyle}><strong>Options</strong><br />{data.options.length}</article>
            <article style={cardStyle}><strong>Confirmation</strong><br />{data.confirmation_text}</article>
            <article style={cardStyle}><strong>Endpoint</strong><br />{data.apply_endpoint || "Disabled"}</article>
          </div>

          {data.warnings?.length ? (
            <article style={{ ...cardStyle, background: "#fff7ed", marginBottom: "1rem" }}>
              <strong>Warnings</strong>
              <ul style={{ marginBottom: 0, paddingLeft: "1.25rem" }}>
                {data.warnings.map((warning) => <li key={warning}>{warning}</li>)}
              </ul>
            </article>
          ) : null}

          <article style={{ ...cardStyle, marginBottom: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Safety rules</h2>
            <ul style={{ margin: 0, paddingLeft: "1.25rem" }}>
              {data.safety_rules.map((rule) => <li key={rule}>{rule}</li>)}
            </ul>
          </article>

          <ReplayHistoryForm
            apiBase={getAdminReplayApiBaseUrl()}
            clubId={clubId}
            enabled={data.enabled}
            options={data.options}
            defaultTarget={data.default_target_reset}
          />

          <article style={{ ...cardStyle, marginTop: "1rem" }} data-testid="replay-job-history">
            <h2 style={{ marginTop: 0 }}>Recent durable replay jobs</h2>
            {data.recent_jobs?.length ? (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "760px" }}>
                  <thead><tr>{["Created", "Scope", "Status", "Actor", "Source", "Job ID"].map((label) => <th key={label} style={{ textAlign: "left", borderBottom: "1px solid #cbd5e1", padding: "0.55rem" }}>{label}</th>)}</tr></thead>
                  <tbody>{data.recent_jobs.map((job) => (
                    <tr key={job.id} data-replay-status={job.status}>
                      <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem" }}>{job.created_at ? new Date(job.created_at).toISOString().slice(0, 19).replace("T", " ") : "—"}</td>
                      <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem" }}>{job.target_reset || "—"}</td>
                      <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem" }}>{job.status}{job.error_text ? ` · ${job.error_text}` : ""}</td>
                      <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem" }}>{job.actor_email || "—"}</td>
                      <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem" }}>{job.source || "—"}</td>
                      <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.55rem", fontFamily: "monospace" }}>{job.id}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            ) : <p style={{ color: "#475569" }}>No replay jobs have been recorded for this club yet.</p>}
          </article>
        </>
      ) : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href="/admin/match-log">Back to Match Log</Link> · <Link href="/admin">Operations cockpit</Link>
      </p>
    </section>
  );
}
