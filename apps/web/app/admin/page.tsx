import Link from "next/link";
import { getAdminOperationsStatus } from "@/lib/adminOperationsApi";
import type { AdminWorkflowStatus } from "@/lib/adminOperationsApi";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const mutedStyle = { color: "#475569" };

function statusLabel(status: string): string {
  return status.replace(/_/g, " ");
}

function riskStyle(risk: string) {
  if (risk === "critical") return { background: "#fee2e2", borderColor: "#fecaca" };
  if (risk === "high") return { background: "#ffedd5", borderColor: "#fed7aa" };
  if (risk === "medium") return { background: "#fef3c7", borderColor: "#fde68a" };
  return { background: "#dcfce7", borderColor: "#bbf7d0" };
}

function WorkflowCard({ workflow }: { workflow: AdminWorkflowStatus }) {
  const routeAvailable = Boolean(workflow.next_route);
  return (
    <article style={cardStyle}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap", alignItems: "flex-start" }}>
        <div>
          <h2 style={{ margin: "0 0 0.25rem", fontSize: "1.08rem" }}>{workflow.label}</h2>
          <p style={{ margin: 0, color: "#64748b", fontSize: "0.85rem" }}>{workflow.streamlit_page_key} · {workflow.api_scope}</p>
        </div>
        <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap" }}>
          <span style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.15rem 0.5rem", fontSize: "0.76rem", background: workflow.enabled ? "#dcfce7" : "white" }}>
            {workflow.enabled ? "Enabled" : statusLabel(workflow.effective_status)}
          </span>
          <span style={{ border: "1px solid", borderRadius: "999px", padding: "0.15rem 0.5rem", fontSize: "0.76rem", ...riskStyle(workflow.risk) }}>
            {workflow.risk}
          </span>
        </div>
      </div>
      <p style={mutedStyle}>{workflow.next_action}</p>
      <p style={{ color: "#64748b", fontSize: "0.85rem" }}><strong>Flag:</strong> <code>{workflow.env_flag}</code></p>
      {workflow.safety_notes?.length ? (
        <ul style={{ color: "#475569", paddingLeft: "1.2rem", marginBottom: "0.75rem" }}>
          {workflow.safety_notes.map((note) => <li key={note}>{note}</li>)}
        </ul>
      ) : null}
      {routeAvailable ? <Link href={workflow.next_route || "/admin"}>Open Next route</Link> : <span style={{ color: "#64748b" }}>Next route pending</span>}
    </article>
  );
}

export default async function AdminEntryPage() {
  const { data, error } = await getAdminOperationsStatus();
  const workflows = data?.workflows ?? [];
  const enabledCount = workflows.filter((workflow) => workflow.enabled).length;
  const statusByKey = new Map(workflows.map((workflow) => [workflow.key, workflow]));
  const sequence = data?.recommended_sequence?.map((key) => statusByKey.get(key)).filter((workflow): workflow is AdminWorkflowStatus => Boolean(workflow));

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin operations
      </p>
      <h1 style={{ marginTop: 0 }}>JUPR Next operations cockpit</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>
        Closed-club migration control for moving operational authority from Streamlit to Next/FastAPI one workflow at a time. This page is status-first: it shows which write workflows are enabled, which remain on Streamlit fallback, and which permanent guardrails still apply.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Admin operations status is temporarily unavailable. {error}</p> : null}

      {data ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>Environment</strong><br />{data.environment}</article>
            <article style={cardStyle}><strong>Mode</strong><br />{statusLabel(data.mode)}</article>
            <article style={cardStyle}><strong>Write pilot</strong><br />{data.write_pilot_enabled ? "Enabled" : "Disabled"}</article>
            <article style={cardStyle}><strong>Enabled workflows</strong><br />{enabledCount}</article>
            <article style={cardStyle}><strong>Strict audit</strong><br />{data.strict_audit_required ? "Required" : "Graceful"}</article>
            <article style={cardStyle}><strong>API service role</strong><br />{data.service_role_configured ? "Configured" : "Not configured"}</article>
          </div>

          <article style={{ ...cardStyle, marginBottom: "1rem", background: "#f8fafc" }}>
            <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Summer migration posture</h2>
            <p style={mutedStyle}>
              Next/FastAPI can now become the staff operations stack through workflow-specific flags. The fallback remains Streamlit until each workflow is proven.
            </p>
            <p style={{ marginBottom: 0 }}>
              <a href={data.streamlit_fallback_url} target="_blank" rel="noreferrer">Open Streamlit fallback</a>
            </p>
          </article>

          <h2>Recommended migration sequence</h2>
          <div style={{ display: "grid", gap: "0.75rem", marginBottom: "1.5rem" }}>
            {(sequence ?? workflows).map((workflow, index) => (
              <article key={`seq-${workflow.key}`} style={{ ...cardStyle, display: "grid", gridTemplateColumns: "auto 1fr", gap: "0.75rem", alignItems: "start" }}>
                <div style={{ fontWeight: 800, color: "#2563eb" }}>{index + 1}</div>
                <div>
                  <strong>{workflow.label}</strong>
                  <div style={{ color: "#64748b", fontSize: "0.86rem" }}>{workflow.next_action}</div>
                </div>
              </article>
            ))}
          </div>

          <h2>Workflow flags</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "1rem", marginBottom: "1.5rem" }}>
            {workflows.map((workflow) => <WorkflowCard key={workflow.key} workflow={workflow} />)}
          </div>

          <h2>Pilot gates</h2>
          <article style={cardStyle}>
            <ol style={{ margin: 0, paddingLeft: "1.25rem" }}>
              {data.pilot_gates.map((gate) => <li key={gate} style={{ marginBottom: "0.35rem" }}>{gate}</li>)}
            </ol>
          </article>

          <h2>Permanent guardrails</h2>
          <article style={cardStyle}>
            <ul style={{ margin: 0, paddingLeft: "1.25rem" }}>
              {data.permanent_guardrails.map((guardrail) => <li key={guardrail} style={{ marginBottom: "0.35rem" }}>{guardrail}</li>)}
            </ul>
          </article>
        </>
      ) : null}
    </section>
  );
}
