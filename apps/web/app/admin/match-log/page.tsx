import Link from "next/link";
import { getAdminApiBaseUrl, getAdminMatchLog } from "@/lib/adminMatchLogApi";
import type { AdminDuplicateGroup, AdminMatchLogMatch } from "@/lib/adminMatchLogApi";
import { getAdminReplayStatus } from "@/lib/adminReplayApi";
import MatchLogApplyPanel from "./MatchLogApplyPanel";
import MatchLogBulkExcludePanel from "./MatchLogBulkExcludePanel";
import MatchLogQuickReplayPanel from "./MatchLogQuickReplayPanel";
import MatchLogSocialPanel from "./MatchLogSocialPanel";

type MatchLogPageProps = {
  searchParams?: {
    filter?: string;
    match_id?: string;
    league?: string;
    week_tag?: string;
    start_date?: string;
    end_date?: string;
    limit?: string;
  };
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const muted = { color: "#475569" };

function playerNames(players: Array<{ id: number | null; name: string }>): string {
  return players.map((player) => player.name || (player.id ? `#${player.id}` : "—")).join(" / ");
}

function dateLabel(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 19);
  return date.toISOString().replace("T", " ").slice(0, 16);
}

function MatchRow({ match }: { match: AdminMatchLogMatch }) {
  return (
    <tr>
      <td>{match.id ?? "—"}</td>
      <td>{dateLabel(match.date)}</td>
      <td>{match.league || "—"}<br /><span style={{ color: "#64748b" }}>{match.week_tag || "—"}</span></td>
      <td>{match.match_type || "—"}</td>
      <td>{playerNames(match.team1)}</td>
      <td><strong>{match.score.display}</strong></td>
      <td>{playerNames(match.team2)}</td>
      <td>{match.is_active === false ? "Inactive" : "Active"}</td>
    </tr>
  );
}

function DuplicateGroupCard({ group, resolved = false }: { group: AdminDuplicateGroup; resolved?: boolean }) {
  return (
    <article style={{ ...cardStyle, background: resolved ? "#f0fdf4" : "#fff7ed" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap" }}>
        <div>
          <strong>{group.league || "—"} · {group.week_tag || "—"}</strong>
          <p style={{ margin: "0.35rem 0", color: "#475569" }}>
            {playerNames(group.team1)} vs {playerNames(group.team2)} · {group.score.display}
          </p>
          <p style={{ margin: 0, color: "#64748b", fontSize: "0.86rem" }}>IDs: {group.ids.join(", ")}</p>
          {resolved && group.resolution?.reason ? (
            <p style={{ margin: "0.35rem 0 0", color: "#166534" }}>Resolved as no issue: {group.resolution.reason}</p>
          ) : null}
        </div>
        <div style={{ textAlign: "right" }}>
          <div><strong>{group.dup_count}</strong> copies</div>
          {resolved ? (
            <>
              <div style={{ color: "#166534" }}>No issue</div>
              {group.resolution?.actor_email ? <div style={{ color: "#64748b" }}>{group.resolution.actor_email}</div> : null}
            </>
          ) : (
            <>
              <div style={{ color: "#166534" }}>Keep #{group.keep_id}</div>
              <div style={{ color: "#b91c1c" }}>Cleanup candidates: {group.delete_ids.map((id) => `#${id}`).join(", ") || "—"}</div>
            </>
          )}
        </div>
      </div>
    </article>
  );
}

export default async function AdminMatchLogPage({ searchParams }: MatchLogPageProps) {
  const clubId = "tres_palapas";
  const { data, error } = await getAdminMatchLog({
    clubId,
    filter: searchParams?.filter || "All",
    matchId: searchParams?.match_id || null,
    league: searchParams?.league || null,
    weekTag: searchParams?.week_tag || null,
    startDate: searchParams?.start_date || null,
    endDate: searchParams?.end_date || null,
    limit: searchParams?.limit || 250
  });
  const { data: replayData, error: replayError } = await getAdminReplayStatus(clubId);

  const selectedFilter = searchParams?.filter || data?.filters.filter || "All";
  const resolvedDuplicateGroups = data?.resolved_duplicate_groups || [];
  const apiBase = getAdminApiBaseUrl();

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin Match Log
      </p>
      <h1 style={{ marginTop: 0 }}>Match Log correction cockpit</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>
        Operational view for duplicate scanning, correction planning, and guarded audited apply flows. Writes stay behind FastAPI feature flags, Supabase JWT role checks, and Python domain services.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Match Log is temporarily unavailable. {error}</p> : null}

      {data && !data.enabled ? (
        <article style={{ ...cardStyle, marginBottom: "1rem", background: "#f8fafc" }}>
          <h2 style={{ marginTop: 0 }}>Next Match Log is disabled</h2>
          <p style={muted}>{data.warnings?.[0] || "Use Streamlit Match Log until this workflow is enabled for the pilot."}</p>
          <p style={{ marginBottom: 0 }}>
            <Link href="/admin">Back to operations cockpit</Link>
          </p>
        </article>
      ) : null}

      {data?.enabled ? (
        <>
          <form style={{ ...cardStyle, marginBottom: "1rem", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
            <label>Filter<br />
              <select name="filter" defaultValue={selectedFilter} style={{ width: "100%" }}>
                <option>All</option>
                <option>League</option>
                <option>Pop-Up</option>
              </select>
            </label>
            <label>Match ID<br /><input name="match_id" defaultValue={searchParams?.match_id || ""} style={{ width: "100%" }} /></label>
            <label>League<br /><input name="league" defaultValue={searchParams?.league || ""} style={{ width: "100%" }} /></label>
            <label>Week tag<br /><input name="week_tag" defaultValue={searchParams?.week_tag || ""} style={{ width: "100%" }} /></label>
            <label>Start date<br /><input name="start_date" type="date" defaultValue={searchParams?.start_date || ""} style={{ width: "100%" }} /></label>
            <label>End date<br /><input name="end_date" type="date" defaultValue={searchParams?.end_date || ""} style={{ width: "100%" }} /></label>
            <label>Limit<br /><input name="limit" type="number" min="1" max="1000" defaultValue={searchParams?.limit || "250"} style={{ width: "100%" }} /></label>
            <button type="submit" style={{ padding: "0.5rem 0.75rem", borderRadius: "8px", border: "1px solid #0f172a", background: "#0f172a", color: "white" }}>Apply</button>
          </form>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>Scanned</strong><br />{data.summary.scanned_matches}</article>
            <article style={cardStyle}><strong>Filtered</strong><br />{data.summary.filtered_matches ?? data.summary.returned_matches}</article>
            <article style={cardStyle}><strong>Shown</strong><br />{data.summary.returned_matches}</article>
            <article style={cardStyle}><strong>Duplicate groups</strong><br />{data.summary.duplicate_groups}</article>
            <article style={cardStyle}><strong>Cleanup candidates</strong><br />{data.summary.duplicate_delete_count}</article>
            <article style={cardStyle}><strong>Resolved no issue</strong><br />{data.summary.resolved_duplicate_groups ?? resolvedDuplicateGroups.length}</article>
          </div>

          {data.warnings?.length ? (
            <article style={{ ...cardStyle, marginBottom: "1rem", background: "#fff7ed" }}>
              <strong>Warnings</strong>
              <ul style={{ marginBottom: 0, paddingLeft: "1.25rem" }}>
                {data.warnings.map((warning) => <li key={warning}>{warning}</li>)}
              </ul>
            </article>
          ) : null}

          <h2>Duplicate scan</h2>
          {data.duplicate_groups.length ? (
            <div style={{ display: "grid", gap: "0.75rem", marginBottom: "1rem" }}>
              {data.duplicate_groups.map((group) => <DuplicateGroupCard key={group.dup_key} group={group} />)}
            </div>
          ) : <p style={muted}>No active duplicate groups found in the current filtered view.</p>}

          {data.duplicate_delete_preview ? (
            <article style={{ ...cardStyle, marginBottom: "1rem" }}>
              <h3 style={{ marginTop: 0 }}>Duplicate cleanup preview</h3>
              <p style={muted}>Future cleanup will keep the oldest row in each duplicate group, remove {data.duplicate_delete_preview.delete_count} row(s), then replay scope <strong>{data.duplicate_delete_preview.recommended_replay_scope}</strong>.</p>
              <p style={{ marginBottom: 0 }}><strong>Candidate IDs:</strong> {data.duplicate_delete_preview.delete_ids.join(", ")}</p>
            </article>
          ) : null}

          {resolvedDuplicateGroups.length ? (
            <>
              <h2>Resolved duplicate candidates</h2>
              <div style={{ display: "grid", gap: "0.75rem", marginBottom: "1rem" }}>
                {resolvedDuplicateGroups.map((group) => <DuplicateGroupCard key={`${group.dup_key}-resolved`} group={group} resolved />)}
              </div>
            </>
          ) : null}

          <h2>Correction and replay planning</h2>
          <article style={{ ...cardStyle, marginBottom: "1rem" }}>
            <p style={muted}>Apply endpoint: {data.correction_plan.apply_endpoint || "not enabled"}</p>
            <p style={muted}>Duplicate cleanup endpoint: {data.correction_plan.duplicate_cleanup_endpoint || "not enabled"}</p>
            <p style={muted}>Duplicate no-issue endpoint: {data.correction_plan.duplicate_no_issue_endpoint || "not enabled"}</p>
            <p><strong>Editable fields:</strong> {data.correction_plan.editable_fields_planned.join(", ")}</p>
            <p><strong>Sample recompute scope:</strong> standings={String(data.correction_plan.recompute_scope_for_sample_edit.standings)}, ratings={String(data.correction_plan.recompute_scope_for_sample_edit.ratings)}</p>
            <ul style={{ paddingLeft: "1.25rem", marginBottom: 0 }}>
              {data.correction_plan.safety_rules.map((rule) => <li key={rule}>{rule}</li>)}
            </ul>
          </article>

          <MatchLogApplyPanel
            apiBase={apiBase}
            clubId={clubId}
            applyEnabled={Boolean(data.apply_enabled)}
            duplicatePreview={data.duplicate_delete_preview}
            duplicateGroups={data.duplicate_groups}
            matches={data.matches}
          />

          <div style={{ marginTop: "1rem" }}>
            <MatchLogSocialPanel apiBase={apiBase} clubId={clubId} enabled={Boolean(data.enabled)} />
          </div>

          <div style={{ marginTop: "1rem" }}>
            <MatchLogBulkExcludePanel apiBase={apiBase} clubId={clubId} enabled={Boolean(data.apply_enabled)} matches={data.matches} />
          </div>

          <div style={{ marginTop: "1rem" }}>
            <MatchLogQuickReplayPanel
              apiBase={apiBase}
              clubId={clubId}
              enabled={Boolean(replayData?.enabled)}
              options={replayData?.options || []}
              defaultTarget={replayData?.default_target_reset || "ALL (Full System Reset)"}
              recommendedTarget={data.duplicate_delete_preview?.recommended_replay_scope || null}
              statusError={replayError}
              warnings={replayData?.warnings || []}
            />
          </div>

          <h2>Matches</h2>
          {data.matches.length ? (
            <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "14px", background: "white" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "900px" }}>
                <thead>
                  <tr style={{ textAlign: "left", background: "#f8fafc" }}>
                    <th style={{ padding: "0.6rem" }}>ID</th>
                    <th style={{ padding: "0.6rem" }}>Date</th>
                    <th style={{ padding: "0.6rem" }}>League / Week</th>
                    <th style={{ padding: "0.6rem" }}>Type</th>
                    <th style={{ padding: "0.6rem" }}>Team 1</th>
                    <th style={{ padding: "0.6rem" }}>Score</th>
                    <th style={{ padding: "0.6rem" }}>Team 2</th>
                    <th style={{ padding: "0.6rem" }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {data.matches.map((match) => <MatchRow key={`${match.id}-${match.dup_key}`} match={match} />)}
                </tbody>
              </table>
            </div>
          ) : <p style={muted}>No matches found for these filters.</p>}
        </>
      ) : null}
    </section>
  );
}
