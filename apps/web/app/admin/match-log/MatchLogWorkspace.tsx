"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { getAdminApiBaseUrl, getAdminMatchLog } from "@/lib/adminMatchLogApi";
import type { AdminDuplicateGroup, AdminMatchExclusionOperation, AdminMatchLogMatch, AdminMatchLogResponse } from "@/lib/adminMatchLogApi";
import { getAdminReplayStatus } from "@/lib/adminReplayApi";
import type { AdminReplayStatusResponse } from "@/lib/adminReplayApi";
import { useAdminSession } from "@/lib/useAdminSession";
import MatchLogApplyPanel from "./MatchLogApplyPanel";
import MatchLogBulkExcludePanel from "./MatchLogBulkExcludePanel";
import MatchLogExclusionRecoveryPanel from "./MatchLogExclusionRecoveryPanel";
import MatchLogQuickReplayPanel from "./MatchLogQuickReplayPanel";
import MatchLogSocialPanel from "./MatchLogSocialPanel";

export type MatchLogSearchParams = {
  filter?: string;
  match_id?: string;
  league?: string;
  week_tag?: string;
  context_type?: string;
  context_id?: string;
  context_ids?: string;
  start_date?: string;
  end_date?: string;
  limit?: string;
};

type MatchLogWorkspaceProps = {
  searchParams?: MatchLogSearchParams;
  mode: "review" | "edit" | "bulk" | "duplicates" | "exclude" | "social" | "replay";
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const muted = { color: "#475569" };
const workspaceModes = [
  { mode: "review", path: "/admin/match-log", label: "Review", title: "Review matches", description: "Find matches, scan duplicates, and choose the right correction tool." },
  { mode: "edit", path: "/admin/match-log/edit", label: "Edit one", title: "Edit a match", description: "Correct one match and review the exact changes before applying them." },
  { mode: "bulk", path: "/admin/match-log/bulk", label: "Bulk edit", title: "Bulk edit matches", description: "Apply the same correction to a selected group of visible matches." },
  { mode: "duplicates", path: "/admin/match-log/duplicates", label: "Duplicates", title: "Resolve duplicates", description: "Mark false positives or remove confirmed duplicate rows with guarded replay." },
  { mode: "exclude", path: "/admin/match-log/exclude", label: "Exclude", title: "Exclude rated matches", description: "Soft-exclude selected matches and complete any required recovery." },
  { mode: "social", path: "/admin/match-log/social", label: "Social", title: "Social match tools", description: "Review and correct social match records." },
  { mode: "replay", path: "/admin/match-log/replay", label: "Replay", title: "Replay ratings", description: "Run a guarded rating replay and review its current posture." }
] as const;

function playerNames(players: Array<{ id: number | null; name: string }>): string {
  return players.map((player) => player.name || (player.id ? `#${player.id}` : "—")).join(" / ");
}

function dateLabel(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 19);
  return date.toISOString().replace("T", " ").slice(0, 16);
}

function stableFilterOptions(values: Array<string | null | undefined>, selected?: string | null): string[] {
  const options = new Set(
    values
      .map((value) => String(value || "").trim())
      .filter(Boolean)
  );
  if (selected?.trim()) options.add(selected.trim());
  return Array.from(options).sort((left, right) => left.localeCompare(right, undefined, { sensitivity: "base" }));
}

function MatchRow({ match }: { match: AdminMatchLogMatch }) {
  return (
    <tr>
      <td>{match.id ?? "—"}</td>
      <td>{match.row_version ?? "—"}</td>
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

export default function MatchLogWorkspace({ searchParams, mode }: MatchLogWorkspaceProps) {
  const clubId = "tres_palapas";
  const { accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const apiBase = getAdminApiBaseUrl();
  const [rawData, setRawData] = useState<AdminMatchLogResponse | null>(null);
  const [dataScope, setDataScope] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [replayData, setReplayData] = useState<AdminReplayStatusResponse | null>(null);
  const [replayError, setReplayError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshNotice, setRefreshNotice] = useState<{ tone: "pending" | "success" | "error"; text: string } | null>(null);
  const [reloadNonce, setReloadNonce] = useState(0);
  const [exclusionOperation, setExclusionOperation] = useState<AdminMatchExclusionOperation | null>(null);
  const mutationRefreshPending = useRef(false);
  const selectedFilterParam = searchParams?.filter || "All";
  const matchIdParam = searchParams?.match_id || null;
  const leagueParam = searchParams?.league || null;
  const weekTagParam = searchParams?.week_tag || null;
  const contextTypeParam = searchParams?.context_type || null;
  const contextIdParam = searchParams?.context_id || null;
  const contextIdsParam = searchParams?.context_ids || contextIdParam || null;
  const startDateParam = searchParams?.start_date || null;
  const endDateParam = searchParams?.end_date || null;
  const limitParam = searchParams?.limit || "250";
  const requestScope = [
    accessToken,
    sessionLoading ? "loading" : "ready",
    selectedFilterParam,
    matchIdParam || "",
    leagueParam || "",
    weekTagParam || "",
    contextTypeParam || "",
    contextIdsParam || "",
    startDateParam || "",
    endDateParam || "",
    limitParam
  ].join("\u0000");
  const data = dataScope === requestScope ? rawData : null;

  const handleMutationComplete = useCallback(() => {
    mutationRefreshPending.current = true;
    setRefreshNotice({ tone: "pending", text: "Refreshing current match and replay status…" });
    setLoading(true);
    setReloadNonce((current) => current + 1);
  }, []);

  useEffect(() => {
    let cancelled = false;

    if (sessionLoading) {
      mutationRefreshPending.current = false;
      setRawData(null);
      setDataScope("");
      setReplayData(null);
      setError(null);
      setReplayError(null);
      setRefreshNotice(null);
      setLoading(true);
      return () => {
        cancelled = true;
      };
    }

    if (!accessToken) {
      mutationRefreshPending.current = false;
      setRawData(null);
      setDataScope("");
      setReplayData(null);
      setError(sessionMessage);
      setReplayError(null);
      setRefreshNotice(null);
      setLoading(false);
      return () => {
        cancelled = true;
      };
    }

    const mutationRefresh = mutationRefreshPending.current;
    if (!mutationRefresh) {
      setRawData(null);
      setDataScope("");
      setReplayData(null);
      setRefreshNotice(null);
    }
    setLoading(true);
    setError(null);
    setReplayError(null);
    Promise.all([
      getAdminMatchLog({
        clubId,
        filter: selectedFilterParam,
        matchId: matchIdParam,
        league: leagueParam,
        weekTag: weekTagParam,
        contextType: contextTypeParam,
        contextIds: contextIdsParam,
        startDate: startDateParam,
        endDate: endDateParam,
        limit: limitParam
      }, accessToken),
      getAdminReplayStatus(clubId, { accessToken, apiBase })
    ])
      .then(([matchLogResult, replayResult]) => {
        if (cancelled) return;
        if (matchLogResult.data) {
          setRawData(matchLogResult.data);
          setDataScope(requestScope);
        } else if (!mutationRefresh) {
          setRawData(null);
          setDataScope("");
        }
        setError(matchLogResult.error);
        if (replayResult.data) setReplayData(replayResult.data);
        else if (!mutationRefresh) setReplayData(null);
        setReplayError(replayResult.error);
        if (mutationRefresh) {
          mutationRefreshPending.current = false;
          if (matchLogResult.data && replayResult.data) {
            setRefreshNotice({ tone: "success", text: "Match and replay status refreshed from the server." });
          } else if (matchLogResult.data) {
            setRefreshNotice({
              tone: "error",
              text: `The change completed and match state refreshed, but replay status could not be reloaded. ${replayResult.error || ""}`.trim()
            });
          } else {
            setRefreshNotice({
              tone: "error",
              text: `The change completed, but refreshed Match Log state could not be loaded. ${matchLogResult.error || ""}`.trim()
            });
          }
        }
      })
      .finally(() => {
        if (!cancelled) {
          mutationRefreshPending.current = false;
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [
    accessToken,
    apiBase,
    contextIdsParam,
    contextTypeParam,
    endDateParam,
    leagueParam,
    limitParam,
    matchIdParam,
    reloadNonce,
    selectedFilterParam,
    sessionLoading,
    sessionMessage,
    requestScope,
    startDateParam,
    weekTagParam
  ]);

  useEffect(() => {
    if (!data) return;
    const unresolved = (data.recent_exclusion_operations || []).find(
      (operation) => operation.status !== "succeeded"
    ) || null;
    setExclusionOperation((current) => {
      if (unresolved) return unresolved;
      if (current?.status === "succeeded") return null;
      return current;
    });
  }, [data]);

  useEffect(() => {
    setExclusionOperation(null);
  }, [clubId]);

  const selectedFilter = selectedFilterParam || data?.filters.filter || "All";
  const resolvedDuplicateGroups = data?.resolved_duplicate_groups || [];
  const leagueOptions = stableFilterOptions(
    data?.filter_options?.leagues || data?.matches.map((match) => match.league) || [],
    leagueParam
  );
  const weekTagOptions = stableFilterOptions(
    data?.filter_options?.week_tags || data?.matches.map((match) => match.week_tag) || [],
    weekTagParam
  );
  const selectedMode = workspaceModes.find((item) => item.mode === mode) || workspaceModes[0];
  const modePath = selectedMode.path;
  const showsMatchContext = mode !== "social" && mode !== "replay";
  const showsMatchSummary = showsMatchContext && mode !== "edit";
  const showsMatchTable = showsMatchContext && mode !== "edit" && mode !== "duplicates";
  const preserveFilters = (path: string) => {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(searchParams || {})) {
      if (value) params.set(key, value);
    }
    const query = params.toString();
    return query ? `${path}?${query}` : path;
  };

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin Match Log
      </p>
      <h1 style={{ marginTop: 0 }}>{selectedMode.title}</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>
        {selectedMode.description}
      </p>

      <nav aria-label="Match Log sections" style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "1rem" }}>
        {workspaceModes.map((item) => (
          <Link
            key={item.mode}
            href={preserveFilters(item.path)}
            aria-current={item.mode === mode ? "page" : undefined}
            style={{
              padding: "0.55rem 0.8rem",
              borderRadius: "999px",
              border: "1px solid #cbd5e1",
              background: item.mode === mode ? "#0f172a" : "white",
              color: item.mode === mode ? "white" : "#0f172a",
              textDecoration: "none",
              fontWeight: 700
            }}
          >
            {item.label}
          </Link>
        ))}
      </nav>

      {loading && !data ? <p style={muted}>Loading protected Match Log…</p> : null}
      {refreshNotice ? (
        <p
          role={refreshNotice.tone === "error" ? "alert" : "status"}
          aria-live="polite"
          style={{ color: refreshNotice.tone === "error" ? "#b91c1c" : refreshNotice.tone === "success" ? "#166534" : "#475569" }}
        >
          {refreshNotice.text}
        </p>
      ) : null}

      {!sessionLoading && !accessToken ? (
        <article style={{ ...cardStyle, marginBottom: "1rem", background: "#fffbeb", borderColor: "#fbbf24" }}>
          <h2 style={{ marginTop: 0 }}>Admin sign-in required</h2>
          <p style={muted}>Sign in with an assigned club admin account before loading Match Log data.</p>
          <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p>
        </article>
      ) : null}

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
          {showsMatchContext ? <form data-testid="match-log-filters" style={{ ...cardStyle, marginBottom: "1rem", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
            {mode === "edit" ? (
              <div style={{ gridColumn: "1 / -1" }}>
                <h2 style={{ margin: 0 }}>Find a match</h2>
                <p style={{ ...muted, marginBottom: 0 }}>Filter the match list, then use the compact selector in the editor below.</p>
              </div>
            ) : null}
            <label>Filter<br />
              <select key={`filter-${selectedFilterParam}`} name="filter" defaultValue={selectedFilter} style={{ width: "100%" }}>
                <option>All</option>
                <option>League</option>
                <option>Pop-Up</option>
              </select>
            </label>
            <label>Match ID<br /><input key={`match-${matchIdParam || "all"}`} name="match_id" defaultValue={matchIdParam || ""} style={{ width: "100%" }} /></label>
            <label>League<br />
              <select key={`league-${leagueParam || "all"}`} name="league" defaultValue={leagueParam || ""} style={{ width: "100%" }}>
                <option value="">All leagues</option>
                {leagueOptions.map((league) => <option key={league} value={league}>{league}</option>)}
              </select>
            </label>
            <label>Week tag<br />
              <select key={`week-${weekTagParam || "all"}`} name="week_tag" defaultValue={weekTagParam || ""} style={{ width: "100%" }}>
                <option value="">All weeks</option>
                {weekTagOptions.map((weekTag) => <option key={weekTag} value={weekTag}>{weekTag}</option>)}
              </select>
            </label>
            <details style={{ gridColumn: "1 / -1" }} open={Boolean(contextTypeParam || contextIdsParam)}>
              <summary style={{ cursor: "pointer", fontWeight: 700 }}>Advanced recovery context</summary>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" }}>
                <label>Context type<br /><input key={`context-type-${contextTypeParam || "all"}`} name="context_type" defaultValue={contextTypeParam || ""} style={{ width: "100%" }} /></label>
                <label>Context IDs (comma-separated)<br /><input key={`context-ids-${contextIdsParam || "all"}`} name="context_ids" defaultValue={contextIdsParam || ""} style={{ width: "100%" }} /></label>
              </div>
            </details>
            <label>Start date<br /><input key={`start-${startDateParam || "all"}`} name="start_date" type="date" defaultValue={startDateParam || ""} style={{ width: "100%" }} /></label>
            <label>End date<br /><input key={`end-${endDateParam || "all"}`} name="end_date" type="date" defaultValue={endDateParam || ""} style={{ width: "100%" }} /></label>
            <label>Limit<br /><input key={`limit-${limitParam}`} name="limit" type="number" min="1" max="1000" defaultValue={limitParam} style={{ width: "100%" }} /></label>
            <button type="submit" style={{ padding: "0.5rem 0.75rem", borderRadius: "8px", border: "1px solid #0f172a", background: "#0f172a", color: "white" }}>Apply filters</button>
            <Link href={modePath} style={{ padding: "0.5rem 0.75rem", borderRadius: "8px", border: "1px solid #64748b", color: "#0f172a", textAlign: "center", textDecoration: "none" }}>Clear filters</Link>
          </form> : null}

          {showsMatchSummary ? <div data-testid="match-log-summary" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>Scanned</strong><br />{data.summary.scanned_matches}</article>
            <article style={cardStyle}><strong>Filtered</strong><br />{data.summary.filtered_matches ?? data.summary.returned_matches}</article>
            <article style={cardStyle}><strong>Shown</strong><br />{data.summary.returned_matches}</article>
            <article style={cardStyle}><strong>Duplicate groups</strong><br />{data.summary.duplicate_groups}</article>
            <article style={cardStyle}><strong>Cleanup candidates</strong><br />{data.summary.duplicate_delete_count}</article>
            <article style={cardStyle}><strong>Resolved no issue</strong><br />{data.summary.resolved_duplicate_groups ?? resolvedDuplicateGroups.length}</article>
          </div> : null}

          {showsMatchTable ? <section data-testid="match-log-results" style={{ marginBottom: "1rem" }}>
            <h2>Matches</h2>
            {data.matches.length ? (
              <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "14px", background: "white" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "900px" }}>
                  <thead>
                    <tr style={{ textAlign: "left", background: "#f8fafc" }}>
                      <th style={{ padding: "0.6rem" }}>ID</th>
                      <th style={{ padding: "0.6rem" }}>Version</th>
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
          </section> : null}

          {data.warnings?.length ? (
            <article style={{ ...cardStyle, marginBottom: "1rem", background: "#fff7ed" }}>
              <strong>Warnings</strong>
              <ul style={{ marginBottom: 0, paddingLeft: "1.25rem" }}>
                {data.warnings.map((warning) => <li key={warning}>{warning}</li>)}
              </ul>
            </article>
          ) : null}

          {mode === "review" ? (
            <>
              <h2>Correction tools</h2>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
                {workspaceModes.filter((item) => item.mode !== "review").map((item) => (
                  <article key={item.mode} style={cardStyle}>
                    <h3 style={{ marginTop: 0 }}>{item.title}</h3>
                    <p style={muted}>{item.description}</p>
                    <Link href={preserveFilters(item.path)}>Open {item.label.toLowerCase()}</Link>
                  </article>
                ))}
              </div>
            </>
          ) : null}

          {mode === "review" || mode === "duplicates" ? (
            <>
              <h2>Duplicate scan</h2>
              {data.duplicate_groups.length ? (
                <div style={{ display: "grid", gap: "0.75rem", marginBottom: "1rem" }}>
                  {data.duplicate_groups.map((group) => <DuplicateGroupCard key={group.dup_key} group={group} />)}
                </div>
              ) : <p style={muted}>No active duplicate groups found in the current filtered view.</p>}

              {data.duplicate_delete_preview ? (
                <article style={{ ...cardStyle, marginBottom: "1rem" }}>
                  <h3 style={{ marginTop: 0 }}>Duplicate cleanup preview</h3>
                  <p style={muted}>Cleanup will keep the oldest row in each duplicate group, soft-exclude {data.duplicate_delete_preview.delete_count} row(s), then replay scope <strong>{data.duplicate_delete_preview.recommended_replay_scope}</strong>.</p>
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
            </>
          ) : null}

          {mode === "review" ? (
            <>
              <h2>Correction and replay planning</h2>
              <article style={{ ...cardStyle, marginBottom: "1rem" }}>
                <p style={muted}>Apply endpoint: {data.correction_plan.apply_endpoint || "not enabled"}</p>
                <p style={muted}>Duplicate cleanup endpoint: {data.correction_plan.duplicate_cleanup_endpoint || "not enabled"}</p>
                <p style={muted}>Rated-match exclude endpoint: {data.correction_plan.exclude_endpoint || "not enabled"}</p>
                <p style={muted}>Duplicate no-issue endpoint: {data.correction_plan.duplicate_no_issue_endpoint || "not enabled"}</p>
                <p><strong>Editable fields:</strong> {data.correction_plan.editable_fields_planned.join(", ")}</p>
                <p><strong>Sample recompute scope:</strong> standings={String(data.correction_plan.recompute_scope_for_sample_edit.standings)}, ratings={String(data.correction_plan.recompute_scope_for_sample_edit.ratings)}</p>
                <ul style={{ paddingLeft: "1.25rem", marginBottom: 0 }}>
                  {data.correction_plan.safety_rules.map((rule) => <li key={rule}>{rule}</li>)}
                </ul>
              </article>
            </>
          ) : null}

          {mode === "edit" || mode === "exclude" ? (
            <MatchLogExclusionRecoveryPanel
              apiBase={apiBase}
              clubId={clubId}
              operation={exclusionOperation}
              onOperationChange={setExclusionOperation}
              onMutationComplete={handleMutationComplete}
            />
          ) : null}

          {mode === "edit" || mode === "bulk" || mode === "duplicates" ? (
            <MatchLogApplyPanel
              mode={mode === "edit" ? "guided" : mode}
              apiBase={apiBase}
              clubId={clubId}
              applyEnabled={Boolean(data.apply_enabled)}
              duplicateCleanupEnabled={Boolean(data.correction_plan.duplicate_cleanup_endpoint)}
              duplicatePreview={data.duplicate_delete_preview}
              duplicateGroups={data.duplicate_groups}
              matches={data.matches}
              recentOperations={data.recent_edit_operations || []}
              exclusionOperation={exclusionOperation}
              onExclusionOperationChange={setExclusionOperation}
              onMutationComplete={handleMutationComplete}
            />
          ) : null}

          {mode === "social" ? (
            <div style={{ marginTop: "1rem" }}>
              <MatchLogSocialPanel apiBase={apiBase} clubId={clubId} enabled={Boolean(data.enabled)} />
            </div>
          ) : null}

          {mode === "exclude" ? (
            <div style={{ marginTop: "1rem" }}>
              <MatchLogBulkExcludePanel
                apiBase={apiBase}
                clubId={clubId}
                enabled={Boolean(data.correction_plan.exclude_endpoint)}
                matches={data.matches}
                exclusionOperation={exclusionOperation}
                onExclusionOperationChange={setExclusionOperation}
                onMutationComplete={handleMutationComplete}
              />
            </div>
          ) : null}

          {mode === "replay" ? (
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
                onMutationComplete={handleMutationComplete}
              />
            </div>
          ) : null}

        </>
      ) : null}
    </section>
  );
}
