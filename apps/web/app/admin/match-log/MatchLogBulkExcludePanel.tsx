"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type { AdminMatchLogMatch } from "@/lib/adminMatchLogApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type MatchLogBulkExcludePanelProps = {
  apiBase: string | null;
  clubId: string;
  enabled: boolean;
  matches: AdminMatchLogMatch[];
};

type ExcludeResult = {
  ok?: boolean;
  mode?: string;
  deleted_count?: number;
  deleted_ids?: number[];
  affected_player_ids?: number[];
  replay_result?: {
    matches_scanned_total?: number;
    matches_rewritten?: number;
    league_ratings_rows?: number;
    skipped_incomplete?: number;
  } | null;
  warning?: string | null;
  replay_error?: string | null;
  recovery_required?: boolean;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const secondaryButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function dateLabel(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 19);
  return date.toISOString().replace("T", " ").slice(0, 16);
}

function playerNames(players: Array<{ id: number | null; name: string }>): string {
  return players.map((player) => player.name || (player.id ? `#${player.id}` : "—")).join(" / ");
}

function resultSummary(result: ExcludeResult | null): string | null {
  if (!result) return null;
  const deleted = result.deleted_count ?? 0;
  if (result.recovery_required || result.mode === "matches_excluded_recovery_required") {
    const detail = result.replay_error ? ` ${result.replay_error}` : "";
    return `Excluded ${deleted} match(es), but recovery did not complete. Do not retry the exclusion; run Replay History immediately.${detail}`;
  }
  if (result.replay_error) return `Excluded ${deleted} match(es), but Replay ALL failed. Run Replay History immediately. ${result.replay_error}`;
  if (result.replay_result) return `Excluded ${deleted} match(es) and Replay ALL completed.`;
  return `Excluded ${deleted} match(es).`;
}

export default function MatchLogBulkExcludePanel({ apiBase, clubId, enabled, matches }: MatchLogBulkExcludePanelProps) {
  const router = useRouter();
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [note, setNote] = useState("");
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [result, setResult] = useState<ExcludeResult | null>(null);

  const selectableMatches = matches.filter((match) => match.id != null).slice(0, 100);
  const selectedSet = new Set(selectedIds);

  function toggleMatch(matchId: number) {
    setSelectedIds((current) => current.includes(matchId) ? current.filter((id) => id !== matchId) : [...current, matchId].sort((a, b) => a - b));
  }

  async function submitExclude(confirmationText: string) {
    setMessage(null);
    setResult(null);
    if (!apiBase) {
      setMessage("API base URL is not configured.");
      return;
    }
    if (!accessToken) {
      setMessage("Sign in at /admin/login before excluding rated matches.");
      return;
    }
    if (!selectedIds.length) {
      setMessage("Select at least one match to exclude.");
      return;
    }
    setPending(true);
    try {
      const response = await fetch(apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/match-log/exclude`), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`
        },
        body: JSON.stringify({
          match_ids: selectedIds,
          confirmation_text: confirmationText,
          note,
          source: "next_match_log_bulk_exclude_panel"
        })
      });
      const payload = await response.json().catch(() => null) as ExcludeResult | { detail?: unknown } | null;
      if (!response.ok) throw new Error(String((payload as { detail?: unknown } | null)?.detail || `API error (${response.status})`));
      const typed = payload as ExcludeResult;
      setResult(typed);
      setMessage(resultSummary(typed));
      setSelectedIds([]);
      router.refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to exclude selected matches.");
    } finally {
      setPending(false);
    }
  }

  if (!enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Bulk exclude is disabled</h2>
        <p style={{ color: "#475569" }}>
          Excluding rated matches requires both <code>JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY=1</code> and <code>JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE=1</code> on FastAPI, plus Supabase JWT delete-match authorization.
        </p>
      </article>
    );
  }

  return (
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Exclude rated matches</h2>
      <p style={{ color: "#475569" }}>
        Streamlit parity for the rated-match delete flow. This soft-excludes selected rated matches, writes audit attribution, recomputes player activity, then runs Replay ALL through FastAPI.
      </p>
      <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
        <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
        <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
          {accessToken ? "Ready to send authorized exclude requests." : sessionLoading ? "Checking admin session…" : "Sign in before excluding rated matches."}
        </p>
        {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
        {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
      </div>

      <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "12px", background: "white", marginBottom: "0.75rem" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "820px" }}>
          <thead>
            <tr style={{ textAlign: "left", background: "#f8fafc" }}>
              <th style={{ padding: "0.55rem" }}>Exclude</th>
              <th style={{ padding: "0.55rem" }}>ID</th>
              <th style={{ padding: "0.55rem" }}>Date</th>
              <th style={{ padding: "0.55rem" }}>League / Week</th>
              <th style={{ padding: "0.55rem" }}>Team 1</th>
              <th style={{ padding: "0.55rem" }}>Score</th>
              <th style={{ padding: "0.55rem" }}>Team 2</th>
            </tr>
          </thead>
          <tbody>
            {selectableMatches.map((match) => {
              const id = Number(match.id);
              return (
                <tr key={id}>
                  <td style={{ padding: "0.55rem" }}><input type="checkbox" checked={selectedSet.has(id)} onChange={() => toggleMatch(id)} /></td>
                  <td style={{ padding: "0.55rem" }}>#{id}</td>
                  <td style={{ padding: "0.55rem" }}>{dateLabel(match.date)}</td>
                  <td style={{ padding: "0.55rem" }}>{match.league || "—"}<br /><span style={{ color: "#64748b" }}>{match.week_tag || "—"}</span></td>
                  <td style={{ padding: "0.55rem" }}>{playerNames(match.team1)}</td>
                  <td style={{ padding: "0.55rem" }}><strong>{match.score?.display || "—"}</strong></td>
                  <td style={{ padding: "0.55rem" }}>{playerNames(match.team2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {!selectableMatches.length ? <p style={{ color: "#475569" }}>No matches are available in the current filtered view.</p> : null}
      <p style={{ color: "#64748b" }}>Selected: {selectedIds.length} / {selectableMatches.length}. The table uses the current filtered Match Log view and caps the bulk action at 100 visible rows.</p>
      <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <button type="button" onClick={() => setSelectedIds(selectableMatches.map((match) => Number(match.id)).filter(Number.isFinite))} disabled={pending || !selectableMatches.length} style={secondaryButtonStyle}>Select all visible</button>
        <button type="button" onClick={() => setSelectedIds([])} disabled={pending || !selectedIds.length} style={secondaryButtonStyle}>Clear selection</button>
      </p>
      <label><strong>Delete/exclude note</strong><br /><input value={note} onChange={(event) => setNote(event.target.value)} style={inputStyle} placeholder="Why these rated matches are being excluded" /></label>
      <p>
        <ConfirmAction
          triggerLabel={`Exclude ${selectedIds.length || "selected"} rated match(es)`}
          title={`Exclude ${selectedIds.length || "the selected"} rated match(es)?`}
          description={<>This will soft-exclude the selected rated matches, write an audit record, recompute player activity, and run Replay ALL. This changes official rating history.</>}
          confirmLabel="Yes, exclude and replay"
          confirmationText="DELETE"
          tone="danger"
          disabled={pending || !accessToken || !selectedIds.length}
          busy={pending}
          onConfirm={submitExclude}
        />
      </p>
      {message ? <p style={{ color: result?.ok && !result?.replay_error ? "#166534" : "#b91c1c" }}>{message}</p> : null}
      {result?.warning ? <p style={{ color: "#92400e" }}>{result.warning}</p> : null}
      {result?.replay_result ? (
        <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", margin: 0 }}>
          <div><dt style={{ fontWeight: 700 }}>Matches scanned</dt><dd style={{ margin: 0 }}>{result.replay_result.matches_scanned_total ?? "—"}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>Matches rewritten</dt><dd style={{ margin: 0 }}>{result.replay_result.matches_rewritten ?? "—"}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>League ratings rows</dt><dd style={{ margin: 0 }}>{result.replay_result.league_ratings_rows ?? "—"}</dd></div>
          <div><dt style={{ fontWeight: 700 }}>Skipped incomplete</dt><dd style={{ margin: 0 }}>{result.replay_result.skipped_incomplete ?? "—"}</dd></div>
        </dl>
      ) : null}
    </article>
  );
}
