"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type {
  AdminLeagueManagerDetailResponse,
  AdminLeagueManagerRosterRow,
  AdminLeagueManagerStatusResponse
} from "@/lib/adminLeagueManagerApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminLeagueManagerStatusResponse;
  initialLeague: string;
};
type BatchResponse = {
  ok: boolean;
  committed?: boolean;
  updated_count?: number;
  detail?: AdminLeagueManagerDetailResponse;
};
type RosterFilter = "all" | "in_league" | "not_in_league" | "inactive";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function operationKey(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return `league-roster:${crypto.randomUUID()}`;
  return `league-roster:${Date.now()}:${Math.random().toString(16).slice(2)}`;
}

export default function LeagueRosterPanel({ apiBase, clubId, status, initialLeague }: Props) {
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [detail, setDetail] = useState<AdminLeagueManagerDetailResponse | null>(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<RosterFilter>("in_league");
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [action, setAction] = useState<"activate" | "deactivate">("activate");
  const [startingRating, setStartingRating] = useState("3.5");
  const [idempotencyKey, setIdempotencyKey] = useState(operationKey);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const detailRequest = useLatestRequestGuard(`${accessToken}\u0000${initialLeague}`, clearProtectedState);
  const actionRequest = useLatestRequestGuard(accessToken);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before editing a league roster.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  function clearProtectedState() {
    actionRequest.invalidate();
    setDetail(null);
    setSelectedIds([]);
    setFilter("in_league");
    setBusy(false);
    setMessage(null);
  }

  async function loadDetail() {
    const generation = detailRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerDetailResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(initialLeague)}`
      );
      if (!detailRequest.isCurrent(generation)) return;
      setDetail(payload);
      setSelectedIds([]);
      setFilter("in_league");
      setIdempotencyKey(operationKey());
    } catch (error) {
      if (detailRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to load the roster.");
      }
    } finally {
      if (detailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function resetOperation(nextAction = action) {
    setAction(nextAction);
    setSelectedIds([]);
    setIdempotencyKey(operationKey());
    setMessage(null);
    setFilter(nextAction === "activate" ? "not_in_league" : "in_league");
  }

  async function saveBatch(confirmationText: string) {
    if (!selectedIds.length) {
      setMessage("Select at least one player.");
      return;
    }
    const rating = action === "activate" ? Number(startingRating) : null;
    if (action === "activate" && (rating === null || !Number.isFinite(rating) || !((rating >= 1 && rating <= 7) || (rating >= 400 && rating <= 2800)))) {
      setMessage("Starting rating must be JUPR 1.0–7.0 or Elo 400–2800.");
      return;
    }

    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<BatchResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(initialLeague)}/roster/batch`,
        {
          method: "POST",
          body: JSON.stringify({
            action,
            player_ids: selectedIds,
            starting_rating: rating,
            idempotency_key: idempotencyKey,
            confirmation_text: confirmationText,
            source: "next_selected_league_roster_page"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      if (payload.detail) setDetail(payload.detail);
      else await loadDetail();
      const count = payload.updated_count ?? selectedIds.length;
      setMessage(`${action === "activate" ? "Added" : "Removed"} ${count} player${count === 1 ? "" : "s"}.`);
      setSelectedIds([]);
      setIdempotencyKey(operationKey());
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(`${error instanceof Error ? error.message : "Unable to update the roster."} The same request key is retained for a safe retry.`);
      }
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(status.enabled ? `${accessToken}\u0000${initialLeague}` : "", loadDetail);

  const roster = detail?.roster || [];
  const visibleRows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return roster.filter((row) => {
      if (needle && !`${row.player_name} ${row.player_id}`.toLowerCase().includes(needle)) return false;
      if (filter === "in_league" && !row.in_league) return false;
      if (filter === "not_in_league" && row.in_league) return false;
      if (filter === "inactive" && row.player_active !== false) return false;
      return true;
    });
  }, [filter, query, roster]);
  const visibleSelectable = visibleRows.filter((row) => row.player_active !== false).map((row) => row.player_id);
  const allVisibleSelected = Boolean(visibleSelectable.length && visibleSelectable.every((id) => selectedIds.includes(id)));
  const rosterMutable = detail?.capabilities?.roster_mutable !== false;

  if (!status.enabled) {
    return <article style={{ ...cardStyle, background: "#f8fafc" }}>League Manager is currently unavailable.</article>;
  }

  if (sessionLoading) return <p role="status">Checking admin access…</p>;

  if (!accessToken) {
    return (
      <article style={{ ...cardStyle, background: "#fffbeb", borderColor: "#fde68a" }}>
        <h2 style={{ marginTop: 0 }}>Admin sign-in required</h2>
        <p><Link href="/admin/login">Open admin login</Link></p>
      </article>
    );
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      {busy && !detail ? <p role="status">Loading {initialLeague} roster…</p> : null}
      {detail ? (
        <article style={cardStyle}>
          {!rosterMutable ? <p style={{ color: "#92400e" }}>This roster is read-only after league close.</p> : null}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
            <label><strong>Search players</strong><br /><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Name or player ID" style={inputStyle} /></label>
            <label><strong>Show</strong><br /><select value={filter} onChange={(event) => setFilter(event.target.value as RosterFilter)} style={inputStyle}><option value="in_league">In this league</option><option value="not_in_league">Add players</option><option value="all">All club players</option><option value="inactive">Inactive club players</option></select></label>
            <label><strong>Action</strong><br /><select value={action} onChange={(event) => resetOperation(event.target.value as "activate" | "deactivate")} disabled={!rosterMutable} style={inputStyle}><option value="activate">Add players</option><option value="deactivate">Remove players</option></select></label>
            {action === "activate" ? <label><strong>Starting JUPR or Elo</strong><br /><input value={startingRating} onChange={(event) => { setStartingRating(event.target.value); setIdempotencyKey(operationKey()); }} disabled={!rosterMutable} style={inputStyle} /></label> : null}
          </div>

          <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            <button type="button" onClick={() => setSelectedIds(allVisibleSelected ? selectedIds.filter((id) => !visibleSelectable.includes(id)) : Array.from(new Set([...selectedIds, ...visibleSelectable])))} disabled={!visibleSelectable.length || !rosterMutable} style={ghostButtonStyle}>{allVisibleSelected ? "Clear visible" : "Select visible"}</button>
            <button type="button" onClick={() => setSelectedIds([])} disabled={!selectedIds.length} style={ghostButtonStyle}>Clear selection</button>
          </p>

          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "650px" }}>
              <thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Select</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Player</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Membership</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Rating</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Record</th></tr></thead>
              <tbody>{visibleRows.map((row: AdminLeagueManagerRosterRow) => {
                const selected = selectedIds.includes(row.player_id);
                return <tr key={row.player_id}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><input aria-label={`Select ${row.player_name}`} type="checkbox" checked={selected} disabled={!rosterMutable || row.player_active === false} onChange={(event) => { setSelectedIds((current) => event.target.checked ? [...current, row.player_id] : current.filter((id) => id !== row.player_id)); setIdempotencyKey(operationKey()); }} /></td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.player_name}<br /><small>#{row.player_id}{row.player_active === false ? " · inactive" : ""}</small></td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.in_league ? "In league" : "Not in league"}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.rating_jupr == null ? "—" : Number(row.rating_jupr).toFixed(2)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}-{row.losses ?? 0}</td></tr>;
              })}</tbody>
            </table>
          </div>
          {!visibleRows.length ? <p style={{ color: "#64748b" }}>No players match these filters.</p> : null}
          <p>
            <ConfirmAction
              triggerLabel={busy ? "Saving…" : action === "activate" ? (selectedIds.length === 1 ? "Add Player" : "Add Players") : (selectedIds.length === 1 ? "Remove Player" : "Remove Players")}
              title={`${action === "activate" ? "Add" : "Remove"} ${selectedIds.length === 1 ? "this player" : "these players"}?`}
              description={`Apply this single atomic roster change to ${selectedIds.length} selected player${selectedIds.length === 1 ? "" : "s"}.`}
              confirmLabel={action === "activate" ? "Yes, add players" : "Yes, remove players"}
              confirmationText="SAVE LEAGUE ROSTER BATCH"
              tone={action === "deactivate" ? "danger" : "default"}
              disabled={!rosterMutable || !selectedIds.length}
              busy={busy}
              onConfirm={saveBatch}
            />
          </p>
        </article>
      ) : null}
      {message ? <p role="status" style={{ color: /unable|error|required|stale|retry/i.test(message) ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </div>
  );
}
