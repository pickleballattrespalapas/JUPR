"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type {
  AdminLeagueManagerDetailResponse,
  AdminLeagueManagerListResponse,
  AdminLeagueManagerRosterRow,
  AdminLeagueManagerStatusResponse
} from "@/lib/adminLeagueManagerApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = { apiBase: string | null; clubId: string; status: AdminLeagueManagerStatusResponse };
type BatchResponse = { ok: boolean; committed?: boolean; updated_count?: number; detail?: AdminLeagueManagerDetailResponse };
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

export default function LeagueRosterPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [leagues, setLeagues] = useState<string[]>([]);
  const [leagueName, setLeagueName] = useState("");
  const [detail, setDetail] = useState<AdminLeagueManagerDetailResponse | null>(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<RosterFilter>("all");
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [action, setAction] = useState<"activate" | "deactivate">("activate");
  const [startingRating, setStartingRating] = useState("3.5");
  const [idempotencyKey, setIdempotencyKey] = useState(operationKey);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const listRequest = useLatestRequestGuard(accessToken, clearProtectedState);
  const detailRequest = useLatestRequestGuard(accessToken);
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
    detailRequest.invalidate();
    actionRequest.invalidate();
    setLeagues([]);
    setLeagueName("");
    setDetail(null);
    setSelectedIds([]);
    setBusy(false);
    setMessage(null);
  }

  async function loadLeagues() {
    const generation = listRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerListResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`
      );
      if (!listRequest.isCurrent(generation)) return;
      const names = (payload.leagues || []).map((league) => league.league_name);
      setLeagues(names);
      if (leagueName && names.includes(leagueName)) await loadDetail(leagueName);
      else if (leagueName) {
        setLeagueName("");
        setDetail(null);
      }
      setMessage(names.length ? `Loaded ${names.length} league(s).` : "No leagues are available.");
    } catch (error) {
      if (listRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load leagues.");
    } finally {
      if (listRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function loadDetail(selectedLeague: string) {
    const generation = detailRequest.begin();
    setLeagueName(selectedLeague);
    setDetail(null);
    setSelectedIds([]);
    setIdempotencyKey(operationKey());
    setMessage(null);
    if (!selectedLeague) return;
    setBusy(true);
    try {
      const payload = await requestJson<AdminLeagueManagerDetailResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(selectedLeague)}`
      );
      if (!detailRequest.isCurrent(generation)) return;
      setDetail(payload);
    } catch (error) {
      if (detailRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load the roster.");
    } finally {
      if (detailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function resetOperation(nextAction = action) {
    setAction(nextAction);
    setSelectedIds([]);
    setIdempotencyKey(operationKey());
    setMessage(null);
  }

  async function saveBatch(confirmationText: string) {
    if (!leagueName || !selectedIds.length) {
      setMessage("Select a league and at least one player.");
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
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(leagueName)}/roster/batch`,
        {
          method: "POST",
          body: JSON.stringify({
            action,
            player_ids: selectedIds,
            starting_rating: rating,
            idempotency_key: idempotencyKey,
            confirmation_text: confirmationText,
            source: "next_league_manager_bulk_roster_page"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      if (payload.detail) setDetail(payload.detail);
      else await loadDetail(leagueName);
      setMessage(`${action === "activate" ? "Added" : "Removed"} ${payload.updated_count ?? selectedIds.length} player(s).`);
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

  useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadLeagues);

  const roster = detail?.roster;
  const visibleRows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return (roster || []).filter((row) => {
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

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Choose a league</h2>
        <p style={{ color: "#475569" }}>Signed in as {adminSessionLabel(session)}. Search and apply one reviewed action to up to 500 players at once.</p>
        {sessionLoading ? <p>Checking session…</p> : null}
        {sessionMessage ? <p style={{ color: "#b91c1c" }}>{sessionMessage}</p> : null}
        {!accessToken && !sessionLoading ? <p><Link href="/admin/login">Open admin login</Link></p> : null}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label><strong>League</strong><br /><select value={leagueName} onChange={(event) => void loadDetail(event.target.value)} disabled={busy || !accessToken} style={inputStyle}><option value="">Select a league</option>{leagues.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
          <button type="button" onClick={() => void loadLeagues()} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Working…" : "Refresh leagues"}</button>
        </div>
      </article>

      {detail ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>{leagueName} roster</h2>
          {!rosterMutable ? <p style={{ color: "#92400e" }}>This roster is read-only after league close.</p> : null}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
            <label><strong>Search players</strong><br /><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Name or player ID" style={inputStyle} /></label>
            <label><strong>Show</strong><br /><select value={filter} onChange={(event) => setFilter(event.target.value as RosterFilter)} style={inputStyle}><option value="all">All club players</option><option value="in_league">In this league</option><option value="not_in_league">Not in this league</option><option value="inactive">Inactive club players</option></select></label>
            <label><strong>Bulk action</strong><br /><select value={action} onChange={(event) => resetOperation(event.target.value as "activate" | "deactivate")} disabled={!rosterMutable} style={inputStyle}><option value="activate">Add / reactivate</option><option value="deactivate">Deactivate</option></select></label>
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
          <p><ConfirmAction triggerLabel={busy ? "Saving…" : `${action === "activate" ? "Add" : "Deactivate"} ${selectedIds.length} selected`} title={`${action === "activate" ? "Add" : "Deactivate"} these league players?`} description={`Apply this single atomic roster change to ${selectedIds.length} selected player(s). A failed request can be retried safely.`} confirmLabel="Yes, save roster batch" confirmationText="SAVE LEAGUE ROSTER BATCH" tone={action === "deactivate" ? "danger" : "default"} disabled={!accessToken || !rosterMutable || !selectedIds.length} busy={busy} onConfirm={saveBatch} /></p>
        </article>
      ) : null}

      {message ? <p role="status" style={{ color: /unable|error|required|sign in|retry/i.test(message) ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </div>
  );
}
