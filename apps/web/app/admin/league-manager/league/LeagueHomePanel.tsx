"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, type ActionCompletion } from "@/components/interaction";
import type {
  AdminLeagueManagerDetailResponse,
  AdminLeagueManagerStatusResponse,
  AdminLeagueManagerWriteResponse
} from "@/lib/adminLeagueManagerApi";
import { isTeamLeagueType, leagueRouteHref, normalizeLeagueType } from "@/lib/leagueRouteContext";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";
import LeagueManagerNav from "../LeagueManagerNav";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminLeagueManagerStatusResponse;
  initialLeagueId?: string | null;
  initialLeague: string;
  initialLeagueType?: string | null;
};
type LifecycleAction = "start" | "pause" | "resume" | "end" | "archive";

const lifecycleConfirmations: Record<LifecycleAction, string> = {
  start: "START LEAGUE",
  pause: "PAUSE LEAGUE",
  resume: "RESUME LEAGUE",
  end: "END LEAGUE",
  archive: "ARCHIVE LEAGUE"
};
const lifecycleLabels: Record<LifecycleAction, string> = {
  start: "Start league",
  pause: "Pause league",
  resume: "Resume league",
  end: "End league",
  archive: "Archive league"
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white",
  minWidth: 0
};
const inputStyle = {
  width: "100%",
  padding: "0.55rem",
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  font: "inherit"
};
const moduleStyle = {
  ...cardStyle,
  display: "grid",
  gap: "0.35rem",
  alignContent: "start",
  textDecoration: "none",
  color: "#0f172a"
};

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function lifecycleActionsFor(status: string): LifecycleAction[] {
  if (status === "draft") return ["start"];
  if (status === "active") return ["pause", "end"];
  if (status === "paused") return ["resume", "end"];
  if (status === "ended") return ["archive"];
  return [];
}

function leagueHref(path: string, leagueId: string, leagueName: string, leagueType: string): string {
  return leagueRouteHref(path, { leagueId, leagueName, leagueType });
}

function moduleDescriptionsFor(status: string) {
  if (status === "paused") {
    return {
      results: "Review standings, records, ratings, and weekly results while league play is paused.",
      settings: "Review the saved schedule, courts, rules, ratings, and defaults before resuming.",
      roster: "Review current members before resuming league play.",
      live: "Review completed rounds and recovery history; new rounds stay paused.",
      awards: "Review award readiness while league play is paused.",
      teams: "Review teams, substitutes, schedules, standings, and playoff readiness before resuming.",
      print: "Print the current schedule, standings, leaders, and roster checklist."
    };
  }
  if (status === "ended") {
    return {
      results: "Review final standings, records, ratings, and weekly results.",
      settings: "Review the final saved schedule, courts, rules, ratings, and defaults.",
      roster: "Review the final league roster.",
      live: "Review completed league-night rounds and recovery history.",
      awards: "Review and finish end-of-league awards before archiving.",
      teams: "Review final teams, schedules, standings, and playoff results.",
      print: "Print the final schedule, standings, leaders, and roster checklist."
    };
  }
  if (status === "archived") {
    return {
      results: "Review archived standings, records, ratings, and weekly results.",
      settings: "Review the archived league configuration.",
      roster: "Review the archived league roster.",
      live: "Review archived league-night rounds and recovery history.",
      awards: "Review archived award results and verification history.",
      teams: "Review archived teams, schedules, standings, and playoff results.",
      print: "Print the archived schedule, standings, leaders, and roster checklist."
    };
  }
  return {
    results: "Standings, records, ratings, and weekly results.",
    settings: "Schedule, courts, rules, ratings, and defaults.",
    roster: "Search, add, remove, and review league members.",
    live: "Run and recover league-night scoring.",
    awards: "Configure, review, mint, and archive awards.",
    teams: "Registration, teams, substitutes, schedules, standings, and playoffs.",
    print: "Open the printable schedule, standings, leaders, and roster checklist."
  };
}

export default function LeagueHomePanel({ apiBase, clubId, status, initialLeagueId, initialLeague, initialLeagueType }: Props) {
  const router = useRouter();
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [detail, setDetail] = useState<AdminLeagueManagerDetailResponse | null>(null);
  const [duplicateName, setDuplicateName] = useState(`${initialLeague} Copy`);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const detailRequest = useLatestRequestGuard(`${accessToken}\u0000${initialLeague}`, clearProtectedState);
  const actionRequest = useLatestRequestGuard(accessToken);

  function clearProtectedState() {
    actionRequest.invalidate();
    setDetail(null);
    setBusy(false);
    setMessage(null);
  }

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before opening this league.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
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
      setDuplicateName(`${payload.league.league_name} Copy`);
    } catch (error) {
      if (detailRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to load league home.");
      }
    } finally {
      if (detailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function transitionLeague(action: LifecycleAction, confirmationText: string): Promise<ActionCompletion> {
    if (!detail) throw new Error("Load the league before changing its status.");
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(initialLeague)}/lifecycle`,
        {
          method: "POST",
          body: JSON.stringify({
            action,
            confirmation_text: confirmationText,
            source: "next_selected_league_home_lifecycle"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the league status response was applied.");
      if (payload.detail) setDetail(payload.detail);
      else await loadDetail();
      setMessage(`${lifecycleLabels[action]} completed.`);
      return actionSuccess("League status updated", `${lifecycleLabels[action]} completed for ${initialLeague}.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to update league status.");
      }
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function duplicateLeague(confirmationText: string): Promise<ActionCompletion> {
    const cleanName = duplicateName.trim();
    if (!cleanName) {
      const error = new Error("New draft name is required.");
      setMessage(error.message);
      throw error;
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(initialLeague)}/duplicate`,
        {
          method: "POST",
          body: JSON.stringify({
            target_league_name: cleanName,
            confirmation_text: confirmationText,
            source: "next_selected_league_home_duplicate"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the duplicated league response was applied.");
      const createdId = String(payload.league?.league_id || payload.league?.league_name || payload.league_name || cleanName);
      const createdName = payload.league?.league_name || payload.league_name || cleanName;
      const createdType = normalizeLeagueType(payload.league?.league_type || detail?.league.league_type || initialLeagueType) || "Individual";
      router.push(leagueHref("/admin/league-manager/league", createdId, createdName, createdType));
      return actionSuccess("League duplicated", `${createdName} was created as a new draft.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to duplicate league.");
      }
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(status.enabled ? `${accessToken}\u0000${initialLeague}` : "", loadDetail);

  const leagueType = normalizeLeagueType(detail?.league.league_type || initialLeagueType) || "Individual";
  const leagueName = detail?.league.league_name || initialLeague;
  const leagueId = String(detail?.league.league_id || initialLeagueId || leagueName).trim();
  const lifecycleActions = detail?.capabilities?.lifecycle_actions || (detail ? lifecycleActionsFor(detail.league.status) : []);
  const leagueStatus = String(detail?.league.status || "").trim().toLowerCase();
  const moduleDescriptions = moduleDescriptionsFor(leagueStatus);
  const hasLifecycleIntegrityError = Boolean(
    detail?.validation?.errors?.some((item) =>
      item.toLowerCase().startsWith("league lifecycle state is inconsistent")
    )
  );
  const setupNotesAreHistorical = (leagueStatus === "ended" || leagueStatus === "archived")
    && !hasLifecycleIntegrityError;
  const setupAttentionHeading = leagueStatus === "paused"
    ? "League setup needs attention before play resumes"
    : setupNotesAreHistorical
      ? `${leagueStatus === "archived" ? "Archived" : "Saved"} setup notes`
      : "League setup needs attention";
  const setupAttentionIntro = leagueStatus === "paused"
    ? "Review these items before resuming league play."
    : setupNotesAreHistorical
      ? `These notes describe the ${leagueStatus} league's saved setup; no lifecycle action is required.`
      : "Review these items before starting or continuing league play.";

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
      <LeagueManagerNav leagueId={leagueId} leagueName={leagueName} leagueType={leagueType} />

      {busy && !detail ? <p role="status">Loading {initialLeague}…</p> : null}
      {message ? <p role="status" style={{ color: /unable|error|required/i.test(message) ? "#b91c1c" : "#166534" }}>{message}</p> : null}

      {detail ? (
        <>
          <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap", alignItems: "flex-start" }}>
              <div>
                <h2 style={{ margin: 0 }}>{leagueName}</h2>
                {detail.league.description ? <p style={{ color: "#475569", marginBottom: 0 }}>{detail.league.description}</p> : null}
              </div>
              <span style={{ border: "1px solid #93c5fd", borderRadius: "999px", padding: "0.2rem 0.55rem", background: "white", fontWeight: 700 }}>{detail.league.status}</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem", marginTop: "1rem" }}>
              <div><strong>Mode</strong><br />{leagueType}</div>
              <div><strong>Format</strong><br />{detail.league.match_format === "singles" ? "Singles" : "Doubles"}</div>
              <div><strong>Roster</strong><br />{detail.league_roster_count ?? 0}</div>
              <div><strong>Standings</strong><br />{detail.standings_count ?? 0}</div>
              <div><strong>Minimum games</strong><br />{detail.league.min_games ?? "—"}</div>
              <div><strong>K-factor</strong><br />{detail.league.k_factor ?? "—"}</div>
            </div>
          </article>

          <section aria-label={`${leagueName} modules`}>
            <h2>League tools</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
              <Link href={leagueHref("/admin/league-manager/results", leagueId, leagueName, leagueType)} style={moduleStyle}><strong>Results</strong><span style={{ color: "#475569" }}>{moduleDescriptions.results}</span></Link>
              <Link href={leagueHref("/admin/league-manager/settings", leagueId, leagueName, leagueType)} style={moduleStyle}><strong>Settings</strong><span style={{ color: "#475569" }}>{moduleDescriptions.settings}</span></Link>
              <Link href={leagueHref("/admin/league-manager/roster", leagueId, leagueName, leagueType)} style={moduleStyle}><strong>Roster</strong><span style={{ color: "#475569" }}>{moduleDescriptions.roster}</span></Link>
              <Link href={leagueHref("/admin/league-manager/live", leagueId, leagueName, leagueType)} style={moduleStyle}><strong>Live rounds</strong><span style={{ color: "#475569" }}>{moduleDescriptions.live}</span></Link>
              <Link href={leagueHref("/admin/league-manager/awards", leagueId, leagueName, leagueType)} style={moduleStyle}><strong>Awards</strong><span style={{ color: "#475569" }}>{moduleDescriptions.awards}</span></Link>
              {isTeamLeagueType(leagueType) ? <Link href={leagueHref("/admin/league-manager/teams", leagueId, leagueName, leagueType)} style={moduleStyle}><strong>Team league</strong><span style={{ color: "#475569" }}>{moduleDescriptions.teams}</span></Link> : null}
              <Link href={leagueHref("/admin/league-manager/print", leagueId, leagueName, leagueType)} style={moduleStyle}><strong>League night printout</strong><span style={{ color: "#475569" }}>{moduleDescriptions.print}</span></Link>
            </div>
          </section>

          <article style={{ ...cardStyle, background: "#fff7ed", borderColor: "#fed7aa" }}>
            <h2 style={{ marginTop: 0 }}>League status</h2>
            {lifecycleActions.length ? (
              <p style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
                {lifecycleActions.map((action) => (
                  <ConfirmAction
                    key={action}
                    triggerLabel={lifecycleLabels[action]}
                    title={`${lifecycleLabels[action]}?`}
                    description={`${lifecycleLabels[action]} changes ${leagueName} from its current ${detail.league.status} state.`}
                    confirmLabel={`Yes, ${lifecycleLabels[action].toLowerCase()}`}
                    confirmationText={lifecycleConfirmations[action]}
                    tone={action === "end" || action === "archive" ? "danger" : "default"}
                    busy={busy}
                    onConfirm={(confirmationText) => transitionLeague(action, confirmationText)}
                  />
                ))}
              </p>
            ) : <p style={{ color: "#64748b" }}>No additional lifecycle actions are available.</p>}
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Duplicate as a new draft</h2>
            <p style={{ color: "#475569" }}>Copies league configuration without copying roster, results, dates, or issued awards.</p>
            <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) auto", gap: "0.75rem", alignItems: "end" }}>
              <label><strong>New draft name</strong><br /><input value={duplicateName} onChange={(event) => setDuplicateName(event.target.value)} maxLength={120} style={inputStyle} /></label>
              <ConfirmAction
                triggerLabel={busy ? "Working…" : "Duplicate league"}
                title="Duplicate this league as a draft?"
                description={`Copy ${leagueName} configuration into ${duplicateName.trim() || "the named draft"}.`}
                confirmLabel="Yes, duplicate league"
                confirmationText="DUPLICATE LEAGUE"
                disabled={!duplicateName.trim()}
                busy={busy}
                onConfirm={duplicateLeague}
              />
            </div>
          </article>

          {detail.validation && !detail.validation.valid ? (
            <article
              role={setupNotesAreHistorical ? undefined : "alert"}
              style={{
                ...cardStyle,
                background: setupNotesAreHistorical ? "#f8fafc" : leagueStatus === "paused" ? "#fffbeb" : "#fef2f2",
                borderColor: setupNotesAreHistorical ? "#cbd5e1" : leagueStatus === "paused" ? "#fde68a" : "#fecaca"
              }}
            >
              <h2 style={{ marginTop: 0 }}>{setupAttentionHeading}</h2>
              <p style={{ color: setupNotesAreHistorical ? "#475569" : leagueStatus === "paused" ? "#92400e" : "#7f1d1d" }}>{setupAttentionIntro}</p>
              <ul style={{ color: setupNotesAreHistorical ? "#475569" : leagueStatus === "paused" ? "#92400e" : "#b91c1c" }}>{detail.validation.errors.map((item) => <li key={item}>{item}</li>)}</ul>
            </article>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
