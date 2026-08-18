"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, type ActionCompletion } from "@/components/interaction";
import type {
  AdminLeagueManagerStatusResponse,
  AdminLeagueManagerWriteResponse
} from "@/lib/adminLeagueManagerApi";
import { leagueRouteHref, normalizeLeagueType } from "@/lib/leagueRouteContext";
import { useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminLeagueManagerStatusResponse;
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};
const inputStyle = {
  width: "100%",
  padding: "0.55rem",
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  font: "inherit"
};

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function leagueHomeHref(leagueId: string, leagueName: string, leagueType: string): string {
  return leagueRouteHref("/admin/league-manager/league", { leagueId, leagueName, leagueType });
}

export default function LeagueCreatePanel({ apiBase, clubId, status }: Props) {
  const router = useRouter();
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const actionRequest = useLatestRequestGuard(accessToken);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [leagueType, setLeagueType] = useState<"Individual" | "Team">("Individual");
  const [matchFormat, setMatchFormat] = useState<"doubles" | "singles">("doubles");
  const [leagueFormat, setLeagueFormat] = useState<"ladder" | "round_robin" | "rotating_partner" | "fixed_team" | "flex_challenge">("ladder");
  const [sessionMode, setSessionMode] = useState<"scheduled_rounds" | "live_court_board" | "self_scheduled">("scheduled_rounds");
  const [minGames, setMinGames] = useState("6");
  const [kFactor, setKFactor] = useState("32");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const createMatchFormat = leagueType === "Team" ? "doubles" : matchFormat;

  function changeLeagueType(nextLeagueType: "Individual" | "Team") {
    setLeagueType(nextLeagueType);
    if (nextLeagueType === "Team") {
      setMatchFormat("doubles");
      setLeagueFormat("fixed_team");
    } else if (leagueFormat === "fixed_team") {
      setLeagueFormat("ladder");
    }
  }

  function changeSeasonFormat(nextFormat: typeof leagueFormat) {
    setLeagueFormat(nextFormat);
    if (nextFormat === "flex_challenge") setSessionMode("self_scheduled");
    else if (sessionMode === "self_scheduled") setSessionMode("scheduled_rounds");
  }

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before creating a league.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  async function createLeague(confirmationText: string): Promise<ActionCompletion> {
    const cleanName = name.trim();
    const minimum = Number(minGames);
    const factor = Number(kFactor);
    if (!cleanName) {
      const error = new Error("League name is required.");
      setMessage(error.message);
      throw error;
    }
    if (!Number.isInteger(minimum) || minimum < 0 || minimum > 1000) {
      const error = new Error("Minimum games must be a whole number from 0 to 1000.");
      setMessage(error.message);
      throw error;
    }
    if (!Number.isInteger(factor) || factor < 1 || factor > 128) {
      const error = new Error("K-factor must be a whole number from 1 to 128.");
      setMessage(error.message);
      throw error;
    }
    if (leagueType === "Team" && matchFormat !== "doubles") {
      const error = new Error("Team leagues must use Doubles.");
      setMessage(error.message);
      throw error;
    }
    if (leagueType === "Individual" && leagueFormat === "ladder" && sessionMode === "self_scheduled") {
      const error = new Error("Ladder leagues need scheduled rounds or a live court board.");
      setMessage(error.message);
      throw error;
    }
    if (leagueType === "Individual" && leagueFormat === "flex_challenge" && sessionMode !== "self_scheduled") {
      const error = new Error("Flex challenge leagues use self-scheduled play.");
      setMessage(error.message);
      throw error;
    }

    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`,
        {
          method: "POST",
          body: JSON.stringify({
            league_name: cleanName,
            league_type: leagueType,
            match_format: createMatchFormat,
            league_format: leagueType === "Team" ? "fixed_team" : leagueFormat,
            session_mode: sessionMode,
            description,
            min_games: minimum,
            k_factor: factor,
            confirmation_text: confirmationText,
            source: "next_league_manager_create_page"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the created league response was applied.");
      const createdName = payload.league?.league_name || payload.league_name || cleanName;
      const createdId = String(payload.league?.league_id || createdName).trim();
      const createdType = normalizeLeagueType(payload.league?.league_type || leagueType) || leagueType;
      router.push(leagueHomeHref(createdId, createdName, createdType));
      return actionSuccess("League created", `${createdName} was created as an inactive league draft.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to create league.");
      }
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

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
    <article style={cardStyle}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
        <label><strong>League name</strong><br /><input value={name} onChange={(event) => setName(event.target.value)} maxLength={120} style={inputStyle} /></label>
        <label><strong>League mode</strong><br /><select value={leagueType} onChange={(event) => changeLeagueType(event.target.value as "Individual" | "Team")} style={inputStyle}><option value="Individual">Individual</option><option value="Team">Team</option></select></label>
        <label>
          <strong>Match modality</strong><br />
          <select
            value={createMatchFormat}
            onChange={(event) => setMatchFormat(event.target.value as "doubles" | "singles")}
            disabled={leagueType === "Team"}
            aria-describedby={leagueType === "Team" ? "team-league-format-note" : undefined}
            style={inputStyle}
          >
            <option value="doubles">Doubles</option>
            {leagueType === "Individual" ? <option value="singles">Singles</option> : null}
          </select>
          {leagueType === "Team" ? <small id="team-league-format-note" style={{ color: "#64748b" }}>Team leagues use Doubles.</small> : null}
        </label>
        <label>
          <strong>Season format</strong><br />
          <select value={leagueType === "Team" ? "fixed_team" : leagueFormat} onChange={(event) => changeSeasonFormat(event.target.value as typeof leagueFormat)} disabled={leagueType === "Team"} style={inputStyle}>
            <option value="ladder">Ladder league</option>
            <option value="round_robin">Season round robin</option>
            <option value="rotating_partner">Rotating-partner individual league</option>
            <option value="flex_challenge">Flex challenge league</option>
            {leagueType === "Team" ? <option value="fixed_team">Fixed-team league</option> : null}
          </select>
          {leagueType === "Team" ? <small style={{ color: "#64748b" }}>Team leagues use the fixed-team format.</small> : null}
        </label>
        <label>
          <strong>Session operation</strong><br />
          <select value={sessionMode} onChange={(event) => setSessionMode(event.target.value as typeof sessionMode)} disabled={leagueType === "Individual" && leagueFormat === "flex_challenge"} style={inputStyle}>
            <option value="scheduled_rounds">Scheduled rounds</option>
            <option value="live_court_board">Live court board</option>
            <option value="self_scheduled">Self-scheduled flex play</option>
          </select>
        </label>
        <label><strong>Minimum games</strong><br /><input type="number" value={minGames} onChange={(event) => setMinGames(event.target.value)} min={0} max={1000} style={inputStyle} /></label>
        <label><strong>K-factor</strong><br /><input type="number" value={kFactor} onChange={(event) => setKFactor(event.target.value)} min={1} max={128} style={inputStyle} /></label>
      </div>
      <label><strong>Description</strong><br /><textarea value={description} onChange={(event) => setDescription(event.target.value)} maxLength={2000} rows={4} style={inputStyle} /></label>
      <p style={{ color: "#475569" }}>Season format determines competition flow; match modality determines singles, doubles, or team scoring. Detailed match-series and court rules are set in the draft settings editor.</p>
      <p style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "center" }}>
        <ConfirmAction
          triggerLabel={busy ? "Creating…" : "Create league"}
          title="Create this league draft?"
          description={`Create ${name.trim() || "this league"} as an inactive ${leagueType.toLowerCase()} ${createMatchFormat} league.`}
          confirmLabel="Yes, create league"
          confirmationText="CREATE LEAGUE"
          disabled={!name.trim() || (leagueType === "Team" && matchFormat !== "doubles")}
          busy={busy}
          onConfirm={createLeague}
        />
        <Link href="/admin/league-manager">Cancel</Link>
      </p>
      {message ? <p role="alert" style={{ color: "#b91c1c" }}>{message}</p> : null}
    </article>
  );
}
