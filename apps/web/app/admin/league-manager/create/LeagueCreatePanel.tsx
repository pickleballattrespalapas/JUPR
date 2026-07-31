"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type {
  AdminLeagueManagerStatusResponse,
  AdminLeagueManagerWriteResponse
} from "@/lib/adminLeagueManagerApi";
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

function leagueHomeHref(leagueName: string, leagueType: string): string {
  const params = new URLSearchParams({ league: leagueName, mode: leagueType });
  return `/admin/league-manager/league?${params.toString()}`;
}

export default function LeagueCreatePanel({ apiBase, clubId, status }: Props) {
  const router = useRouter();
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const actionRequest = useLatestRequestGuard(accessToken);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [leagueType, setLeagueType] = useState<"Individual" | "Team">("Individual");
  const [matchFormat, setMatchFormat] = useState<"doubles" | "singles">("doubles");
  const [minGames, setMinGames] = useState("6");
  const [kFactor, setKFactor] = useState("32");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

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

  async function createLeague(confirmationText: string) {
    const cleanName = name.trim();
    const minimum = Number(minGames);
    const factor = Number(kFactor);
    if (!cleanName) {
      setMessage("League name is required.");
      return;
    }
    if (!Number.isInteger(minimum) || minimum < 0 || minimum > 1000) {
      setMessage("Minimum games must be a whole number from 0 to 1000.");
      return;
    }
    if (!Number.isInteger(factor) || factor < 1 || factor > 128) {
      setMessage("K-factor must be a whole number from 1 to 128.");
      return;
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
            match_format: matchFormat,
            description,
            min_games: minimum,
            k_factor: factor,
            confirmation_text: confirmationText,
            source: "next_league_manager_create_page"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      const createdName = payload.league?.league_name || payload.league_name || cleanName;
      const createdType = String(payload.league?.league_type || leagueType);
      router.push(leagueHomeHref(createdName, createdType));
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to create league.");
      }
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
        <label><strong>League mode</strong><br /><select value={leagueType} onChange={(event) => setLeagueType(event.target.value as "Individual" | "Team")} style={inputStyle}><option value="Individual">Individual</option><option value="Team">Team</option></select></label>
        <label><strong>League format</strong><br /><select value={matchFormat} onChange={(event) => setMatchFormat(event.target.value as "doubles" | "singles")} style={inputStyle}><option value="doubles">Doubles</option><option value="singles">Singles</option></select></label>
        <label><strong>Minimum games</strong><br /><input type="number" value={minGames} onChange={(event) => setMinGames(event.target.value)} min={0} max={1000} style={inputStyle} /></label>
        <label><strong>K-factor</strong><br /><input type="number" value={kFactor} onChange={(event) => setKFactor(event.target.value)} min={1} max={128} style={inputStyle} /></label>
      </div>
      <label><strong>Description</strong><br /><textarea value={description} onChange={(event) => setDescription(event.target.value)} maxLength={2000} rows={4} style={inputStyle} /></label>
      <p style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "center" }}>
        <ConfirmAction
          triggerLabel={busy ? "Creating…" : "Create league"}
          title="Create this league draft?"
          description={`Create ${name.trim() || "this league"} as an inactive ${leagueType.toLowerCase()} ${matchFormat} league.`}
          confirmLabel="Yes, create league"
          confirmationText="CREATE LEAGUE"
          disabled={!name.trim()}
          busy={busy}
          onConfirm={createLeague}
        />
        <Link href="/admin/league-manager">Cancel</Link>
      </p>
      {message ? <p role="alert" style={{ color: "#b91c1c" }}>{message}</p> : null}
    </article>
  );
}
