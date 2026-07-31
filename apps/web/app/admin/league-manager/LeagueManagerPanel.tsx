"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import type {
  AdminLeagueManagerLeague,
  AdminLeagueManagerListResponse,
  AdminLeagueManagerStatusResponse
} from "@/lib/adminLeagueManagerApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
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
const buttonStyle = {
  display: "inline-block",
  padding: "0.6rem 0.9rem",
  borderRadius: "999px",
  border: "1px solid #0f172a",
  background: "#0f172a",
  color: "white",
  fontWeight: 800,
  textDecoration: "none"
};
const ghostButtonStyle = {
  ...buttonStyle,
  background: "white",
  color: "#0f172a"
};

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function leagueHomeHref(league: AdminLeagueManagerLeague): string {
  const params = new URLSearchParams({
    league: league.league_name,
    mode: String(league.league_type || "Individual")
  });
  return `/admin/league-manager/league?${params.toString()}`;
}

export default function LeagueManagerPanel({ apiBase, clubId, status }: Props) {
  const router = useRouter();
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [leagues, setLeagues] = useState<AdminLeagueManagerLeague[]>([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const listRequest = useLatestRequestGuard(accessToken, clearProtectedState);

  function clearProtectedState() {
    setLeagues([]);
    setBusy(false);
    setMessage(null);
  }

  async function requestJson<T>(path: string): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before opening League Manager.");
    const response = await fetch(apiUrl(apiBase, path), {
      headers: { Authorization: `Bearer ${accessToken}` }
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
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
      setLeagues(payload.leagues || []);
      if (!(payload.leagues || []).length) setMessage("No leagues are available yet.");
    } catch (error) {
      if (listRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to load leagues.");
      }
    } finally {
      if (listRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function selectLeague(leagueName: string) {
    const league = leagues.find((item) => item.league_name === leagueName);
    if (league) router.push(leagueHomeHref(league));
  }

  useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadLeagues);

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>League Manager is unavailable</h2>
        <p style={{ color: "#475569" }}>{status.warnings?.[0] || "League Manager is currently disabled."}</p>
      </article>
    );
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
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
        <h2 style={{ marginTop: 0 }}>Create league draft</h2>
        <p style={{ color: "#475569" }}>Start a new Individual or Team league.</p>
        <Link href="/admin/league-manager/create" style={buttonStyle}>Create league draft</Link>
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Open league</h2>
        <label>
          <strong>Select league</strong><br />
          <select
            defaultValue=""
            onChange={(event) => selectLeague(event.target.value)}
            disabled={busy}
            style={inputStyle}
          >
            <option value="">Choose a league</option>
            {leagues.map((league) => (
              <option key={league.league_name} value={league.league_name}>
                {league.league_name} · {league.league_type || "Individual"} · {league.match_format === "singles" ? "Singles" : "Doubles"} · {league.status}
              </option>
            ))}
          </select>
        </label>
        <p><button type="button" onClick={() => void loadLeagues()} disabled={busy} style={ghostButtonStyle}>{busy ? "Refreshing…" : "Refresh list"}</button></p>
        {message ? <p role="status" style={{ color: /unable|error|required/i.test(message) ? "#b91c1c" : "#475569" }}>{message}</p> : null}
      </article>
    </div>
  );
}
