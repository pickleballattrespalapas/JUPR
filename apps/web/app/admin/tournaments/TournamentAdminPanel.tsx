"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import type {
  AdminTournament,
  AdminTournamentListResponse,
  AdminTournamentStatusResponse
} from "@/lib/adminTournamentApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminTournamentStatusResponse;
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
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function tournamentHomeHref(tournament: AdminTournament): string {
  const params = new URLSearchParams({ tournament: tournament.id, name: tournament.name });
  return `/admin/tournaments/tournament?${params.toString()}`;
}

export default function TournamentAdminPanel({ apiBase, clubId, status }: Props) {
  const router = useRouter();
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [includeArchived, setIncludeArchived] = useState(false);
  const [tournaments, setTournaments] = useState<AdminTournament[]>([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const listRequest = useLatestRequestGuard(`${accessToken}\u0000${includeArchived ? "archived" : "active"}`, clearProtectedState);

  function clearProtectedState() {
    setTournaments([]);
    setBusy(false);
    setMessage(null);
  }

  async function requestJson<T>(path: string): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before opening Tournament Manager.");
    const response = await fetch(apiUrl(apiBase, path), {
      headers: { Authorization: `Bearer ${accessToken}` }
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  async function loadTournaments() {
    const generation = listRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const suffix = includeArchived ? "?include_archived=true" : "";
      const payload = await requestJson<AdminTournamentListResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments${suffix}`
      );
      if (!listRequest.isCurrent(generation)) return;
      setTournaments(payload.tournaments || []);
      if (!(payload.tournaments || []).length) setMessage("No tournaments match this view.");
    } catch (error) {
      if (listRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to load tournaments.");
      }
    } finally {
      if (listRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function selectTournament(tournamentId: string) {
    const tournament = tournaments.find((row) => row.id === tournamentId);
    if (tournament) router.push(tournamentHomeHref(tournament));
  }

  useAuthenticatedAutoLoad(status.enabled ? `${accessToken}\u0000${includeArchived ? "archived" : "active"}` : "", loadTournaments);

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Tournament Manager is unavailable</h2>
        <p style={{ color: "#475569" }}>{status.warnings?.[0] || "Tournament administration is currently disabled."}</p>
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
        <h2 style={{ marginTop: 0 }}>Create tournament</h2>
        <p style={{ color: "#475569" }}>Create a draft tournament shell, then complete setup from its tournament home.</p>
        <Link href="/admin/tournaments/create" style={buttonStyle}>Create tournament</Link>
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Open tournament</h2>
        <label>
          <strong>Select tournament</strong><br />
          <select defaultValue="" onChange={(event) => selectTournament(event.target.value)} disabled={busy} style={inputStyle}>
            <option value="">Choose a tournament</option>
            {tournaments.map((tournament) => (
              <option key={tournament.id} value={tournament.id}>
                {tournament.name} · {tournament.status} · {tournament.registration_count ?? 0} registrations
              </option>
            ))}
          </select>
        </label>
        <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginTop: "0.75rem" }}>
          <input
            type="checkbox"
            checked={includeArchived}
            onChange={(event) => setIncludeArchived(event.target.checked)}
            disabled={busy}
          />
          Include archived tournaments
        </label>
        <p><button type="button" onClick={() => void loadTournaments()} disabled={busy} style={ghostButtonStyle}>{busy ? "Refreshing…" : "Refresh tournaments"}</button></p>
        {message ? <p role="status" style={{ color: /unable|error|required/i.test(message) ? "#b91c1c" : "#475569" }}>{message}</p> : null}
      </article>
    </div>
  );
}
