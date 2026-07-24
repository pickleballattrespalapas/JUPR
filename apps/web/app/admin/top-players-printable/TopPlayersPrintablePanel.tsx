"use client";

import Link from "next/link";
import { useState } from "react";
import type {
  AdminLeagueManagerStatusResponse,
  AdminTopPlayersPrintableResponse
} from "@/lib/adminLeagueManagerApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  limit: number;
  status: AdminLeagueManagerStatusResponse;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

export default function TopPlayersPrintablePanel({ apiBase, clubId, limit, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [payload, setPayload] = useState<AdminTopPlayersPrintableResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const rankingsRequest = useLatestRequestGuard(accessToken, () => {
    setBusy(false); setMessage(null); setPayload(null);
  });

  async function loadRankings() {
    const generation = rankingsRequest.begin();
    if (!apiBase) { setMessage("API base URL is not configured."); return; }
    if (!accessToken) { setMessage("Sign in before loading the authenticated ranking export."); return; }
    setBusy(true);
    setMessage(null);
    setPayload(null);
    try {
      const response = await fetch(apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/top-players-printable?limit=${encodeURIComponent(String(limit))}`), {
        headers: { Authorization: `Bearer ${accessToken}` }
      });
      const body = await response.json().catch(() => null);
      if (!response.ok) throw new Error(String(body?.detail || `API error (${response.status})`));
      const next = body as AdminTopPlayersPrintableResponse;
      if (!rankingsRequest.isCurrent(generation)) return;
      setPayload(next);
      setMessage(`Loaded ${next.ranking_count} eligible player(s) for ${next.period.label}.`);
    } catch (error) {
      if (rankingsRequest.isCurrent(generation)) {
        setPayload(null);
        setMessage(error instanceof Error ? error.message : "Unable to load rankings.");
      }
    } finally {
      if (rankingsRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadRankings);

  if (!status.enabled) {
    return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2>League Manager is disabled</h2><p>{status.warnings?.[0]}</p></article>;
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <style>{`@media print { nav, header, footer, .no-print { display: none !important; } body { background: white !important; } [data-print-surface] { display: block !important; } table { font-size: 10px; } thead { display: table-header-group; } tr { break-inside: avoid; page-break-inside: avoid; } @page { size: landscape; margin: 10mm; } }`}</style>
      <article className="no-print" style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Authenticated ranking export</h2>
        <p><strong>{adminSessionLabel(session)}</strong></p>
        <p style={{ color: "#475569" }}>FastAPI filters the previous UTC calendar month, requires at least 10 scored games, excludes inactive players, and ranks the eligible set by current JUPR.</p>
        {sessionLoading ? <p>Checking admin session…</p> : null}
        {sessionMessage ? <p style={{ color: "#b91c1c" }}>{sessionMessage}</p> : null}
        {!accessToken ? <p><Link href="/admin/login">Open admin login</Link></p> : null}
        <p style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}><button type="button" onClick={loadRankings} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Refreshing…" : "Refresh rankings"}</button><button type="button" onClick={() => window.print()} disabled={busy || !payload} style={buttonStyle}>Print or save PDF</button></p>
        {message ? <p style={{ color: /unable|error|sign in/i.test(message) ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>

      {payload ? <section data-print-surface="top-active-players">
        <h1 style={{ marginBottom: "0.25rem" }}>Top active players · {payload.period.label}</h1>
        <p style={{ color: "#475569" }}>Previous calendar month · minimum {payload.minimum_games} games · UTC boundaries · {payload.ranking_count} eligible</p>
        <article style={cardStyle}>
          {payload.rankings.length ? <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead><tr><th align="left">Rank</th><th align="left">Player</th><th align="right">JUPR</th><th align="right">Record</th><th align="right">Games</th></tr></thead>
            <tbody>{payload.rankings.map((row) => <tr key={row.player_id}><td style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{row.rank}</td><td style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{row.player_name}</td><td align="right" style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{Number(row.rating_jupr).toFixed(3)}</td><td align="right" style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{row.record}</td><td align="right" style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{row.games}</td></tr>)}</tbody>
          </table> : <p>{payload.empty_message || "No eligible players."}</p>}
        </article>
      </section> : null}
    </div>
  );
}
