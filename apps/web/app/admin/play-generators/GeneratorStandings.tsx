"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import PlayGeneratorStandingsTable, {
  PlayGeneratorStanding,
  standingsSortLabel
} from "@/components/PlayGeneratorStandingsTable";
import { useAdminSession } from "@/lib/useAdminSession";

type StandingsSort = "wins" | "points" | "differential";

type Session = {
  session_key: string;
  title: string;
  status: string;
  generator_kind: string;
  play_format: string;
  current_round_number?: number | null;
  total_rounds?: number | null;
  standings_sort?: StandingsSort;
  standings?: PlayGeneratorStanding[];
  event: {
    standingsSort?: StandingsSort;
    currentRoundNumber?: number;
    rounds?: Array<{ number: number; status: string }>;
  };
};

type Props = { apiBase: string | null; clubId: string; sessionKey: string };

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const linkButton = {
  display: "inline-flex",
  alignItems: "center",
  minHeight: "38px",
  padding: "0.45rem 0.75rem",
  border: "1px solid #cbd5e1",
  borderRadius: "999px",
  color: "#0f172a",
  fontWeight: 800,
  textDecoration: "none"
};

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

export default function AdminGeneratorStandings({ apiBase, clubId, sessionKey }: Props) {
  const { accessToken } = useAdminSession();
  const [session, setSession] = useState<Session | null>(null);
  const [message, setMessage] = useState("Loading standings…");

  useEffect(() => {
    if (!apiBase || !accessToken) return;
    let active = true;
    void fetch(
      apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}`),
      { headers: { Authorization: `Bearer ${accessToken}` }, cache: "no-store" }
    )
      .then(async (response) => {
        const payload = await response.json().catch(() => null);
        if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
        if (payload?.session?.generator_kind !== "round_robin") {
          throw new Error("Standings are available for Round-Robin Generator sessions.");
        }
        if (active) setSession(payload.session as Session);
      })
      .catch((error) => {
        if (active) setMessage(error instanceof Error ? error.message : "Unable to load standings.");
      });
    return () => { active = false; };
  }, [accessToken, apiBase, clubId, sessionKey]);

  if (!session) {
    return <article style={cardStyle}><h1>Round-Robin standings</h1><p>{message}</p></article>;
  }

  const currentRound = Number(session.current_round_number || session.event.currentRoundNumber || 1);
  const sortMode = session.standings_sort || session.event.standingsSort || "wins";
  const visibleRounds = (session.event.rounds || []).filter((row) => row.number <= currentRound);

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <p style={{ margin: "0 0 0.4rem" }}>
          <Link href="/admin/round-robin-generator">← Round-Robin Generator</Link>
        </p>
        <h1 style={{ margin: "0 0 0.35rem" }}>{session.title} standings</h1>
        <p style={{ margin: 0, color: "#475569" }}>
          {session.play_format === "singles" ? "Singles" : "Doubles"} · {standingsSortLabel(sortMode)} · {session.status}
        </p>
      </article>

      <nav aria-label="Round-Robin session navigation" style={{ ...cardStyle, display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <Link
          href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`}
          style={linkButton}
        >
          Current round
        </Link>
        {visibleRounds.map((row) => (
          <Link
            key={row.number}
            href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${row.number}`}
            style={linkButton}
          >
            Round {row.number}
          </Link>
        ))}
      </nav>

      <PlayGeneratorStandingsTable rows={session.standings || []} sortMode={sortMode} />
    </div>
  );
}
