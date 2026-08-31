"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess } from "@/components/interaction";
import {
  blankTeam,
  createSession,
  duplicateRosterNames,
  lastTeamMatchSessionKey,
  legacyTeamMatchDraftKey,
  teamMatchDraftKey,
  teamReady,
  writeSession,
  type Team
} from "./teamMatchState";

type StoredDraft = {
  title?: string;
  teams?: Team[];
};

const card = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const input = {
  width: "100%",
  boxSizing: "border-box" as const,
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  padding: "0.55rem",
  font: "inherit"
};

const button = {
  border: 0,
  borderRadius: "999px",
  padding: "0.62rem 0.95rem",
  background: "#0f172a",
  color: "white",
  fontWeight: 800,
  cursor: "pointer"
};

const secondaryButton = {
  ...button,
  border: "1px solid #cbd5e1",
  background: "white",
  color: "#0f172a"
};

function cleanTeam(team: Team): Team {
  return {
    ...team,
    name: team.name.trim(),
    women: [team.women[0].trim(), team.women[1].trim()],
    men: [team.men[0].trim(), team.men[1].trim()]
  };
}

export default function TeamMatchGenerator({ clubId }: { clubId: string }) {
  const router = useRouter();
  const storageKey = teamMatchDraftKey(clubId);
  const [title, setTitle] = useState(`Team Match ${new Date().toISOString().slice(0, 10)}`);
  const [teams, setTeams] = useState<Team[]>([blankTeam(0), blankTeam(1)]);
  const [message, setMessage] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const [lastSessionId, setLastSessionId] = useState<string | null>(null);

  useEffect(() => {
    try {
      const current = window.localStorage.getItem(storageKey);
      const legacy = window.localStorage.getItem(legacyTeamMatchDraftKey(clubId));
      const stored = JSON.parse(current || legacy || "{}") as StoredDraft;
      if (stored.title) setTitle(stored.title);
      if (stored.teams && stored.teams.length >= 2) {
        setTeams(stored.teams.map(cleanTeam));
      }
      const recentId = window.localStorage.getItem(lastTeamMatchSessionKey(clubId));
      if (recentId) setLastSessionId(recentId);
    } catch {
      // A damaged local draft should never block a clean setup.
    }
    setHydrated(true);
  }, [clubId, storageKey]);

  useEffect(() => {
    if (!hydrated) return;
    window.localStorage.setItem(storageKey, JSON.stringify({ title, teams } satisfies StoredDraft));
  }, [hydrated, storageKey, title, teams]);

  const duplicateNames = useMemo(() => duplicateRosterNames(teams), [teams]);

  function patchTeam(teamId: string, patch: Partial<Team>): void {
    setTeams((current) => current.map((team) => (team.id === teamId ? { ...team, ...patch } : team)));
    setMessage(null);
  }

  function patchPlayer(teamId: string, group: "women" | "men", index: 0 | 1, value: string): void {
    setTeams((current) => current.map((team) => {
      if (team.id !== teamId) return team;
      const next = [...team[group]] as [string, string];
      next[index] = value;
      return { ...team, [group]: next };
    }));
    setMessage(null);
  }

  function addTeam(): void {
    setTeams((current) => [...current, blankTeam(current.length)]);
    setMessage(null);
  }

  function removeTeam(teamId: string): void {
    if (teams.length <= 2) return;
    setTeams((current) => current.filter((team) => team.id !== teamId));
    setMessage(null);
  }

  function generate(): void {
    const cleanedTeams = teams.map(cleanTeam);
    const notReady = cleanedTeams.filter((team) => !teamReady(team));
    if (notReady.length) {
      setMessage("Each team needs a name plus four distinct players: two women and two men.");
      return;
    }
    const repeated = duplicateRosterNames(cleanedTeams);
    if (repeated.length) {
      setMessage(`A player can appear on only one team. Check: ${repeated.join(", ")}.`);
      return;
    }

    const session = createSession(title, cleanedTeams);
    writeSession(clubId, session);
    window.localStorage.setItem(storageKey, JSON.stringify({ title: session.title, teams: cleanedTeams } satisfies StoredDraft));
    setLastSessionId(session.id);
    router.push(
      `/clubs/${encodeURIComponent(clubId)}/team-match-generator/sessions/${encodeURIComponent(session.id)}`
    );
  }

  function clearDraft(): void {
    window.localStorage.removeItem(storageKey);
    window.localStorage.removeItem(legacyTeamMatchDraftKey(clubId));
    setTitle(`Team Match ${new Date().toISOString().slice(0, 10)}`);
    setTeams([blankTeam(0), blankTeam(1)]);
    setMessage("Draft cleared.");
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <div style={card}>
        <label style={{ display: "grid", gap: "0.35rem", fontWeight: 700 }}>
          Session title
          <input value={title} onChange={(event) => setTitle(event.target.value)} style={input} />
        </label>
      </div>

      <div style={{ ...card, background: "#f8fafc" }}>
        <strong>How mixed lineups work</strong>
        <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>
          Enter only the four-player rosters here. After Women’s Doubles and Men’s Doubles are submitted,
          each team chooses its Mixed Doubles 1 woman and man by player name. The remaining two players
          automatically become Mixed Doubles 2.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "1rem" }}>
        {teams.map((team, teamIndex) => (
          <div key={team.id} style={card}>
            <div style={{ display: "flex", gap: "0.5rem", justifyContent: "space-between", alignItems: "center" }}>
              <input
                value={team.name}
                onChange={(event) => patchTeam(team.id, { name: event.target.value })}
                aria-label={`Team ${teamIndex + 1} name`}
                style={{ ...input, fontWeight: 800, fontSize: "1.05rem" }}
              />
              {teams.length > 2 ? (
                <button type="button" onClick={() => removeTeam(team.id)} style={secondaryButton}>
                  Remove
                </button>
              ) : null}
            </div>
            <p style={{ marginBottom: "0.35rem", fontWeight: 800 }}>Women</p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
              <input
                aria-label={`${team.name || `Team ${teamIndex + 1}`} woman 1`}
                placeholder="Woman 1 name"
                value={team.women[0]}
                onChange={(event) => patchPlayer(team.id, "women", 0, event.target.value)}
                style={input}
              />
              <input
                aria-label={`${team.name || `Team ${teamIndex + 1}`} woman 2`}
                placeholder="Woman 2 name"
                value={team.women[1]}
                onChange={(event) => patchPlayer(team.id, "women", 1, event.target.value)}
                style={input}
              />
            </div>
            <p style={{ marginBottom: "0.35rem", fontWeight: 800 }}>Men</p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
              <input
                aria-label={`${team.name || `Team ${teamIndex + 1}`} man 1`}
                placeholder="Man 1 name"
                value={team.men[0]}
                onChange={(event) => patchPlayer(team.id, "men", 0, event.target.value)}
                style={input}
              />
              <input
                aria-label={`${team.name || `Team ${teamIndex + 1}`} man 2`}
                placeholder="Man 2 name"
                value={team.men[1]}
                onChange={(event) => patchPlayer(team.id, "men", 1, event.target.value)}
                style={input}
              />
            </div>
          </div>
        ))}
      </div>

      {duplicateNames.length ? (
        <div role="alert" style={{ ...card, borderColor: "#fecaca", background: "#fef2f2", color: "#991b1b" }}>
          A player can appear on only one team. Check: {duplicateNames.join(", ")}.
        </div>
      ) : null}

      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.6rem", alignItems: "center" }}>
        <button type="button" onClick={addTeam} style={secondaryButton}>Add team</button>
        <button type="button" onClick={generate} style={button}>Generate team schedule</button>
        <ConfirmAction
          triggerLabel="Clear draft"
          title="Clear this Team Match draft?"
          description="This removes the setup roster from this browser. Previously generated sessions remain available by their session URLs."
          confirmLabel="Yes, clear draft"
          cancelLabel="No, keep draft"
          confirmationText=""
          tone="danger"
          onConfirm={async () => {
            clearDraft();
            return actionSuccess("Draft cleared", "The Team Match setup draft was removed from this browser.");
          }}
        />
        {lastSessionId ? (
          <Link
            href={`/clubs/${encodeURIComponent(clubId)}/team-match-generator/sessions/${encodeURIComponent(lastSessionId)}`}
            style={{ fontWeight: 800 }}
          >
            Resume most recent session →
          </Link>
        ) : null}
      </div>

      {message ? (
        <div
          role={message.includes("needs") || message.includes("only one team") ? "alert" : "status"}
          aria-live="polite"
          style={{ ...card, borderColor: "#bfdbfe", background: "#eff6ff", color: "#1e3a8a" }}
        >
          {message}
        </div>
      ) : null}
    </div>
  );
}
