"use client";

import { useMemo, useState } from "react";
import type { PublicPlayer } from "@/lib/api";

type PublicLiveCreatorProps = {
  apiBase: string | null;
  clubSlug: string;
  players?: PublicPlayer[];
};

type LiveMode = "quick" | "club_social";
type EventType = "round_robin" | "league_ladder";

const defaultNames = "Amy\nBrooke\nChris\nDana";
const participantCounts = Array.from({ length: 17 }, (_, index) => index + 4);

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function namesFromText(text: string): string[] {
  const seen = new Set<string>();
  const names: string[] = [];
  for (const rawName of text.split(/\r?\n|,/)) {
    const name = rawName.trim();
    if (!name) continue;
    const key = name.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    names.push(name);
  }
  return names;
}

function appendNames(currentText: string, newNames: string[]): string {
  const existing = namesFromText(currentText);
  const seen = new Set(existing.map((name) => name.toLowerCase()));
  const merged = [...existing];
  for (const name of newNames) {
    const clean = name.trim();
    if (!clean || seen.has(clean.toLowerCase())) continue;
    seen.add(clean.toLowerCase());
    merged.push(clean);
  }
  return merged.join("\n");
}

export default function PublicLiveCreator({ apiBase, clubSlug, players = [] }: PublicLiveCreatorProps) {
  const [liveMode, setLiveMode] = useState<LiveMode>("quick");
  const [eventType, setEventType] = useState<EventType>("round_robin");
  const [targetCount, setTargetCount] = useState(8);
  const [eventName, setEventName] = useState("Saturday Event");
  const [participantText, setParticipantText] = useState(defaultNames);
  const [playerSearch, setPlayerSearch] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const participantNames = useMemo(() => namesFromText(participantText), [participantText]);
  const participantCount = participantNames.length;
  const participantNameSet = useMemo(() => new Set(participantNames.map((name) => name.toLowerCase())), [participantNames]);
  const playerOptions = useMemo(
    () => [...players].filter((player) => player.is_active !== false).sort((a, b) => a.name.localeCompare(b.name)),
    [players]
  );
  const filteredPlayerOptions = useMemo(() => {
    const query = playerSearch.trim().toLowerCase();
    if (query.length < 2) return [];
    return playerOptions
      .filter((player) => player.name.toLowerCase().includes(query))
      .slice(0, 10);
  }, [playerOptions, playerSearch]);
  const countMessage = participantCount < 4
    ? "Add at least 4 unique players."
    : participantCount > 20
      ? "Public quick sessions support up to 20 players."
      : participantCount !== targetCount
        ? `Current roster has ${participantCount}; selected count is ${targetCount}.`
        : `Ready with ${participantCount} players.`;
  const canSubmit = liveMode === "quick" && eventType === "round_robin" && participantCount >= 4 && participantCount <= 20;

  function addPlayerName(name: string) {
    setParticipantText((current) => appendNames(current, [name]));
    setPlayerSearch("");
  }

  async function createSession() {
    if (!apiBase) {
      setError("The public API base URL is not configured for this deployment.");
      return;
    }
    if (liveMode !== "quick") {
      setError("Club Social setup is still handled in the Streamlit/JUPR Live admin workflow.");
      return;
    }
    if (eventType !== "round_robin") {
      setError("League / Ladder setup is still handled in the Streamlit/JUPR Live admin workflow.");
      return;
    }
    if (!canSubmit) {
      setError(countMessage);
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const response = await fetch(apiUrl(apiBase, `/clubs/${clubSlug}/live-sessions`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          event_name: eventName,
          event_type: "round_robin",
          participant_names: participantNames
        })
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(String(payload?.detail || `API error (${response.status})`));
      }
      const sessionKey = String(payload?.session?.session_key || "");
      const editToken = String(payload?.edit_token || "");
      if (!sessionKey || !editToken) {
        throw new Error("The API did not return a live session edit link.");
      }
      window.location.href = `/clubs/${clubSlug}/live/${sessionKey}?edit=${encodeURIComponent(editToken)}`;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to create live session.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section style={{ border: "1px solid #bfdbfe", borderRadius: "14px", padding: "1rem", background: "#eff6ff", marginBottom: "1rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap", alignItems: "flex-start" }}>
        <div>
          <h2 style={{ margin: "0 0 0.35rem", fontSize: "1.35rem" }}>🔴 JUPR Live</h2>
          <p style={{ color: "#334155", marginTop: 0 }}>
            Run a lightweight JUPR Live quick session with session-only scoring. No persistence to rated match history and no official rating updates are used in this mode.
          </p>
        </div>
        <span style={{ border: "1px solid #bfdbfe", borderRadius: "999px", padding: "0.25rem 0.75rem", background: "white", color: "#1d4ed8", fontWeight: 800, fontSize: "0.85rem" }}>Public</span>
      </div>

      <div style={{ display: "grid", gap: "1rem" }}>
        <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
          <legend style={{ fontWeight: 800, marginBottom: "0.5rem" }}>Live mode</legend>
          <label style={{ marginRight: "1rem" }}>
            <input type="radio" checked={liveMode === "quick"} onChange={() => setLiveMode("quick")} /> Quick Session
          </label>
          <label>
            <input type="radio" checked={liveMode === "club_social"} onChange={() => setLiveMode("club_social")} /> Club Social
          </label>
        </fieldset>

        <p style={{ color: "#334155", margin: 0 }}>
          {liveMode === "quick"
            ? "Quick Session creates a public round-robin scoreboard that expires automatically."
            : "Club Social remains in the Streamlit/JUPR Live admin workflow until organizer permissions are ported."}
        </p>

        <div style={{ height: "0.7rem", borderRadius: "999px", background: "white", border: "1px solid #dbeafe" }} />

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
          <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
            <legend style={{ fontWeight: 800, marginBottom: "0.5rem" }}>Event type</legend>
            <label style={{ marginRight: "1rem" }}>
              <input type="radio" checked={eventType === "round_robin"} onChange={() => setEventType("round_robin")} /> Round Robin
            </label>
            <label>
              <input type="radio" checked={eventType === "league_ladder"} onChange={() => setEventType("league_ladder")} /> League / Ladder
            </label>
          </fieldset>

          <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
            Count
            <select value={targetCount} onChange={(event) => setTargetCount(Number(event.target.value))} style={{ padding: "0.6rem", borderRadius: "8px", border: "1px solid #cbd5e1", font: "inherit" }}>
              {participantCounts.map((count) => <option key={count} value={count}>{count}</option>)}
            </select>
          </label>
        </div>

        <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
          Event name
          <input
            value={eventName}
            onChange={(event) => setEventName(event.target.value)}
            style={{ padding: "0.6rem", borderRadius: "8px", border: "1px solid #cbd5e1", font: "inherit" }}
          />
        </label>

        <div style={{ display: "grid", gap: "0.5rem" }}>
          <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
            Search current players
            <input
              value={playerSearch}
              onChange={(event) => setPlayerSearch(event.target.value)}
              placeholder="Type at least 2 letters, then add a player"
              style={{ padding: "0.6rem", borderRadius: "8px", border: "1px solid #cbd5e1", font: "inherit" }}
            />
          </label>
          {playerSearch.trim().length < 2 ? (
            <p style={{ margin: 0, color: "#64748b", fontSize: "0.9rem" }}>Type at least 2 characters to search the current player list. Guest names can still be typed directly below.</p>
          ) : filteredPlayerOptions.length ? (
            <div style={{ display: "grid", gap: "0.4rem" }}>
              {filteredPlayerOptions.map((player) => {
                const alreadyAdded = participantNameSet.has(player.name.toLowerCase());
                return (
                  <div key={String(player.id)} style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "center", border: "1px solid #cbd5e1", borderRadius: "10px", padding: "0.5rem", background: "white" }}>
                    <span>{player.name}</span>
                    <button
                      type="button"
                      onClick={() => addPlayerName(player.name)}
                      disabled={alreadyAdded}
                      style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: alreadyAdded ? "#f1f5f9" : "white", color: "#0f172a", fontWeight: 800, cursor: alreadyAdded ? "default" : "pointer" }}
                    >
                      {alreadyAdded ? "Added" : "Add"}
                    </button>
                  </div>
                );
              })}
            </div>
          ) : (
            <p style={{ margin: 0, color: "#b45309", fontSize: "0.9rem" }}>No current players match “{playerSearch.trim()}”. Type a guest name in the roster box below.</p>
          )}
        </div>

        <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
          Names or roster entry ({participantCount})
          <textarea
            value={participantText}
            onChange={(event) => setParticipantText(event.target.value)}
            rows={8}
            style={{ padding: "0.6rem", borderRadius: "8px", border: "1px solid #cbd5e1", font: "inherit" }}
          />
        </label>
        <p style={{ margin: 0, color: participantCount >= 4 && participantCount <= 20 ? "#166534" : "#b45309" }}>{countMessage}</p>

        <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
          <button
            type="button"
            onClick={createSession}
            disabled={submitting || !canSubmit}
            style={{ border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: canSubmit ? "#2563eb" : "#94a3b8", color: "white", fontWeight: 800, cursor: submitting || !canSubmit ? "default" : "pointer" }}
          >
            {submitting ? "Creating…" : "Create event"}
          </button>
          <button
            type="button"
            onClick={() => {
              setEventName("Saturday Event");
              setParticipantText(defaultNames);
              setTargetCount(8);
              setLiveMode("quick");
              setEventType("round_robin");
              setPlayerSearch("");
              setError(null);
            }}
            style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.65rem 1rem", background: "white", color: "#0f172a", fontWeight: 800, cursor: "pointer" }}
          >
            Reset
          </button>
        </div>
        {error ? <p style={{ color: "#b91c1c", margin: 0 }}>{error}</p> : null}
      </div>
    </section>
  );
}
