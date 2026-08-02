"use client";

import { useEffect, useMemo, useState } from "react";
import type { PublicPlayer } from "@/lib/api";

type PublicLiveCreatorProps = {
  apiBase: string | null;
  clubSlug: string;
  players?: PublicPlayer[];
};

type LiveMode = "quick" | "club_social";
type EventType = "round_robin" | "league_ladder";
type PendingCreatePayload = Record<string, unknown> & { idempotency_key: string };

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

function newOperationKey(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return crypto.randomUUID();
  return `public-live-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function hasExactLeagueCourtFit(count: number): boolean {
  for (let fives = 0; fives * 5 <= count; fives += 1) {
    if ((count - fives * 5) % 4 === 0) return true;
  }
  return false;
}

export default function PublicLiveCreator({ apiBase, clubSlug, players = [] }: PublicLiveCreatorProps) {
  const [liveMode, setLiveMode] = useState<LiveMode>("quick");
  const [eventType, setEventType] = useState<EventType>("round_robin");
  const [targetCount, setTargetCount] = useState(8);
  const [eventName, setEventName] = useState("Saturday Event");
  const [participantText, setParticipantText] = useState(defaultNames);
  const [playerSearch, setPlayerSearch] = useState("");
  const [totalRounds, setTotalRounds] = useState(3);
  const [courtSizesText, setCourtSizesText] = useState("");
  const [hostName, setHostName] = useState("");
  const [skillLevels, setSkillLevels] = useState<string[]>(["All"]);
  const [participantPlayerIds, setParticipantPlayerIds] = useState<Record<string, number>>({});
  const [createOperationKey, setCreateOperationKey] = useState("");
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
  const leagueFit = eventType !== "league_ladder" || hasExactLeagueCourtFit(participantCount);
  const currentFormCanSubmit = participantCount >= 4 && participantCount <= 20 && participantCount === targetCount && leagueFit && (liveMode !== "club_social" || Boolean(hostName.trim()));
  const canSubmit = Boolean(createOperationKey) || currentFormCanSubmit;
  const createOperationStorageKey = `jupr-live-create-operation:${clubSlug}`;

  useEffect(() => {
    const raw = sessionStorage.getItem(createOperationStorageKey) || "";
    if (!raw) return;
    try {
      const pending = JSON.parse(raw) as { idempotency_key?: unknown };
      if (typeof pending.idempotency_key === "string") setCreateOperationKey(pending.idempotency_key);
    } catch {
      setCreateOperationKey(raw);
    }
  }, [createOperationStorageKey]);

  function addPlayerName(player: PublicPlayer) {
    setParticipantText((current) => appendNames(current, [player.name]));
    const numericId = Number(player.id);
    if (Number.isInteger(numericId)) {
      setParticipantPlayerIds((current) => ({ ...current, [player.name.toLowerCase()]: numericId }));
    }
    setPlayerSearch("");
  }

  async function createSession() {
    if (!apiBase) {
      setError("The public API base URL is not configured for this deployment.");
      return;
    }
    if (!canSubmit) {
      setError(countMessage);
      return;
    }
    setSubmitting(true);
    setError(null);
    const rawPending = sessionStorage.getItem(createOperationStorageKey) || "";
    let requestPayload: PendingCreatePayload | null = null;
    if (rawPending) {
      try {
        const parsed = JSON.parse(rawPending) as PendingCreatePayload;
        if (typeof parsed.idempotency_key === "string" && parsed.idempotency_key) requestPayload = parsed;
      } catch {
        // Migrate an older key-only browser record below.
      }
    }
    const operationKey = requestPayload?.idempotency_key || createOperationKey || rawPending || newOperationKey();
    const selectedPlayerLinks: Record<string, number> = {};
    if (liveMode === "club_social") {
      for (const name of participantNames) {
        const playerId = participantPlayerIds[name.toLowerCase()];
        if (Number.isInteger(playerId)) selectedPlayerLinks[name] = playerId;
      }
    }
    requestPayload ||= {
      event_name: eventName,
      event_type: eventType,
      participant_names: participantNames,
      live_mode: liveMode,
      total_rounds: totalRounds,
      court_sizes: courtSizesText.split(",").map((value) => Number(value.trim())).filter((value) => Number.isInteger(value) && value > 0),
      host_name: liveMode === "club_social" ? hostName.trim() : null,
      skill_levels: liveMode === "club_social" ? skillLevels : [],
      participant_player_ids: selectedPlayerLinks,
      idempotency_key: operationKey
    };
    sessionStorage.setItem(createOperationStorageKey, JSON.stringify(requestPayload));
    setCreateOperationKey(operationKey);
    try {
      const response = await fetch(apiUrl(apiBase, `/clubs/${clubSlug}/live-sessions`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestPayload)
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        if ([400, 403, 409, 422].includes(response.status)) {
          sessionStorage.removeItem(createOperationStorageKey);
          setCreateOperationKey("");
        }
        throw new Error(String(payload?.detail || `API error (${response.status})`));
      }
      const sessionKey = String(payload?.session?.session_key || "");
      const editToken = String(payload?.edit_token || "");
      if (!sessionKey || !editToken) {
        throw new Error("The API did not return a live session edit link.");
      }
      sessionStorage.removeItem(createOperationStorageKey);
      setCreateOperationKey("");
      window.location.href = `/clubs/${clubSlug}/live/${sessionKey}#edit=${encodeURIComponent(editToken)}`;
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
          <h2 style={{ margin: "0 0 0.35rem", fontSize: "1.35rem" }}>Round-Robin and Ladder Generators</h2>
          <p style={{ color: "#334155", marginTop: 0 }}>
            Create a durable Round-Robin or Ladder session. Quick sessions stay unrated; Club Social sends completed results to moderation without changing ratings.
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
            ? "Quick Session creates an unrated public scoreboard that survives refreshes and expires automatically."
            : "Club Social persists the scoreboard, then submits completed unrated results to the staff moderation queue."}
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

        {eventType === "league_ladder" ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
            <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
              League rounds
              <input type="number" min={1} max={20} value={totalRounds} onChange={(event) => setTotalRounds(Number(event.target.value))} style={{ padding: "0.6rem", borderRadius: "8px", border: "1px solid #cbd5e1", font: "inherit" }} />
            </label>
            <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
              Court sizes (optional)
              <input value={courtSizesText} onChange={(event) => setCourtSizesText(event.target.value)} placeholder="4,4 or 5,5" style={{ padding: "0.6rem", borderRadius: "8px", border: "1px solid #cbd5e1", font: "inherit" }} />
            </label>
          </div>
        ) : null}

        {liveMode === "club_social" ? (
          <div style={{ display: "grid", gap: "0.75rem" }}>
            <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
              Host / Submitter Name
              <input value={hostName} onChange={(event) => setHostName(event.target.value)} maxLength={160} style={{ padding: "0.6rem", borderRadius: "8px", border: "1px solid #cbd5e1", font: "inherit" }} />
            </label>
            <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
              <legend style={{ fontWeight: 800, marginBottom: "0.35rem" }}>Skill tags</legend>
              {["All", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"].map((level) => (
                <label key={level} style={{ marginRight: "0.85rem" }}>
                  <input
                    type="checkbox"
                    checked={skillLevels.includes(level)}
                    onChange={(event) => setSkillLevels((current) => {
                      if (!event.target.checked) return current.filter((value) => value !== level);
                      if (level === "All") return ["All"];
                      return [...new Set([...current.filter((value) => value !== "All"), level])];
                    })}
                  /> {level}
                </label>
              ))}
            </fieldset>
          </div>
        ) : null}

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
                const nameKey = player.name.toLowerCase();
                const namePresent = participantNameSet.has(nameKey);
                const alreadyLinked = participantPlayerIds[nameKey] === Number(player.id);
                return (
                  <div key={String(player.id)} style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "center", border: "1px solid #cbd5e1", borderRadius: "10px", padding: "0.5rem", background: "white" }}>
                    <span>{player.name}</span>
                    <button
                      type="button"
                      onClick={() => addPlayerName(player)}
                      disabled={alreadyLinked}
                      style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: alreadyLinked ? "#f1f5f9" : "white", color: "#0f172a", fontWeight: 800, cursor: alreadyLinked ? "default" : "pointer" }}
                    >
                      {alreadyLinked ? "Linked" : namePresent ? "Link profile" : "Add"}
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
        <p style={{ margin: 0, color: participantCount >= 4 && participantCount <= 20 && leagueFit ? "#166534" : "#b45309" }}>
          {!leagueFit ? "League / Ladder needs an exact combination of 4-player and 5-player courts." : countMessage}
        </p>
        {liveMode === "club_social" ? <p style={{ margin: 0, color: "#475569" }}>Use current-player search for rated members so Club Social links the selected profile. Manually typed names are treated as guests and near-duplicates are rejected before creation.</p> : null}

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
            disabled={Boolean(createOperationKey)}
            onClick={() => {
              if (createOperationKey) return;
              setEventName("Saturday Event");
              setParticipantText(defaultNames);
              setTargetCount(8);
              setLiveMode("quick");
              setEventType("round_robin");
              setTotalRounds(3);
              setCourtSizesText("");
              setHostName("");
              setSkillLevels(["All"]);
              setParticipantPlayerIds({});
              sessionStorage.removeItem(createOperationStorageKey);
              setCreateOperationKey("");
              setPlayerSearch("");
              setError(null);
            }}
            style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.65rem 1rem", background: "white", color: "#0f172a", fontWeight: 800, cursor: createOperationKey ? "not-allowed" : "pointer", opacity: createOperationKey ? 0.55 : 1 }}
          >
            {createOperationKey ? "Reset locked during recovery" : "Reset"}
          </button>
        </div>
        {createOperationKey ? (
          <p style={{ color: "#92400e", margin: 0 }}>
            <strong>Unresolved create retained.</strong> Create will retry the exact preserved request before accepting new inputs. Operation <code>{createOperationKey}</code>.
          </p>
        ) : null}
        {error ? <p style={{ color: "#b91c1c", margin: 0 }}>{error}</p> : null}
      </div>
    </section>
  );
}
