"use client";

import { useEffect, useMemo, useState } from "react";

export type GeneratorKind = "round_robin" | "ladder";
export type PlayFormat = "singles" | "doubles";
export type LinkedPlayerIds = Record<string, number>;

type DirectoryPlayer = {
  id: number | string;
  name: string;
  is_active?: boolean | null;
};

type DirectoryResponse = {
  players?: DirectoryPlayer[];
};

type Props = {
  apiBase: string | null;
  clubKey: string;
  generatorKind: GeneratorKind;
  playFormat: PlayFormat;
  targetCount: number;
  participantText: string;
  linkedPlayerIds: LinkedPlayerIds;
  onTargetCountChange: (count: number) => void;
  onParticipantTextChange: (text: string) => void;
  onLinkedPlayerIdsChange: (links: LinkedPlayerIds) => void;
  onInvalidate: () => void;
};

const inputStyle = {
  width: "100%",
  boxSizing: "border-box" as const,
  padding: "0.6rem",
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  font: "inherit"
};

const secondaryButton = {
  border: "1px solid #cbd5e1",
  borderRadius: "999px",
  padding: "0.4rem 0.75rem",
  background: "white",
  color: "#0f172a",
  fontWeight: 800,
  cursor: "pointer"
};

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

export function normalizeRosterName(value: string): string {
  return String(value || "").trim().replace(/\s+/g, " ").toLowerCase();
}

export function rosterNamesFromText(text: string): string[] {
  const names: string[] = [];
  const seen = new Set<string>();
  for (const rawName of String(text || "").split(/\r?\n|,/)) {
    const name = rawName.trim().replace(/\s+/g, " ");
    if (!name) continue;
    const key = normalizeRosterName(name);
    if (seen.has(key)) continue;
    seen.add(key);
    names.push(name);
  }
  return names;
}

function ladderCourtCount(playerCount: number, playFormat: PlayFormat): number {
  const minimum = playFormat === "singles" ? 2 : 4;
  if (playerCount < minimum) return 0;
  if (playFormat === "doubles") {
    let best = 0;
    for (let fives = 0; fives <= Math.floor(playerCount / 5); fives += 1) {
      const remainder = playerCount - fives * 5;
      if (remainder >= 0 && remainder % 4 === 0) {
        const groups = fives + remainder / 4;
        best = Math.max(best, groups);
      }
    }
    if (best > 0) return best;
  }
  return Math.min(
    Math.max(1, Math.ceil(playerCount / 5)),
    Math.max(1, Math.floor(playerCount / minimum))
  );
}

export function recommendedGeneratorSetup(
  generatorKind: GeneratorKind,
  playFormat: PlayFormat,
  playerCount: number
): { courtCount: number; totalRounds: number } {
  const count = Math.max(0, Math.min(40, Math.floor(Number(playerCount) || 0)));
  const minimum = playFormat === "singles" ? 2 : 4;
  if (count < minimum) {
    return { courtCount: 0, totalRounds: generatorKind === "ladder" ? 4 : 1 };
  }
  if (generatorKind === "ladder") {
    return {
      courtCount: ladderCourtCount(count, playFormat),
      totalRounds: 4
    };
  }
  const courtCount = Math.max(1, Math.floor(count / minimum));
  if (playFormat === "singles") {
    return {
      courtCount,
      totalRounds: count % 2 === 0 ? count - 1 : count
    };
  }
  const uniquePartnerPairs = (count * (count - 1)) / 2;
  const partnerPairsPerRound = courtCount * 2;
  return {
    courtCount,
    totalRounds: Math.min(50, Math.max(1, Math.ceil(uniquePartnerPairs / partnerPairsPerRound)))
  };
}

function explanation(
  generatorKind: GeneratorKind,
  playFormat: PlayFormat,
  targetCount: number,
  courtCount: number,
  totalRounds: number
): string {
  if (generatorKind === "ladder") {
    return `${targetCount} players create ${courtCount} balanced ladder court${courtCount === 1 ? "" : "s"} and ${totalRounds} result-driven rounds.`;
  }
  if (playFormat === "singles") {
    return `${targetCount} players create ${courtCount} court${courtCount === 1 ? "" : "s"} and a complete ${totalRounds}-round singles rotation.`;
  }
  return `${targetCount} players create ${courtCount} court${courtCount === 1 ? "" : "s"} and ${totalRounds} rounds for a complete partner rotation.`;
}

export default function GeneratorRosterSetup({
  apiBase,
  clubKey,
  generatorKind,
  playFormat,
  targetCount,
  participantText,
  linkedPlayerIds,
  onTargetCountChange,
  onParticipantTextChange,
  onLinkedPlayerIdsChange,
  onInvalidate
}: Props) {
  const [playerSearch, setPlayerSearch] = useState("");
  const [directoryPlayers, setDirectoryPlayers] = useState<DirectoryPlayer[]>([]);
  const [directoryError, setDirectoryError] = useState<string | null>(null);
  const [pickerMessage, setPickerMessage] = useState<string | null>(null);

  const participantNames = useMemo(
    () => rosterNamesFromText(participantText),
    [participantText]
  );
  const participantNameSet = useMemo(
    () => new Set(participantNames.map(normalizeRosterName)),
    [participantNames]
  );
  const minimumPlayers = playFormat === "singles" ? 2 : 4;
  const participantCounts = useMemo(
    () => Array.from({ length: 40 - minimumPlayers + 1 }, (_, index) => minimumPlayers + index),
    [minimumPlayers]
  );
  const setup = useMemo(
    () => recommendedGeneratorSetup(generatorKind, playFormat, targetCount),
    [generatorKind, playFormat, targetCount]
  );
  const linkedCount = participantNames.filter(
    (name) => Number(linkedPlayerIds[normalizeRosterName(name)] || 0) > 0
  ).length;
  const exactCount = participantNames.length === targetCount;
  const publicClubSlug = clubKey.replace(/_/g, "-");

  useEffect(() => {
    if (!apiBase) return;
    let cancelled = false;
    setDirectoryError(null);
    void fetch(
      apiUrl(
        apiBase,
        `/clubs/${encodeURIComponent(publicClubSlug)}/players?status=active&sort=name&limit=1000`
      ),
      { cache: "no-store" }
    )
      .then(async (response) => {
        const payload = (await response.json().catch(() => null)) as DirectoryResponse | null;
        if (!response.ok) {
          throw new Error(String((payload as { detail?: string } | null)?.detail || `API error (${response.status})`));
        }
        if (cancelled) return;
        const rows = [...(payload?.players || [])]
          .filter((player) => player && player.is_active !== false && String(player.name || "").trim())
          .sort((left, right) => String(left.name).localeCompare(String(right.name)));
        setDirectoryPlayers(rows);
      })
      .catch((error) => {
        if (cancelled) return;
        setDirectoryPlayers([]);
        setDirectoryError(error instanceof Error ? error.message : "Current-player search is unavailable.");
      });
    return () => {
      cancelled = true;
    };
  }, [apiBase, publicClubSlug]);

  const filteredPlayerOptions = useMemo(() => {
    const query = normalizeRosterName(playerSearch);
    if (query.length < 2) return [];
    return directoryPlayers
      .filter((player) => normalizeRosterName(player.name).includes(query))
      .slice(0, 10);
  }, [directoryPlayers, playerSearch]);

  function changeTargetCount(nextCount: number): void {
    onTargetCountChange(nextCount);
    onInvalidate();
    setPickerMessage(null);
  }

  function changeParticipantText(nextText: string): void {
    const nextNames = rosterNamesFromText(nextText);
    const nextNameSet = new Set(nextNames.map(normalizeRosterName));
    const nextLinks = Object.fromEntries(
      Object.entries(linkedPlayerIds).filter(([key]) => nextNameSet.has(key))
    );
    onParticipantTextChange(nextText);
    onLinkedPlayerIdsChange(nextLinks);
    onInvalidate();
    setPickerMessage(null);
  }

  function addCurrentPlayer(player: DirectoryPlayer): void {
    const name = String(player.name || "").trim().replace(/\s+/g, " ");
    const key = normalizeRosterName(name);
    if (!name || participantNameSet.has(key)) return;
    if (participantNames.length >= targetCount) {
      setPickerMessage(`The roster already has the selected ${targetCount} players. Increase the player count or remove a name first.`);
      return;
    }
    const nextNames = [...participantNames, name];
    const playerId = Number(player.id);
    onParticipantTextChange(nextNames.join("\n"));
    if (Number.isFinite(playerId) && playerId > 0) {
      onLinkedPlayerIdsChange({ ...linkedPlayerIds, [key]: playerId });
    }
    onInvalidate();
    setPlayerSearch("");
    setPickerMessage(`${name} added to the roster.`);
  }

  const countMessage = exactCount
    ? `Ready with ${targetCount} players.`
    : participantNames.length < targetCount
      ? `Add ${targetCount - participantNames.length} more player${targetCount - participantNames.length === 1 ? "" : "s"}.`
      : `Remove ${participantNames.length - targetCount} player${participantNames.length - targetCount === 1 ? "" : "s"} or increase the selected count.`;

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: "0.75rem"
        }}
      >
        <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
          Number of players
          <select
            value={targetCount}
            onChange={(event) => changeTargetCount(Number(event.target.value))}
            style={inputStyle}
          >
            {participantCounts.map((count) => (
              <option key={count} value={count}>{count}</option>
            ))}
          </select>
        </label>
        <div
          aria-live="polite"
          style={{
            border: "1px solid #bfdbfe",
            borderRadius: "10px",
            padding: "0.75rem",
            background: "#eff6ff"
          }}
        >
          <strong>Automatic setup</strong>
          <p style={{ margin: "0.3rem 0 0", color: "#334155" }}>
            {setup.courtCount} court{setup.courtCount === 1 ? "" : "s"} · {setup.totalRounds} round{setup.totalRounds === 1 ? "" : "s"}
          </p>
          <small style={{ color: "#475569" }}>
            {explanation(generatorKind, playFormat, targetCount, setup.courtCount, setup.totalRounds)}
          </small>
        </div>
      </div>

      <div style={{ display: "grid", gap: "0.5rem" }}>
        <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
          Search current players
          <input
            value={playerSearch}
            onChange={(event) => {
              setPlayerSearch(event.target.value);
              setPickerMessage(null);
            }}
            placeholder="Type at least 2 letters, then add a player"
            style={inputStyle}
          />
        </label>
        {playerSearch.trim().length < 2 ? (
          <p style={{ margin: 0, color: "#64748b", fontSize: "0.9rem" }}>
            Search the current player list, or type guest names directly in the roster box below.
          </p>
        ) : filteredPlayerOptions.length ? (
          <div style={{ display: "grid", gap: "0.4rem" }}>
            {filteredPlayerOptions.map((player) => {
              const alreadyAdded = participantNameSet.has(normalizeRosterName(player.name));
              const rosterFull = participantNames.length >= targetCount;
              return (
                <div
                  key={String(player.id)}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    gap: "0.75rem",
                    alignItems: "center",
                    border: "1px solid #cbd5e1",
                    borderRadius: "10px",
                    padding: "0.5rem",
                    background: "white"
                  }}
                >
                  <span>{player.name}</span>
                  <button
                    type="button"
                    onClick={() => addCurrentPlayer(player)}
                    disabled={alreadyAdded || rosterFull}
                    style={{
                      ...secondaryButton,
                      background: alreadyAdded || rosterFull ? "#f1f5f9" : "white",
                      cursor: alreadyAdded || rosterFull ? "default" : "pointer"
                    }}
                  >
                    {alreadyAdded ? "Added" : rosterFull ? "Roster full" : "Add"}
                  </button>
                </div>
              );
            })}
          </div>
        ) : (
          <p style={{ margin: 0, color: "#b45309", fontSize: "0.9rem" }}>
            No current players match “{playerSearch.trim()}”. Type a guest name in the roster box below.
          </p>
        )}
        {directoryError ? (
          <p style={{ margin: 0, color: "#b45309", fontSize: "0.9rem" }}>
            Current-player search is unavailable. Guest roster entry still works. {directoryError}
          </p>
        ) : null}
      </div>

      <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
        Names or roster entry ({participantNames.length} of {targetCount})
        <textarea
          value={participantText}
          onChange={(event) => changeParticipantText(event.target.value)}
          rows={Math.min(14, Math.max(7, targetCount))}
          placeholder="One player per line, in starting order"
          style={inputStyle}
        />
      </label>
      <div>
        <p style={{ margin: 0, color: exactCount ? "#166534" : "#b45309", fontWeight: 700 }}>
          {countMessage}
        </p>
        <p style={{ margin: "0.25rem 0 0", color: "#64748b", fontSize: "0.9rem" }}>
          {linkedCount} selected from the current player list · {participantNames.length - linkedCount} guest or unlinked.
          Roster line order controls starting order and bye priority.
        </p>
        {pickerMessage ? (
          <p role="status" aria-live="polite" style={{ margin: "0.25rem 0 0", color: "#1d4ed8" }}>
            {pickerMessage}
          </p>
        ) : null}
      </div>
    </section>
  );
}
