"use client";

import { fetchAllPublicPlayers, matchesPlayerSearch, playerSearchKey } from "@/lib/playerSearch";

import CourtReserveRosterImport from "@/components/CourtReserveRosterImport";
import GeneratorRatingReview from "@/components/GeneratorRatingReview";

import GeneratorPlayerCreate from "@/app/admin/play-generators/submissions/GeneratorPlayerCreate";

import { useCallback, useEffect, useMemo, useState } from "react";

export type GeneratorKind = "round_robin" | "ladder";
export type PlayFormat = "singles" | "doubles" | "doubles_singles";
export type LinkedPlayerIds = Record<string, number>;

export type GeneratorSetup = {
  courtCount: number;
  totalRounds: number;
  doublesCourtCount: number;
  singlesCourtCount: number;
  scheduledPlayers: number;
  byesPerRound: number;
};

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
  creationClubId?: string;
  creationAccessToken?: string;
  creationDraftKey?: string;
  disabled?: boolean;
  onCreationPendingChange?: (pending: boolean) => void;
  ratedSetup?: boolean;
  ratingReviewAccessToken?: string;
  onRatingReviewReadyChange?: (ready: boolean) => void;
  generatorKind: GeneratorKind;
  playFormat: PlayFormat;
  targetCount: number;
  fixedPlayerCount?: number;
  moneyballOnly?: boolean;
  participantText: string;
  linkedPlayerIds: LinkedPlayerIds;
  doublesCourtCount: number;
  singlesCourtCount: number;
  onTargetCountChange: (count: number) => void;
  onDoublesCourtCountChange: (count: number) => void;
  onSinglesCourtCountChange: (count: number) => void;
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

function gcd(left: number, right: number): number {
  let a = Math.abs(Math.floor(left));
  let b = Math.abs(Math.floor(right));
  while (b) {
    [a, b] = [b, a % b];
  }
  return a || 1;
}

function lcm(left: number, right: number): number {
  return Math.abs(left * right) / gcd(left, right);
}

function lexicographicallyGreater(left: number[], right: number[]): boolean {
  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    if (left[index] === right[index]) continue;
    return left[index] > right[index];
  }
  return left.length > right.length;
}

export function recommendedMixedCourtSetup(playerCount: number): {
  doublesCourtCount: number;
  singlesCourtCount: number;
} {
  const count = Math.max(0, Math.min(40, Math.floor(Number(playerCount) || 0)));
  let best: { doublesCourtCount: number; singlesCourtCount: number; score: number[] } | null = null;
  for (let doubles = 1; doubles <= Math.floor((count - 2) / 4); doubles += 1) {
    for (let singles = 1; singles <= Math.floor((count - doubles * 4) / 2); singles += 1) {
      const scheduled = doubles * 4 + singles * 2;
      const score = [
        scheduled,
        -Math.abs(doubles * 4 - singles * 2),
        doubles,
        -(doubles + singles)
      ];
      if (!best || lexicographicallyGreater(score, best.score)) {
        best = { doublesCourtCount: doubles, singlesCourtCount: singles, score };
      }
    }
  }
  return best
    ? { doublesCourtCount: best.doublesCourtCount, singlesCourtCount: best.singlesCourtCount }
    : { doublesCourtCount: 1, singlesCourtCount: 1 };
}

function mixedRoundCount(playerCount: number, doublesCourts: number, singlesCourts: number): number {
  const count = Math.max(1, playerCount);
  const doublesSlots = Math.max(0, doublesCourts) * 4;
  const singlesSlots = Math.max(0, singlesCourts) * 2;
  const scheduled = doublesSlots + singlesSlots;
  if (scheduled < 6 || scheduled > count) return 1;
  const byes = count - scheduled;
  const roleCycle = [
    count / gcd(count, doublesSlots),
    count / gcd(count, singlesSlots),
    byes ? count / gcd(count, byes) : 1
  ].reduce((cycle, value) => lcm(cycle, value), 1);
  const relationshipPairsPerRound = doublesCourts * 2 + singlesCourts;
  const coverageRounds = Math.ceil(((count * (count - 1)) / 2) / relationshipPairsPerRound);
  return Math.min(50, Math.max(roleCycle, Math.ceil(coverageRounds / roleCycle) * roleCycle));
}

export function recommendedGeneratorSetup(
  generatorKind: GeneratorKind,
  playFormat: PlayFormat,
  playerCount: number,
  doublesCourtCount?: number,
  singlesCourtCount?: number,
  moneyball = false
): GeneratorSetup {
  const count = Math.max(0, Math.min(40, Math.floor(Number(playerCount) || 0)));
  const minimum = playFormat === "singles" ? 2 : playFormat === "doubles_singles" ? 6 : 4;
  if (count < minimum) {
    return {
      courtCount: 0,
      totalRounds: generatorKind === "ladder" ? 4 : 1,
      doublesCourtCount: 0,
      singlesCourtCount: 0,
      scheduledPlayers: 0,
      byesPerRound: count
    };
  }
  if (generatorKind === "ladder") {
    const courtCount = ladderCourtCount(count, playFormat);
    return {
      courtCount,
      totalRounds: 4,
      doublesCourtCount: playFormat === "doubles" ? courtCount : 0,
      singlesCourtCount: playFormat === "singles" ? courtCount : 0,
      scheduledPlayers: count,
      byesPerRound: 0
    };
  }
  if (playFormat === "doubles_singles") {
    const recommendation = recommendedMixedCourtSetup(count);
    let doubles = Math.max(1, Math.floor(Number(doublesCourtCount) || recommendation.doublesCourtCount));
    let singles = Math.max(1, Math.floor(Number(singlesCourtCount) || recommendation.singlesCourtCount));
    if (doubles * 4 + singles * 2 > count) {
      doubles = recommendation.doublesCourtCount;
      singles = recommendation.singlesCourtCount;
    }
    const scheduledPlayers = doubles * 4 + singles * 2;
    return {
      courtCount: doubles + singles,
      totalRounds: mixedRoundCount(count, doubles, singles),
      doublesCourtCount: doubles,
      singlesCourtCount: singles,
      scheduledPlayers,
      byesPerRound: count - scheduledPlayers
    };
  }
  const courtCount = Math.max(1, Math.floor(count / minimum));
  if (playFormat === "singles") {
    return {
      courtCount,
      totalRounds: count % 2 === 0 ? count - 1 : count,
      doublesCourtCount: 0,
      singlesCourtCount: courtCount,
      scheduledPlayers: courtCount * 2,
      byesPerRound: count - courtCount * 2
    };
  }
  const uniquePartnerPairs = (count * (count - 1)) / 2;
  const partnerPairsPerRound = courtCount * 2;
  return {
    courtCount,
    totalRounds: moneyball && count === 6 ? 6 : Math.min(50, Math.max(1, Math.ceil(uniquePartnerPairs / partnerPairsPerRound))),
    doublesCourtCount: courtCount,
    singlesCourtCount: 0,
    scheduledPlayers: courtCount * 4,
    byesPerRound: count - courtCount * 4
  };
}

function explanation(
  generatorKind: GeneratorKind,
  playFormat: PlayFormat,
  courtCount: number,
  totalRounds: number
): string {
  const roundLabel = `${totalRounds} round${totalRounds === 1 ? "" : "s"}`;
  if (generatorKind === "ladder") {
    return `This setup uses ${courtCount} court${courtCount === 1 ? "" : "s"} for ${totalRounds} ladder round${totalRounds === 1 ? "" : "s"}. Each new round is based on the last result.`;
  }
  if (playFormat === "singles") {
    return `This setup uses ${courtCount} court${courtCount === 1 ? "" : "s"} for a ${totalRounds}-round singles rotation.`;
  }
  if (playFormat === "doubles_singles") {
    return `This setup uses ${courtCount} courts for ${roundLabel} of singles and doubles, balancing partners, opponents, and byes.`;
  }
  return `This setup uses ${courtCount} court${courtCount === 1 ? "" : "s"} for ${roundLabel}, rotating partners throughout.`;
}

export default function GeneratorRosterSetup({
  apiBase,
  clubKey,
  creationClubId,
  creationAccessToken,
  creationDraftKey,
  disabled = false,
  onCreationPendingChange,
  ratedSetup = false,
  ratingReviewAccessToken,
  onRatingReviewReadyChange,
  generatorKind,
  playFormat,
  targetCount,
  fixedPlayerCount,
  moneyballOnly,
  participantText,
  linkedPlayerIds,
  doublesCourtCount,
  singlesCourtCount,
  onTargetCountChange,
  onDoublesCourtCountChange,
  onSinglesCourtCountChange,
  onParticipantTextChange,
  onLinkedPlayerIdsChange,
  onInvalidate
}: Props) {
  const [creationBusy, setCreationBusy] = useState(false);
  const [creationPending, setCreationPending] = useState(false);
  const rosterLocked = disabled || creationBusy || creationPending;
  const handleCreationPending = useCallback((_id: string, pending: boolean) => setCreationPending(pending), []);
  useEffect(() => { onCreationPendingChange?.(creationBusy || creationPending); }, [creationBusy, creationPending, onCreationPendingChange]);
  const [importOpen, setImportOpen] = useState(false);
  const [requestedRatings, setRequestedRatings] = useState<Record<string, string>>({});
  const [ratingsReady, setRatingsReady] = useState(false);
  const [playerSearch, setPlayerSearch] = useState("");
  const [directoryPlayers, setDirectoryPlayers] = useState<DirectoryPlayer[]>([]);
  const [directoryError, setDirectoryError] = useState<string | null>(null);
  const [pickerMessage, setPickerMessage] = useState<string | null>(null);

  const participantNames = useMemo(
    () => rosterNamesFromText(participantText),
    [participantText]
  );
  const minimumPlayers = playFormat === "singles" ? 2 : playFormat === "doubles_singles" ? 6 : 4;
  const participantCounts = useMemo(
    () => Array.from({ length: 40 - minimumPlayers + 1 }, (_, index) => minimumPlayers + index).filter(count => !moneyballOnly || count === 6 || count % 4 === 0 || count % 4 === 1),
    [minimumPlayers, moneyballOnly]
  );
  const setup = useMemo(
    () => recommendedGeneratorSetup(generatorKind, playFormat, targetCount, doublesCourtCount, singlesCourtCount, moneyballOnly),
    [generatorKind, playFormat, targetCount, doublesCourtCount, singlesCourtCount, moneyballOnly]
  );
  const linkedCount = participantNames.filter(
    (name) => Number(linkedPlayerIds[normalizeRosterName(name)] || 0) > 0
  ).length;
  const unlinkedNames = participantNames.filter(name => !(Number(linkedPlayerIds[normalizeRosterName(name)] || 0) > 0));
  const manualCount = unlinkedNames.length;
  const needsPlayerProfiles = Boolean(moneyballOnly || fixedPlayerCount || ratedSetup) && manualCount > 0;
  const reviewPlayerIds = participantNames.map(name => Number(linkedPlayerIds[normalizeRosterName(name)] || 0)).filter(id => id > 0);
  const exactCount = participantNames.length === targetCount;
  const canReviewRatings = Boolean(apiBase && creationClubId && ratingReviewAccessToken);
  useEffect(() => {
    onRatingReviewReadyChange?.(!ratedSetup || (!manualCount && (!canReviewRatings || ratingsReady)));
  }, [ratedSetup, manualCount, canReviewRatings, ratingsReady, onRatingReviewReadyChange]);
  const publicClubSlug = clubKey.replace(/_/g, "-");
  const maximumDoublesCourts = Math.max(1, Math.floor((targetCount - 2) / 4));
  const maximumSinglesCourts = Math.max(1, Math.floor((targetCount - 4) / 2));

  useEffect(() => {
    if (playFormat !== "doubles_singles") return;
    const valid = doublesCourtCount >= 1 && singlesCourtCount >= 1 && doublesCourtCount * 4 + singlesCourtCount * 2 <= targetCount;
    if (valid) return;
    const recommended = recommendedMixedCourtSetup(targetCount);
    onDoublesCourtCountChange(recommended.doublesCourtCount);
    onSinglesCourtCountChange(recommended.singlesCourtCount);
    onInvalidate();
  }, [
    playFormat,
    targetCount,
    doublesCourtCount,
    singlesCourtCount,
    onDoublesCourtCountChange,
    onSinglesCourtCountChange,
    onInvalidate
  ]);

  useEffect(() => {
    if (!apiBase) return;
    const controller = new AbortController();
    setDirectoryError(null);
    setDirectoryPlayers([]);
    void fetchAllPublicPlayers<DirectoryPlayer>(
      apiUrl(apiBase, `/clubs/${encodeURIComponent(publicClubSlug)}/players?status=all&sort=name`), controller.signal
    ).then(players => {
      if (!controller.signal.aborted) setDirectoryPlayers(players);
    }).catch(() => {
      if (!controller.signal.aborted) setDirectoryError("Player search isn’t available right now. You can still type names below.");
    });
    return () => controller.abort();
  }, [apiBase, publicClubSlug]);

  const filteredPlayerOptions = useMemo(() => {
    const query = normalizeRosterName(playerSearch);
    if (query.length < 2) return [];
    return directoryPlayers
      .filter((player) => matchesPlayerSearch(player.name, query))
      .slice(0, 10);
  }, [directoryPlayers, playerSearch]);

  function changeTargetCount(nextCount: number): void {
    onTargetCountChange(nextCount);
    if (playFormat === "doubles_singles") {
      const recommended = recommendedMixedCourtSetup(nextCount);
      onDoublesCourtCountChange(recommended.doublesCourtCount);
      onSinglesCourtCountChange(recommended.singlesCourtCount);
    }
    onInvalidate();
    setPickerMessage(null);
  }

  function changeDoublesCourts(nextCount: number): void {
    const nextDoubles = Math.max(1, nextCount);
    const maxSingles = Math.max(1, Math.floor((targetCount - nextDoubles * 4) / 2));
    onDoublesCourtCountChange(nextDoubles);
    if (singlesCourtCount > maxSingles) onSinglesCourtCountChange(maxSingles);
    onInvalidate();
  }

  function changeSinglesCourts(nextCount: number): void {
    const nextSingles = Math.max(1, nextCount);
    const maxDoubles = Math.max(1, Math.floor((targetCount - nextSingles * 2) / 4));
    onSinglesCourtCountChange(nextSingles);
    if (doublesCourtCount > maxDoubles) onDoublesCourtCountChange(maxDoubles);
    onInvalidate();
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

  function addCurrentPlayer(player: DirectoryPlayer, lineupName?: string): boolean {
    const name = String(player.name || "").trim().replace(/\s+/g, " ");
    const key = normalizeRosterName(name);
    const playerId = Number(player.id);
    if (!name || !Number.isInteger(playerId) || playerId <= 0) return false;
    const existingName = participantNames.find(entry => normalizeRosterName(entry) === normalizeRosterName(lineupName || ""))
      || participantNames.find(entry => playerSearchKey(entry) === playerSearchKey(name));
    const existingKey = normalizeRosterName(existingName || "");
    const duplicate = participantNames.some(entry => normalizeRosterName(entry) !== existingKey && (
      normalizeRosterName(entry) === key || Number(linkedPlayerIds[normalizeRosterName(entry)]) === playerId
    ));
    if (duplicate) {
      setPickerMessage(`${name} is already linked to another place in this lineup.`);
      return false;
    }
    if (!existingName && participantNames.length >= targetCount) {
      setPickerMessage(`The roster already has the selected ${targetCount} players. Increase the player count or remove a name first.`);
      return false;
    }
    const nextNames = existingName ? participantNames.map(entry => entry === existingName ? name : entry) : [...participantNames, name];
    const nextLinks = { ...linkedPlayerIds };
    if (existingKey) delete nextLinks[existingKey];
    nextLinks[key] = playerId;
    onParticipantTextChange(nextNames.join("\n"));
    onLinkedPlayerIdsChange(nextLinks);
    onInvalidate();
    setPlayerSearch("");
    setPickerMessage(existingName ? `${name} linked to their club profile.` : `${name} added to the roster.`);
    return true;
  }

  async function selectCreatedPlayer(playerId: number, lineupName?: string): Promise<void> {
    const response = await fetch(apiUrl(apiBase!, `/admin/clubs/${encodeURIComponent(creationClubId!)}/players/editor/players/${playerId}`), {
      cache: "no-store", headers: { Authorization: `Bearer ${creationAccessToken}` }
    });
    if (!response.ok) throw new Error("The player could not be loaded. Check player creation to try again.");
    const payload = await response.json() as { player?: DirectoryPlayer };
    if (!payload.player || Number(payload.player.id) !== playerId) throw new Error("Could not confirm the player's profile.");
    const player = payload.player;
    setDirectoryPlayers(current => [...current.filter(row => Number(row.id) !== playerId), player]);
    if (!addCurrentPlayer(player, lineupName)) throw new Error("This club profile could not be linked to the lineup. Check the selected name before continuing.");
  }

  const countMessage = exactCount
    ? needsPlayerProfiles ? `Link or create club profiles for ${manualCount} player${manualCount === 1 ? "" : "s"} before previewing.` : `Ready with ${targetCount} players.`
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
            disabled={rosterLocked || Boolean(fixedPlayerCount)}
            value={targetCount}
            onChange={(event) => changeTargetCount(Number(event.target.value))}
            style={inputStyle}
          >
            {participantCounts.map((count) => (
              <option key={count} value={count}>{count}</option>
            ))}
          </select>
        </label>
        {playFormat === "doubles_singles" ? (
          <>
            <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
              Doubles courts
              <select
                disabled={rosterLocked}
                value={doublesCourtCount}
                onChange={(event) => changeDoublesCourts(Number(event.target.value))}
                style={inputStyle}
              >
                {Array.from({ length: maximumDoublesCourts }, (_, index) => index + 1).map((count) => (
                  <option
                    key={count}
                    value={count}
                    disabled={count * 4 + singlesCourtCount * 2 > targetCount}
                  >
                    {count}
                  </option>
                ))}
              </select>
            </label>
            <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
              Singles courts
              <select
                disabled={rosterLocked}
                value={singlesCourtCount}
                onChange={(event) => changeSinglesCourts(Number(event.target.value))}
                style={inputStyle}
              >
                {Array.from({ length: maximumSinglesCourts }, (_, index) => index + 1).map((count) => (
                  <option
                    key={count}
                    value={count}
                    disabled={doublesCourtCount * 4 + count * 2 > targetCount}
                  >
                    {count}
                  </option>
                ))}
              </select>
            </label>
          </>
        ) : null}
        <div
          aria-live="polite"
          style={{
            border: "1px solid #bfdbfe",
            borderRadius: "10px",
            padding: "0.75rem",
            background: "#eff6ff"
          }}
        >
          <strong>{generatorKind === "round_robin" ? "Starting schedule" : "Automatic setup"}</strong>
          <p style={{ margin: "0.3rem 0 0", color: "#334155" }}>
            {playFormat === "doubles_singles" ? (
              <>
                {setup.doublesCourtCount} doubles court{setup.doublesCourtCount === 1 ? "" : "s"} · {setup.singlesCourtCount} singles court{setup.singlesCourtCount === 1 ? "" : "s"}
                <br />
                {setup.scheduledPlayers} playing · {setup.byesPerRound} bye{setup.byesPerRound === 1 ? "" : "s"} per round · {setup.totalRounds} rounds
              </>
            ) : (
              <>{setup.courtCount} court{setup.courtCount === 1 ? "" : "s"} · {setup.totalRounds} round{setup.totalRounds === 1 ? "" : "s"}</>
            )}
          </p>
          <small style={{ color: "#475569" }}>
            {explanation(generatorKind, playFormat, setup.courtCount, setup.totalRounds)}
            {generatorKind === "round_robin" && !moneyballOnly && !fixedPlayerCount ? " You can keep adding rounds during play." : ""}
            {moneyballOnly || fixedPlayerCount ? (
              <>
                {targetCount === 6
                  ? " Balanced six-player rotation: 4 games with 4 different partners and 2 byes each, one in each three-round block, with no consecutive byes. Each opponent is faced once or twice. This is not a perfect round robin."
                  : " Everyone partners once and faces each opponent twice."}
                {` Moneyball uses ${setup.totalRounds} fixed rounds. The starting lineup and scheduled slots stay fixed. Substitutions make both players ineligible to win.`}
              </>
            ) : ""}
            {generatorKind === "round_robin" && playFormat === "doubles" && !moneyballOnly && !fixedPlayerCount ? " A perfect partner-and-opponent rotation is the first priority, including compatible roster changes. Adaptive rounds balance assigned byes before game counts." : ""}
          </small>
        </div>
      </div>

      <div style={{ display: "grid", gap: "0.5rem" }}>
        <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
          Search club players
          <input
            disabled={rosterLocked}
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
            {moneyballOnly || fixedPlayerCount ? "Choose club players or create a new player with a starting level before previewing Moneyball." : "Search the club player list, or type names directly below."}
          </p>
        ) : filteredPlayerOptions.length ? (
          <div style={{ display: "grid", gap: "0.4rem" }}>
            {filteredPlayerOptions.map((player) => {
              const listedName = participantNames.find(name => playerSearchKey(name) === playerSearchKey(player.name));
              const alreadyAdded = Boolean(listedName && Number(linkedPlayerIds[normalizeRosterName(listedName)]) > 0)
                || Object.values(linkedPlayerIds).some(id => id === Number(player.id));
              const rosterFull = participantNames.length >= targetCount && !listedName;
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
                    disabled={rosterLocked || alreadyAdded || rosterFull}
                    style={{
                      ...secondaryButton,
                      background: alreadyAdded || rosterFull ? "#f1f5f9" : "white",
                      cursor: alreadyAdded || rosterFull ? "default" : "pointer"
                    }}
                  >
                    {alreadyAdded ? "Added" : listedName ? "Link player" : rosterFull ? "Roster full" : "Add"}
                  </button>
                </div>
              );
            })}
          </div>
        ) : (
          <p style={{ margin: 0, color: "#b45309", fontSize: "0.9rem" }}>
            No club players match “{playerSearch.trim()}”. {moneyballOnly || fixedPlayerCount ? "Use Create new player below to add their profile and starting level." : "You can type the name below instead."}
          </p>
        )}
        {directoryError ? (
          <p style={{ margin: 0, color: "#b45309", fontSize: "0.9rem" }}>
            {directoryError}
          </p>
        ) : null}
      </div>

      {moneyballOnly || fixedPlayerCount || ratedSetup ? (
        <div style={{ border: "1px solid #cbd5e1", borderRadius: 10, padding: "0.75rem", minWidth: 0 }}>
          {apiBase && creationClubId && creationAccessToken ? <GeneratorPlayerCreate
            context="setup" clubId={creationClubId} sessionKey={creationDraftKey || `setup:${generatorKind}`} participantId="new-player"
            lineupNames={participantNames.length >= targetCount ? unlinkedNames : undefined}
            participantName="this lineup" apiBase={apiBase} accessToken={creationAccessToken}
            players={directoryPlayers.map(player => ({ id: Number(player.id), name: player.name }))}
            disabled={disabled || creationBusy || (participantNames.length >= targetCount && !manualCount && !creationPending)} matched={false}
            onPendingChange={handleCreationPending} onBusyChange={setCreationBusy} onCreated={selectCreatedPlayer}
          /> : <p style={{ margin: 0 }}>
            <a href="/admin/login" target="_blank" rel="noreferrer">Sign in as club staff to create a new player</a>, then return to this lineup.
          </p>}
          {needsPlayerProfiles ? <p style={{ margin: "0.5rem 0 0", color: "#b45309" }}>
            Still to link: {unlinkedNames.join(", ")}. Search and choose Link player for an existing profile, or use Create new player. These names already have places in your lineup.
          </p> : null}
          {participantNames.length >= targetCount && !manualCount && !creationPending ? <small style={{ display: "block", marginTop: 6 }}>Remove a player or increase the player count to make room for someone new.</small> : null}
        </div>
      ) : null}

      {apiBase && creationClubId && ratingReviewAccessToken ? <div>
        <button type="button" style={secondaryButton} disabled={rosterLocked || Boolean(directoryError)} onClick={() => setImportOpen(value => !value)}>Import CourtReserve roster</button>
        {importOpen ? <CourtReserveRosterImport apiBase={apiBase} clubId={creationClubId} accessToken={ratingReviewAccessToken}
          players={directoryPlayers.map(player => ({ id: Number(player.id), name: player.name }))}
          matchFormat={playFormat === "singles" ? "singles" : "doubles"} requireRatingReview={ratedSetup}
          disabled={rosterLocked} submitLabel="Use imported roster" onCancel={() => setImportOpen(false)}
          onImport={({ playerIds, requestedRatings: importedRequests }) => {
            if (playerIds.length > 40) throw new Error("Generators support up to 40 players. Select up to 40 names for this round robin.");
            if (fixedPlayerCount && playerIds.length !== fixedPlayerCount) throw new Error(`Import exactly ${fixedPlayerCount} players for this lineup.`);
            const selected = playerIds.map(id => directoryPlayers.find(player => Number(player.id) === id));
            if (selected.some(player => !player)) throw new Error("The club list changed. Refresh before importing.");
            const names = selected.map(player => player!.name);
            if (new Set(names.map(normalizeRosterName)).size !== names.length) throw new Error("These club profiles have the same display name. Resolve the name conflict before adding both to this generator.");
            onParticipantTextChange(names.join("\n"));
            onLinkedPlayerIdsChange(Object.fromEntries(selected.map(player => [normalizeRosterName(player!.name), Number(player!.id)])));
            onTargetCountChange(Math.max(minimumPlayers, playerIds.length));
            setRequestedRatings(importedRequests); setRatingsReady(false); setImportOpen(false); onInvalidate();
          }} /> : null}
      </div> : null}
      {ratedSetup && participantNames.length > 0 ? <div>
        {canReviewRatings && reviewPlayerIds.length > 0 ? <GeneratorRatingReview
          apiBase={apiBase!} clubId={creationClubId!} accessToken={ratingReviewAccessToken!} playerIds={reviewPlayerIds}
          playFormat={playFormat} requestedRatings={requestedRatings} disabled={rosterLocked}
          onReady={setRatingsReady} onConfirmed={onInvalidate} /> : <p>Before rated play, a club administrator must confirm starting estimates for players with no rated history. Players with rated history use their calculated JUPR.</p>}
        {manualCount > 0 ? <p role="status">Choose club profiles for {unlinkedNames.join(", ")} before rated play.</p> : null}
      </div> : null}

      <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
        Players ({participantNames.length} of {targetCount})
        <textarea
          disabled={rosterLocked}
          value={participantText}
          onChange={(event) => changeParticipantText(event.target.value)}
          rows={Math.min(14, Math.max(7, targetCount))}
          placeholder="One player per line, in starting order"
          style={inputStyle}
        />
      </label>
      <div>
        <p style={{ margin: 0, color: exactCount && !needsPlayerProfiles ? "#166534" : "#b45309", fontWeight: 700 }}>
          {countMessage}
        </p>
        <p style={{ margin: "0.25rem 0 0", color: "#64748b", fontSize: "0.9rem" }}>
          {linkedCount} chosen from the club list · {manualCount} entered manually. {moneyballOnly || fixedPlayerCount ? "The order below sets the fixed Moneyball schedule." : "The order below sets the starting order and who gets the first bye."}
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
