"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { InteractionDialog, StaticActionFeedback, actionSuccess } from "@/components/interaction";
import type { PublicPlayer } from "@/lib/api";
import type {
  AdminMatchUploaderCreatePlayersResult,
  AdminMatchUploaderPlayerBatchOperation,
  AdminMatchUploaderRoundRobinPreview,
  AdminMatchUploaderStatusResponse,
  AdminMatchUploaderWriteResult
} from "@/lib/adminMatchUploaderApi";
import {
  clearDirectMatchIdempotencyKey,
  directMatchIdempotencyKey
} from "@/lib/directMatchIdempotency";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";
import styles from "./layout.module.css";

type Props = {
  apiBase: string | null;
  clubId: string;
  players: PublicPlayer[];
  status: AdminMatchUploaderStatusResponse;
};

type MatchRow = {
  rowId: string;
  date: string;
  league: string;
  weekTag: string;
  ratingScope: "" | "overall_only" | "unrated";
  t1p1: string;
  t1p2: string;
  t2p1: string;
  t2p2: string;
  s1: string;
  s2: string;
};

type SinglesRow = {
  date: string;
  league: string;
  weekTag: string;
  playerA: string;
  playerB: string;
  scoreA: string;
  scoreB: string;
  ratingScope: "" | "unrated";
};

type RrCourtInput = {
  rowId: string;
  formatType: string;
  playerNames: string[];
};
type RrScoreRow = {
  rowId: string;
  court: number;
  label: string;
  t1: Array<{ id: number; name: string }>;
  t2: Array<{ id: number; name: string }>;
  t1p1: number;
  t1p2: number;
  t2p1: number;
  t2p2: number;
  s1: string;
  s2: string;
};
type RrCourtSchedule = { court: number; formatType: string; expectedGames?: number | null; matches: RrScoreRow[] };
type RrPayload = { source: string; custom_schedule: string; schedule_mode: string; courts: Array<{ court: number; format_type: string; player_names: string[] }> };
type NewPlayerDraft = { name: string; startingJupr: string };
type PlayerBatchRecovery = {
  operationKey: string;
  operationScope: string;
  status: string;
  message: string;
  continueRoundRobin: boolean;
};
type StoredPlayerBatchRecovery = PlayerBatchRecovery & { version: 1 };
type PlayerRoundRobinRecord = { wins: number; losses: number };
type PlayerRoundRobinRecords = Record<string, PlayerRoundRobinRecord>;
type SearchablePlayerInputProps = {
  inputId: string;
  label: string;
  value: string;
  players: PublicPlayer[];
  allPlayers?: PublicPlayer[];
  disabled?: boolean;
  invalid?: boolean;
  onChange: (playerId: string) => void;
  onCreate: (
    name: string,
    startingJupr: number,
  ) => Promise<PublicPlayer | null>;
};
type SearchablePlayerMultiInputProps = {
  inputId: string;
  label: string;
  values: string[];
  players: PublicPlayer[];
  disabled?: boolean;
  onChange: (playerNames: string[]) => void;
  onCreate: (
    name: string,
    startingJupr: number,
  ) => Promise<PublicPlayer | null>;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", minWidth: 0, boxSizing: "border-box" as const, padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
const dangerButtonStyle = { ...buttonStyle, background: "#b91c1c", borderColor: "#b91c1c" };

class MatchUploaderApiError extends Error {
  operationKey: string | null;
  uncertain: boolean;

  constructor(message: string, operationKey: string | null = null, uncertain = false) {
    super(message);
    this.name = "MatchUploaderApiError";
    this.operationKey = operationKey;
    this.uncertain = uncertain;
  }
}

function apiErrorDetail(payload: unknown, status: number): { message: string; operationKey: string | null; uncertain: boolean } {
  const fallback = `API error (${status}).`;
  if (!payload || typeof payload !== "object") return { message: fallback, operationKey: null, uncertain: status >= 500 || status === 409 };
  const record = payload as Record<string, unknown>;
  const detail = record.detail ?? record.message ?? record.error;
  if (detail && typeof detail === "object" && !Array.isArray(detail)) {
    const detailRecord = detail as Record<string, unknown>;
    const explicitlyFailed = detailRecord.kind === "failed" && detailRecord.recovery_required !== true;
    return {
      message: typeof detailRecord.message === "string" ? detailRecord.message : fallback,
      operationKey: typeof detailRecord.operation_key === "string" ? detailRecord.operation_key : null,
      uncertain: detailRecord.recovery_required === true
        || detailRecord.kind === "uncertain"
        || detailRecord.code === "RECOVERY_REQUIRED"
        || (!explicitlyFailed && (status >= 500 || [408, 425, 429].includes(status))),
    };
  }
  if (Array.isArray(detail)) {
    return {
      message: detail.map((item) => typeof item === "string" ? item : JSON.stringify(item)).join("; ") || fallback,
      operationKey: null,
      uncertain: status >= 500 || status === 409,
    };
  }
  return { message: typeof detail === "string" ? detail : fallback, operationKey: null, uncertain: status >= 500 || status === 409 };
}

function todayIsoDate(): string {
  const today = new Date();
  const year = today.getFullYear();
  const month = String(today.getMonth() + 1).padStart(2, "0");
  const day = String(today.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function randomId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function normalizeNewPlayerName(value: string): string {
  return value.replace(/\s+/gu, " ").trim();
}

function normalizeNewPlayerBatch(players: Array<{ name: string; starting_jupr: number }>): Array<{ name: string; starting_jupr: number }> {
  const seen = new Set<string>();
  const normalized: Array<{ name: string; starting_jupr: number }> = [];
  for (const player of players) {
    const name = normalizeNewPlayerName(player.name);
    const nameKey = name.toLocaleLowerCase("en-US");
    if (seen.has(nameKey)) continue;
    seen.add(nameKey);
    normalized.push({ name, starting_jupr: Number(player.starting_jupr) });
  }
  return normalized;
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    const object = value as Record<string, unknown>;
    return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(object[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

async function reviewedPlayerBatchFingerprint(players: Array<{ name: string; starting_jupr: number }>): Promise<string> {
  const reviewed = {
    players: normalizeNewPlayerBatch(players).map((player) => ({
      name: player.name,
      starting_jupr: Number(player.starting_jupr).toFixed(4)
    }))
  };
  const digest = await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(canonicalJson(reviewed)));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function newMatchRow(
  date: string = todayIsoDate(),
  weekTag: string = "Week 1",
  ratingScope: MatchRow["ratingScope"] = "",
  league: string = "Open",
): MatchRow {
  return { rowId: randomId("row"), date, league, weekTag, ratingScope, t1p1: "", t1p2: "", t2p1: "", t2p2: "", s1: "0", s2: "0" };
}

function newSinglesRow(
  date: string = todayIsoDate(),
  league: string = "Open",
  weekTag: string = "Week 1",
): SinglesRow {
  return { date, league, weekTag, playerA: "", playerB: "", scoreA: "0", scoreB: "0", ratingScope: "" };
}

function newRoundRobin(formatType: string): RrCourtInput {
  return {
    rowId: randomId("court"),
    formatType,
    playerNames: [],
  };
}

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function ratingLabel(value?: number | null): string {
  if (value == null) return "—";
  return (Number(value) / 400).toFixed(2);
}

function deltaLabel(value?: number | null): string {
  if (value == null) return "—";
  const juprDelta = Number(value) / 400;
  const normalized = Math.abs(juprDelta) < 0.005 ? 0 : juprDelta;
  return `${normalized >= 0 ? "+" : ""}${normalized.toFixed(2)}`;
}

function isUploaderErrorMessage(message: string): boolean {
  return /\b(unable|unavailable|disabled|required|must|cannot|could not|not configured|sign in|error|invalid|select|enter|choose|failed|conflict|changed|reload|retry|nothing)\b/i.test(message);
}

function isFilled(row: MatchRow): boolean {
  return Boolean(row.t1p1 || row.t1p2 || row.t2p1 || row.t2p2 || Number(row.s1 || 0) + Number(row.s2 || 0) > 0);
}

function validateRequiredRow(row: MatchRow, index: number): string | null {
  const ids = [row.t1p1, row.t1p2, row.t2p1, row.t2p2].filter(Boolean);
  if (ids.length !== 4) return `Row ${index + 1}: complete the highlighted player fields.`;
  if (new Set(ids).size !== 4) return `Row ${index + 1}: each player may appear only once.`;
  const s1 = Number(row.s1 || 0);
  const s2 = Number(row.s2 || 0);
  if (!Number.isFinite(s1) || !Number.isFinite(s2) || s1 < 0 || s2 < 0) return `Row ${index + 1}: scores must be non-negative numbers.`;
  if (s1 + s2 <= 0) return `Row ${index + 1}: enter a non-zero score in the highlighted score fields.`;
  return null;
}

function validateRow(row: MatchRow, index: number): string | null {
  if (!isFilled(row)) return null;
  return validateRequiredRow(row, index);
}

function isReadyRow(row: MatchRow, index: number): boolean {
  return isFilled(row) && validateRow(row, index) === null;
}

function validateSingles(row: SinglesRow): string | null {
  if (!row.playerA || !row.playerB) return "Select two singles players.";
  if (row.playerA === row.playerB) return "Singles players must be different.";
  const s1 = Number(row.scoreA || 0);
  const s2 = Number(row.scoreB || 0);
  if (!Number.isFinite(s1) || !Number.isFinite(s2) || s1 < 0 || s2 < 0) return "Singles scores must be non-negative numbers.";
  if (s1 + s2 <= 0) return "Enter a non-zero singles score.";
  if (s1 === s2) return "Singles scores cannot be tied.";
  return null;
}

function validateStartingJupr(value: string): string | null {
  const numericValue = Number(value);
  if (!value.trim() || !Number.isFinite(numericValue)) {
    return "Enter a Starting JUPR from 1.00 to 7.00.";
  }
  if (numericValue < 1 || numericValue > 7) {
    return "Starting JUPR must be between 1.00 and 7.00.";
  }
  return null;
}

function mergePlayers(current: PublicPlayer[], incoming: NonNullable<AdminMatchUploaderCreatePlayersResult["players"]>): PublicPlayer[] {
  const byId = new Map<string, PublicPlayer>();
  for (const player of current) byId.set(String(player.id), player);
  for (const player of incoming) byId.set(String(player.id), player as PublicPlayer);
  return Array.from(byId.values()).sort((left, right) => String(left.name).localeCompare(String(right.name)));
}

function playerBatchOperationResult(operation: AdminMatchUploaderPlayerBatchOperation): AdminMatchUploaderCreatePlayersResult | null {
  return operation.result || operation.result_json || null;
}

function SearchablePlayerInput({
  inputId,
  label,
  value,
  players,
  allPlayers,
  disabled = false,
  invalid = false,
  onChange,
  onCreate,
}: SearchablePlayerInputProps) {
  const playerUniverse = allPlayers || players;
  const selected = playerUniverse.find((player) => String(player.id) === value);
  const selectedName = selected ? String(selected.name) : "";
  const [query, setQuery] = useState(selectedName);
  const [startingJupr, setStartingJupr] = useState("");
  const [creating, setCreating] = useState(false);
  const cleanedQuery = query.replace(/\s+/g, " ").trim();
  const exactPlayer = players.find(
    (player) =>
      String(player.name).trim().toLocaleLowerCase()
      === cleanedQuery.toLocaleLowerCase(),
  );
  const existingClubPlayer = playerUniverse.find(
    (player) =>
      String(player.name).trim().toLocaleLowerCase()
      === cleanedQuery.toLocaleLowerCase(),
  );
  const unavailableExactPlayer = existingClubPlayer && !exactPlayer
    ? existingClubPlayer
    : null;
  const matchingPlayers = cleanedQuery
    ? players.filter((player) =>
        String(player.name).trim().toLocaleLowerCase().includes(cleanedQuery.toLocaleLowerCase()),
      )
    : players;
  const numericStartingJupr = Number(startingJupr);
  const startingJuprMessage = cleanedQuery && !existingClubPlayer && matchingPlayers.length === 0
    ? validateStartingJupr(startingJupr)
    : null;
  const validatedInputStyle = invalid
    ? { ...inputStyle, border: "2px solid #dc2626", background: "#fef2f2" }
    : inputStyle;

  const exactPlayerId = exactPlayer ? String(exactPlayer.id) : "";
  const previousExactPlayerId = useRef(exactPlayerId);

  useEffect(() => {
    setQuery(selectedName);
  }, [selectedName]);

  useEffect(() => {
    const previousId = previousExactPlayerId.current;
    previousExactPlayerId.current = exactPlayerId;
    if (!value && exactPlayerId && cleanedQuery && exactPlayerId !== previousId) {
      onChange(exactPlayerId);
    }
  }, [cleanedQuery, exactPlayerId, onChange, value]);

  async function createAndSelect() {
    if (
      !cleanedQuery
      || existingClubPlayer
      || startingJuprMessage
    ) return;
    setCreating(true);
    try {
      const player = await onCreate(cleanedQuery, numericStartingJupr);
      if (player) {
        setQuery(String(player.name));
        onChange(String(player.id));
      }
    } finally {
      setCreating(false);
    }
  }

  return (
    <div>
      <label htmlFor={inputId}><strong>{label}</strong></label>
      {value ? (
        <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) auto", gap: "0.35rem", alignItems: "stretch" }}>
          <div
            title={selectedName}
            style={{ ...validatedInputStyle, minHeight: "2.4rem", display: "flex", alignItems: "center", whiteSpace: "normal", overflowWrap: "anywhere", background: invalid ? "#fef2f2" : "#f8fafc" }}
          >
            {selectedName}
          </div>
          <button
            type="button"
            aria-label={`Clear ${label}`}
            disabled={disabled || creating}
            onClick={() => {
              setQuery("");
              onChange("");
            }}
          >
            Clear
          </button>
        </div>
      ) : (
        <input
          id={inputId}
          list={`${inputId}-options`}
          value={query}
          placeholder="Search player…"
          autoComplete="off"
          disabled={disabled || creating}
          aria-invalid={invalid || undefined}
          onChange={(event) => {
            const next = event.target.value;
            setQuery(next);
            const match = players.find(
              (player) =>
                String(player.name).trim().toLocaleLowerCase()
                === next.replace(/\s+/g, " ").trim().toLocaleLowerCase(),
            );
            onChange(match ? String(match.id) : "");
          }}
          style={validatedInputStyle}
        />
      )}
      <datalist id={`${inputId}-options`}>
        {players.map((player) => (
          <option key={String(player.id)} value={String(player.name)} />
        ))}
      </datalist>
        {unavailableExactPlayer ? (
          <p role="status" style={{ color: "#92400e", margin: "0.35rem 0 0", fontWeight: 700 }}>
            {String(unavailableExactPlayer.name)} is already used in this match. Clear that position before selecting this player here.
          </p>
        ) : null}
        {cleanedQuery && !existingClubPlayer && matchingPlayers.length === 0 ? (
        <div style={{ display: "grid", gridTemplateColumns: "minmax(100px, 1fr) auto", gap: "0.35rem", marginTop: "0.35rem", alignItems: "end" }}>
          <label htmlFor={`${inputId}-starting-jupr`}>
            <span style={{ display: "block", color: "#475569", fontSize: "0.8rem" }}>Starting JUPR *</span>
            <input
              required
              id={`${inputId}-starting-jupr`}
              type="number"
              min={1}
              max={7}
              step={0.01}
              value={startingJupr}
              disabled={disabled || creating}
              aria-invalid={Boolean(startingJuprMessage) || undefined}
              onChange={(event) => setStartingJupr(event.target.value)}
              style={startingJuprMessage ? { ...inputStyle, border: "2px solid #dc2626", background: "#fef2f2" } : inputStyle}
            />
          </label>
          <button
            type="button"
            onClick={createAndSelect}
            disabled={disabled || creating || Boolean(startingJuprMessage)}
            style={ghostButtonStyle}
          >
            {creating ? "Creating…" : `Create “${cleanedQuery}”`}
          </button>
          {startingJuprMessage ? <p role="alert" style={{ gridColumn: "1 / -1", color: "#b91c1c", margin: 0, fontWeight: 700 }}>{startingJuprMessage}</p> : null}
        </div>
      ) : null}
    </div>
  );
}

function SearchablePlayerMultiInput({
  inputId,
  label,
  values,
  players,
  disabled = false,
  onChange,
  onCreate,
}: SearchablePlayerMultiInputProps) {
  const [query, setQuery] = useState("");
  const [startingJupr, setStartingJupr] = useState("");
  const [creating, setCreating] = useState(false);
  const cleanedQuery = query.replace(/\s+/g, " ").trim();
  const selectedNames = new Set(
    values.map((name) => name.trim().toLocaleLowerCase()),
  );
  const exactPlayer = players.find(
    (player) =>
      String(player.name).trim().toLocaleLowerCase()
      === cleanedQuery.toLocaleLowerCase(),
  );
  const numericStartingJupr = Number(startingJupr);
  const startingJuprMessage = cleanedQuery && !exactPlayer
    ? validateStartingJupr(startingJupr)
    : null;

  function addPlayerName(name: string) {
    const cleanedName = name.replace(/\s+/g, " ").trim();
    if (
      !cleanedName
      || selectedNames.has(cleanedName.toLocaleLowerCase())
    ) return;
    onChange([...values, cleanedName]);
    setQuery("");
  }

  async function createAndAdd() {
    if (
      !cleanedQuery
      || exactPlayer
      || startingJuprMessage
    ) return;
    setCreating(true);
    try {
      const player = await onCreate(cleanedQuery, numericStartingJupr);
      if (player) addPlayerName(String(player.name));
    } finally {
      setCreating(false);
    }
  }

  return (
    <div>
      <label htmlFor={inputId}><strong>{label}</strong></label>
      {values.length ? (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", margin: "0.35rem 0 0.5rem" }}>
          {values.map((name) => (
            <span key={name.toLocaleLowerCase()} style={{ display: "inline-flex", alignItems: "center", gap: "0.35rem", border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.25rem 0.35rem 0.25rem 0.65rem", background: "white" }}>
              {name}
              <button
                type="button"
                aria-label={`Remove ${name}`}
                onClick={() => onChange(values.filter((item) => item.toLocaleLowerCase() !== name.toLocaleLowerCase()))}
                disabled={disabled || creating}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      ) : (
        <p style={{ color: "#64748b", fontSize: "0.85rem", margin: "0.35rem 0 0.5rem" }}>No players selected yet.</p>
      )}
      <div style={{ display: "grid", gridTemplateColumns: "minmax(140px, 1fr) auto", gap: "0.4rem" }}>
        <input
          id={inputId}
          list={`${inputId}-options`}
          value={query}
          onChange={(event) => {
          const next = event.target.value;
          setQuery(next);
          const normalizedNext = next.replace(/\s+/g, " ").trim().toLocaleLowerCase();
          const match = players.find((player) =>
            String(player.name).trim().toLocaleLowerCase() === normalizedNext
            && !selectedNames.has(String(player.name).trim().toLocaleLowerCase()),
          );
          if (match) addPlayerName(String(match.name));
        }}
        onKeyDown={(event) => {
            if (event.key === "Enter" && exactPlayer) {
              event.preventDefault();
              addPlayerName(String(exactPlayer.name));
            }
          }}
          placeholder="Search player name…"
          autoComplete="off"
          disabled={disabled || creating}
          style={inputStyle}
        />
        <button
          type="button"
          onClick={() => exactPlayer && addPlayerName(String(exactPlayer.name))}
          disabled={!exactPlayer || selectedNames.has(String(exactPlayer?.name || "").trim().toLocaleLowerCase())}
          style={ghostButtonStyle}
        >
          Add player
        </button>
      </div>
      <datalist id={`${inputId}-options`}>
        {players
          .filter((player) => !selectedNames.has(String(player.name).trim().toLocaleLowerCase()))
          .map((player) => <option key={String(player.id)} value={String(player.name)} />)}
      </datalist>
      {cleanedQuery && !exactPlayer ? (
        <div style={{ display: "grid", gridTemplateColumns: "minmax(100px, 1fr) auto", gap: "0.4rem", marginTop: "0.4rem", alignItems: "end" }}>
          <label htmlFor={`${inputId}-starting-jupr`}>
            <span style={{ display: "block", color: "#475569", fontSize: "0.8rem" }}>Starting JUPR *</span>
            <input
              required
              id={`${inputId}-starting-jupr`}
              type="number"
              min={1}
              max={7}
              step={0.01}
              value={startingJupr}
              onChange={(event) => setStartingJupr(event.target.value)}
              disabled={disabled || creating}
              aria-invalid={Boolean(startingJuprMessage) || undefined}
              style={startingJuprMessage ? { ...inputStyle, border: "2px solid #dc2626", background: "#fef2f2" } : inputStyle}
            />
          </label>
          <button type="button" onClick={createAndAdd} disabled={disabled || creating || Boolean(startingJuprMessage)} style={ghostButtonStyle}>
            {creating ? "Creating…" : `Create & add “${cleanedQuery}”`}
          </button>
          {startingJuprMessage ? <p role="alert" style={{ gridColumn: "1 / -1", color: "#b91c1c", margin: 0, fontWeight: 700 }}>{startingJuprMessage}</p> : null}
        </div>
      ) : null}
    </div>
  );
}

function previewToSchedule(preview: AdminMatchUploaderRoundRobinPreview): RrCourtSchedule[] {
  return (preview.courts || []).map((court) => ({
    court: court.court,
    formatType: court.format_type,
    expectedGames: court.expected_games,
    matches: (court.matches || []).map((match) => ({
      rowId: match.row_id || randomId("rr"),
      court: match.court,
      label: match.label,
      t1: match.t1,
      t2: match.t2,
      t1p1: match.t1_p1,
      t1p2: match.t1_p2,
      t2p1: match.t2_p1,
      t2p2: match.t2_p2,
      s1: "0",
      s2: "0"
    }))
  }));
  }

  function roundRobinPlayerRecords(schedule: RrCourtSchedule[]): PlayerRoundRobinRecords {
    const records: PlayerRoundRobinRecords = {};
    const increment = (playerId: number, field: keyof PlayerRoundRobinRecord) => {
      const key = String(playerId);
      records[key] = records[key] || { wins: 0, losses: 0 };
      records[key][field] += 1;
    };
    for (const court of schedule) {
      for (const match of court.matches) {
        const score1 = Number(match.s1 || 0);
        const score2 = Number(match.s2 || 0);
        if (!Number.isFinite(score1) || !Number.isFinite(score2) || score1 < 0 || score2 < 0 || score1 + score2 <= 0 || score1 === score2) continue;
        const winners = score1 > score2 ? match.t1 : match.t2;
        const losers = score1 > score2 ? match.t2 : match.t1;
        for (const player of winners) increment(player.id, "wins");
        for (const player of losers) increment(player.id, "losses");
      }
    }
    return records;
  }


function RemoveAllMatchesDialog({
  onClose,
  onKeepRows,
  onRemoveAll,
}: {
  onClose: () => void;
  onKeepRows: () => void;
  onRemoveAll: () => void;
}) {
  return (
    <InteractionDialog
      open
      phase="ready"
      title="Remove entered matches?"
      description="Choose whether to keep completed or partially entered rows, remove only blank rows, or clear the entire batch."
      onRequestClose={onClose}
      actions={(
        <>
          <button type="button" onClick={onClose} style={ghostButtonStyle}>No, go back</button>
          <button type="button" onClick={onKeepRows} style={ghostButtonStyle}>Keep rows with data</button>
          <button type="button" onClick={onRemoveAll} style={dangerButtonStyle}>Yes, remove all</button>
        </>
      )}
    >
      <p>Rows that already contain match data are protected unless you explicitly choose <strong>Yes, remove all</strong>.</p>
    </InteractionDialog>
  );
}

function SubmissionResultDialog({
    result,
    roundRobinRecords,
    submissionKind,
    onClose,
  }: {
    result: AdminMatchUploaderWriteResult;
    roundRobinRecords?: PlayerRoundRobinRecords | null;
    submissionKind: "manual" | "round_robin" | "singles" | null;
    onClose: () => void;
  }) {
  const inserted = result.result?.inserted ?? 0;
  const email = result.auto_player_updates;
  const emailSummary = email?.mode === "auto_sent"
      ? `${email.sent ?? 0} sent, ${email.skipped ?? 0} skipped, ${email.errors ?? 0} error(s).`
      : "Not sent in staging.";
    const matchIds = (result.operation?.match_ids || []).map((value) => String(value)).filter(Boolean);
    const correctionMatchId = matchIds[0] || (result.feedback?.latest_match_id == null ? "" : String(result.feedback.latest_match_id));
    const [chooseMatchesToEdit, setChooseMatchesToEdit] = useState(false);
    const [selectedCorrectionIds, setSelectedCorrectionIds] = useState<string[]>(() => [...matchIds]);
    const bulkCorrectionHref = (ids: string[]) => {
      const params = new URLSearchParams();
      params.set("match_ids", ids.join(","));
      params.set("selected_ids", ids.join(","));
      params.set("limit", String(Math.max(250, ids.length)));
      return `/admin/match-log/bulk?${params.toString()}`;
    };
    const isRoundRobinBulk = submissionKind === "round_robin" && matchIds.length > 1;
    const isManualMulti = submissionKind === "manual" && matchIds.length > 1;
    const correctionHref = isRoundRobinBulk
      ? bulkCorrectionHref(matchIds)
      : correctionMatchId
        ? `/admin/match-log/edit?match_id=${encodeURIComponent(correctionMatchId)}`
        : (result.recovery?.match_log_route || "/admin/match-log");
    const showRoundRobinRecords = Boolean(roundRobinRecords && Object.keys(roundRobinRecords).length);

  const dialogActions = (
    <>
      {correctionMatchId && !isManualMulti ? <Link href={correctionHref} style={{ ...ghostButtonStyle, textDecoration: "none", display: "inline-flex", alignItems: "center" }}>Edit results</Link> : null}
      {correctionMatchId && isManualMulti && !chooseMatchesToEdit ? <button type="button" onClick={() => setChooseMatchesToEdit(true)} style={ghostButtonStyle}>Edit results</button> : null}
      <button type="button" onClick={onClose} style={buttonStyle}>OK</button>
    </>
  );

  return (
    <InteractionDialog
      open
      phase="success"
      size="wide"
      title="Match submission complete"
      onRequestClose={onClose}
      actions={dialogActions}
    >
        <div tabIndex={-1} data-dialog-focus>
          <StaticActionFeedback tone="success" title="Matches saved" description={`Successfully inserted ${inserted} rated match${inserted === 1 ? "" : "es"}.`} />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem" }}>
          <div><strong>Inserted</strong><br />{inserted}</div>
          <div><strong>Match write</strong><br />{result.match_write_committed ? "Committed" : "Review required"}</div>
          <div><strong>Rating type</strong><br />{result.feedback?.rating_type || result.result?.match_format || "doubles/overall"}</div>
          <div><strong>Skipped incomplete</strong><br />{result.result?.skipped_incomplete ?? 0}</div>
          <div><strong>Skipped empty</strong><br />{result.result?.skipped_empty ?? 0}</div>
          <div><strong>Skipped unrated</strong><br />{result.result?.skipped_unrated ?? 0}</div>
        </div>
        {email ? <p style={{ marginTop: "1rem" }}><strong>Player-update email:</strong> {emailSummary}</p> : null}
        {matchIds.length ? <p><strong>Created match IDs:</strong> {matchIds.map((id) => `#${id}`).join(", ")}</p> : null}
        {result.feedback?.affected_players?.length ? (
          <div style={{ overflowX: "auto", marginTop: "1rem" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Player</th>{showRoundRobinRecords ? <th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Wins</th> : null}{showRoundRobinRecords ? <th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Losses</th> : null}<th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Before</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>After</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Change</th></tr></thead>
              <tbody>
                {result.feedback.affected_players.map((player) => {
        const record = roundRobinRecords?.[String(player.id)];
        return (
          <tr key={player.id}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{player.name}</td>{showRoundRobinRecords ? <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{record?.wins ?? 0}</td> : null}{showRoundRobinRecords ? <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{record?.losses ?? 0}</td> : null}<td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{ratingLabel(player.rating_before)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{ratingLabel(player.rating_after)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{deltaLabel(player.rating_delta)}</td></tr>
        );
      })}
              </tbody>
            </table>
          </div>
        ) : null}
        {result.warnings?.length ? <ul style={{ color: "#92400e" }}>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
        {isManualMulti && chooseMatchesToEdit ? (
          <div style={{ border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.85rem", marginTop: "1rem", background: "#f8fafc" }}>
            <h3 style={{ marginTop: 0 }}>Choose matches to edit</h3>
            <p style={{ color: "#475569" }}>Select the uploaded matches that need correction. They will open together in Bulk edit.</p>
            <div style={{ display: "grid", gap: "0.4rem" }}>
              {matchIds.map((id) => (
                <label key={id} style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                  <input
                    type="checkbox"
                    checked={selectedCorrectionIds.includes(id)}
                    onChange={() => setSelectedCorrectionIds((current) => current.includes(id) ? current.filter((value) => value !== id) : [...current, id])}
                  />
                  Match #{id}
                </label>
              ))}
            </div>
            <p style={{ display: "flex", justifyContent: "flex-end", gap: "0.5rem", flexWrap: "wrap", marginBottom: 0 }}>
              <button type="button" onClick={() => setChooseMatchesToEdit(false)} style={ghostButtonStyle}>Back</button>
              {selectedCorrectionIds.length ? (
                <Link href={bulkCorrectionHref(selectedCorrectionIds)} style={{ ...ghostButtonStyle, textDecoration: "none", display: "inline-flex", alignItems: "center" }}>
                  Open selected in bulk editor
                </Link>
              ) : (
                <button type="button" disabled style={ghostButtonStyle}>Open selected in bulk editor</button>
              )}
            </p>
          </div>
        ) : null}
    </InteractionDialog>
  );
}

export default function MatchUploaderForm({ apiBase, clubId, players, status }: Props) {
  const firstFormat = status.round_robin_format_options?.[0] || "4-Player";
  const legacyOfficialLeagueOptions = status.league_options.filter((item) => item.trim().toUpperCase() !== "POPUP");
  const doublesLeagueOptions = status.doubles_league_options?.length
    ? status.doubles_league_options
    : legacyOfficialLeagueOptions;
  const singlesLeagueOptions = status.singles_league_options || [];
  const initialLeague = doublesLeagueOptions[0] || singlesLeagueOptions[0] || "";
  const initialWeekTag = status.week_tag_options[0] || "Week 1";
  const singlesEnabled = Boolean(status.singles_write_enabled && status.singles_submit_endpoint);
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [knownPlayers, setKnownPlayers] = useState<PublicPlayer[]>(players);
  const [entryMethod, setEntryMethod] = useState<"singles" | "manual" | "round_robin">("manual");
  const [context, setContext] = useState<"league" | "popup">("league");
  const [defaultDate, setDefaultDate] = useState(todayIsoDate());
  const [defaultLeague, setDefaultLeague] = useState(initialLeague);
  const [defaultWeekTag, setDefaultWeekTag] = useState(initialWeekTag);
  const [popupEventName, setPopupEventName] = useState("Saturday Social");
  const [singlesRow, setSinglesRow] = useState<SinglesRow>(() => newSinglesRow(todayIsoDate(), initialLeague, initialWeekTag));
  const [rows, setRows] = useState<MatchRow[]>(() => [newMatchRow(todayIsoDate(), initialWeekTag, "", initialLeague)]);
  const [rrCourts, setRrCourts] = useState<RrCourtInput[]>(() => [newRoundRobin(firstFormat)]);
  const [rrCustomSchedule, setRrCustomSchedule] = useState("");
  const [rrSchedule, setRrSchedule] = useState<RrCourtSchedule[]>([]);
  const [rrPendingPayload, setRrPendingPayload] = useState<RrPayload | null>(null);
  const [newPlayerDrafts, setNewPlayerDrafts] = useState<NewPlayerDraft[]>([]);
  const [saving, setSaving] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [creatingPlayers, setCreatingPlayers] = useState(false);
  const [playerBatchRecovery, setPlayerBatchRecovery] = useState<PlayerBatchRecovery | null>(null);
  const [checkingPlayerBatchRecovery, setCheckingPlayerBatchRecovery] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [result, setResult] = useState<AdminMatchUploaderWriteResult | null>(null);
  const [submissionKind, setSubmissionKind] = useState<"manual" | "round_robin" | "singles" | null>(null);
  const [manualValidationAttempted, setManualValidationAttempted] = useState(false);
  const [singlesValidationAttempted, setSinglesValidationAttempted] = useState(false);
  const [removeAllDialogOpen, setRemoveAllDialogOpen] = useState(false);

  const playerRecoveryStorageKey = `jupr-match-uploader-player-recovery:${clubId}`;

  useEffect(() => {
    setPlayerBatchRecovery(null);
    try {
      const raw = globalThis.sessionStorage?.getItem(playerRecoveryStorageKey);
      if (!raw) return;
      const stored = JSON.parse(raw) as Partial<StoredPlayerBatchRecovery>;
      if (
        stored.version === 1
        && typeof stored.operationKey === "string"
        && typeof stored.operationScope === "string"
        && typeof stored.status === "string"
        && typeof stored.message === "string"
      ) {
        setPlayerBatchRecovery({
          operationKey: stored.operationKey,
          operationScope: stored.operationScope,
          status: stored.status,
          message: stored.message,
          continueRoundRobin: stored.continueRoundRobin === true,
        });
      }
    } catch {
      // A blocked or stale session store must not hide the rest of Match Uploader.
    }
  }, [playerRecoveryStorageKey]);

  const activeLeagueOptions = entryMethod === "singles" ? singlesLeagueOptions : doublesLeagueOptions;
  const activeLeagueFormatLabel = entryMethod === "singles" ? "singles" : "doubles";
  const hasActiveOfficialLeague = activeLeagueOptions.length > 0;
  const filledRows = rows.filter(isFilled);
  const readyRows = rows.filter((row, index) => isReadyRow(row, index));
  const hasInvalidFilledRows = filledRows.length !== readyRows.length;
  const defaultManualWeekTag = context === "popup" ? "" : defaultWeekTag;
  const defaultManualRatingScope: MatchRow["ratingScope"] = context === "popup" ? "overall_only" : "";
  const singlesError = singlesValidationAttempted ? validateSingles(singlesRow) : null;
  const singlesScoreA = Number(singlesRow.scoreA || 0);
  const singlesScoreB = Number(singlesRow.scoreB || 0);
  const singlesPlayersDuplicate = Boolean(
    singlesRow.playerA
    && singlesRow.playerB
    && singlesRow.playerA === singlesRow.playerB,
  );
  const singlesScoreInvalid = singlesValidationAttempted && (
    !Number.isFinite(singlesScoreA)
    || !Number.isFinite(singlesScoreB)
    || singlesScoreA < 0
    || singlesScoreB < 0
    || singlesScoreA + singlesScoreB <= 0
    || singlesScoreA === singlesScoreB
  );
  const rowHasEnteredData = (row: MatchRow) =>
    isFilled(row)
    || row.date !== defaultDate
    || (context === "league" && row.league !== defaultLeague)
    || (context === "league" && row.weekTag !== defaultWeekTag)
    || row.ratingScope !== defaultManualRatingScope;
  const scoredRrRows = rrSchedule.flatMap((court) => court.matches).filter((match) => Number(match.s1 || 0) + Number(match.s2 || 0) > 0);
  const rrResultRecords = submissionKind === "round_robin" ? roundRobinPlayerRecords(rrSchedule) : null;
  const matchType = context === "popup" ? "PopUp" : "Live Match";
  const messageIsError = Boolean(message && (result?.ok === false || isUploaderErrorMessage(message)));

  function clearEntryFeedback() {
    setMessage(null);
    setResult(null);
  }

  function retainPlayerBatchRecovery(recovery: PlayerBatchRecovery) {
    setPlayerBatchRecovery(recovery);
    try {
      globalThis.sessionStorage?.setItem(
        playerRecoveryStorageKey,
        JSON.stringify({ version: 1, ...recovery } satisfies StoredPlayerBatchRecovery),
      );
    } catch {
      // The in-memory guard still prevents a second write in this page session.
    }
  }

  function clearPlayerBatchRecovery() {
    setPlayerBatchRecovery(null);
    try {
      globalThis.sessionStorage?.removeItem(playerRecoveryStorageKey);
    } catch {
      // The completed server operation is authoritative if storage cleanup fails.
    }
  }

  function resetManualRows() {
    const priorScope = (readyRows[0] || filledRows[0])?.ratingScope;
    const preservedScope = context === "popup"
      ? (priorScope === "unrated" ? "unrated" : "overall_only")
      : (priorScope || "");
    setRows([newMatchRow(defaultDate, defaultManualWeekTag, preservedScope, defaultLeague)]);
    setManualValidationAttempted(false);
  }

  function changeEntryMethod(nextMethod: "singles" | "manual" | "round_robin") {
    clearEntryFeedback();
    setManualValidationAttempted(false);
    setSinglesValidationAttempted(false);
    setEntryMethod(nextMethod);
    const nextOptions = nextMethod === "singles" ? singlesLeagueOptions : doublesLeagueOptions;
    const nextLeague = nextOptions.includes(defaultLeague) ? defaultLeague : (nextOptions[0] || "");
    if (nextLeague !== defaultLeague) {
      setDefaultLeague(nextLeague);
      setRows((current) => current.map((row) => ({ ...row, league: nextLeague })));
      setSinglesRow((current) => ({ ...current, league: nextLeague }));
    }
  }

  function changeContext(nextContext: "league" | "popup") {
    clearEntryFeedback();
    setManualValidationAttempted(false);
    setSinglesValidationAttempted(false);
    setContext(nextContext);
    setRows((current) => current.map((row) => ({
      ...row,
      league: row.league || defaultLeague,
      weekTag: nextContext === "popup" ? "" : (row.weekTag || defaultWeekTag),
      ratingScope: nextContext === "popup"
        ? (row.ratingScope === "unrated" ? "unrated" : "overall_only")
        : (row.ratingScope === "unrated" ? "unrated" : ""),
    })));
    setSinglesRow((current) => ({
      ...current,
      league: nextContext === "popup"
        ? "POPUP"
        : (!current.league || current.league === "POPUP" ? defaultLeague : current.league),
      weekTag: nextContext === "popup" ? "" : (current.weekTag || defaultWeekTag),
    }));
  }

  function changeDefaultDate(nextDate: string) {
    const previousDate = defaultDate;
    clearEntryFeedback();
    setDefaultDate(nextDate);
    setRows((current) => current.map((row) => isFilled(row) ? row : { ...row, date: nextDate }));
    setSinglesRow((current) => !current.date || current.date === previousDate ? { ...current, date: nextDate } : current);
  }

  function changeDefaultLeague(nextLeague: string) {
    const previousLeague = defaultLeague;
    clearEntryFeedback();
    setDefaultLeague(nextLeague);
    setRows((current) => current.map((row) =>
      !row.league || row.league === previousLeague
        ? { ...row, league: nextLeague }
        : row
    ));
    setSinglesRow((current) =>
      !current.league || current.league === previousLeague
        ? { ...current, league: nextLeague }
        : current
    );
  }

  function changeDefaultWeekTag(nextWeekTag: string) {
    const previousWeekTag = defaultWeekTag;
    clearEntryFeedback();
    setDefaultWeekTag(nextWeekTag);
    setRows((current) => current.map((row) =>
      !row.weekTag || row.weekTag === previousWeekTag
        ? { ...row, weekTag: nextWeekTag }
        : row
    ));
    setSinglesRow((current) =>
      !current.weekTag || current.weekTag === previousWeekTag
        ? { ...current, weekTag: nextWeekTag }
        : current
    );
  }

  function acknowledgeSubmission() {
    if (submissionKind === "manual") resetManualRows();
    if (submissionKind === "singles") {
      setSinglesRow((current) => ({
        ...newSinglesRow(current.date, current.league, current.weekTag),
        ratingScope: current.ratingScope,
      }));
      setSinglesValidationAttempted(false);
    }
    if (submissionKind === "round_robin") {
      setRrSchedule([]);
      setRrPendingPayload(null);
    }
    setSubmissionKind(null);
    setMessage(null);
    setResult(null);
  }

  function removeRow(rowId: string) {
    clearEntryFeedback();
    setRows((current) => {
      const remaining = current.filter((row) => row.rowId !== rowId);
      return remaining.length ? remaining : [newMatchRow(defaultDate, defaultManualWeekTag, defaultManualRatingScope, defaultLeague)];
    });
  }

  function removeAllRows() {
    clearEntryFeedback();
    setRemoveAllDialogOpen(false);
    resetManualRows();
  }

  function keepRowsWithData() {
    clearEntryFeedback();
    setRemoveAllDialogOpen(false);
    setManualValidationAttempted(false);
    setRows((current) => {
      const kept = current.filter(rowHasEnteredData);
      return kept.length ? kept : [newMatchRow(defaultDate, defaultManualWeekTag, defaultManualRatingScope, defaultLeague)];
    });
  }

  function playerOptionsFor(row: MatchRow, currentValue: string): PublicPlayer[] {
    const selectedElsewhere = new Set(
      [row.t1p1, row.t1p2, row.t2p1, row.t2p2].filter((value) => value && value !== currentValue),
    );
    return knownPlayers.filter((player) => !selectedElsewhere.has(String(player.id)));
  }

  function singlesPlayerOptions(currentValue: string): PublicPlayer[] {
    const selectedElsewhere = new Set(
      [singlesRow.playerA, singlesRow.playerB].filter((value) => value && value !== currentValue),
    );
    return knownPlayers.filter((player) => !selectedElsewhere.has(String(player.id)));
  }

  function requireReady(): boolean {
    if (!apiBase) {
      setMessage("API base URL is not configured.");
      return false;
    }
    if (!accessToken) {
      setMessage("Sign in at /admin/login before using Match Uploader.");
      return false;
    }
    if (!status.enabled) {
      setMessage("Next Match Uploader is disabled on the API.");
      return false;
    }
    return true;
  }

  function patchRow(rowId: string, patch: Partial<MatchRow>) {
    clearEntryFeedback();
    setRows((current) => current.map((row) => row.rowId === rowId ? { ...row, ...patch } : row));
  }

  function patchSingles(patch: Partial<SinglesRow>) {
    clearEntryFeedback();
    setSinglesRow((current) => ({ ...current, ...patch }));
  }

  function patchRrCourt(rowId: string, patch: Partial<RrCourtInput>) {
    clearEntryFeedback();
    setRrCourts((current) => current.map((court) => court.rowId === rowId ? { ...court, ...patch } : court));
    if (patch.formatType !== undefined || patch.playerNames !== undefined) {
      setRrSchedule([]);
      setRrPendingPayload(null);
      setNewPlayerDrafts([]);
    }
  }

  function patchRrScore(rowId: string, patch: Partial<Pick<RrScoreRow, "s1" | "s2">>) {
    clearEntryFeedback();
    setRrSchedule((current) => current.map((court) => ({
      ...court,
      matches: court.matches.map((match) => match.rowId === rowId ? { ...match, ...patch } : match)
    })));
  }

  function buildRoundRobinPayload(): RrPayload | null {
    const courts = rrCourts.map((court, index) => ({
      court: index + 1,
      format_type: court.formatType,
      player_names: [...court.playerNames],
    }));
    const empty = courts.find((court) => court.player_names.length === 0);
    if (empty) {
      setMessage(`Court ${empty.court}: enter player names before generating a schedule.`);
      return null;
    }
    return { source: "next_match_uploader_round_robin_preview", custom_schedule: rrCustomSchedule, schedule_mode: "full", courts };
  }

  async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Match Uploader.");
    const response = await fetch(apiUrl(apiBase, path), {
      ...init,
      headers: {
        ...(init.body === undefined ? {} : { "Content-Type": "application/json" }),
        Authorization: `Bearer ${accessToken}`,
        ...init.headers,
      },
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = apiErrorDetail(payload, response.status);
      throw new MatchUploaderApiError(detail.message, detail.operationKey, detail.uncertain);
    }
    return payload as T;
  }

  async function postJson<T>(path: string, body: unknown): Promise<T> {
    return requestJson<T>(path, { method: "POST", body: JSON.stringify(body) });
  }

  function playerOperationPath(operationKey: string): string {
    const template = status.player_operation_endpoint
      || "/admin/clubs/{club_id}/match-uploader/player-operations/{operation_key}";
    return template
      .replace("{club_id}", encodeURIComponent(clubId))
      .replace("{operation_key}", encodeURIComponent(operationKey));
  }

  async function inspectPlayerBatchOperation() {
    const recovery = playerBatchRecovery;
    if (!recovery) return;
    setCheckingPlayerBatchRecovery(true);
    try {
      const operation = await requestJson<AdminMatchUploaderPlayerBatchOperation>(
        playerOperationPath(recovery.operationKey),
      );
      const operationStatus = String(operation.status || "unknown");
      const recoveredResult = playerBatchOperationResult(operation);
      if (operationStatus === "completed") {
        if (recoveredResult?.players?.length) {
          setKnownPlayers((current) => mergePlayers(current, recoveredResult.players || []));
        }
        clearDirectMatchIdempotencyKey(recovery.operationScope, recovery.operationKey);
        clearPlayerBatchRecovery();
        setMessage(
          `Exact player operation ${recovery.operationKey} completed. ${recoveredResult?.accepted_count ?? recoveredResult?.players?.length ?? 0} player profile(s) were reconciled.`,
        );
        if (recovery.continueRoundRobin && rrPendingPayload) {
          await previewRoundRobin(rrPendingPayload);
        }
        return;
      }
      if (operationStatus === "failed") {
        clearDirectMatchIdempotencyKey(recovery.operationScope, recovery.operationKey);
        clearPlayerBatchRecovery();
        setMessage(
          `Exact player operation ${recovery.operationKey} is confirmed failed and did not complete. Review the player list, then submit a new reviewed request if it is still needed.`,
        );
        return;
      }
      const reconciled = await requestJson<AdminMatchUploaderCreatePlayersResult>(
        `${playerOperationPath(recovery.operationKey)}/reconcile`,
        {
          method: "POST",
          body: JSON.stringify({
            confirmation_text: "RECONCILE PLAYER BATCH",
            source: "next_match_uploader_player_operation_reconcile",
          }),
        },
      );
      if (reconciled.ok) {
        if (reconciled.players?.length) {
          setKnownPlayers((current) => mergePlayers(current, reconciled.players || []));
        }
        clearDirectMatchIdempotencyKey(recovery.operationScope, recovery.operationKey);
        clearPlayerBatchRecovery();
        setMessage(
          `Exact player operation ${recovery.operationKey} completed during reconciliation. ${reconciled.accepted_count ?? reconciled.players?.length ?? 0} player profile(s) were applied from authoritative readback.`,
        );
        if (recovery.continueRoundRobin && rrPendingPayload) {
          await previewRoundRobin(rrPendingPayload);
        }
        return;
      }
      if (reconciled.status === "failed" && reconciled.recovery_required !== true) {
        clearDirectMatchIdempotencyKey(recovery.operationScope, recovery.operationKey);
        clearPlayerBatchRecovery();
        setMessage(
          `Exact player operation ${recovery.operationKey} is proven failed. Review the player list before submitting a new reviewed request.`,
        );
        return;
      }
      const operationMessage = operation.error || operation.error_text || recovery.message;
      retainPlayerBatchRecovery({ ...recovery, status: operationStatus, message: operationMessage });
      setMessage(`Exact player operation ${recovery.operationKey} still needs recovery. Do not create another batch or use a new key.`);
    } catch (error) {
      setMessage(
        `${error instanceof Error ? error.message : "Unable to inspect the player operation."} Operation ${recovery.operationKey} remains retained; do not retry it with a new key.`,
      );
    } finally {
      setCheckingPlayerBatchRecovery(false);
    }
  }

  async function createAndSelectPlayer(
    name: string,
    startingJupr: number,
  ): Promise<PublicPlayer | null> {
    if (!requireReady()) return null;
    if (playerBatchRecovery) {
      setMessage(`Check exact operation ${playerBatchRecovery.operationKey} before creating another player.`);
      return null;
    }
    setCreatingPlayers(true);
    const playersToCreate = normalizeNewPlayerBatch([{ name, starting_jupr: startingJupr }]);
    const request = {
      source: "next_match_uploader_inline_new_player",
      players: playersToCreate,
      reviewed_fingerprint: await reviewedPlayerBatchFingerprint(playersToCreate),
      confirmation_text: "CREATE PLAYERS"
    };
    const operationScope = `match-uploader:${clubId}:create-player:inline`;
    const idempotencyKey = directMatchIdempotencyKey(operationScope, request);
    try {
      const payload = await postJson<AdminMatchUploaderCreatePlayersResult>(
        `/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/players`,
        {
          ...request,
          idempotency_key: idempotencyKey
        },
      );
      clearDirectMatchIdempotencyKey(operationScope, idempotencyKey);
      clearPlayerBatchRecovery();
      const incoming = payload.players || [];
      if (incoming.length) {
        setKnownPlayers((current) => mergePlayers(current, incoming));
      }
      const created = incoming.find(
        (player) =>
          String(player.name).trim().toLocaleLowerCase()
          === name.trim().toLocaleLowerCase(),
      );
      setMessage(
        created
          ? `Created ${created.name}. Continue entering the match.`
          : "The player profile could not be confirmed.",
      );
      return (created as PublicPlayer | undefined) || null;
    } catch (error) {
      const operationKey = error instanceof MatchUploaderApiError && error.operationKey
        ? error.operationKey
        : idempotencyKey;
      const errorMessage = error instanceof Error ? error.message : "Unable to create player.";
      const uncertain = !(error instanceof MatchUploaderApiError) || error.uncertain;
      if (uncertain) {
        retainPlayerBatchRecovery({
          operationKey,
          operationScope,
          status: "uncertain",
          message: errorMessage,
          continueRoundRobin: false,
        });
        setMessage(
          `${errorMessage} The exact create request is retained as ${operationKey}. Check that operation before retrying or creating another player.`,
        );
      } else {
        clearDirectMatchIdempotencyKey(operationScope, idempotencyKey);
        setMessage(errorMessage);
      }
      return null;
    } finally {
      setCreatingPlayers(false);
    }
  }

  async function submitSinglesMatch() {
    setMessage(null);
    setResult(null);
    setSinglesValidationAttempted(true);
    if (!requireReady()) return;
    if (context === "league" && !singlesRow.league) {
      setMessage("Create or activate a Singles league in League Manager, or use Pop-Up / Social.");
      return;
    }
    if (!singlesEnabled) {
      setMessage("Direct singles submission is unavailable until its transactional write and replay path is complete.");
      return;
    }
    const error = validateSingles(singlesRow);
    if (error) {
      setMessage(error);
      return;
    }
    setSaving(true);
    const request = {
      source: "next_match_uploader_singles",
      date: singlesRow.date,
      league: context === "popup" ? "POPUP" : (singlesRow.league || defaultLeague),
      week_tag: context === "popup" ? "" : (singlesRow.weekTag || defaultWeekTag),
      match_type: context === "popup" ? "PopUp" : "Singles",
      is_popup: context === "popup",
      context_type: context === "popup" ? "event" : null,
      context_name: context === "popup" ? popupEventName : undefined,
      t1_p1: Number(singlesRow.playerA),
      t2_p1: Number(singlesRow.playerB),
      score_t1: Number(singlesRow.scoreA),
      score_t2: Number(singlesRow.scoreB),
      rating_scope: singlesRow.ratingScope || undefined
    };
    const idempotencyKey = directMatchIdempotencyKey(
      `match-uploader:${clubId}:singles`,
      request
    );
    try {
      const payload = await postJson<AdminMatchUploaderWriteResult>(`/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/singles`, {
        ...request,
        idempotency_key: idempotencyKey
      });
      if (!payload.ok || payload.match_write_committed === false) {
      setMessage(payload.warnings?.[0] || "The singles match submission could not be confirmed. Review Match Log before retrying.");
      return;
    }
    setResult(payload);
    setSubmissionKind("singles");
      clearDirectMatchIdempotencyKey(
        `match-uploader:${clubId}:singles`,
        idempotencyKey
      );
      setMessage(`Submitted singles match; inserted ${payload.result?.inserted ?? 0} rated singles match.`);
    } catch (error) {
      setMessage(`${error instanceof Error ? error.message : "Unable to submit singles match."} Retry this unchanged form; duplicate protection is active.`);
    } finally {
      setSaving(false);
    }
  }

  async function previewRoundRobin(payload: RrPayload) {
    setGenerating(true);
    setResult(null);
    try {
      const preview = await postJson<AdminMatchUploaderRoundRobinPreview>(`/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/round-robin/preview`, payload);
      if (preview.missing_players?.length) {
        setRrSchedule([]);
        setRrPendingPayload(payload);
        setNewPlayerDrafts(preview.missing_players.map((name) => ({ name, startingJupr: "" })));
        setMessage(`Found ${preview.missing_players.length} new player(s). Create profiles to continue.`);
        return;
      }
      setRrPendingPayload(null);
      setNewPlayerDrafts([]);
      setRrSchedule(previewToSchedule(preview));
      setMessage(`Generated ${preview.match_count ?? 0} round-robin game(s). Enter non-zero scores, then submit scored games.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to generate round-robin schedule.");
    } finally {
      setGenerating(false);
    }
  }

  async function generateRoundRobin() {
    setMessage(null);
    setResult(null);
    if (!requireReady()) return;
    const payload = buildRoundRobinPayload();
    if (payload) await previewRoundRobin(payload);
  }

  async function createPlayersAndContinue() {
    setMessage(null);
    setResult(null);
    if (!requireReady() || !rrPendingPayload) return;
    if (playerBatchRecovery) {
      setMessage(`Check exact operation ${playerBatchRecovery.operationKey} before creating another player batch.`);
      return;
    }
    const playersToCreate = normalizeNewPlayerBatch(newPlayerDrafts.map((draft) => ({ name: draft.name, starting_jupr: Number(draft.startingJupr) })));
    const invalid = playersToCreate.find((player) => !player.name || !Number.isFinite(player.starting_jupr) || player.starting_jupr < 1 || player.starting_jupr > 7);
    if (invalid) {
      setMessage("Each new player needs a name and a Starting JUPR between 1.0 and 7.0.");
      return;
    }
    setCreatingPlayers(true);
    const request = {
      source: "next_match_uploader_new_players",
      players: playersToCreate,
      reviewed_fingerprint: await reviewedPlayerBatchFingerprint(playersToCreate),
      confirmation_text: "CREATE PLAYERS"
    };
    const operationScope = `match-uploader:${clubId}:create-players:round-robin`;
    const idempotencyKey = directMatchIdempotencyKey(operationScope, request);
    try {
      const payload = await postJson<AdminMatchUploaderCreatePlayersResult>(`/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/players`, { ...request, idempotency_key: idempotencyKey });
      clearDirectMatchIdempotencyKey(operationScope, idempotencyKey);
      clearPlayerBatchRecovery();
      if (payload.players?.length) setKnownPlayers((current) => mergePlayers(current, payload.players || []));
      setMessage(`Created or confirmed ${payload.accepted_count ?? playersToCreate.length} player profile(s). Regenerating schedule…`);
      await previewRoundRobin(rrPendingPayload);
    } catch (error) {
      const operationKey = error instanceof MatchUploaderApiError && error.operationKey
        ? error.operationKey
        : idempotencyKey;
      const errorMessage = error instanceof Error ? error.message : "Unable to create players.";
      const uncertain = !(error instanceof MatchUploaderApiError) || error.uncertain;
      if (uncertain) {
        retainPlayerBatchRecovery({
          operationKey,
          operationScope,
          status: "uncertain",
          message: errorMessage,
          continueRoundRobin: true,
        });
        setMessage(`${errorMessage} The exact reviewed batch is retained as ${operationKey}. Check that operation before retrying or creating another batch.`);
      } else {
        clearDirectMatchIdempotencyKey(operationScope, idempotencyKey);
        setMessage(errorMessage);
      }
    } finally {
      setCreatingPlayers(false);
    }
  }

  async function submitMatches(matches: Array<Record<string, unknown>>, source: string, kind: "manual" | "round_robin") {
    setSaving(true);
    const request = { source, matches };
    const operationScope = `match-uploader:${clubId}:batch`;
    const idempotencyKey = directMatchIdempotencyKey(
      operationScope,
      request
    );
    try {
      const payload = await postJson<AdminMatchUploaderWriteResult>(`/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/batch`, {
        ...request,
        idempotency_key: idempotencyKey
      });
      if (!payload.ok || payload.match_write_committed === false) {
      setMessage(payload.warnings?.[0] || "The match submission could not be confirmed. Review Match Log before retrying.");
      return;
    }
    setResult(payload);
    setSubmissionKind(kind);
      clearDirectMatchIdempotencyKey(operationScope, idempotencyKey);
      const handoff = payload.auto_player_updates;
      const handoffSummary = handoff?.mode === "auto_sent"
        ? ` Player-update email: ${handoff.sent ?? 0} sent, ${handoff.skipped ?? 0} skipped, ${handoff.errors ?? 0} error(s).`
        : handoff?.mode
          ? " Player-update email was not sent in staging."
          : "";
      setMessage(`Submitted ${payload.submitted_count ?? matches.length} row(s); inserted ${payload.result?.inserted ?? 0} rated match(es).${handoffSummary}`);
    } catch (error) {
      setMessage(`${error instanceof Error ? error.message : "Unable to submit matches."} Retry this unchanged batch; duplicate protection is active.`);
    } finally {
      setSaving(false);
    }
  }

  async function submitManualBatch() {
    setMessage(null);
    setResult(null);
    setManualValidationAttempted(true);
    if (!requireReady()) return;
    if (context === "league" && !defaultLeague) {
      setMessage("Create or activate a Doubles league in League Manager, or use Pop-Up / Social.");
      return;
    }
    const enteredRows = rows.filter(rowHasEnteredData);
    const validationRows = enteredRows.length ? enteredRows : rows.slice(0, 1);
    const errors = validationRows.map((row) => validateRequiredRow(row, rows.indexOf(row))).filter(Boolean) as string[];
    if (errors.length) {
      setMessage(errors[0]);
      return;
    }
    const matches = readyRows.map((row) => ({
      date: row.date,
      league: context === "popup" ? "POPUP" : (row.league || defaultLeague),
      week_tag: context === "popup" ? "" : row.weekTag,
      match_type: matchType,
      rating_scope: context === "popup"
        ? (row.ratingScope === "unrated" ? "unrated" : "overall_only")
        : (row.ratingScope || undefined),
      is_popup: context === "popup",
      context_type: context === "popup" ? "event" : null,
      context_name: context === "popup" ? popupEventName : undefined,
      t1_p1: Number(row.t1p1),
      t1_p2: Number(row.t1p2),
      t2_p1: Number(row.t2p1),
      t2_p2: Number(row.t2p2),
      score_t1: Number(row.s1),
      score_t2: Number(row.s2)
    }));
    if (!matches.length) {
      setMessage("Enter at least one complete match row.");
      return;
    }
    await submitMatches(matches, "next_match_uploader_manual_batch", "manual");
  }

  async function submitRoundRobinScores() {
    setMessage(null);
    setResult(null);
    if (!requireReady()) return;
    if (context === "league" && !defaultLeague) {
      setMessage("Create or activate a Doubles league in League Manager, or use Pop-Up / Social.");
      return;
    }
    const matches = scoredRrRows.map((row) => ({
      date: defaultDate,
      league: context === "popup" ? "POPUP" : defaultLeague,
      week_tag: context === "popup" ? "" : defaultWeekTag,
      match_type: matchType,
      rating_scope: context === "popup" ? "overall_only" : undefined,
      is_popup: context === "popup",
      context_type: context === "popup" ? "event" : null,
      context_name: context === "popup" ? popupEventName : undefined,
      t1_p1: row.t1p1,
      t1_p2: row.t1p2,
      t2_p1: row.t2p1,
      t2_p2: row.t2p2,
      score_t1: Number(row.s1),
      score_t2: Number(row.s2)
    }));
    if (!matches.length) {
      setMessage("Enter at least one non-zero round-robin score before submitting.");
      return;
    }
    await submitMatches(matches, "next_match_uploader_round_robin", "round_robin");
  }

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Next Match Uploader is disabled</h2>
        <p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the Match Uploader pilot flag on FastAPI."}</p>
      </article>
    );
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Match entry setup</h2>
        <p style={{ color: "#475569" }}>
          Doubles manual/batch and round-robin entry use the existing doubles/overall rating path.
          {singlesEnabled ? " Singles input writes to the separate singles rating." : " Direct singles entry remains unavailable until its write and replay path is transaction-safe."}
        </p>
        {status.warnings?.length ? (
          <ul style={{ color: "#92400e" }}>
            {status.warnings.map((warning) => <li key={warning}>{warning}</li>)}
          </ul>
        ) : null}
        <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "0.75rem" }}>
          <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
          <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
            {accessToken ? "Ready to send authorized Match Uploader requests." : sessionLoading ? "Checking admin session…" : "Sign in before generating schedules, creating players, or submitting matches."}
          </p>
          {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
          {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
        </div>
        <div className={styles.setupGrid}>
          <label className={styles.field}><strong>Entry method</strong><br /><select value={entryMethod} onChange={(event) => changeEntryMethod(event.target.value as "singles" | "manual" | "round_robin")} style={inputStyle}>{singlesEnabled ? <option value="singles">Singles match</option> : null}<option value="manual">Doubles manual / batch</option><option value="round_robin">Doubles round robin</option></select></label>
          <label className={styles.field}><strong>Context</strong><br /><select value={context} onChange={(event) => changeContext(event.target.value as "league" | "popup")} style={inputStyle}><option value="league">Official League</option><option value="popup">Pop-Up / Social</option></select></label>
          <label className={styles.field}><strong>Default date</strong><br /><input value={defaultDate} onChange={(event) => changeDefaultDate(event.target.value)} type="date" style={inputStyle} /></label>
          {context === "league" ? <label className={styles.field}><strong>Default {activeLeagueFormatLabel} league</strong><br /><select value={defaultLeague} onChange={(event) => changeDefaultLeague(event.target.value)} disabled={!hasActiveOfficialLeague} style={inputStyle}>{hasActiveOfficialLeague ? activeLeagueOptions.map((item) => <option key={item}>{item}</option>) : <option value="">No active {activeLeagueFormatLabel} leagues</option>}</select></label> : null}
          {context === "league" ? <label className={styles.field}><strong>Default week/session</strong><br /><select value={defaultWeekTag} onChange={(event) => changeDefaultWeekTag(event.target.value)} style={inputStyle}>{status.week_tag_options.map((item) => <option key={item}>{item}</option>)}</select></label> : null}
          {context === "popup" ? <label className={styles.field}><strong>Pop-Up event name</strong><br /><input value={popupEventName} onChange={(event) => { clearEntryFeedback(); setPopupEventName(event.target.value); }} style={inputStyle} /></label> : null}
        </div>
        {context === "league" && !hasActiveOfficialLeague ? <p role="alert" style={{ color: "#92400e", marginBottom: 0 }}><strong>No active {activeLeagueFormatLabel} leagues.</strong> Create one in League Manager or use Pop-Up / Social.</p> : null}
      </article>

      {playerBatchRecovery ? (
        <article aria-live="polite" style={{ ...cardStyle, borderColor: "#f59e0b", background: "#fffbeb" }}>
          <h2 style={{ marginTop: 0 }}>Player creation needs exact-operation recovery</h2>
          <p style={{ color: "#92400e" }}>
            The create response was not conclusive. Do not submit another player batch or use a new key until this exact operation is checked.
          </p>
          <p><strong>Operation key:</strong> <code style={{ overflowWrap: "anywhere" }}>{playerBatchRecovery.operationKey}</code></p>
          <p><strong>Last known status:</strong> {playerBatchRecovery.status.replace(/_/g, " ")}</p>
          <p>{playerBatchRecovery.message}</p>
          <button
            type="button"
            onClick={inspectPlayerBatchOperation}
            disabled={checkingPlayerBatchRecovery || !accessToken}
            style={ghostButtonStyle}
          >
            {checkingPlayerBatchRecovery ? "Checking and reconciling…" : "Check exact operation and reconcile"}
          </button>
        </article>
      ) : null}

      {entryMethod === "singles" && singlesEnabled ? (
        <article style={{ ...cardStyle, background: "#f8fafc" }}>
          <h2 style={{ marginTop: 0 }}>Singles match entry</h2>
          <p style={{ color: "#475569" }}>Use this for one-on-one games only. Official League and Pop-Up / Social control match organization; rating changes always use the separate singles rating.</p>
          <div className={`${styles.metadataGrid} ${context === "popup" ? styles.popupMetadataGrid : ""}`}>
            <label className={styles.field}><strong>Date</strong><br /><input type="date" value={singlesRow.date} onChange={(event) => patchSingles({ date: event.target.value })} style={inputStyle} /></label>
            {context === "league" ? <label className={styles.field}><strong>Singles league</strong><br /><select value={singlesRow.league} onChange={(event) => patchSingles({ league: event.target.value })} disabled={!hasActiveOfficialLeague} style={inputStyle}>{hasActiveOfficialLeague ? activeLeagueOptions.map((item) => <option key={item}>{item}</option>) : <option value="">No active singles leagues</option>}</select></label> : null}
            {context === "league" ? <label className={styles.field}><strong>Week / session</strong><br /><input value={singlesRow.weekTag} onChange={(event) => patchSingles({ weekTag: event.target.value })} style={inputStyle} /></label> : null}
            <label className={styles.field}><strong>Rating scope</strong><br /><select value={singlesRow.ratingScope} onChange={(event) => patchSingles({ ratingScope: event.target.value as SinglesRow["ratingScope"] })} style={inputStyle}><option value="">Rated singles</option><option value="unrated">Unrated / record only</option></select></label>
          </div>
          <div className={styles.teamsViewport}>
            <div className={styles.teamsGrid}>
              <section aria-label="Singles Player 1" className={styles.teamPanel}>
                <h4 style={{ margin: 0 }}>Player 1</h4>
                <SearchablePlayerInput key={`singles-player-a-${singlesRow.playerA || "empty"}`} inputId="singles-player-a" label="Player" value={singlesRow.playerA} players={singlesPlayerOptions(singlesRow.playerA)} allPlayers={knownPlayers} invalid={singlesValidationAttempted && (!singlesRow.playerA || singlesPlayersDuplicate)} disabled={saving || creatingPlayers} onChange={(playerA) => patchSingles({ playerA })} onCreate={createAndSelectPlayer} />
              </section>
              <section aria-label="Singles scores" className={styles.scorePanel}>
                <label className={styles.scoreField}><strong>Player 1 score</strong><br /><input type="number" min={0} max={99} value={singlesRow.scoreA} onChange={(event) => patchSingles({ scoreA: event.target.value })} aria-invalid={singlesScoreInvalid || undefined} style={singlesScoreInvalid ? { ...inputStyle, border: "2px solid #dc2626", background: "#fef2f2" } : inputStyle} /></label>
                <label className={styles.scoreField}><strong>Player 2 score</strong><br /><input type="number" min={0} max={99} value={singlesRow.scoreB} onChange={(event) => patchSingles({ scoreB: event.target.value })} aria-invalid={singlesScoreInvalid || undefined} style={singlesScoreInvalid ? { ...inputStyle, border: "2px solid #dc2626", background: "#fef2f2" } : inputStyle} /></label>
              </section>
              <section aria-label="Singles Player 2" className={styles.teamPanel}>
                <h4 style={{ margin: 0 }}>Player 2</h4>
                <SearchablePlayerInput key={`singles-player-b-${singlesRow.playerB || "empty"}`} inputId="singles-player-b" label="Player" value={singlesRow.playerB} players={singlesPlayerOptions(singlesRow.playerB)} allPlayers={knownPlayers} invalid={singlesValidationAttempted && (!singlesRow.playerB || singlesPlayersDuplicate)} disabled={saving || creatingPlayers} onChange={(playerB) => patchSingles({ playerB })} onCreate={createAndSelectPlayer} />
              </section>
            </div>
          </div>
          {singlesError ? <p role="alert" style={{ color: "#b91c1c", marginBottom: 0 }}><strong>{singlesError}</strong></p> : null}
          <p><button type="button" onClick={submitSinglesMatch} disabled={saving || !accessToken || !singlesEnabled} style={buttonStyle}>{saving ? "Submitting…" : "Submit singles match"}</button></p>
        </article>
      ) : null}

      {entryMethod === "manual" ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Doubles manual / batch score entry</h2>
          <p style={{ color: "#475569" }}>Empty rows are ignored. Submitted rows must have four distinct players and a non-zero score.</p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "0.75rem" }}>
            <button type="button" onClick={() => { clearEntryFeedback(); setManualValidationAttempted(false); setRows((current) => [...current, newMatchRow(defaultDate, defaultManualWeekTag, defaultManualRatingScope, defaultLeague)]); }} disabled={rows.length >= status.max_batch_rows} style={ghostButtonStyle}>Add 1 Match</button>
            <button type="button" onClick={() => { clearEntryFeedback(); setManualValidationAttempted(false); setRows((current) => [...current, ...Array.from({ length: 5 }, () => newMatchRow(defaultDate, defaultManualWeekTag, defaultManualRatingScope, defaultLeague))].slice(0, status.max_batch_rows)); }} disabled={rows.length >= status.max_batch_rows} style={ghostButtonStyle}>Add 5 Matches</button>
            {rows.some(rowHasEnteredData) ? (
              <button type="button" onClick={() => setRemoveAllDialogOpen(true)} disabled={saving} style={dangerButtonStyle}>Remove All</button>
            ) : (
              <button type="button" onClick={removeAllRows} disabled={rows.length <= 1} style={ghostButtonStyle}>Remove All</button>
            )}
            {removeAllDialogOpen ? (
              <RemoveAllMatchesDialog
                onClose={() => setRemoveAllDialogOpen(false)}
                onKeepRows={keepRowsWithData}
                onRemoveAll={removeAllRows}
              />
            ) : null}
          </div>
          <div style={{ display: "grid", gap: "0.75rem" }}>
            {rows.map((row, index) => {
              const validateThisRow = manualValidationAttempted && (rowHasEnteredData(row) || (!rows.some(rowHasEnteredData) && index === 0));
              const rowError = validateThisRow ? validateRequiredRow(row, index) : null;
              const scoreInvalid = validateThisRow && Number(row.s1 || 0) + Number(row.s2 || 0) <= 0;
              return (
              <div key={row.rowId} style={{ border: rowError ? "2px solid #dc2626" : "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: rowError ? "#fff7f7" : rowHasEnteredData(row) ? "#f8fafc" : "white" }}>
                <div style={{ display: "flex", justifyContent: "space-between", gap: "0.5rem", alignItems: "center", marginBottom: "0.5rem" }}>
                  <strong>Match {index + 1}</strong>
                  {rowHasEnteredData(row) ? (
                    <ConfirmAction
                      triggerLabel="Remove match"
                      title={`Remove Match ${index + 1}?`}
                      description="This match contains entered data. Removing it cannot be undone from this screen."
                      confirmLabel="Yes, remove match"
                      confirmationText="REMOVE"
                      tone="danger"
                      disabled={saving}
                      busy={saving}
                      onConfirm={async () => {
                        removeRow(row.rowId);
                        return actionSuccess("Match removed", "The match row was removed from this unsaved batch.");
                      }}
                    />
                  ) : (
                    <button type="button" onClick={() => removeRow(row.rowId)} disabled={rows.length <= 1}>Remove match</button>
                  )}
                </div>
                <div className={`${styles.metadataGrid} ${context === "popup" ? styles.popupMetadataGrid : ""}`}>
                  <label className={styles.field}><strong>Date</strong><br /><input type="date" value={row.date} onChange={(event) => patchRow(row.rowId, { date: event.target.value })} style={inputStyle} /></label>
                  {context === "league" ? <label className={styles.field}><strong>League</strong><br /><select value={row.league} onChange={(event) => patchRow(row.rowId, { league: event.target.value })} style={inputStyle}>{activeLeagueOptions.map((item) => <option key={item}>{item}</option>)}</select></label> : null}
                  {context === "league" ? <label className={styles.field}><strong>Week / session</strong><br /><input value={row.weekTag} onChange={(event) => patchRow(row.rowId, { weekTag: event.target.value })} style={inputStyle} /></label> : null}
                  <label className={styles.field}><strong>Rating scope</strong><br /><select value={row.ratingScope} onChange={(event) => patchRow(row.rowId, { ratingScope: event.target.value as MatchRow["ratingScope"] })} style={inputStyle}>{context === "popup" ? <><option value="overall_only">Overall only (rated)</option><option value="unrated">Unrated / record only</option></> : <><option value="">Overall + league</option><option value="overall_only">Overall only</option><option value="unrated">Unrated / record only</option></>}</select></label>
                </div>
                <div className={styles.teamsViewport}>
                  <div className={styles.teamsGrid}>
                    <section aria-label={`Match ${index + 1} Team 1`} className={styles.teamPanel}>
                      <h4 style={{ margin: 0 }}>Team 1</h4>
                      <SearchablePlayerInput inputId={`${row.rowId}-t1p1`} label="Player 1" value={row.t1p1} players={playerOptionsFor(row, row.t1p1)} allPlayers={knownPlayers} invalid={validateThisRow && !row.t1p1} disabled={saving || creatingPlayers} onChange={(t1p1) => patchRow(row.rowId, { t1p1 })} onCreate={createAndSelectPlayer} />
                      <SearchablePlayerInput inputId={`${row.rowId}-t1p2`} label="Player 2" value={row.t1p2} players={playerOptionsFor(row, row.t1p2)} allPlayers={knownPlayers} invalid={validateThisRow && !row.t1p2} disabled={saving || creatingPlayers} onChange={(t1p2) => patchRow(row.rowId, { t1p2 })} onCreate={createAndSelectPlayer} />
                    </section>
                    <section aria-label={`Match ${index + 1} scores`} className={styles.scorePanel}>
                      <label className={styles.scoreField}><strong>Team 1 score</strong><br /><input value={row.s1} onChange={(event) => patchRow(row.rowId, { s1: event.target.value })} aria-invalid={scoreInvalid || undefined} type="number" min={0} max={99} style={scoreInvalid ? { ...inputStyle, border: "2px solid #dc2626", background: "#fef2f2" } : inputStyle} /></label>
                      <label className={styles.scoreField}><strong>Team 2 score</strong><br /><input value={row.s2} onChange={(event) => patchRow(row.rowId, { s2: event.target.value })} aria-invalid={scoreInvalid || undefined} type="number" min={0} max={99} style={scoreInvalid ? { ...inputStyle, border: "2px solid #dc2626", background: "#fef2f2" } : inputStyle} /></label>
                    </section>
                    <section aria-label={`Match ${index + 1} Team 2`} className={styles.teamPanel}>
                      <h4 style={{ margin: 0 }}>Team 2</h4>
                      <SearchablePlayerInput inputId={`${row.rowId}-t2p1`} label="Player 1" value={row.t2p1} players={playerOptionsFor(row, row.t2p1)} allPlayers={knownPlayers} invalid={validateThisRow && !row.t2p1} disabled={saving || creatingPlayers} onChange={(t2p1) => patchRow(row.rowId, { t2p1 })} onCreate={createAndSelectPlayer} />
                      <SearchablePlayerInput inputId={`${row.rowId}-t2p2`} label="Player 2" value={row.t2p2} players={playerOptionsFor(row, row.t2p2)} allPlayers={knownPlayers} invalid={validateThisRow && !row.t2p2} disabled={saving || creatingPlayers} onChange={(t2p2) => patchRow(row.rowId, { t2p2 })} onCreate={createAndSelectPlayer} />
                    </section>
                  </div>
                </div>
                {rowError ? <p role="alert" style={{ color: "#b91c1c", marginBottom: 0 }}><strong>{rowError}</strong></p> : null}
              </div>
              );
            })}
          </div>
          <p><strong>Ready rows:</strong> {readyRows.length} / {rows.length}</p>
          <button type="button" onClick={submitManualBatch} disabled={saving || !accessToken} style={buttonStyle}>{saving ? "Submitting…" : "Submit batch"}</button>
          {message && !result ? <p aria-live="polite" role={messageIsError ? "alert" : "status"} style={{ color: messageIsError ? "#b91c1c" : "#166534" }}>{message}</p> : null}
        </article>
      ) : null}

      {entryMethod === "round_robin" ? (
        <>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Doubles round-robin generator</h2>
            <p style={{ color: "#475569" }}>Enter one court per group. The API checks names, creates missing players when requested, then returns the Python-generated schedule for score entry.</p>
            <div style={{ display: "grid", gap: "0.75rem" }}>
              {rrCourts.map((court, index) => (
                <div key={court.rowId} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: "#f8fafc" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", gap: "0.5rem", alignItems: "center", marginBottom: "0.5rem" }}><strong>Court {index + 1}</strong><button type="button" onClick={() => setRrCourts((current) => current.filter((item) => item.rowId !== court.rowId))} disabled={rrCourts.length <= 1}>Remove</button></div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
                    <label><strong>Format</strong><br /><select value={court.formatType} onChange={(event) => patchRrCourt(court.rowId, { formatType: event.target.value })} style={inputStyle}>{(status.round_robin_format_options || [firstFormat]).map((format) => <option key={format}>{format}</option>)}</select><span style={{ display: "block", color: "#64748b", fontSize: "0.85rem", marginTop: "0.25rem" }}>Expected games: {status.round_robin_expected_games?.[court.formatType] ?? "—"}</span></label>
                    <SearchablePlayerMultiInput
                      inputId={`${court.rowId}-players`}
                      label="Players"
                      values={court.playerNames}
                      players={knownPlayers}
                      disabled={generating || creatingPlayers}
                      onChange={(playerNames) => patchRrCourt(court.rowId, { playerNames })}
                      onCreate={createAndSelectPlayer}
                    />
                  </div>
                </div>
              ))}
            </div>
            <p><button type="button" onClick={() => setRrCourts((current) => [...current, newRoundRobin(firstFormat)])} disabled={rrCourts.length >= 10} style={ghostButtonStyle}>Add round robin</button></p>
            <label><strong>Custom schedule override</strong><br /><textarea value={rrCustomSchedule} onChange={(event) => setRrCustomSchedule(event.target.value)} rows={3} placeholder="Optional lines like: 1 2 3 4" style={inputStyle} /></label>
            <p><button type="button" onClick={generateRoundRobin} disabled={generating || creatingPlayers || !accessToken} style={buttonStyle}>{generating ? "Generating…" : "Generate schedule"}</button></p>
          </article>

          {newPlayerDrafts.length ? (
            <article style={{ ...cardStyle, borderColor: "#f59e0b", background: "#fffbeb" }}>
              <h2 style={{ marginTop: 0 }}>New players found — create profiles to continue</h2>
              <p style={{ color: "#92400e" }}>Review each new player and set a Starting JUPR.</p>
              <div style={{ display: "grid", gap: "0.5rem" }}>
                {newPlayerDrafts.map((draft, index) => (
                  <div key={`${draft.name}-${index}`} style={{ display: "grid", gridTemplateColumns: "minmax(180px, 1fr) 140px", gap: "0.5rem" }}>
                    <input value={draft.name} onChange={(event) => setNewPlayerDrafts((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, name: event.target.value } : item))} style={inputStyle} />
                    <input value={draft.startingJupr} onChange={(event) => setNewPlayerDrafts((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, startingJupr: event.target.value } : item))} type="number" min={1} max={7} step={0.01} style={inputStyle} />
                  </div>
                ))}
              </div>
              <p><button type="button" onClick={createPlayersAndContinue} disabled={creatingPlayers || generating || !accessToken || Boolean(playerBatchRecovery)} style={buttonStyle}>{creatingPlayers ? "Creating…" : "Create Players & Continue"}</button></p>
            </article>
          ) : null}

          {rrSchedule.length ? (
            <article style={cardStyle}>
              <h2 style={{ marginTop: 0 }}>Round-robin scores</h2>
              <p style={{ color: "#475569" }}>Zero-zero games are left unsubmitted, matching the Streamlit flow.</p>
              <div style={{ display: "grid", gap: "1rem" }}>
                {rrSchedule.map((court) => (
                  <div key={`court-${court.court}`} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem" }}>
                    <h3 style={{ marginTop: 0 }}>Court {court.court} · {court.formatType}</h3>
                    <div style={{ display: "grid", gap: "0.5rem" }}>
                      {court.matches.map((match) => (
                        <div key={match.rowId} style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: "0.5rem", alignItems: "center" }}>
                          <div><strong>{match.label}</strong><br />{match.t1.map((player) => player.name).join(" / ")}</div>
                          <label><strong>Team 1 score</strong><br /><input value={match.s1} onChange={(event) => patchRrScore(match.rowId, { s1: event.target.value })} type="number" min={0} max={99} style={inputStyle} /></label>
                          <label><strong>Team 2 score</strong><br /><input value={match.s2} onChange={(event) => patchRrScore(match.rowId, { s2: event.target.value })} type="number" min={0} max={99} style={inputStyle} /></label>
                          <div>{match.t2.map((player) => player.name).join(" / ")}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              <p><strong>Scored games:</strong> {scoredRrRows.length} / {rrSchedule.reduce((total, court) => total + court.matches.length, 0)}</p>
              <button type="button" onClick={submitRoundRobinScores} disabled={saving || !accessToken || !scoredRrRows.length} style={buttonStyle}>{saving ? "Submitting…" : "Submit scored round-robin games"}</button>
            </article>
          ) : null}
        </>
      ) : null}

      {message && !result && entryMethod !== "manual" ? <p aria-live="polite" role={messageIsError ? "alert" : "status"} style={{ color: messageIsError ? "#b91c1c" : "#166534" }}>{message}</p> : null}

      {result ? <SubmissionResultDialog result={result} roundRobinRecords={rrResultRecords} submissionKind={submissionKind} onClose={acknowledgeSubmission} /> : null}
    </section>
  );
}
