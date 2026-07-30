"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type { PublicPlayer } from "@/lib/api";
import type {
  AdminMatchUploaderCreatePlayersResult,
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
type SearchablePlayerInputProps = {
  inputId: string;
  label: string;
  value: string;
  players: PublicPlayer[];
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

function newMatchRow(
  date: string = todayIsoDate(),
  weekTag: string = "Week 1",
  ratingScope: MatchRow["ratingScope"] = "",
  league: string = "Open",
): MatchRow {
  return { rowId: randomId("row"), date, league, weekTag, ratingScope, t1p1: "", t1p2: "", t2p1: "", t2p2: "", s1: "0", s2: "0" };
}

function newSinglesRow(): SinglesRow {
  return { date: todayIsoDate(), league: "Singles", weekTag: "Singles", playerA: "", playerB: "", scoreA: "0", scoreB: "0", ratingScope: "" };
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
  return /\b(unable|unavailable|disabled|required|must|cannot|could not|not configured|sign in|error|invalid|select|enter|choose|failed)\b/i.test(message);
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

function mergePlayers(current: PublicPlayer[], incoming: NonNullable<AdminMatchUploaderCreatePlayersResult["players"]>): PublicPlayer[] {
  const byId = new Map<string, PublicPlayer>();
  for (const player of current) byId.set(String(player.id), player);
  for (const player of incoming) byId.set(String(player.id), player as PublicPlayer);
  return Array.from(byId.values()).sort((left, right) => String(left.name).localeCompare(String(right.name)));
}

function SearchablePlayerInput({
  inputId,
  label,
  value,
  players,
  disabled = false,
  invalid = false,
  onChange,
  onCreate,
}: SearchablePlayerInputProps) {
  const selected = players.find((player) => String(player.id) === value);
  const selectedName = selected ? String(selected.name) : "";
  const [query, setQuery] = useState(selectedName);
  const [startingJupr, setStartingJupr] = useState("3.5");
  const [creating, setCreating] = useState(false);
  const cleanedQuery = query.replace(/\s+/g, " ").trim();
  const exactPlayer = players.find(
    (player) =>
      String(player.name).trim().toLocaleLowerCase()
      === cleanedQuery.toLocaleLowerCase(),
  );
  const matchingPlayers = cleanedQuery
    ? players.filter((player) =>
        String(player.name).trim().toLocaleLowerCase().includes(cleanedQuery.toLocaleLowerCase()),
      )
    : players;
  const numericStartingJupr = Number(startingJupr);
  const validatedInputStyle = invalid
    ? { ...inputStyle, border: "2px solid #dc2626", background: "#fef2f2" }
    : inputStyle;

  useEffect(() => {
    setQuery(selectedName);
  }, [selectedName]);

  async function createAndSelect() {
    if (
      !cleanedQuery
      || exactPlayer
      || !Number.isFinite(numericStartingJupr)
      || numericStartingJupr < 1
      || numericStartingJupr > 7
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
      {cleanedQuery && !exactPlayer && matchingPlayers.length === 0 ? (
        <div style={{ display: "grid", gridTemplateColumns: "minmax(100px, 1fr) auto", gap: "0.35rem", marginTop: "0.35rem", alignItems: "end" }}>
          <label htmlFor={`${inputId}-starting-jupr`}>
            <span style={{ display: "block", color: "#475569", fontSize: "0.8rem" }}>Starting JUPR</span>
            <input
              id={`${inputId}-starting-jupr`}
              type="number"
              min={1}
              max={7}
              step={0.01}
              value={startingJupr}
              disabled={disabled || creating}
              onChange={(event) => setStartingJupr(event.target.value)}
              style={inputStyle}
            />
          </label>
          <button
            type="button"
            onClick={createAndSelect}
            disabled={disabled || creating}
            style={ghostButtonStyle}
          >
            {creating ? "Creating…" : `Create “${cleanedQuery}”`}
          </button>
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
  const [startingJupr, setStartingJupr] = useState("3.5");
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
      || !Number.isFinite(numericStartingJupr)
      || numericStartingJupr < 1
      || numericStartingJupr > 7
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
          onChange={(event) => setQuery(event.target.value)}
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
            <span style={{ display: "block", color: "#475569", fontSize: "0.8rem" }}>Starting JUPR</span>
            <input
              id={`${inputId}-starting-jupr`}
              type="number"
              min={1}
              max={7}
              step={0.01}
              value={startingJupr}
              onChange={(event) => setStartingJupr(event.target.value)}
              disabled={disabled || creating}
              style={inputStyle}
            />
          </label>
          <button type="button" onClick={createAndAdd} disabled={disabled || creating} style={ghostButtonStyle}>
            {creating ? "Creating…" : `Create & add “${cleanedQuery}”`}
          </button>
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


function RemoveAllMatchesDialog({
  onClose,
  onKeepRows,
  onRemoveAll,
}: {
  onClose: () => void;
  onKeepRows: () => void;
  onRemoveAll: () => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
    return () => {
      if (dialog?.open) dialog.close();
    };
  }, []);

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby="remove-all-matches-title"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      style={{ width: "min(620px, calc(100vw - 2rem))", border: 0, borderRadius: "16px", padding: 0, boxShadow: "0 24px 70px rgba(15, 23, 42, 0.35)" }}
    >
      <div style={{ padding: "1.25rem" }}>
        <h2 id="remove-all-matches-title" style={{ marginTop: 0 }}>Remove entered matches?</h2>
        <p>Choose whether to keep completed or partially entered rows, remove only blank rows, or clear the entire batch.</p>
        <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.5rem", flexWrap: "wrap" }}>
          <button type="button" onClick={onClose} style={ghostButtonStyle}>No, go back</button>
          <button type="button" onClick={onKeepRows} style={ghostButtonStyle}>Keep rows with data</button>
          <button type="button" onClick={onRemoveAll} style={dangerButtonStyle}>Yes, remove all</button>
        </div>
      </div>
    </dialog>
  );
}

function SubmissionResultDialog({
  result,
  onClose,
}: {
  result: AdminMatchUploaderWriteResult;
  onClose: () => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
    return () => {
      if (dialog?.open) dialog.close();
    };
  }, []);

  const inserted = result.result?.inserted ?? 0;
  const email = result.auto_player_updates;
  const emailSummary = email?.mode === "auto_sent"
    ? `${email.sent ?? 0} sent, ${email.skipped ?? 0} skipped, ${email.errors ?? 0} error(s).`
    : "Not sent in staging.";

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby="match-submission-result-title"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      style={{ width: "min(720px, calc(100vw - 2rem))", maxHeight: "calc(100vh - 2rem)", overflowY: "auto", border: 0, borderRadius: "16px", padding: 0, boxShadow: "0 24px 70px rgba(15, 23, 42, 0.35)" }}
    >
      <div style={{ padding: "1.25rem" }}>
        <h2 id="match-submission-result-title" style={{ marginTop: 0 }}>Match submission complete</h2>
        <p role="status" style={{ color: "#166534" }}>
          Successfully inserted {inserted} rated match{inserted === 1 ? "" : "es"}.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem" }}>
          <div><strong>Inserted</strong><br />{inserted}</div>
          <div><strong>Match write</strong><br />{result.match_write_committed ? "Committed" : "Review required"}</div>
          <div><strong>Rating type</strong><br />{result.feedback?.rating_type || result.result?.match_format || "doubles/overall"}</div>
          <div><strong>Skipped incomplete</strong><br />{result.result?.skipped_incomplete ?? 0}</div>
          <div><strong>Skipped empty</strong><br />{result.result?.skipped_empty ?? 0}</div>
          <div><strong>Skipped unrated</strong><br />{result.result?.skipped_unrated ?? 0}</div>
        </div>
        {email ? <p style={{ marginTop: "1rem" }}><strong>Player-update email:</strong> {emailSummary}</p> : null}
        {result.feedback?.affected_players?.length ? (
          <div style={{ overflowX: "auto", marginTop: "1rem" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Player</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Before</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>After</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Change</th></tr></thead>
              <tbody>
                {result.feedback.affected_players.map((player) => (
                  <tr key={player.id}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{player.name}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{ratingLabel(player.rating_before)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{ratingLabel(player.rating_after)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{deltaLabel(player.rating_delta)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
        {result.warnings?.length ? <ul style={{ color: "#92400e" }}>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
        <p style={{ display: "flex", justifyContent: "flex-end", marginBottom: 0 }}>
          <button type="button" onClick={onClose} style={buttonStyle}>OK</button>
        </p>
      </div>
    </dialog>
  );
}

export default function MatchUploaderForm({ apiBase, clubId, players, status }: Props) {
  const firstFormat = status.round_robin_format_options?.[0] || "4-Player";
  const officialLeagueOptions = status.league_options.filter((item) => item.trim().toUpperCase() !== "POPUP");
  const selectableLeagueOptions = officialLeagueOptions.length ? officialLeagueOptions : ["Open"];
  const initialLeague = selectableLeagueOptions[0];
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
  const [singlesRow, setSinglesRow] = useState<SinglesRow>(() => newSinglesRow());
  const [rows, setRows] = useState<MatchRow[]>(() => [newMatchRow(todayIsoDate(), initialWeekTag, "", initialLeague)]);
  const [rrCourts, setRrCourts] = useState<RrCourtInput[]>(() => [newRoundRobin(firstFormat)]);
  const [rrCustomSchedule, setRrCustomSchedule] = useState("");
  const [rrSchedule, setRrSchedule] = useState<RrCourtSchedule[]>([]);
  const [rrPendingPayload, setRrPendingPayload] = useState<RrPayload | null>(null);
  const [newPlayerDrafts, setNewPlayerDrafts] = useState<NewPlayerDraft[]>([]);
  const [saving, setSaving] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [creatingPlayers, setCreatingPlayers] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [result, setResult] = useState<AdminMatchUploaderWriteResult | null>(null);
  const [submissionKind, setSubmissionKind] = useState<"manual" | "round_robin" | "singles" | null>(null);
  const [manualValidationAttempted, setManualValidationAttempted] = useState(false);
  const [removeAllDialogOpen, setRemoveAllDialogOpen] = useState(false);

  const filledRows = rows.filter(isFilled);
  const readyRows = rows.filter((row, index) => isReadyRow(row, index));
  const hasInvalidFilledRows = filledRows.length !== readyRows.length;
  const defaultManualWeekTag = context === "popup" ? "" : defaultWeekTag;
  const defaultManualRatingScope: MatchRow["ratingScope"] = context === "popup" ? "overall_only" : "";
  const rowHasEnteredData = (row: MatchRow) =>
    isFilled(row)
    || row.date !== defaultDate
    || (context === "league" && row.league !== defaultLeague)
    || (context === "league" && row.weekTag !== defaultWeekTag)
    || row.ratingScope !== defaultManualRatingScope;
  const scoredRrRows = rrSchedule.flatMap((court) => court.matches).filter((match) => Number(match.s1 || 0) + Number(match.s2 || 0) > 0);
  const matchType = context === "popup" ? "PopUp" : "Live Match";
  const messageIsError = Boolean(message && (result?.ok === false || isUploaderErrorMessage(message)));

  function clearEntryFeedback() {
    setMessage(null);
    setResult(null);
  }

  function resetManualRows() {
    const priorScope = (readyRows[0] || filledRows[0])?.ratingScope;
    const preservedScope = context === "popup"
      ? (priorScope === "unrated" ? "unrated" : "overall_only")
      : (priorScope || "");
    setRows([newMatchRow(defaultDate, defaultManualWeekTag, preservedScope, defaultLeague)]);
    setManualValidationAttempted(false);
  }

  function changeContext(nextContext: "league" | "popup") {
    clearEntryFeedback();
    setManualValidationAttempted(false);
    setContext(nextContext);
    setRows((current) => current.map((row) => ({
      ...row,
      league: row.league || defaultLeague,
      weekTag: nextContext === "popup" ? "" : (row.weekTag || defaultWeekTag),
      ratingScope: nextContext === "popup"
        ? (row.ratingScope === "unrated" ? "unrated" : "overall_only")
        : (row.ratingScope === "unrated" ? "unrated" : ""),
    })));
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
  }

  function acknowledgeSubmission() {
    if (submissionKind === "manual") resetManualRows();
    if (submissionKind === "singles") {
      setSinglesRow((current) => ({
        ...newSinglesRow(),
        date: current.date,
        league: current.league,
        weekTag: current.weekTag,
        ratingScope: current.ratingScope,
      }));
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

  async function postJson<T>(path: string, body: unknown): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Match Uploader.");
    const response = await fetch(apiUrl(apiBase, path), {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${accessToken}` },
      body: JSON.stringify(body)
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  async function createAndSelectPlayer(
    name: string,
    startingJupr: number,
  ): Promise<PublicPlayer | null> {
    if (!requireReady()) return null;
    setCreatingPlayers(true);
    try {
      const payload = await postJson<AdminMatchUploaderCreatePlayersResult>(
        `/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/players`,
        {
          source: "next_match_uploader_inline_new_player",
          players: [{ name, starting_jupr: startingJupr }],
        },
      );
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
      setMessage(
        error instanceof Error ? error.message : "Unable to create player.",
      );
      return null;
    } finally {
      setCreatingPlayers(false);
    }
  }

  async function submitSinglesMatch() {
    setMessage(null);
    setResult(null);
    if (!requireReady()) return;
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
      league: singlesRow.league || "Singles",
      week_tag: singlesRow.weekTag || "Singles",
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
        setNewPlayerDrafts(preview.missing_players.map((name) => ({ name, startingJupr: "3.5" })));
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
    const playersToCreate = newPlayerDrafts.map((draft) => ({ name: draft.name.trim(), starting_jupr: Number(draft.startingJupr) }));
    const invalid = playersToCreate.find((player) => !player.name || !Number.isFinite(player.starting_jupr) || player.starting_jupr < 1 || player.starting_jupr > 7);
    if (invalid) {
      setMessage("Each new player needs a name and a Starting JUPR between 1.0 and 7.0.");
      return;
    }
    setCreatingPlayers(true);
    try {
      const payload = await postJson<AdminMatchUploaderCreatePlayersResult>(`/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/players`, { source: "next_match_uploader_new_players", players: playersToCreate });
      if (payload.players?.length) setKnownPlayers((current) => mergePlayers(current, payload.players || []));
      setMessage(`Created or confirmed ${payload.accepted_count ?? playersToCreate.length} player profile(s). Regenerating schedule…`);
      await previewRoundRobin(rrPendingPayload);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to create players.");
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
          <label className={styles.field}><strong>Entry method</strong><br /><select value={entryMethod} onChange={(event) => { clearEntryFeedback(); setManualValidationAttempted(false); setEntryMethod(event.target.value as "singles" | "manual" | "round_robin"); }} style={inputStyle}>{singlesEnabled ? <option value="singles">Singles match</option> : null}<option value="manual">Doubles manual / batch</option><option value="round_robin">Doubles round robin</option></select></label>
          {entryMethod !== "singles" ? <label className={styles.field}><strong>Context</strong><br /><select value={context} onChange={(event) => changeContext(event.target.value as "league" | "popup")} style={inputStyle}><option value="league">Official League</option><option value="popup">Pop-Up / Social</option></select></label> : null}
          {entryMethod !== "singles" ? <label className={styles.field}><strong>Default date</strong><br /><input value={defaultDate} onChange={(event) => { const value = event.target.value; clearEntryFeedback(); setDefaultDate(value); setRows((current) => current.map((row) => isFilled(row) ? row : { ...row, date: value })); }} type="date" style={inputStyle} /></label> : null}
          {entryMethod !== "singles" && context === "league" ? <label className={styles.field}><strong>Default league</strong><br /><select value={defaultLeague} onChange={(event) => changeDefaultLeague(event.target.value)} style={inputStyle}>{selectableLeagueOptions.map((item) => <option key={item}>{item}</option>)}</select></label> : null}
          {entryMethod !== "singles" && context === "league" ? <label className={styles.field}><strong>Default week/session</strong><br /><select value={defaultWeekTag} onChange={(event) => changeDefaultWeekTag(event.target.value)} style={inputStyle}>{status.week_tag_options.map((item) => <option key={item}>{item}</option>)}</select></label> : null}
          {entryMethod !== "singles" && context === "popup" ? <label className={styles.field}><strong>Pop-Up event name</strong><br /><input value={popupEventName} onChange={(event) => { clearEntryFeedback(); setPopupEventName(event.target.value); }} style={inputStyle} /></label> : null}
        </div>
      </article>

      {entryMethod === "singles" && singlesEnabled ? (
        <article style={{ ...cardStyle, background: "#f8fafc" }}>
          <h2 style={{ marginTop: 0 }}>Singles match input</h2>
          <p style={{ color: "#475569" }}>Use this for one-on-one games only. This updates each player’s singles rating and writes an official match row with <code>match_format=singles</code>.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem" }}>
            <label><strong>Date</strong><br /><input type="date" value={singlesRow.date} onChange={(event) => patchSingles({ date: event.target.value })} style={inputStyle} /></label>
            <label><strong>Singles league/tag</strong><br /><input value={singlesRow.league} onChange={(event) => patchSingles({ league: event.target.value })} style={inputStyle} /></label>
            <label><strong>Session</strong><br /><input value={singlesRow.weekTag} onChange={(event) => patchSingles({ weekTag: event.target.value })} style={inputStyle} /></label>
            <label><strong>Rating scope</strong><br /><select value={singlesRow.ratingScope} onChange={(event) => patchSingles({ ratingScope: event.target.value as SinglesRow["ratingScope"] })} style={inputStyle}><option value="">Rated singles</option><option value="unrated">Unrated / record only</option></select></label>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem", alignItems: "end", marginTop: "0.75rem" }}>
            <SearchablePlayerInput
              inputId="singles-player-a"
              label="Player A"
              value={singlesRow.playerA}
              players={knownPlayers}
              disabled={saving || creatingPlayers}
              onChange={(playerA) => patchSingles({ playerA })}
              onCreate={createAndSelectPlayer}
            />
            <label><strong>Score A</strong><br /><input type="number" min={0} max={99} value={singlesRow.scoreA} onChange={(event) => patchSingles({ scoreA: event.target.value })} style={inputStyle} /></label>
            <label><strong>Score B</strong><br /><input type="number" min={0} max={99} value={singlesRow.scoreB} onChange={(event) => patchSingles({ scoreB: event.target.value })} style={inputStyle} /></label>
            <SearchablePlayerInput
              inputId="singles-player-b"
              label="Player B"
              value={singlesRow.playerB}
              players={knownPlayers}
              disabled={saving || creatingPlayers}
              onChange={(playerB) => patchSingles({ playerB })}
              onCreate={createAndSelectPlayer}
            />
          </div>
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
                      onConfirm={() => removeRow(row.rowId)}
                    />
                  ) : (
                    <button type="button" onClick={() => removeRow(row.rowId)} disabled={rows.length <= 1}>Remove match</button>
                  )}
                </div>
                <div className={`${styles.metadataGrid} ${context === "popup" ? styles.popupMetadataGrid : ""}`}>
                  <label className={styles.field}><strong>Date</strong><br /><input type="date" value={row.date} onChange={(event) => patchRow(row.rowId, { date: event.target.value })} style={inputStyle} /></label>
                  {context === "league" ? <label className={styles.field}><strong>League</strong><br /><select value={row.league} onChange={(event) => patchRow(row.rowId, { league: event.target.value })} style={inputStyle}>{selectableLeagueOptions.map((item) => <option key={item}>{item}</option>)}</select></label> : null}
                  {context === "league" ? <label className={styles.field}><strong>Week / session</strong><br /><input value={row.weekTag} onChange={(event) => patchRow(row.rowId, { weekTag: event.target.value })} style={inputStyle} /></label> : null}
                  <label className={styles.field}><strong>Rating scope</strong><br /><select value={row.ratingScope} onChange={(event) => patchRow(row.rowId, { ratingScope: event.target.value as MatchRow["ratingScope"] })} style={inputStyle}>{context === "popup" ? <><option value="overall_only">Overall only (rated)</option><option value="unrated">Unrated / record only</option></> : <><option value="">Overall + league</option><option value="overall_only">Overall only</option><option value="unrated">Unrated / record only</option></>}</select></label>
                </div>
                <div className={styles.teamsViewport}>
                  <div className={styles.teamsGrid}>
                    <section aria-label={`Match ${index + 1} Team 1`} className={styles.teamPanel}>
                      <h4 style={{ margin: 0 }}>Team 1</h4>
                      <SearchablePlayerInput inputId={`${row.rowId}-t1p1`} label="Player 1" value={row.t1p1} players={playerOptionsFor(row, row.t1p1)} invalid={validateThisRow && !row.t1p1} disabled={saving || creatingPlayers} onChange={(t1p1) => patchRow(row.rowId, { t1p1 })} onCreate={createAndSelectPlayer} />
                      <SearchablePlayerInput inputId={`${row.rowId}-t1p2`} label="Player 2" value={row.t1p2} players={playerOptionsFor(row, row.t1p2)} invalid={validateThisRow && !row.t1p2} disabled={saving || creatingPlayers} onChange={(t1p2) => patchRow(row.rowId, { t1p2 })} onCreate={createAndSelectPlayer} />
                    </section>
                    <section aria-label={`Match ${index + 1} scores`} className={styles.scorePanel}>
                      <label className={styles.scoreField}><strong>Team 1 score</strong><br /><input value={row.s1} onChange={(event) => patchRow(row.rowId, { s1: event.target.value })} aria-invalid={scoreInvalid || undefined} type="number" min={0} max={99} style={scoreInvalid ? { ...inputStyle, border: "2px solid #dc2626", background: "#fef2f2" } : inputStyle} /></label>
                      <label className={styles.scoreField}><strong>Team 2 score</strong><br /><input value={row.s2} onChange={(event) => patchRow(row.rowId, { s2: event.target.value })} aria-invalid={scoreInvalid || undefined} type="number" min={0} max={99} style={scoreInvalid ? { ...inputStyle, border: "2px solid #dc2626", background: "#fef2f2" } : inputStyle} /></label>
                    </section>
                    <section aria-label={`Match ${index + 1} Team 2`} className={styles.teamPanel}>
                      <h4 style={{ margin: 0 }}>Team 2</h4>
                      <SearchablePlayerInput inputId={`${row.rowId}-t2p1`} label="Player 1" value={row.t2p1} players={playerOptionsFor(row, row.t2p1)} invalid={validateThisRow && !row.t2p1} disabled={saving || creatingPlayers} onChange={(t2p1) => patchRow(row.rowId, { t2p1 })} onCreate={createAndSelectPlayer} />
                      <SearchablePlayerInput inputId={`${row.rowId}-t2p2`} label="Player 2" value={row.t2p2} players={playerOptionsFor(row, row.t2p2)} invalid={validateThisRow && !row.t2p2} disabled={saving || creatingPlayers} onChange={(t2p2) => patchRow(row.rowId, { t2p2 })} onCreate={createAndSelectPlayer} />
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
              <p><button type="button" onClick={createPlayersAndContinue} disabled={creatingPlayers || generating || !accessToken} style={buttonStyle}>{creatingPlayers ? "Creating…" : "Create Players & Continue"}</button></p>
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

      {result ? <SubmissionResultDialog result={result} onClose={acknowledgeSubmission} /> : null}
    </section>
  );
}
