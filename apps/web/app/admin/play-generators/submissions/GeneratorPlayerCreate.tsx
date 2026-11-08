"use client";

import { useEffect, useRef, useState } from "react";
import { playerSearchKey } from "@/lib/playerSearch";

type Player = { id: number; name: string };
type Creation = { version: 1; key: string; name: string; level?: number; playerId?: number; estimateRequest?: { participant_id: string; player_ids: Record<string, number>; expected_version: number } };
type Props = {
  context?: "setup" | "submission" | "courtreserve" | "league";
  initialStartingLevel?: string;
  lineupNames?: string[];
  playerIds?: Record<string, number>; expectedVersion?: number; ratingMode?: "rated" | "unrated";
  clubId: string; sessionKey: string; participantId: string; participantName: string;
  apiBase: string; accessToken: string; players: Player[]; disabled: boolean; matched: boolean;
  onPendingChange: (participantId: string, pending: boolean) => void;
  onBusyChange: (busy: boolean) => void; onCreated: (playerId: number, lineupName?: string) => Promise<void>;
};
const field = { padding: "0.6rem", border: "1px solid #cbd5e1", borderRadius: 8, font: "inherit", maxWidth: "100%", boxSizing: "border-box" as const };

class CreationError extends Error {
  constructor(message: string, public status: number, public uncertain: boolean, public definitelyNotCreated = false) { super(message); }
}

export default function GeneratorPlayerCreate({ context = "submission", initialStartingLevel = "", lineupNames, playerIds = {}, expectedVersion, ratingMode = "unrated", clubId, sessionKey, participantId, participantName, apiBase, accessToken, players, disabled, matched, onPendingChange, onBusyChange, onCreated }: Props) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState(context === "setup" || context === "league" ? "" : participantName);
  const [level, setLevel] = useState(initialStartingLevel);
  const [operation, setOperation] = useState<Creation | null>(null);
  const [message, setMessage] = useState("");
  const [found, setFound] = useState<Player[]>([]);
  const running = useRef(false);
  const mounted = useRef(false);
  const operationRef = useRef<Creation | null>(null);
  const source = context === "league" ? "league_court_roster" : context === "courtreserve" ? "courtreserve_roster_import" : context === "setup" ? "generator_roster_setup" : "generator_submission_review";
  const storageKey = `jupr-generator-player-create:v1:${clubId}:${sessionKey}:${participantId}`;

  useEffect(() => {
    mounted.current = true;
    try {
      const saved = JSON.parse(sessionStorage.getItem(storageKey) || "null") as Creation | null;
      if (saved?.version === 1 && typeof saved.key === "string" && saved.key.startsWith("generator-player:")
        && typeof saved.name === "string" && ((typeof saved.level === "number" && Number.isFinite(saved.level) && saved.level >= 1 && saved.level <= 7)
          || (context === "submission" && saved.estimateRequest?.participant_id === participantId && Number.isInteger(saved.estimateRequest.expected_version)))) {
        operationRef.current = saved; setOperation(saved); setName(saved.name); setLevel(saved.level == null ? "" : String(saved.level)); setOpen(true);
        onPendingChange(participantId, true);
        setMessage("Check the previous creation before continuing.");
      }
    } catch { /* In-memory recovery remains available if storage is unavailable. */ }
    return () => { mounted.current = false; };
  }, [storageKey, participantId, onPendingChange, context]);

  function retain(next: Creation | null) {
    operationRef.current = next;
    if (mounted.current) { setOperation(next); onPendingChange(participantId, Boolean(next)); }
    try {
      if (next) sessionStorage.setItem(storageKey, JSON.stringify(next));
      else sessionStorage.removeItem(storageKey);
    } catch { /* Keep the exact request in memory for an uninterrupted page. */ }
  }

  async function request(path: string, body?: unknown) {
    const response = await fetch(`${apiBase.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/${path}`, {
      method: body ? "POST" : "GET", cache: "no-store",
      headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = payload.detail;
      throw new CreationError(typeof detail === "string" ? detail : detail?.message || "Could not confirm player creation.", response.status,
        detail?.recovery_required === true || detail?.kind === "uncertain" || response.status >= 500 || [408, 425, 429].includes(response.status), detail?.recovery_required === false);
    }
    return payload;
  }

  async function finish(playerId: number) {
    if (!Number.isInteger(playerId) || playerId <= 0) throw new Error("Could not confirm the new player's profile.");
    if (operationRef.current) retain({ ...operationRef.current, playerId });
    if (!mounted.current) return;
    await onCreated(playerId, context === "setup" || context === "league" ? name : undefined);
    retain(null);
    if ((context === "setup" || context === "league") && mounted.current) { setOpen(false); setName(""); setLevel(""); setFound([]); }
  }

  async function sendCreation(saved: Creation, newRequest = false) {
    let payload;
    try {
      payload = saved.estimateRequest
        ? await request(`play-generators/sessions/${encodeURIComponent(sessionKey)}/players`, {
          ...saved.estimateRequest, name: saved.name, idempotency_key: saved.key,
        })
        : await request("players/editor/players", {
          name: saved.name, starting_jupr: saved.level, idempotency_key: saved.key, source,
        });
    } catch (error) {
      if (error instanceof CreationError && !error.uncertain && ((newRequest && error.status !== 409) || error.definitelyNotCreated)) retain(null);
      throw error;
    }
    await finish(payload.player?.id);
  }

  async function recover(saved: Creation) {
    if (saved.playerId) { await finish(saved.playerId); return; }
    const path = `players/editor/operations/${encodeURIComponent(saved.key)}`;
    let result;
    try { result = await request(path); }
    catch (error) {
      if (error instanceof CreationError && error.status === 404) { await sendCreation(saved, true); return; }
      throw error;
    }
    if (result.status === "completed") { await finish((result.result || result.result_json)?.player?.id); return; }
    if (result.status !== "failed" || result.recovery_required) {
      result = await request(`${path}/reconcile`, { confirmation_text: "RECONCILE PLAYER OPERATION", source });
      if (result.ok !== false && result.player?.id) { await finish(result.player.id); return; }
    }
    if (result.status === "failed" && !result.recovery_required) {
      retain(null);
      if (mounted.current) setMessage("The previous attempt did not create a player. Check the details and try again.");
      return;
    }
    throw new Error("Player creation still needs checking. Try checking again before creating another profile.");
  }

  const matching = [...players, ...found].filter((player, index, all) =>
    playerSearchKey(player.name) === playerSearchKey(name) && all.findIndex(p => p.id === player.id) === index);
  const valid = (!lineupNames || lineupNames.includes(name)) && Boolean(name.trim()) && (context === "submission" ? Number.isInteger(expectedVersion) : level.trim() !== "" && Number.isFinite(Number(level)) && Number(level) >= 1 && Number(level) <= 7);

  async function create() {
    if (running.current || disabled || !accessToken) return;
    if (!operationRef.current && (!valid || matching.length)) return;
    running.current = true; onBusyChange(true); setMessage("");
    try {
      if (operationRef.current) { await recover(operationRef.current); return; }
      // Recheck current club profiles immediately before a new creation request.
      const latest = await request(context === "submission" ? "play-generators/submissions?status=pending" : "players/editor/players");
      if (!mounted.current) return;
      const existing = (latest.players as Player[]).filter(player => playerSearchKey(player.name) === playerSearchKey(name));
      if (existing.length) { setFound(existing); setMessage("A matching club profile already exists. Select it below."); return; }
      const saved: Creation = { version: 1, key: `generator-player:${globalThis.crypto.randomUUID()}`, name: name.trim(),
        ...(context === "submission" ? { estimateRequest: { participant_id: participantId, player_ids: { ...playerIds }, expected_version: expectedVersion! } } : { level: Number(level) }) };
      retain(saved);
      await sendCreation(saved, true);
    } catch (error) {
      if (mounted.current) setMessage(operationRef.current?.playerId
        ? "The player was created, but the list could not reload. Use the created player below."
        : `${error instanceof Error ? error.message : "Could not confirm player creation."}${operationRef.current ? " Check player creation before trying a new request." : ""}`);
    } finally { running.current = false; onBusyChange(false); }
  }

  async function selectExisting(playerId: number) {
    if (disabled || running.current) return;
    running.current = true; onBusyChange(true); setMessage("");
    try { await finish(playerId); }
    catch (error) { if (mounted.current) setMessage(error instanceof Error ? error.message : "Could not load the existing player."); }
    finally { running.current = false; onBusyChange(false); }
  }

  if (matched && !operation) return null;
  if (!open) return <button type="button" disabled={disabled} style={field} aria-label={`Create new player for ${participantName}`} onClick={() => { if (lineupNames?.length) setName(lineupNames[0]); else if (context !== "setup" && context !== "league") setName(participantName); setOpen(true); }}>Create new player</button>;
  return <form aria-label={`New club player for ${participantName}`} onSubmit={event => { event.preventDefault(); void create(); }} style={{ display: "grid", gap: "0.6rem", minWidth: 0 }}>
    {lineupNames?.length && !operation ? <label style={{ display: "grid", gap: 4 }}>Player already in your lineup
      <select aria-label={`Player already in lineup for ${participantName}`} value={name} disabled={disabled} style={field}
        onChange={event => { setName(event.target.value); setLevel(""); setFound([]); setMessage(""); }}>
        {lineupNames.map(playerName => <option key={playerName} value={playerName}>{playerName}</option>)}
      </select>
    </label> : null}
    <label style={{ display: "grid", gap: 4 }}>Player name
      <input aria-label={`New player name for ${participantName}`} value={name} readOnly={Boolean(lineupNames)} required maxLength={160} disabled={disabled || Boolean(operation)} style={field}
        onChange={event => { setName(event.target.value); setFound([]); setMessage(""); }} />
    </label>
    {context !== "submission" ? <label style={{ display: "grid", gap: 4 }}>Doubles starting level
      <input aria-label={`New player starting level for ${participantName}`} type="number" inputMode="decimal" min={1} max={7} step="any" required placeholder="e.g. 3.50" value={level}
        disabled={disabled || Boolean(operation)} onChange={event => setLevel(event.target.value)} style={field} />
    </label> : null}
    {!operation && matching.length ? <div role="status">A matching club profile exists.
      {matching.map(player => <button key={player.id} type="button" disabled={disabled} style={{ ...field, display: "block", width: "100%", marginTop: 4 }} onClick={() => void selectExisting(player.id)}>Use {player.name}</button>)}
    </div> : null}
    <small>{context === "league" ? "Creates a club profile and adds it to this court. Enter a starting estimate; players without rated history are reviewed before play." : context === "courtreserve" ? "Creates a PCS profile and selects it for this report. A single CourtReserve skill level is filled in as the starting estimate; confirm or edit it before creating." : context === "setup" ? "Creates a club profile and adds it to your lineup. Moneyball uses this starting level to set expectations." : ratingMode === "unrated" ? "We’ll estimate a provisional rating from the group’s ratings, partners, opponents, and saved scores. Unrated games keep this rating provisional." : "We’ll estimate a starting rating from the group’s ratings, partners, opponents, and saved scores. Rated results are applied when approved."}</small>
    {message ? <p role="status" style={{ margin: 0 }}>{message}</p> : null}
    <button type="submit" style={field} disabled={disabled || (!operation && (!valid || matching.length > 0))}>
      {operation?.playerId ? "Use created player" : operation ? "Check player creation" : "Create and select player"}
    </button>
    {!operation ? <button type="button" disabled={disabled} style={field} onClick={() => setOpen(false)}>Cancel</button> : null}
  </form>;
}
