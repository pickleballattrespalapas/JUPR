"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { PublicLiveMatch, PublicLiveSessionDetail } from "@/lib/api";

type LiveSessionRunnerProps = {
  apiBase: string | null;
  clubSlug: string;
  initialSession: PublicLiveSessionDetail;
};

type PendingMutationPayload = Record<string, unknown> & { idempotency_key: string };

const thStyle = { textAlign: "left" as const, borderBottom: "1px solid #cbd5e1", padding: "0.5rem", whiteSpace: "nowrap" as const };
const tdStyle = { borderBottom: "1px solid #e2e8f0", padding: "0.5rem", whiteSpace: "nowrap" as const };
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function formatTimestamp(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
}

function eventTypeLabel(value?: string | null): string {
  const normalized = String(value || "").replace(/_/g, " ").trim();
  return normalized ? normalized.replace(/\b\w/g, (char) => char.toUpperCase()) : "Play Session";
}

function teamLabel(names: string[]): string {
  return names.filter(Boolean).join(" / ") || "TBD";
}

function scoreLabel(match: PublicLiveMatch): string {
  const scoreA = match.score_a ?? null;
  const scoreB = match.score_b ?? null;
  if (scoreA == null && scoreB == null) return "—";
  return `${scoreA ?? 0}–${scoreB ?? 0}`;
}

function scoreInputKey(matchId: string, side: "a" | "b"): string {
  return `${matchId}:${side}`;
}

function newOperationKey(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return crypto.randomUUID();
  return `public-live-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function LiveSessionRunner({ apiBase, clubSlug, initialSession }: LiveSessionRunnerProps) {
  const [session, setSession] = useState(initialSession);
  const [editToken, setEditToken] = useState("");
  const [scoreValues, setScoreValues] = useState<Record<string, string>>(() => {
    const initial: Record<string, string> = {};
    for (const round of initialSession.rounds || []) {
      for (const match of round.matches || []) {
        initial[scoreInputKey(match.id, "a")] = match.score_a == null ? "" : String(match.score_a);
        initial[scoreInputKey(match.id, "b")] = match.score_b == null ? "" : String(match.score_b);
      }
      for (const court of round.courts || []) {
        for (const match of court.matches || []) {
          initial[scoreInputKey(match.id, "a")] = match.score_a == null ? "" : String(match.score_a);
          initial[scoreInputKey(match.id, "b")] = match.score_b == null ? "" : String(match.score_b);
        }
      }
    }
    return initial;
  });
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [messageTone, setMessageTone] = useState<"success" | "error" | null>(null);
  const [operationKeys, setOperationKeys] = useState<Record<string, string>>({});
  const [subScope, setSubScope] = useState<"round" | "game">("round");
  const [subOriginalId, setSubOriginalId] = useState("");
  const [substituteName, setSubstituteName] = useState("");
  const [subMatchId, setSubMatchId] = useState("");
  const canEdit = Boolean(editToken) && session.status === "active";
  const canRecoverCompletion = Boolean(editToken) && Boolean(operationKeys.complete);
  const publicPath = `/clubs/${clubSlug}/live/${session.session_key}`;
  const editPath = `${publicPath}#edit=${encodeURIComponent(editToken)}`;

  const operationStorageKey = useCallback((action: string): string => {
    return `jupr-live-operation:${clubSlug}:${session.session_key}:${action}`;
  }, [clubSlug, session.session_key]);

  useEffect(() => {
    const storageKey = `jupr-live-edit:${clubSlug}:${initialSession.session_key}`;
    const hash = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    const discovered = hash.get("edit") || sessionStorage.getItem(storageKey) || "";
    if (discovered) {
      sessionStorage.setItem(storageKey, discovered);
      setEditToken(discovered);
    }
    if (hash.has("edit")) {
      window.history.replaceState({}, "", `${window.location.pathname}${window.location.search}`);
    }
    const pending: Record<string, string> = {};
    for (const action of ["scores", "advance", "complete", "substitute"]) {
      const raw = sessionStorage.getItem(operationStorageKey(action));
      if (!raw) continue;
      try {
        const payload = JSON.parse(raw) as { idempotency_key?: unknown };
        if (typeof payload.idempotency_key === "string") pending[action] = payload.idempotency_key;
      } catch {
        pending[action] = raw;
      }
    }
    setOperationKeys(pending);
  }, [clubSlug, initialSession.session_key, operationStorageKey]);

  const allMatches = useMemo(() => {
    const seen = new Set<string>();
    const matches: PublicLiveMatch[] = [];
    for (const round of session.rounds || []) {
      const roundMatches = round.courts?.length
        ? round.courts.flatMap((court) => court.matches || [])
        : round.matches || [];
      for (const match of roundMatches) {
        if (seen.has(match.id)) continue;
        seen.add(match.id);
        matches.push(match);
      }
    }
    return matches;
  }, [session]);
  const scoredMatches = allMatches.filter((match) => match.is_scored).length;
  const progressLabel = allMatches.length ? `${scoredMatches}/${allMatches.length}` : "0/0";
  const isLeague = session.event_type === "league" || session.event_type === "league_ladder";
  const editableMatches = allMatches.filter((match) => !isLeague || match.round_number === (session.current_round || 1));

  function rememberOperation(action: string, request: Record<string, unknown>): PendingMutationPayload {
    const raw = sessionStorage.getItem(operationStorageKey(action)) || "";
    if (raw) {
      try {
        const pending = JSON.parse(raw) as PendingMutationPayload;
        if (typeof pending.idempotency_key === "string" && pending.idempotency_key) return pending;
      } catch {
        const migrated = { ...request, idempotency_key: raw };
        sessionStorage.setItem(operationStorageKey(action), JSON.stringify(migrated));
        setOperationKeys((current) => ({ ...current, [action]: raw }));
        return migrated;
      }
    }
    const idempotencyKey = operationKeys[action] || newOperationKey();
    const pending = { ...request, idempotency_key: idempotencyKey };
    sessionStorage.setItem(operationStorageKey(action), JSON.stringify(pending));
    setOperationKeys((current) => ({ ...current, [action]: idempotencyKey }));
    return pending;
  }

  function clearOperation(action: string) {
    sessionStorage.removeItem(operationStorageKey(action));
    setOperationKeys((current) => {
      const next = { ...current };
      delete next[action];
      return next;
    });
  }

  function applySession(nextSession: PublicLiveSessionDetail) {
    setSession(nextSession);
    const nextScores: Record<string, string> = {};
    for (const round of nextSession.rounds || []) {
      const matches = round.courts?.length ? round.courts.flatMap((court) => court.matches || []) : round.matches || [];
      for (const match of matches) {
        nextScores[scoreInputKey(match.id, "a")] = match.score_a == null ? "" : String(match.score_a);
        nextScores[scoreInputKey(match.id, "b")] = match.score_b == null ? "" : String(match.score_b);
      }
    }
    setScoreValues(nextScores);
  }

  async function copyPath(path: string, label: string) {
    try {
      const absolute = typeof window === "undefined" ? path : `${window.location.origin}${path}`;
      await navigator.clipboard.writeText(absolute);
      setMessage(`${label} copied.`);
      setMessageTone("success");
    } catch {
      setMessage(`Unable to copy ${label.toLowerCase()}.`);
      setMessageTone("error");
    }
  }

  async function saveScores() {
    if (!apiBase) {
      setMessage("The public API base URL is not configured for this deployment.");
      setMessageTone("error");
      return;
    }
    if (!editToken) {
      setMessage("This link is view-only. Use the original edit link to enter scores.");
      setMessageTone("error");
      return;
    }
    setSaving(true);
    setMessage(null);
    setMessageTone(null);
    const action = "scores";
    try {
      const scores = editableMatches.map((match) => {
        const aRaw = scoreValues[scoreInputKey(match.id, "a")] ?? "";
        const bRaw = scoreValues[scoreInputKey(match.id, "b")] ?? "";
        return {
          match_id: match.id,
          score_a: aRaw === "" ? null : Number(aRaw),
          score_b: bRaw === "" ? null : Number(bRaw)
        };
      });
      const requestPayload = rememberOperation(action, {
        edit_token: editToken,
        expected_version: session.version,
        scores
      });
      const response = await fetch(apiUrl(apiBase, `/clubs/${clubSlug}/live-sessions/${session.session_key}/scores`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestPayload)
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        if ([400, 403, 409, 422].includes(response.status)) clearOperation(action);
        throw new Error(String(payload?.detail || `API error (${response.status})`));
      }
      applySession(payload.session);
      clearOperation(action);
      setMessage("Scores saved.");
      setMessageTone("success");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Unable to save scores.");
      setMessageTone("error");
    } finally {
      setSaving(false);
    }
  }

  async function refreshSession() {
    if (!apiBase) return;
    setSaving(true);
    setMessage(null);
    setMessageTone(null);
    try {
      const response = await fetch(apiUrl(apiBase, `/clubs/${clubSlug}/live-sessions/${session.session_key}`), { cache: "no-store" });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
      applySession(payload.session);
      setMessage("Session refreshed from durable state.");
      setMessageTone("success");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to refresh session.");
      setMessageTone("error");
    } finally {
      setSaving(false);
    }
  }

  async function runAction(action: "advance" | "complete") {
    if (!apiBase || !editToken) return;
    setSaving(true);
    setMessage(null);
    setMessageTone(null);
    try {
      const requestPayload = rememberOperation(action, {
        edit_token: editToken,
        expected_version: session.version
      });
      const response = await fetch(apiUrl(apiBase, `/clubs/${clubSlug}/live-sessions/${session.session_key}/${action}`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestPayload)
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        if ([400, 403, 409, 422].includes(response.status)) clearOperation(action);
        throw new Error(String(payload?.detail || `API error (${response.status})`));
      }
      applySession(payload.session);
      clearOperation(action);
      setMessage(action === "advance" ? `Advanced to round ${payload.advanced_to_round || payload.session.current_round}.` : (payload.social_submission ? "Session completed and sent to Club Social moderation." : "Session completed."));
      setMessageTone("success");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : `Unable to ${action} session.`);
      setMessageTone("error");
    } finally {
      setSaving(false);
    }
  }

  async function saveSubstitution() {
    const retryingPending = Boolean(operationKeys.substitute);
    if (!apiBase || !editToken || (!retryingPending && (!subOriginalId || !substituteName.trim()))) return;
    const action = "substitute";
    setSaving(true);
    setMessage(null);
    setMessageTone(null);
    try {
      const requestPayload = rememberOperation(action, {
        edit_token: editToken,
        expected_version: session.version,
        scope: subScope,
        round_number: session.current_round || 1,
        original_participant_id: subOriginalId,
        substitute_name: substituteName.trim(),
        match_id: subScope === "game" ? subMatchId : null
      });
      const response = await fetch(apiUrl(apiBase, `/clubs/${clubSlug}/live-sessions/${session.session_key}/substitutions`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestPayload)
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        if ([400, 403, 409, 422].includes(response.status)) clearOperation(action);
        throw new Error(String(payload?.detail || `API error (${response.status})`));
      }
      applySession(payload.session);
      clearOperation(action);
      setSubstituteName("");
      setMessage("Substitution saved for remaining unscored play.");
      setMessageTone("success");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to save substitution.");
      setMessageTone("error");
    } finally {
      setSaving(false);
    }
  }

  function renderMatch(match: PublicLiveMatch) {
    const matchCanEdit = canEdit && (!isLeague || match.round_number === (session.current_round || 1));
    return (
      <article key={match.id} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.8rem", background: "#f8fafc" }}>
        <p style={{ margin: "0 0 0.35rem", color: "#64748b", fontSize: "0.85rem" }}>{match.label}</p>
        <div style={{ display: "grid", gridTemplateColumns: matchCanEdit ? "1fr 4.2rem 4.2rem 1fr" : "1fr auto 1fr", alignItems: "center", gap: "0.75rem" }}>
          <strong>{teamLabel(match.team_a)}</strong>
          {matchCanEdit ? (
            <>
              <input
                aria-label={`${match.label} team A score`}
                type="number"
                min={0}
                max={99}
                value={scoreValues[scoreInputKey(match.id, "a")] ?? ""}
                onChange={(event) => setScoreValues((current) => ({ ...current, [scoreInputKey(match.id, "a")]: event.target.value }))}
                style={{ width: "100%", padding: "0.45rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" }}
              />
              <input
                aria-label={`${match.label} team B score`}
                type="number"
                min={0}
                max={99}
                value={scoreValues[scoreInputKey(match.id, "b")] ?? ""}
                onChange={(event) => setScoreValues((current) => ({ ...current, [scoreInputKey(match.id, "b")]: event.target.value }))}
                style={{ width: "100%", padding: "0.45rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" }}
              />
            </>
          ) : (
            <span style={{ fontWeight: 800, fontSize: "1.1rem" }}>{scoreLabel(match)}</span>
          )}
          <strong style={{ textAlign: "right" }}>{teamLabel(match.team_b)}</strong>
        </div>
        {match.winner ? <p style={{ margin: "0.4rem 0 0", color: "#166534", fontSize: "0.9rem" }}>Winner: {match.winner}</p> : null}
      </article>
    );
  }

  return (
    <>
      <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap", alignItems: "flex-start", marginBottom: "1rem" }}>
        <div>
          <h1 style={{ margin: "0 0 0.35rem", fontSize: "2.2rem", lineHeight: 1.1 }}>{session.title}</h1>
          <p style={{ margin: 0, color: "#475569" }}>
            {eventTypeLabel(session.event_type)}
            {session.current_round ? ` · Current round ${session.current_round}` : ""}
          </p>
          <p style={{ margin: "0.35rem 0 0", color: "#64748b", fontSize: "0.9rem" }}>
            Last updated {formatTimestamp(session.updated_at ?? session.last_seen_at)}
          </p>
          {!canEdit ? <p style={{ color: "#64748b" }}>{editToken ? "This scoreboard is complete; score fields are locked." : "View-only scoreboard. The edit link is only shown to the person who created this event."}</p> : null}
        </div>
        <span style={{ border: "1px solid #bfdbfe", borderRadius: "999px", padding: "0.25rem 0.75rem", color: "#1d4ed8", background: "#eff6ff", fontSize: "0.85rem", fontWeight: 800 }}>
          {session.status}
        </span>
        <button type="button" onClick={refreshSession} disabled={saving} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: "white", fontWeight: 800, cursor: saving ? "default" : "pointer" }}>
          Refresh durable state
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
        <article style={cardStyle}><strong>Rounds</strong><br />{session.rounds.length}</article>
        <article style={cardStyle}><strong>Matches scored</strong><br />{progressLabel}</article>
        <article style={cardStyle}><strong>Current round</strong><br />{session.current_round ?? "—"}</article>
        <article style={cardStyle}><strong>Mode</strong><br />{session.live_mode === "club_social" ? "Club Social" : (canEdit ? "Quick score entry" : "Public view")}</article>
      </div>

      <div style={{ ...cardStyle, marginBottom: "1rem", background: "#f8fafc" }}>
        <strong>Share links</strong>
        <p style={{ color: "#475569" }}>Share the public link with players. Keep the edit link private for score entry.</p>
        <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
          <Link href={publicPath}>Open public view</Link>
          <button type="button" onClick={() => copyPath(publicPath, "Public link")} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: "white", fontWeight: 800, cursor: "pointer" }}>Copy public link</button>
          {canEdit ? <button type="button" onClick={() => copyPath(editPath, "Edit link")} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: "white", fontWeight: 800, cursor: "pointer" }}>Copy edit link</button> : null}
          <a href={`/api/clubs/${clubSlug}/live-sessions/${session.session_key}/export?format=csv`}>Export CSV</a>
          <a href={`/api/clubs/${clubSlug}/live-sessions/${session.session_key}/export?format=json`}>Export JSON</a>
        </div>
      </div>

      {!canEdit && canRecoverCompletion ? (
        <div style={{ ...cardStyle, marginBottom: "1rem", background: "#fff7ed", borderColor: "#fdba74" }}>
          <strong>Completion reconciliation is still pending.</strong>
          <p style={{ color: "#9a3412" }}>Retry the preserved completion request to reconcile its durable operation record. Do not clear browser session data or create a new operation key.</p>
          <code style={{ display: "block", overflowWrap: "anywhere", marginBottom: "0.65rem" }}>{operationKeys.complete}</code>
          <button type="button" onClick={() => runAction("complete")} disabled={saving} style={{ border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: "#9a3412", color: "white", fontWeight: 800 }}>
            {saving ? "Reconciling…" : "Retry preserved completion"}
          </button>
          {message ? <p style={{ color: messageTone === "success" ? "#166534" : "#b91c1c" }}>{message}</p> : null}
        </div>
      ) : null}

      {canEdit ? (
        <div style={{ ...cardStyle, marginBottom: "1rem", background: "#eff6ff", borderColor: "#bfdbfe" }}>
          <strong>Score entry enabled.</strong> Save as you go. The private edit token is held in this browser session and omitted from the public URL.
          {Object.keys(operationKeys).length ? (
            <div style={{ marginTop: "0.65rem", color: "#92400e" }}>
              <strong>Unresolved request retained.</strong> Retry the same action before changing it; JUPR will send the exact preserved payload and operation key.
              {Object.entries(operationKeys).map(([action, key]) => <code key={action} style={{ display: "block", overflowWrap: "anywhere" }}>{action}: {key}</code>)}
            </div>
          ) : null}
          <div style={{ marginTop: "0.75rem", display: "flex", gap: "0.65rem", flexWrap: "wrap", alignItems: "center" }}>
            <button
              type="button"
              onClick={saveScores}
              disabled={saving}
              style={{ border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: "#2563eb", color: "white", fontWeight: 800, cursor: saving ? "default" : "pointer" }}
            >
              {saving ? "Saving…" : "Save scores"}
            </button>
            {session.event_type === "league" || session.event_type === "league_ladder" ? (
              <button type="button" onClick={() => runAction("advance")} disabled={saving} style={{ border: "1px solid #2563eb", borderRadius: "999px", padding: "0.65rem 1rem", background: "white", color: "#1d4ed8", fontWeight: 800, cursor: saving ? "default" : "pointer" }}>
                Advance round
              </button>
            ) : null}
            <button type="button" onClick={() => runAction("complete")} disabled={saving} style={{ border: "1px solid #166534", borderRadius: "999px", padding: "0.65rem 1rem", background: "white", color: "#166534", fontWeight: 800, cursor: saving ? "default" : "pointer" }}>
              Complete session
            </button>
            {message ? <span style={{ marginLeft: "0.75rem", color: messageTone === "success" ? "#166534" : "#b91c1c" }}>{message}</span> : null}
          </div>
        </div>
      ) : message ? <p style={{ color: messageTone === "success" ? "#166534" : "#b91c1c" }}>{message}</p> : null}

      {canEdit && session.live_mode !== "club_social" ? (
        <section style={{ ...cardStyle, marginBottom: "1rem" }}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Substitute for remaining play</h2>
          <p style={{ color: "#475569" }}>Substitutions affect only unscored matches. Scored matches remain locked.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.65rem" }}>
            <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
              Original player
              <select value={subOriginalId} onChange={(event) => setSubOriginalId(event.target.value)} disabled={Boolean(operationKeys.substitute)} style={{ padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px" }}>
                <option value="">Select player</option>
                {session.participants.map((participant) => <option key={participant.id} value={participant.id}>{participant.name}</option>)}
              </select>
            </label>
            <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
              Substitute name
              <input value={substituteName} onChange={(event) => setSubstituteName(event.target.value)} disabled={Boolean(operationKeys.substitute)} maxLength={80} style={{ padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px" }} />
            </label>
            <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
              Scope
              <select value={subScope} onChange={(event) => setSubScope(event.target.value as "round" | "game")} disabled={Boolean(operationKeys.substitute)} style={{ padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px" }}>
                <option value="round">Current round</option>
                <option value="game">One game</option>
              </select>
            </label>
            {subScope === "game" ? (
              <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
                Match
                <select value={subMatchId} onChange={(event) => setSubMatchId(event.target.value)} disabled={Boolean(operationKeys.substitute)} style={{ padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px" }}>
                  <option value="">Select match</option>
                  {editableMatches.filter((match) => !match.is_scored).map((match) => <option key={match.id} value={match.id}>{match.label}</option>)}
                </select>
              </label>
            ) : null}
          </div>
          <button type="button" onClick={saveSubstitution} disabled={saving || (!operationKeys.substitute && (!subOriginalId || !substituteName.trim() || (subScope === "game" && !subMatchId)))} style={{ marginTop: "0.75rem", border: 0, borderRadius: "999px", padding: "0.55rem 0.9rem", background: "#0f766e", color: "white", fontWeight: 800 }}>
            {operationKeys.substitute ? "Retry preserved substitution" : "Save substitution"}
          </button>
          {session.substitutions.length ? <p style={{ color: "#475569" }}>{session.substitutions.length} substitution record(s) are retained with this session.</p> : null}
        </section>
      ) : null}

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.5fr) minmax(280px, 1fr)", gap: "1rem", alignItems: "start" }}>
        <div style={{ display: "grid", gap: "1rem" }}>
          {session.rounds.length === 0 ? (
            <div style={cardStyle}>
              <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>No public match state yet</h2>
              <p style={{ color: "#475569" }}>This session exists, but the event schedule is not ready yet.</p>
            </div>
          ) : null}

          {session.rounds.map((round) => (
            <section key={round.number} style={cardStyle}>
              <h2 style={{ marginTop: 0, fontSize: "1.2rem" }}>Round {round.number}</h2>
              {round.courts && round.courts.length > 0 ? (
                <div style={{ display: "grid", gap: "0.75rem" }}>
                  {round.courts.map((court) => (
                    <div key={court.court_number}>
                      <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem" }}>Court {court.court_number}</h3>
                      <div style={{ display: "grid", gap: "0.5rem" }}>
                        {court.matches.map((match) => renderMatch(match))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ display: "grid", gap: "0.5rem" }}>
                  {round.matches.map((match) => renderMatch(match))}
                </div>
              )}
            </section>
          ))}
        </div>

        <aside style={{ display: "grid", gap: "1rem" }}>
          <section style={cardStyle}>
            <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Standings</h2>
            {session.standings.length === 0 ? <p style={{ color: "#475569" }}>No standings yet.</p> : null}
            {session.standings.length > 0 ? (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Rank</th>
                      <th style={thStyle}>Player</th>
                      <th style={thStyle}>W/L</th>
                      <th style={thStyle}>Diff</th>
                    </tr>
                  </thead>
                  <tbody>
                    {session.standings.map((row, index) => (
                      <tr key={`${row.participantId ?? row.name ?? index}`}>
                        <td style={tdStyle}>{String(row.rank ?? index + 1)}</td>
                        <td style={tdStyle}>{String(row.name ?? "—")}</td>
                        <td style={tdStyle}>{String(row.wins ?? 0)}/{String(row.losses ?? 0)}</td>
                        <td style={tdStyle}>{String(row.differential ?? "—")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </section>
        </aside>
      </div>
    </>
  );
}
