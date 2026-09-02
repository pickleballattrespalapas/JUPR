"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  applyMixedLineups,
  gameByKey,
  matchupResult,
  matchupStep,
  readSession,
  standingsForSession,
  teamPlayers,
  validDreamBreakerOrder,
  writeSession,
  type DreamBreaker,
  type MixedLineup,
  type RegulationGame,
  type RegulationGameKey,
  type Team,
  type TeamMatchSession,
  type TeamMatchup
} from "../../teamMatchState";

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
  padding: "0.65rem",
  font: "inherit"
};

const scoreInput = {
  ...input,
  fontSize: "1.35rem",
  fontWeight: 800,
  textAlign: "center" as const
};

const button = {
  border: 0,
  borderRadius: "999px",
  padding: "0.68rem 1rem",
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

function scoreValue(raw: string): number | null {
  if (raw.trim() === "") return null;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed < 0) return null;
  return parsed;
}

function scoreLabel(game: RegulationGame): string {
  if (!game.submitted || game.scoreA === null || game.scoreB === null) return "Pending";
  return `${game.scoreA}–${game.scoreB}`;
}

function matchupName(matchup: TeamMatchup, teams: Map<string, Team>): string {
  return `${teams.get(matchup.teamAId)?.name || "Team A"} vs ${teams.get(matchup.teamBId)?.name || "Team B"}`;
}

function csvEscape(value: unknown): string {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function gameNumber(key: RegulationGameKey): number {
  return { women: 1, men: 2, "mixed-1": 3, "mixed-2": 4 }[key];
}

function nextIncompleteMatchup(session: TeamMatchSession, afterId: string): TeamMatchup | null {
  const start = session.matchups.findIndex((matchup) => matchup.id === afterId);
  const ordered = start >= 0
    ? [...session.matchups.slice(start + 1), ...session.matchups.slice(0, start)]
    : session.matchups;
  return ordered.find((matchup) => !matchupResult(matchup).complete) || null;
}

function TeamScoreEntry({
  game,
  teamA,
  teamB,
  initialScoreA,
  initialScoreB,
  submitLabel,
  onSubmit,
  onCancel
}: {
  game: RegulationGame;
  teamA: Team;
  teamB: Team;
  initialScoreA: number | null;
  initialScoreB: number | null;
  submitLabel: string;
  onSubmit: (scoreA: number, scoreB: number) => void;
  onCancel?: () => void;
}) {
  const [scoreA, setScoreA] = useState(initialScoreA === null ? "" : String(initialScoreA));
  const [scoreB, setScoreB] = useState(initialScoreB === null ? "" : String(initialScoreB));
  const [error, setError] = useState<string | null>(null);

  function submit(): void {
    const parsedA = scoreValue(scoreA);
    const parsedB = scoreValue(scoreB);
    if (parsedA === null || parsedB === null) {
      setError("Enter both scores before submitting this result.");
      return;
    }
    if (parsedA === parsedB) {
      setError("A regulation game cannot end in a tie.");
      return;
    }
    if (parsedA + parsedB <= 0) {
      setError("Enter a played score before submitting this result.");
      return;
    }
    setError(null);
    onSubmit(parsedA, parsedB);
  }

  return (
    <div style={{ ...card, borderColor: "#bfdbfe" }}>
      <p style={{ margin: "0 0 0.35rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.06em", fontSize: "0.78rem" }}>
        Game {gameNumber(game.key)} of 4
      </p>
      <h2 style={{ margin: "0 0 1rem" }}>{game.label}</h2>
      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 92px 30px 92px minmax(0, 1fr)", gap: "0.65rem", alignItems: "center" }}>
        <div>
          <strong>{teamA.name}</strong>
          <div style={{ color: "#475569", marginTop: "0.2rem" }}>{game.sideA.join(" / ")}</div>
        </div>
        <input
          aria-label={`${game.label} score for ${teamA.name}`}
          type="number"
          min={0}
          inputMode="numeric"
          value={scoreA}
          onChange={(event) => setScoreA(event.target.value)}
          style={scoreInput}
          autoFocus
        />
        <strong style={{ textAlign: "center" }}>–</strong>
        <input
          aria-label={`${game.label} score for ${teamB.name}`}
          type="number"
          min={0}
          inputMode="numeric"
          value={scoreB}
          onChange={(event) => setScoreB(event.target.value)}
          style={scoreInput}
        />
        <div style={{ textAlign: "right" }}>
          <strong>{teamB.name}</strong>
          <div style={{ color: "#475569", marginTop: "0.2rem" }}>{game.sideB.join(" / ")}</div>
        </div>
      </div>
      {error ? <p role="alert" style={{ color: "#b91c1c", fontWeight: 700 }}>{error}</p> : null}
      <div style={{ display: "flex", gap: "0.6rem", marginTop: "1rem", flexWrap: "wrap" }}>
        <button type="button" onClick={submit} style={button}>{submitLabel}</button>
        {onCancel ? <button type="button" onClick={onCancel} style={secondaryButton}>Cancel edit</button> : null}
      </div>
    </div>
  );
}

function DreamBreakerScoreEntry({
  breaker,
  teamA,
  teamB,
  onSubmit
}: {
  breaker: DreamBreaker;
  teamA: Team;
  teamB: Team;
  onSubmit: (scoreA: number, scoreB: number) => void;
}) {
  const [scoreA, setScoreA] = useState(breaker.scoreA === null ? "" : String(breaker.scoreA));
  const [scoreB, setScoreB] = useState(breaker.scoreB === null ? "" : String(breaker.scoreB));
  const [error, setError] = useState<string | null>(null);

  function submit(): void {
    const parsedA = scoreValue(scoreA);
    const parsedB = scoreValue(scoreB);
    if (parsedA === null || parsedB === null) {
      setError("Enter both DreamBreaker scores.");
      return;
    }
    if (parsedA === parsedB) {
      setError("The DreamBreaker must produce a winner.");
      return;
    }
    if (parsedA + parsedB <= 0) {
      setError("Enter a played DreamBreaker score.");
      return;
    }
    setError(null);
    onSubmit(parsedA, parsedB);
  }

  return (
    <div style={{ ...card, borderColor: "#f59e0b", background: "#fffbeb" }}>
      <p style={{ margin: "0 0 0.35rem", color: "#92400e", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.06em", fontSize: "0.78rem" }}>
        Tiebreaker
      </p>
      <h2 style={{ margin: "0 0 0.4rem" }}>DreamBreaker</h2>
      <p style={{ color: "#78350f", marginTop: 0 }}>Rally scoring to 21, win by two. Rotate to the next player after every four rallies.</p>
      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 92px 30px 92px minmax(0, 1fr)", gap: "0.65rem", alignItems: "center" }}>
        <div>
          <strong>{teamA.name}</strong>
          <div style={{ color: "#78350f", marginTop: "0.2rem" }}>{breaker.orderA.join(" → ")}</div>
        </div>
        <input aria-label={`DreamBreaker score for ${teamA.name}`} type="number" min={0} inputMode="numeric" value={scoreA} onChange={(event) => setScoreA(event.target.value)} style={scoreInput} autoFocus />
        <strong style={{ textAlign: "center" }}>–</strong>
        <input aria-label={`DreamBreaker score for ${teamB.name}`} type="number" min={0} inputMode="numeric" value={scoreB} onChange={(event) => setScoreB(event.target.value)} style={scoreInput} />
        <div style={{ textAlign: "right" }}>
          <strong>{teamB.name}</strong>
          <div style={{ color: "#78350f", marginTop: "0.2rem" }}>{breaker.orderB.join(" → ")}</div>
        </div>
      </div>
      {error ? <p role="alert" style={{ color: "#b91c1c", fontWeight: 700 }}>{error}</p> : null}
      <button type="button" onClick={submit} style={{ ...button, marginTop: "1rem" }}>Submit DreamBreaker result</button>
    </div>
  );
}

export default function TeamMatchSessionRunner({ clubId, sessionKey }: { clubId: string; sessionKey: string }) {
  const [session, setSession] = useState<TeamMatchSession | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editingGameKey, setEditingGameKey] = useState<RegulationGameKey | null>(null);
  const [lineupA, setLineupA] = useState<MixedLineup | null>(null);
  const [lineupB, setLineupB] = useState<MixedLineup | null>(null);
  const [orderA, setOrderA] = useState<string[]>([]);
  const [orderB, setOrderB] = useState<string[]>([]);

  useEffect(() => {
    const stored = readSession(clubId, sessionKey);
    if (!stored) {
      setError("This Team Match session was not found in this browser. Return to setup and generate it again.");
    } else {
      const firstOpen = stored.matchups.find((matchup) => !matchupResult(matchup).complete);
      if (!stored.activeMatchupId && firstOpen) stored.activeMatchupId = firstOpen.id;
      setSession(stored);
    }
    setHydrated(true);
  }, [clubId, sessionKey]);

  const teamMap = useMemo(
    () => new Map((session?.teams || []).map((team) => [team.id, team])),
    [session?.teams]
  );
  const activeMatchup = useMemo(() => {
    if (!session) return null;
    if (session.activeMatchupId) {
      const selected = session.matchups.find((matchup) => matchup.id === session.activeMatchupId);
      if (selected) return selected;
    }
    return session.matchups.find((matchup) => !matchupResult(matchup).complete) || null;
  }, [session]);
  const teamA = activeMatchup ? teamMap.get(activeMatchup.teamAId) || null : null;
  const teamB = activeMatchup ? teamMap.get(activeMatchup.teamBId) || null : null;
  const step = activeMatchup && teamA && teamB ? matchupStep(activeMatchup, teamA, teamB) : null;
  const currentGameKey = editingGameKey || (step?.kind === "game" ? step.gameKey : null);
  const currentGame = activeMatchup && currentGameKey ? gameByKey(activeMatchup, currentGameKey) : null;
  const standings = useMemo(() => (session ? standingsForSession(session) : []), [session]);
  const completeCount = useMemo(
    () => session?.matchups.filter((matchup) => matchupResult(matchup).complete).length || 0,
    [session]
  );
  const eventComplete = Boolean(session && session.matchups.length > 0 && completeCount === session.matchups.length);

  useEffect(() => {
    if (!activeMatchup || !teamA || !teamB) return;
    setEditingGameKey(null);
    setLineupA(activeMatchup.mixedLineups.teamA || { woman: teamA.women[0], man: teamA.men[0] });
    setLineupB(activeMatchup.mixedLineups.teamB || { woman: teamB.women[0], man: teamB.men[0] });
    setOrderA(validDreamBreakerOrder(activeMatchup.dreamBreaker.orderA, teamA) ? activeMatchup.dreamBreaker.orderA : teamPlayers(teamA));
    setOrderB(validDreamBreakerOrder(activeMatchup.dreamBreaker.orderB, teamB) ? activeMatchup.dreamBreaker.orderB : teamPlayers(teamB));
  }, [activeMatchup?.id, teamA?.id, teamB?.id]);

  function persist(next: TeamMatchSession): void {
    const updated = { ...next, updatedAt: new Date().toISOString() };
    writeSession(clubId, updated);
    setSession(updated);
  }

  function replaceMatchup(nextMatchup: TeamMatchup, statusMessage: string): void {
    if (!session) return;
    persist({
      ...session,
      matchups: session.matchups.map((matchup) => matchup.id === nextMatchup.id ? nextMatchup : matchup)
    });
    setMessage(statusMessage);
    setError(null);
  }

  function submitGame(gameKey: RegulationGameKey, scoreAValue: number, scoreBValue: number): void {
    if (!activeMatchup) return;
    const nextMatchup: TeamMatchup = {
      ...activeMatchup,
      games: activeMatchup.games.map((game) => game.key === gameKey ? {
        ...game,
        scoreA: scoreAValue,
        scoreB: scoreBValue,
        submitted: true,
        submittedAt: new Date().toISOString()
      } : game)
    };
    setEditingGameKey(null);

    if (gameKey === "women") {
      replaceMatchup(nextMatchup, "Women’s Doubles submitted. Men’s Doubles is next.");
    } else if (gameKey === "men") {
      replaceMatchup(nextMatchup, "Men’s Doubles submitted. Each team now chooses its mixed lineup.");
    } else if (gameKey === "mixed-1") {
      replaceMatchup(nextMatchup, "Mixed Doubles 1 submitted. Mixed Doubles 2 is next.");
    } else {
      const result = matchupResult(nextMatchup);
      replaceMatchup(
        nextMatchup,
        result.complete
          ? "Regulation complete. The team matchup has a winner."
          : "Regulation is tied 2–2. Set the DreamBreaker rotations next."
      );
    }
  }

  function saveMixedLineups(): void {
    if (!activeMatchup || !teamA || !teamB || !lineupA || !lineupB) return;
    if (!teamA.women.includes(lineupA.woman) || !teamA.men.includes(lineupA.man)) {
      setError(`Choose ${teamA.name} players from its roster.`);
      return;
    }
    if (!teamB.women.includes(lineupB.woman) || !teamB.men.includes(lineupB.man)) {
      setError(`Choose ${teamB.name} players from its roster.`);
      return;
    }
    const nextMatchup = applyMixedLineups(activeMatchup, teamA, teamB, lineupA, lineupB);
    replaceMatchup(nextMatchup, "Mixed lineups submitted. Mixed Doubles 1 is ready.");
  }

  function saveDreamBreakerOrders(): void {
    if (!activeMatchup || !teamA || !teamB) return;
    if (!validDreamBreakerOrder(orderA, teamA) || !validDreamBreakerOrder(orderB, teamB)) {
      setError("Each DreamBreaker order must contain all four team players exactly once.");
      return;
    }
    replaceMatchup(
      {
        ...activeMatchup,
        dreamBreaker: { ...activeMatchup.dreamBreaker, orderA: [...orderA], orderB: [...orderB] }
      },
      "DreamBreaker rotations submitted. Enter the tiebreaker result."
    );
  }

  function submitDreamBreaker(scoreAValue: number, scoreBValue: number): void {
    if (!activeMatchup) return;
    replaceMatchup(
      {
        ...activeMatchup,
        dreamBreaker: {
          ...activeMatchup.dreamBreaker,
          scoreA: scoreAValue,
          scoreB: scoreBValue,
          submitted: true,
          submittedAt: new Date().toISOString()
        }
      },
      "DreamBreaker submitted. The team matchup is complete."
    );
  }

  function advanceToNextMatchup(): void {
    if (!session || !activeMatchup) return;
    const next = nextIncompleteMatchup(session, activeMatchup.id);
    persist({ ...session, activeMatchupId: next?.id || null });
    setMessage(next ? `Opening ${matchupName(next, teamMap)}.` : "All team matchups are complete.");
    setError(null);
  }

  function downloadCsv(): void {
    if (!session) return;
    const rows: Array<Array<string | number>> = [["Round", "Team A", "Game", "Side A", "Score A", "Score B", "Side B", "Team B", "Submitted"]];
    for (const matchup of session.matchups) {
      const left = teamMap.get(matchup.teamAId);
      const right = teamMap.get(matchup.teamBId);
      for (const game of matchup.games) {
        rows.push([
          matchup.round,
          left?.name || "",
          game.label,
          game.sideA.join(" / "),
          game.scoreA ?? "",
          game.scoreB ?? "",
          game.sideB.join(" / "),
          right?.name || "",
          game.submitted ? "yes" : "no"
        ]);
      }
      if (matchup.dreamBreaker.submitted) {
        rows.push([
          matchup.round,
          left?.name || "",
          "DreamBreaker",
          matchup.dreamBreaker.orderA.join(" → "),
          matchup.dreamBreaker.scoreA ?? "",
          matchup.dreamBreaker.scoreB ?? "",
          matchup.dreamBreaker.orderB.join(" → "),
          right?.name || "",
          "yes"
        ]);
      }
    }
    const csv = rows.map((row) => row.map(csvEscape).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${session.title.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "team-match"}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  if (!hydrated) return <div style={card}>Loading Team Match session…</div>;
  if (error && !session) {
    return (
      <div role="alert" style={{ ...card, borderColor: "#fecaca", background: "#fef2f2", color: "#991b1b" }}>
        <p style={{ marginTop: 0 }}>{error}</p>
        <Link href={`/clubs/${encodeURIComponent(clubId)}/team-match-generator`}>Return to Team Match setup</Link>
      </div>
    );
  }
  if (!session) return null;

  const activeIndex = activeMatchup ? session.matchups.findIndex((matchup) => matchup.id === activeMatchup.id) : -1;
  const result = activeMatchup ? matchupResult(activeMatchup) : null;

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <style>{`@media print { .no-print { display:none !important; } body { background:white !important; } } @media (max-width: 760px) { .team-score-grid { grid-template-columns: 1fr 72px 20px 72px 1fr !important; } }`}</style>

      <div style={{ ...card, display: "flex", justifyContent: "space-between", gap: "1rem", alignItems: "flex-start", flexWrap: "wrap" }}>
        <div>
          <p style={{ margin: "0 0 0.35rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.06em", fontSize: "0.78rem" }}>Live Team Match</p>
          <h1 style={{ margin: 0 }}>{session.title}</h1>
          <p style={{ color: "#475569", marginBottom: 0 }}>
            {completeCount} of {session.matchups.length} team matchups complete
          </p>
        </div>
        <div className="no-print" style={{ display: "flex", gap: "0.55rem", flexWrap: "wrap" }}>
          <button type="button" onClick={downloadCsv} style={secondaryButton}>Download CSV</button>
          <button type="button" onClick={() => window.print()} style={secondaryButton}>Print</button>
          <Link href={`/clubs/${encodeURIComponent(clubId)}/team-match-generator`} style={{ ...secondaryButton, textDecoration: "none" }}>Back to setup</Link>
        </div>
      </div>

      {message ? <div role="status" aria-live="polite" style={{ ...card, borderColor: "#bfdbfe", background: "#eff6ff", color: "#1e3a8a" }}>{message}</div> : null}
      {error ? <div role="alert" style={{ ...card, borderColor: "#fecaca", background: "#fef2f2", color: "#991b1b" }}>{error}</div> : null}

      {eventComplete && !activeMatchup ? (
        <div style={{ ...card, borderColor: "#86efac", background: "#f0fdf4" }}>
          <h2 style={{ marginTop: 0 }}>All team matchups complete</h2>
          <p style={{ marginBottom: 0 }}>The full round robin has finished. Final standings appear below.</p>
        </div>
      ) : null}

      {activeMatchup && teamA && teamB ? (
        <>
          <div style={card}>
            <p style={{ margin: "0 0 0.35rem", color: "#64748b", fontWeight: 800 }}>
              Round {activeMatchup.round} · Team matchup {activeIndex + 1} of {session.matchups.length}
            </p>
            <h2 style={{ margin: 0 }}>{teamA.name} vs {teamB.name}</h2>
            <p style={{ color: "#475569", marginBottom: 0 }}>
              Regulation: {result?.regulationWinsA || 0}–{result?.regulationWinsB || 0}
              {result?.decidedByDreamBreaker ? " · DreamBreaker required" : ""}
            </p>
          </div>

          {currentGame ? (
            <TeamScoreEntry
              key={`${activeMatchup.id}:${currentGame.key}:${editingGameKey ? "edit" : "live"}`}
              game={currentGame}
              teamA={teamA}
              teamB={teamB}
              initialScoreA={currentGame.scoreA}
              initialScoreB={currentGame.scoreB}
              submitLabel={editingGameKey ? `Save corrected ${currentGame.label} result` : `Submit ${currentGame.label} result`}
              onSubmit={(scoreAValue, scoreBValue) => submitGame(currentGame.key, scoreAValue, scoreBValue)}
              onCancel={editingGameKey ? () => setEditingGameKey(null) : undefined}
            />
          ) : null}

          {!editingGameKey && step?.kind === "mixed_lineups" && lineupA && lineupB ? (
            <div style={{ ...card, borderColor: "#bfdbfe" }}>
              <p style={{ margin: "0 0 0.35rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.06em", fontSize: "0.78rem" }}>After gender doubles</p>
              <h2 style={{ marginTop: 0 }}>Choose Mixed Doubles lineups</h2>
              <p style={{ color: "#475569" }}>Each team chooses the woman and man for Mixed Doubles 1. The remaining woman and man automatically play Mixed Doubles 2.</p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem" }}>
                <div style={{ ...card, background: "#f8fafc" }}>
                  <h3 style={{ marginTop: 0 }}>{teamA.name}</h3>
                  <label style={{ display: "grid", gap: "0.35rem", fontWeight: 700 }}>
                    Mixed Doubles 1 woman
                    <select value={lineupA.woman} onChange={(event) => setLineupA({ ...lineupA, woman: event.target.value })} style={input}>
                      {teamA.women.map((name) => <option key={name} value={name}>{name}</option>)}
                    </select>
                  </label>
                  <label style={{ display: "grid", gap: "0.35rem", fontWeight: 700, marginTop: "0.7rem" }}>
                    Mixed Doubles 1 man
                    <select value={lineupA.man} onChange={(event) => setLineupA({ ...lineupA, man: event.target.value })} style={input}>
                      {teamA.men.map((name) => <option key={name} value={name}>{name}</option>)}
                    </select>
                  </label>
                  <p style={{ marginBottom: 0, color: "#475569" }}>
                    Mixed 2: {teamA.women.find((name) => name !== lineupA.woman)} / {teamA.men.find((name) => name !== lineupA.man)}
                  </p>
                </div>
                <div style={{ ...card, background: "#f8fafc" }}>
                  <h3 style={{ marginTop: 0 }}>{teamB.name}</h3>
                  <label style={{ display: "grid", gap: "0.35rem", fontWeight: 700 }}>
                    Mixed Doubles 1 woman
                    <select value={lineupB.woman} onChange={(event) => setLineupB({ ...lineupB, woman: event.target.value })} style={input}>
                      {teamB.women.map((name) => <option key={name} value={name}>{name}</option>)}
                    </select>
                  </label>
                  <label style={{ display: "grid", gap: "0.35rem", fontWeight: 700, marginTop: "0.7rem" }}>
                    Mixed Doubles 1 man
                    <select value={lineupB.man} onChange={(event) => setLineupB({ ...lineupB, man: event.target.value })} style={input}>
                      {teamB.men.map((name) => <option key={name} value={name}>{name}</option>)}
                    </select>
                  </label>
                  <p style={{ marginBottom: 0, color: "#475569" }}>
                    Mixed 2: {teamB.women.find((name) => name !== lineupB.woman)} / {teamB.men.find((name) => name !== lineupB.man)}
                  </p>
                </div>
              </div>
              <button type="button" onClick={saveMixedLineups} style={{ ...button, marginTop: "1rem" }}>Submit mixed lineups</button>
            </div>
          ) : null}

          {!editingGameKey && step?.kind === "dreambreaker_lineups" ? (
            <div style={{ ...card, borderColor: "#f59e0b", background: "#fffbeb" }}>
              <p style={{ margin: "0 0 0.35rem", color: "#92400e", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.06em", fontSize: "0.78rem" }}>Regulation tied 2–2</p>
              <h2 style={{ marginTop: 0 }}>Set DreamBreaker rotations</h2>
              <p style={{ color: "#78350f" }}>Put all four players in order. Each player competes for four rallies before the next player enters.</p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem" }}>
                {[
                  { team: teamA, order: orderA, setOrder: setOrderA, side: "A" },
                  { team: teamB, order: orderB, setOrder: setOrderB, side: "B" }
                ].map(({ team, order, setOrder, side }) => (
                  <div key={team.id} style={{ ...card, background: "white" }}>
                    <h3 style={{ marginTop: 0 }}>{team.name}</h3>
                    {order.map((value, index) => (
                      <label key={`${side}-${index}`} style={{ display: "grid", gridTemplateColumns: "70px 1fr", gap: "0.5rem", alignItems: "center", marginTop: index ? "0.55rem" : 0, fontWeight: 700 }}>
                        Slot {index + 1}
                        <select
                          value={value}
                          onChange={(event) => {
                            const next = [...order];
                            next[index] = event.target.value;
                            setOrder(next);
                          }}
                          style={input}
                        >
                          {teamPlayers(team).map((name) => <option key={name} value={name}>{name}</option>)}
                        </select>
                      </label>
                    ))}
                  </div>
                ))}
              </div>
              <button type="button" onClick={saveDreamBreakerOrders} style={{ ...button, marginTop: "1rem" }}>Submit DreamBreaker rotations</button>
            </div>
          ) : null}

          {!editingGameKey && step?.kind === "dreambreaker_score" ? (
            <DreamBreakerScoreEntry
              key={`${activeMatchup.id}:dreambreaker`}
              breaker={activeMatchup.dreamBreaker}
              teamA={teamA}
              teamB={teamB}
              onSubmit={submitDreamBreaker}
            />
          ) : null}

          {!editingGameKey && step?.kind === "complete" && result?.complete ? (
            <div style={{ ...card, borderColor: "#86efac", background: "#f0fdf4" }}>
              <p style={{ margin: "0 0 0.35rem", color: "#166534", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.06em", fontSize: "0.78rem" }}>Team matchup complete</p>
              <h2 style={{ margin: "0 0 0.4rem" }}>{teamMap.get(result.winnerTeamId || "")?.name} wins</h2>
              <p style={{ marginTop: 0 }}>
                Regulation {result.regulationWinsA}–{result.regulationWinsB}
                {result.decidedByDreamBreaker ? ` · DreamBreaker ${activeMatchup.dreamBreaker.scoreA}–${activeMatchup.dreamBreaker.scoreB}` : ""}
              </p>
              <button type="button" onClick={advanceToNextMatchup} style={button}>
                {completeCount === session.matchups.length ? "Finish and view final standings" : "Continue to next team matchup"}
              </button>
            </div>
          ) : null}

          <div style={card}>
            <h3 style={{ marginTop: 0 }}>Submitted results</h3>
            <div style={{ display: "grid", gap: "0.45rem" }}>
              {activeMatchup.games.map((game) => (
                <div key={game.key} style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) auto auto", gap: "0.75rem", alignItems: "center", borderTop: "1px solid #f1f5f9", paddingTop: "0.5rem" }}>
                  <div>
                    <strong>{game.label}</strong>
                    <div style={{ color: "#64748b", fontSize: "0.9rem" }}>{game.sideA.join(" / ")} vs {game.sideB.join(" / ")}</div>
                  </div>
                  <strong>{scoreLabel(game)}</strong>
                  {game.submitted ? (
                    <button type="button" onClick={() => { setEditingGameKey(game.key); setMessage(`Editing ${game.label}.`); }} style={secondaryButton} className="no-print">Edit</button>
                  ) : <span />}
                </div>
              ))}
              {activeMatchup.dreamBreaker.submitted ? (
                <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) auto", gap: "0.75rem", alignItems: "center", borderTop: "1px solid #f1f5f9", paddingTop: "0.5rem" }}>
                  <strong>DreamBreaker</strong>
                  <strong>{activeMatchup.dreamBreaker.scoreA}–{activeMatchup.dreamBreaker.scoreB}</strong>
                </div>
              ) : null}
            </div>
          </div>
        </>
      ) : null}

      <div style={card}>
        <h2 style={{ marginTop: 0 }}>Standings</h2>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "650px" }}>
            <thead>
              <tr>
                {['Team', 'Match W-L', 'Game W-L', 'DreamBreaker W', 'Point Diff', 'Played'].map((head) => (
                  <th key={head} style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{head}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {standings.map((row) => (
                <tr key={row.id}>
                  <td style={{ padding: "0.55rem", borderBottom: "1px solid #f1f5f9", fontWeight: 800 }}>{row.name}</td>
                  <td style={{ padding: "0.55rem", borderBottom: "1px solid #f1f5f9" }}>{row.teamWins}-{row.teamLosses}</td>
                  <td style={{ padding: "0.55rem", borderBottom: "1px solid #f1f5f9" }}>{row.gameWins}-{row.gameLosses}</td>
                  <td style={{ padding: "0.55rem", borderBottom: "1px solid #f1f5f9" }}>{row.dreamBreakerWins}</td>
                  <td style={{ padding: "0.55rem", borderBottom: "1px solid #f1f5f9" }}>{row.pointDiff > 0 ? "+" : ""}{row.pointDiff}</td>
                  <td style={{ padding: "0.55rem", borderBottom: "1px solid #f1f5f9" }}>{row.played}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
