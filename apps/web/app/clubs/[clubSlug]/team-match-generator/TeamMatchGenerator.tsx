"use client";

import { useEffect, useMemo, useState } from "react";

type Team = {
  id: string;
  name: string;
  women: [string, string];
  men: [string, string];
  mixedMode: "straight" | "cross";
};

type Game = {
  key: string;
  label: string;
  sideA: string;
  sideB: string;
  scoreA: number | null;
  scoreB: number | null;
};

type Matchup = {
  id: string;
  round: number;
  teamAId: string;
  teamBId: string;
  games: Game[];
};

type StoredState = {
  title: string;
  teams: Team[];
  matchups: Matchup[];
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

function uid(prefix: string): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function blankTeam(index: number): Team {
  return {
    id: uid("team"),
    name: `Team ${index + 1}`,
    women: ["", ""],
    men: ["", ""],
    mixedMode: "straight"
  };
}

function teamReady(team: Team): boolean {
  const names = [...team.women, ...team.men].map((name) => name.trim()).filter(Boolean);
  return team.name.trim().length > 0 && names.length === 4 && new Set(names.map((name) => name.toLowerCase())).size === 4;
}

function mixedPairs(team: Team): [[string, string], [string, string]] {
  if (team.mixedMode === "cross") {
    return [
      [team.women[0], team.men[1]],
      [team.women[1], team.men[0]]
    ];
  }
  return [
    [team.women[0], team.men[0]],
    [team.women[1], team.men[1]]
  ];
}

function buildGames(teamA: Team, teamB: Team): Game[] {
  const aMixed = mixedPairs(teamA);
  const bMixed = mixedPairs(teamB);
  return [
    {
      key: "women",
      label: "Women’s Doubles",
      sideA: `${teamA.women[0]} / ${teamA.women[1]}`,
      sideB: `${teamB.women[0]} / ${teamB.women[1]}`,
      scoreA: null,
      scoreB: null
    },
    {
      key: "men",
      label: "Men’s Doubles",
      sideA: `${teamA.men[0]} / ${teamA.men[1]}`,
      sideB: `${teamB.men[0]} / ${teamB.men[1]}`,
      scoreA: null,
      scoreB: null
    },
    {
      key: "mixed-1",
      label: "Mixed Doubles 1",
      sideA: `${aMixed[0][0]} / ${aMixed[0][1]}`,
      sideB: `${bMixed[0][0]} / ${bMixed[0][1]}`,
      scoreA: null,
      scoreB: null
    },
    {
      key: "mixed-2",
      label: "Mixed Doubles 2",
      sideA: `${aMixed[1][0]} / ${aMixed[1][1]}`,
      sideB: `${bMixed[1][0]} / ${bMixed[1][1]}`,
      scoreA: null,
      scoreB: null
    }
  ];
}

function roundRobinPairs(teamIds: string[]): Array<{ round: number; a: string; b: string }> {
  const ids = [...teamIds];
  if (ids.length % 2 === 1) ids.push("BYE");
  const rounds = ids.length - 1;
  const half = ids.length / 2;
  const rotation = [...ids];
  const result: Array<{ round: number; a: string; b: string }> = [];

  for (let round = 1; round <= rounds; round += 1) {
    for (let i = 0; i < half; i += 1) {
      const a = rotation[i];
      const b = rotation[rotation.length - 1 - i];
      if (a !== "BYE" && b !== "BYE") result.push({ round, a, b });
    }
    const fixed = rotation[0];
    const rest = rotation.slice(1);
    rest.unshift(rest.pop() as string);
    rotation.splice(0, rotation.length, fixed, ...rest);
  }
  return result;
}

function csvEscape(value: unknown): string {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export default function TeamMatchGenerator({ clubId }: { clubId: string }) {
  const storageKey = `team-match-generator:${clubId}`;
  const [title, setTitle] = useState(`Team Match ${new Date().toISOString().slice(0, 10)}`);
  const [teams, setTeams] = useState<Team[]>([blankTeam(0), blankTeam(1)]);
  const [matchups, setMatchups] = useState<Matchup[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      if (raw) {
        const stored = JSON.parse(raw) as StoredState;
        if (stored.title) setTitle(stored.title);
        if (stored.teams?.length >= 2) setTeams(stored.teams);
        if (stored.matchups) setMatchups(stored.matchups);
      }
    } catch {
      // Ignore a bad local draft and start clean.
    }
    setHydrated(true);
  }, [storageKey]);

  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(storageKey, JSON.stringify({ title, teams, matchups } satisfies StoredState));
  }, [hydrated, storageKey, title, teams, matchups]);

  const teamMap = useMemo(() => new Map(teams.map((team) => [team.id, team])), [teams]);

  const standings = useMemo(() => {
    const rows = new Map(
      teams.map((team) => [
        team.id,
        { id: team.id, name: team.name, teamWins: 0, teamLosses: 0, ties: 0, gameWins: 0, gameLosses: 0, pointDiff: 0, played: 0 }
      ])
    );

    for (const matchup of matchups) {
      const a = rows.get(matchup.teamAId);
      const b = rows.get(matchup.teamBId);
      if (!a || !b) continue;
      let aGames = 0;
      let bGames = 0;
      let completed = 0;
      for (const game of matchup.games) {
        if (game.scoreA === null || game.scoreB === null || game.scoreA === game.scoreB) continue;
        completed += 1;
        a.pointDiff += game.scoreA - game.scoreB;
        b.pointDiff += game.scoreB - game.scoreA;
        if (game.scoreA > game.scoreB) {
          aGames += 1;
          a.gameWins += 1;
          b.gameLosses += 1;
        } else {
          bGames += 1;
          b.gameWins += 1;
          a.gameLosses += 1;
        }
      }
      if (completed === 4) {
        a.played += 1;
        b.played += 1;
        if (aGames > bGames) {
          a.teamWins += 1;
          b.teamLosses += 1;
        } else if (bGames > aGames) {
          b.teamWins += 1;
          a.teamLosses += 1;
        } else {
          a.ties += 1;
          b.ties += 1;
        }
      }
    }

    return [...rows.values()].sort((a, b) =>
      b.teamWins - a.teamWins ||
      b.gameWins - a.gameWins ||
      b.pointDiff - a.pointDiff ||
      a.name.localeCompare(b.name)
    );
  }, [matchups, teams]);

  function patchTeam(teamId: string, patch: Partial<Team>): void {
    setTeams((current) => current.map((team) => (team.id === teamId ? { ...team, ...patch } : team)));
    if (matchups.length) setMessage("Team details changed. Regenerate the schedule to refresh player pairings.");
  }

  function patchPlayer(teamId: string, group: "women" | "men", index: 0 | 1, value: string): void {
    setTeams((current) => current.map((team) => {
      if (team.id !== teamId) return team;
      const next = [...team[group]] as [string, string];
      next[index] = value;
      return { ...team, [group]: next };
    }));
    if (matchups.length) setMessage("Player names changed. Regenerate the schedule to refresh matchups.");
  }

  function addTeam(): void {
    setTeams((current) => [...current, blankTeam(current.length)]);
    setMatchups([]);
  }

  function removeTeam(teamId: string): void {
    if (teams.length <= 2) return;
    setTeams((current) => current.filter((team) => team.id !== teamId));
    setMatchups([]);
  }

  function generate(): void {
    const notReady = teams.filter((team) => !teamReady(team));
    if (notReady.length) {
      setMessage("Each team needs a name plus four distinct players: two women and two men.");
      return;
    }
    const pairs = roundRobinPairs(teams.map((team) => team.id));
    const next = pairs.map((pair) => {
      const teamA = teamMap.get(pair.a) as Team;
      const teamB = teamMap.get(pair.b) as Team;
      return {
        id: uid("matchup"),
        round: pair.round,
        teamAId: pair.a,
        teamBId: pair.b,
        games: buildGames(teamA, teamB)
      };
    });
    setMatchups(next);
    setMessage(`Generated ${next.length} team matchup${next.length === 1 ? "" : "s"}.`);
  }

  function scoreGame(matchupId: string, gameKey: string, side: "A" | "B", raw: string): void {
    const parsed = raw === "" ? null : Math.max(0, Number.parseInt(raw, 10) || 0);
    setMatchups((current) => current.map((matchup) => {
      if (matchup.id !== matchupId) return matchup;
      return {
        ...matchup,
        games: matchup.games.map((game) => game.key === gameKey ? {
          ...game,
          [side === "A" ? "scoreA" : "scoreB"]: parsed
        } : game)
      };
    }));
  }

  function downloadCsv(): void {
    const rows = [["Round", "Team A", "Game", "Side A", "Score A", "Score B", "Side B", "Team B"]];
    for (const matchup of matchups) {
      const teamA = teamMap.get(matchup.teamAId);
      const teamB = teamMap.get(matchup.teamBId);
      for (const game of matchup.games) {
        rows.push([
          String(matchup.round), teamA?.name || "", game.label, game.sideA,
          game.scoreA ?? "", game.scoreB ?? "", game.sideB, teamB?.name || ""
        ] as string[]);
      }
    }
    const csv = rows.map((row) => row.map(csvEscape).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${title.trim().replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "team-match"}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function reset(): void {
    if (!window.confirm("Clear this team match draft and all entered scores?")) return;
    localStorage.removeItem(storageKey);
    setTitle(`Team Match ${new Date().toISOString().slice(0, 10)}`);
    setTeams([blankTeam(0), blankTeam(1)]);
    setMatchups([]);
    setMessage("Draft cleared.");
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <style>{`@media print { .no-print { display:none !important; } body { background:white !important; } }`}</style>
      <div style={card} className="no-print">
        <label style={{ display: "grid", gap: "0.35rem", fontWeight: 700 }}>
          Session title
          <input value={title} onChange={(event) => setTitle(event.target.value)} style={input} />
        </label>
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
              {teams.length > 2 ? <button type="button" onClick={() => removeTeam(team.id)} style={secondaryButton} className="no-print">Remove</button> : null}
            </div>
            <p style={{ marginBottom: "0.35rem", fontWeight: 800 }}>Women</p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
              <input placeholder="Woman 1" value={team.women[0]} onChange={(event) => patchPlayer(team.id, "women", 0, event.target.value)} style={input} />
              <input placeholder="Woman 2" value={team.women[1]} onChange={(event) => patchPlayer(team.id, "women", 1, event.target.value)} style={input} />
            </div>
            <p style={{ marginBottom: "0.35rem", fontWeight: 800 }}>Men</p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
              <input placeholder="Man 1" value={team.men[0]} onChange={(event) => patchPlayer(team.id, "men", 0, event.target.value)} style={input} />
              <input placeholder="Man 2" value={team.men[1]} onChange={(event) => patchPlayer(team.id, "men", 1, event.target.value)} style={input} />
            </div>
            <label style={{ display: "grid", gap: "0.35rem", marginTop: "0.8rem", fontWeight: 700 }}>
              Mixed pairing
              <select value={team.mixedMode} onChange={(event) => patchTeam(team.id, { mixedMode: event.target.value as Team["mixedMode"] })} style={input}>
                <option value="straight">Woman 1 + Man 1 / Woman 2 + Man 2</option>
                <option value="cross">Woman 1 + Man 2 / Woman 2 + Man 1</option>
              </select>
            </label>
          </div>
        ))}
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.6rem" }} className="no-print">
        <button type="button" onClick={addTeam} style={secondaryButton}>Add team</button>
        <button type="button" onClick={generate} style={button}>Generate team schedule</button>
        {matchups.length ? <button type="button" onClick={downloadCsv} style={secondaryButton}>Download CSV</button> : null}
        {matchups.length ? <button type="button" onClick={() => window.print()} style={secondaryButton}>Print</button> : null}
        <button type="button" onClick={reset} style={secondaryButton}>Clear draft</button>
      </div>

      {message ? <div style={{ ...card, borderColor: "#bfdbfe", background: "#eff6ff", color: "#1e3a8a" }} className="no-print">{message}</div> : null}

      {matchups.length ? (
        <>
          <div style={card}>
            <h2 style={{ marginTop: 0 }}>{title || "Team Match"}</h2>
            <p style={{ color: "#475569" }}>Each team matchup contains four regulation games: women’s doubles, men’s doubles, mixed doubles 1, and mixed doubles 2.</p>
          </div>

          <div style={card}>
            <h2 style={{ marginTop: 0 }}>Standings</h2>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "620px" }}>
                <thead><tr>{["Team", "Match W-L-T", "Game W-L", "Point Diff", "Played"].map((head) => <th key={head} style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{head}</th>)}</tr></thead>
                <tbody>{standings.map((row) => <tr key={row.id}>
                  <td style={{ padding: "0.55rem", borderBottom: "1px solid #f1f5f9", fontWeight: 800 }}>{row.name}</td>
                  <td style={{ padding: "0.55rem", borderBottom: "1px solid #f1f5f9" }}>{row.teamWins}-{row.teamLosses}-{row.ties}</td>
                  <td style={{ padding: "0.55rem", borderBottom: "1px solid #f1f5f9" }}>{row.gameWins}-{row.gameLosses}</td>
                  <td style={{ padding: "0.55rem", borderBottom: "1px solid #f1f5f9" }}>{row.pointDiff > 0 ? "+" : ""}{row.pointDiff}</td>
                  <td style={{ padding: "0.55rem", borderBottom: "1px solid #f1f5f9" }}>{row.played}</td>
                </tr>)}</tbody>
              </table>
            </div>
          </div>

          {Array.from(new Set(matchups.map((matchup) => matchup.round))).map((round) => (
            <section key={round} style={{ display: "grid", gap: "0.8rem" }}>
              <h2 style={{ marginBottom: 0 }}>Round {round}</h2>
              {matchups.filter((matchup) => matchup.round === round).map((matchup) => {
                const teamA = teamMap.get(matchup.teamAId);
                const teamB = teamMap.get(matchup.teamBId);
                return <div key={matchup.id} style={card}>
                  <h3 style={{ marginTop: 0 }}>{teamA?.name} vs {teamB?.name}</h3>
                  <div style={{ display: "grid", gap: "0.55rem" }}>
                    {matchup.games.map((game) => <div key={game.key} style={{ display: "grid", gridTemplateColumns: "minmax(180px, 1fr) 72px 72px minmax(180px, 1fr)", gap: "0.5rem", alignItems: "center", padding: "0.5rem 0", borderTop: "1px solid #f1f5f9" }}>
                      <div><strong>{game.label}</strong><br /><span style={{ color: "#475569" }}>{game.sideA}</span></div>
                      <input aria-label={`${game.label} score for ${teamA?.name}`} type="number" min={0} value={game.scoreA ?? ""} onChange={(event) => scoreGame(matchup.id, game.key, "A", event.target.value)} style={input} />
                      <input aria-label={`${game.label} score for ${teamB?.name}`} type="number" min={0} value={game.scoreB ?? ""} onChange={(event) => scoreGame(matchup.id, game.key, "B", event.target.value)} style={input} />
                      <div style={{ textAlign: "right" }}><strong>{game.label}</strong><br /><span style={{ color: "#475569" }}>{game.sideB}</span></div>
                    </div>)}
                  </div>
                </div>;
              })}
            </section>
          ))}
        </>
      ) : null}
    </div>
  );
}
