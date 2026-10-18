"use client";

import { useEffect, useId, useState, type CSSProperties } from "react";

export type LeagueCourtRosterDraft = { court: string; formatType: string; playerNames: string };

type Props = {
  courts: LeagueCourtRosterDraft[];
  disabled: boolean;
  formats: string[];
  playerNames?: string[];
  onChange: (courts: LeagueCourtRosterDraft[]) => void;
  onEditRoster?: () => void;
};

type PlayerPosition = { courtIndex: number; playerIndex: number; name: string };
type UndoDraft = { before: LeagueCourtRosterDraft[]; after: string };

const buttonStyle: CSSProperties = { minHeight: 42, padding: "0.45rem 0.7rem", border: "1px solid #cbd5e1", borderRadius: 8, background: "white", color: "#0f172a", font: "inherit", fontWeight: 700, cursor: "pointer" };
const inputStyle: CSSProperties = { width: "100%", minHeight: 42, padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: 8, font: "inherit", boxSizing: "border-box" };
const rowStyle: CSSProperties = { display: "flex", gap: "0.35rem", alignItems: "center" };

function splitNames(value: string): string[] {
  return value.replace(/,/g, "\n").split("\n").map((name) => name.trim()).filter(Boolean);
}

function nameKey(value: string): string {
  return value.replace(/\s+/gu, " ").trim().toLocaleLowerCase("en-US");
}

export function LeagueCourtRosterEditor({ courts, disabled, formats, playerNames, onChange, onEditRoster }: Props) {
  const instructionsId = useId();
  const [selected, setSelected] = useState<PlayerPosition | null>(null);
  const [undo, setUndo] = useState<UndoDraft | null>(null);
  const [bulkDrafts, setBulkDrafts] = useState<Record<number, string>>({});
  const [notice, setNotice] = useState<{ text: string; error?: boolean } | null>(null);
  const fingerprint = JSON.stringify(courts);
  const namesByCourt = courts.map((court) => splitNames(court.playerNames));
  const rosterNames = playerNames ?? namesByCourt.flat();
  const allowedNames = new Map(rosterNames.map((name) => [nameKey(name), name]));
  const assignedNames = new Set(namesByCourt.flat().map(nameKey));
  const unassignedNames = [...allowedNames.entries()].filter(([key]) => !assignedNames.has(key)).map(([, name]) => name);
  const activeSelection = selected && namesByCourt[selected.courtIndex]?.[selected.playerIndex] === selected.name ? selected : null;
  const availableFormats = formats.length ? formats : ["4-player", "5-player"];

  // Discard position-based controls when the parent loads a new court arrangement.
  useEffect(() => {
    setSelected(null);
    setBulkDrafts({});
  }, [fingerprint]);

  function withNames(court: LeagueCourtRosterDraft, names: string[]): LeagueCourtRosterDraft {
    const matchingFormat = [4, 5].includes(names.length)
      ? availableFormats.find((format) => Number.parseInt(format, 10) === names.length)
      : undefined;
    return { ...court, playerNames: names.join("\n"), formatType: matchingFormat ?? court.formatType };
  }

  function commit(next: LeagueCourtRosterDraft[], message: string) {
    if (disabled || JSON.stringify(next) === fingerprint) return;
    setUndo({ before: courts.map((court) => ({ ...court })), after: JSON.stringify(next) });
    setSelected(null);
    setBulkDrafts({});
    setNotice({ text: message });
    onChange(next);
  }

  function selectOrSwap(position: PlayerPosition) {
    if (disabled) return;
    if (!activeSelection) {
      setSelected(position);
      setNotice(null);
      return;
    }
    if (activeSelection.courtIndex === position.courtIndex && activeSelection.playerIndex === position.playerIndex) {
      setSelected(null);
      return;
    }
    const nextNames = namesByCourt.map((names) => [...names]);
    nextNames[activeSelection.courtIndex][activeSelection.playerIndex] = position.name;
    nextNames[position.courtIndex][position.playerIndex] = activeSelection.name;
    commit(courts.map((court, index) => withNames(court, nextNames[index])), `Swapped ${activeSelection.name} and ${position.name}.`);
  }

  function moveSelected(targetIndex: number) {
    if (!activeSelection || targetIndex === activeSelection.courtIndex || !courts[targetIndex]) return;
    const nextNames = namesByCourt.map((names) => [...names]);
    nextNames[activeSelection.courtIndex].splice(activeSelection.playerIndex, 1);
    nextNames[targetIndex].push(activeSelection.name);
    commit(courts.map((court, index) => withNames(court, nextNames[index])), `Moved ${activeSelection.name} to Court ${courts[targetIndex].court}.`);
  }

  function reorder(courtIndex: number, playerIndex: number, direction: -1 | 1) {
    const names = [...namesByCourt[courtIndex]];
    const targetIndex = playerIndex + direction;
    if (targetIndex < 0 || targetIndex >= names.length) return;
    [names[playerIndex], names[targetIndex]] = [names[targetIndex], names[playerIndex]];
    commit(courts.map((court, index) => index === courtIndex ? withNames(court, names) : court), `Moved ${names[targetIndex]} to position ${targetIndex + 1} on Court ${courts[courtIndex].court}.`);
  }

  function unassign(courtIndex: number, playerIndex: number) {
    const name = namesByCourt[courtIndex][playerIndex];
    commit(courts.map((court, index) => index === courtIndex ? withNames(court, namesByCourt[index].filter((_, slot) => slot !== playerIndex)) : court), `${name} is unassigned. Choose a court below to place them.`);
  }

  function assign(name: string, courtIndex: number) {
    if (!allowedNames.has(nameKey(name)) || assignedNames.has(nameKey(name)) || !courts[courtIndex]) return;
    commit(courts.map((court, index) => index === courtIndex ? withNames(court, [...namesByCourt[index], name]) : court), `Assigned ${name} to Court ${courts[courtIndex].court}.`);
  }

  function applyBulk(courtIndex: number) {
    const names = splitNames(bulkDrafts[courtIndex] ?? courts[courtIndex].playerNames);
    const usedNames = new Set(namesByCourt.flatMap((list, index) => index === courtIndex ? [] : list).map(nameKey));
    for (const name of names) {
      const key = nameKey(name);
      if (usedNames.has(key)) {
        setNotice({ text: `${name} is listed more than once. Swap or move an assigned player using their row.`, error: true });
        return;
      }
      if (playerNames && !allowedNames.has(key)) {
        setNotice({ text: `${name} is not in the active session roster. Use Edit session roster to add them first.`, error: true });
        return;
      }
      usedNames.add(key);
    }
    commit(courts.map((court, index) => index === courtIndex ? withNames(court, names) : court), `Updated Court ${courts[courtIndex].court}.`);
  }

  return (
    <section aria-label="Edit court roster" aria-describedby={instructionsId} style={{ display: "grid", gap: "0.85rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "0.65rem", flexWrap: "wrap" }}>
        <p id={instructionsId} style={{ margin: 0, color: "#475569", flex: "1 1 280px" }}><strong>Tap a player, then another to swap.</strong> Use the arrows to change positions, or select a player to move them to another court.</p>
        <button type="button" disabled={disabled || !undo || undo.after !== fingerprint} onClick={() => {
          if (!undo || disabled || undo.after !== fingerprint) return;
          onChange(undo.before.map((court) => ({ ...court })));
          setUndo(null);
          setSelected(null);
          setNotice({ text: "Undid the last court edit." });
        }} style={buttonStyle}>Undo last edit</button>
        {onEditRoster ? <button type="button" onClick={onEditRoster} disabled={disabled} style={buttonStyle}>Edit session roster</button> : null}
      </div>

      {activeSelection ? (
        <div style={{ ...rowStyle, padding: "0.7rem", border: "1px solid #2563eb", borderRadius: 10, background: "#eff6ff", flexWrap: "wrap" }}>
          <strong style={{ flex: "1 1 220px" }}>{activeSelection.name}: tap another player to swap.</strong>
          <label>Move to court{" "}<select aria-label={`Move ${activeSelection.name} to court`} value="" onChange={(event) => { if (event.target.value !== "") moveSelected(Number(event.target.value)); }} disabled={disabled} style={{ ...inputStyle, width: "auto" }}>
            <option value="">Choose court…</option>
            {courts.map((court, index) => index !== activeSelection.courtIndex ? <option key={index} value={index}>Court {court.court} ({namesByCourt[index].length} players)</option> : null)}
          </select></label>
          <button type="button" onClick={() => setSelected(null)} disabled={disabled} style={buttonStyle}>Cancel selection</button>
        </div>
      ) : null}

      {notice ? <p role={notice.error ? "alert" : "status"} style={{ margin: 0, color: notice.error ? "#b91c1c" : "#334155" }}>{notice.text}</p> : null}

      {unassignedNames.length ? (
        <div style={{ padding: "0.85rem", border: "1px solid #f59e0b", borderRadius: 10, background: "#fffbeb" }}>
          <h3 style={{ margin: "0 0 0.4rem" }}>Unassigned players ({unassignedNames.length})</h3>
          <p style={{ margin: "0 0 0.65rem", color: "#92400e" }}>Place every active player on a court before generating the preview. To remove an attendee, use Edit session roster.</p>
          <div style={{ display: "grid", gap: "0.5rem", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))" }}>
            {unassignedNames.map((name) => <label key={nameKey(name)} style={{ ...rowStyle, flexWrap: "wrap" }}><strong style={{ flex: "1 1 110px" }}>{name}</strong><select aria-label={`Assign ${name} to court`} value="" disabled={disabled} onChange={(event) => { if (event.target.value !== "") assign(name, Number(event.target.value)); }} style={{ ...inputStyle, width: "auto", flex: "1 1 120px" }}><option value="">Choose court…</option>{courts.map((court, index) => <option key={index} value={index}>Court {court.court} ({namesByCourt[index].length} players)</option>)}</select></label>)}
          </div>
        </div>
      ) : null}

      <div data-responsive-court-grid style={{ display: "grid", gap: "0.85rem", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 290px), 1fr))" }}>
        {courts.map((court, courtIndex) => {
          const names = namesByCourt[courtIndex];
          const validCount = [4, 5].includes(names.length);
          const courtFormats = [court.formatType, ...availableFormats.filter((format) => format.toLowerCase() !== court.formatType.toLowerCase())];
          return (
            <article key={courtIndex} aria-label={`Court ${court.court} roster`} style={{ minWidth: 0, padding: "0.85rem", border: "1px solid #cbd5e1", borderRadius: 12, background: "#f8fafc" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: "0.5rem", marginBottom: "0.7rem" }}><h3 style={{ margin: 0 }}>Court {court.court || courtIndex + 1}</h3><span style={{ color: validCount ? "#475569" : "#b45309", fontWeight: 700 }}>{names.length} players</span></div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1.5fr", gap: "0.5rem", marginBottom: "0.7rem" }}>
                <label>Court position<select aria-label={`Court position for Court ${court.court}`} value={courtIndex} onChange={(event) => {
                  const targetIndex = Number(event.target.value);
                  if (targetIndex === courtIndex || !courts[targetIndex]) return;
                  const next = [...courts];
                  [next[courtIndex], next[targetIndex]] = [next[targetIndex], next[courtIndex]];
                  commit(next.map((row, index) => ({ ...row, court: String(index + 1) })), `Swapped the groups on Court ${court.court} and Court ${courts[targetIndex].court}.`);
                }} disabled={disabled} style={inputStyle}>{courts.map((_, index) => <option key={index} value={index}>Court {index + 1}</option>)}</select></label>
                <label>Format<select aria-label={`Format for Court ${court.court}`} value={court.formatType} onChange={(event) => commit(courts.map((row, index) => index === courtIndex ? { ...row, formatType: event.target.value } : row), `Updated Court ${court.court} format.`)} disabled={disabled} style={inputStyle}>{courtFormats.map((format) => <option key={format} value={format}>{format}</option>)}</select></label>
              </div>
              <ol aria-label={`Player positions on Court ${court.court}`} style={{ display: "grid", gap: "0.4rem", margin: 0, padding: 0, listStyle: "none" }}>
                {names.map((name, playerIndex) => {
                  const isSelected = activeSelection?.courtIndex === courtIndex && activeSelection.playerIndex === playerIndex;
                  return <li key={`${playerIndex}:${name}`} style={rowStyle}>
                    <button type="button" aria-pressed={isSelected} aria-label={activeSelection && !isSelected ? `Swap ${activeSelection.name} with ${name} on Court ${court.court}` : `Select ${name} on Court ${court.court}, position ${playerIndex + 1}`} onClick={() => selectOrSwap({ courtIndex, playerIndex, name })} disabled={disabled} style={{ ...buttonStyle, flex: "1 1 0", minWidth: 0, display: "flex", gap: "0.5rem", alignItems: "center", textAlign: "left", overflowWrap: "anywhere", borderColor: isSelected ? "#2563eb" : "#cbd5e1", background: isSelected ? "#dbeafe" : "white" }}><span style={{ color: "#64748b", fontSize: "0.85rem" }}>{playerIndex + 1}</span><span>{name}</span></button>
                    <button type="button" aria-label={`Move ${name} up on Court ${court.court}`} title="Move up one position" onClick={() => reorder(courtIndex, playerIndex, -1)} disabled={disabled || playerIndex === 0} style={{ ...buttonStyle, padding: "0.4rem 0.55rem" }}>↑</button>
                    <button type="button" aria-label={`Move ${name} down on Court ${court.court}`} title="Move down one position" onClick={() => reorder(courtIndex, playerIndex, 1)} disabled={disabled || playerIndex === names.length - 1} style={{ ...buttonStyle, padding: "0.4rem 0.55rem" }}>↓</button>
                    <button type="button" aria-label={`Unassign ${name} from Court ${court.court}`} title="Move to unassigned players" onClick={() => unassign(courtIndex, playerIndex)} disabled={disabled} style={{ ...buttonStyle, padding: "0.4rem 0.55rem" }}>×</button>
                  </li>;
                })}
              </ol>
              {!names.length ? <p style={{ color: "#64748b" }}>Move a player here or assign one from the unassigned list.</p> : null}
              {!validCount ? <p style={{ color: "#92400e", margin: "0.65rem 0 0" }}>Needs 4 or 5 players before preview.</p> : null}
              <details style={{ marginTop: "0.8rem" }}>
                <summary style={{ cursor: "pointer", color: "#475569" }}>Edit player list as text</summary>
                <label style={{ display: "block", marginTop: "0.5rem" }}>Players, one per line<textarea aria-label={`Players, one per line for Court ${court.court}`} rows={5} value={bulkDrafts[courtIndex] ?? court.playerNames} onChange={(event) => setBulkDrafts((current) => ({ ...current, [courtIndex]: event.target.value }))} disabled={disabled} style={inputStyle} /></label>
                <button type="button" onClick={() => applyBulk(courtIndex)} disabled={disabled || (bulkDrafts[courtIndex] ?? court.playerNames) === court.playerNames} style={buttonStyle}>Apply player list</button>
              </details>
              <button type="button" onClick={() => commit(courts.filter((_, index) => index !== courtIndex).map((row, index) => ({ ...row, court: String(index + 1) })), `Removed Court ${court.court}. Its players are now unassigned.`)} disabled={disabled || courts.length === 1} style={{ ...buttonStyle, marginTop: "0.75rem", fontSize: "0.85rem" }}>Remove court</button>
            </article>
          );
        })}
      </div>
      <div><button type="button" onClick={() => {
        const usedNumbers = new Set(courts.map((court) => Number(court.court)));
        let courtNumber = 1;
        while (usedNumbers.has(courtNumber)) courtNumber += 1;
        commit([...courts, { court: String(courtNumber), formatType: availableFormats.find((format) => Number.parseInt(format, 10) === 4) ?? availableFormats[0], playerNames: "" }], `Added Court ${courtNumber}.`);
      }} disabled={disabled} style={buttonStyle}>Add court</button></div>
    </section>
  );
}
