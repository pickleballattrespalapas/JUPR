"use client";

import { useState } from "react";
import { DragDropContext, Draggable, Droppable, type DropResult } from "@hello-pangea/dnd";
import styles from "./LeagueMovementCourtBoard.module.css";

export type MovementBoardAssignment = {
  playerId: number;
  toCourt: number | null;
  toSlot: number | null;
};

type MovementRow = {
  player_id: number;
  player_name: string;
  from_court: number | null;
  suggested_court: number | null;
  suggested_slot?: number | null;
  to_court: number | null;
  to_slot?: number | null;
  wins: number;
  differential: number;
  overridden: boolean;
};

type RosterPlayer = {
  player_id: number;
  player_name: string;
  rating?: number | null;
};

type MovementCourt = {
  court_number: number;
  player_names: string[];
  players_json?: Array<Record<string, unknown>>;
};

type Props = {
  rows: MovementRow[];
  courts: MovementCourt[];
  bench: RosterPlayer[];
  disabled?: boolean;
  onAssignmentsChange: (assignments: MovementBoardAssignment[]) => void;
};

type BoardPlayer = {
  id: number;
  name: string;
  rating: number | null;
  fromCourt: number | null;
  suggestedCourt: number | null;
  suggestedSlot: number | null;
  wins: number | null;
  differential: number | null;
};

type BoardColumn = {
  id: string;
  courtNumber: number | null;
  title: string;
  players: BoardPlayer[];
};

const BENCH_COLUMN_ID = "bench";

function courtColumnId(courtNumber: number): string {
  return `court-${courtNumber}`;
}

function numericValue(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function boardPlayer(
  raw: Record<string, unknown>,
  rowById: Map<number, MovementRow>,
  fallbackName = "Player",
  fallbackCourt: number | null = null,
  fallbackSlot: number | null = null,
): BoardPlayer | null {
  const id = numericValue(raw.player_id ?? raw.id);
  if (id == null || id <= 0) return null;
  const row = rowById.get(id);
  return {
    id,
    name: String(raw.player_name ?? raw.name ?? row?.player_name ?? fallbackName),
    rating: numericValue(raw.rating ?? raw.rating_jupr),
    fromCourt: row?.from_court ?? null,
    suggestedCourt: row?.suggested_court ?? fallbackCourt,
    suggestedSlot: row?.suggested_slot ?? fallbackSlot,
    wins: row ? Number(row.wins) : null,
    differential: row ? Number(row.differential) : null,
  };
}

function initialColumns(rows: MovementRow[], courts: MovementCourt[], bench: RosterPlayer[]): BoardColumn[] {
  const rowById = new Map(rows.map((row) => [Number(row.player_id), row]));
  const rowByName = new Map(rows.map((row) => [row.player_name.trim().toLocaleLowerCase(), row]));
  const columns: BoardColumn[] = [...courts]
    .sort((left, right) => Number(left.court_number) - Number(right.court_number))
    .map((court) => {
      const rawPlayers = Array.isArray(court.players_json) ? court.players_json : [];
      const players = rawPlayers.length
        ? rawPlayers.flatMap((raw, index) => {
            const player = boardPlayer(raw, rowById, court.player_names[index] || "Player", court.court_number, index + 1);
            return player ? [player] : [];
          })
        : court.player_names.flatMap((name, index) => {
            const row = rowByName.get(name.trim().toLocaleLowerCase());
            if (!row) return [];
            return [{
              id: Number(row.player_id),
              name: row.player_name,
              rating: null,
              fromCourt: row.from_court,
              suggestedCourt: row.suggested_court,
              suggestedSlot: row.suggested_slot ?? index + 1,
              wins: Number(row.wins),
              differential: Number(row.differential),
            }];
          });
      return {
        id: courtColumnId(Number(court.court_number)),
        courtNumber: Number(court.court_number),
        title: `Court ${court.court_number}`,
        players,
      };
    });
  columns.push({
    id: BENCH_COLUMN_ID,
    courtNumber: null,
    title: "Bench",
    players: bench.flatMap((raw) => {
      const player = boardPlayer(raw as unknown as Record<string, unknown>, rowById, raw.player_name, null, null);
      return player ? [player] : [];
    }),
  });
  return columns;
}

function assignmentsFor(columns: BoardColumn[]): MovementBoardAssignment[] {
  return columns.flatMap((column) => column.players.map((player, index) => ({
    playerId: player.id,
    toCourt: column.courtNumber,
    toSlot: column.courtNumber == null ? null : index + 1,
  })));
}

function signed(value: number): string {
  return value > 0 ? `+${value}` : String(value);
}

function movementMeta(player: BoardPlayer, targetCourt: number | null, targetSlot: number | null) {
  if (targetCourt == null) {
    return {
      direction: "bench",
      label: player.fromCourt == null ? "On Bench" : `Benched from Court ${player.fromCourt}`,
      symbol: "•",
    };
  }
  if (player.fromCourt == null) {
    return { direction: "new", label: `New or returning · Court ${targetCourt}`, symbol: "+" };
  }
  if (targetCourt < player.fromCourt) {
    return { direction: "up", label: `Moved up from Court ${player.fromCourt}`, symbol: "↑" };
  }
  if (targetCourt > player.fromCourt) {
    return { direction: "down", label: `Moved down from Court ${player.fromCourt}`, symbol: "↓" };
  }
  return { direction: "stay", label: `Staying on Court ${targetCourt}`, symbol: "•" };
}

function averageRating(players: BoardPlayer[]): string | null {
  const ratings = players.flatMap((player) => player.rating == null ? [] : [player.rating]);
  if (!ratings.length) return null;
  const average = ratings.reduce((total, value) => total + value, 0) / ratings.length;
  return average >= 100 ? average.toFixed(0) : average.toFixed(2);
}

export function LeagueMovementCourtBoard({ rows, courts, bench, disabled = false, onAssignmentsChange }: Props) {
  const [columns, setColumns] = useState<BoardColumn[]>(() => initialColumns(rows, courts, bench));
  const [announcement, setAnnouncement] = useState("Drag a player card to reorder a court, move between courts, or exchange a player with Bench.");

  function handleDragEnd(result: DropResult) {
    if (disabled || !result.destination) return;
    const { source, destination, draggableId } = result;
    if (source.droppableId === destination.droppableId && source.index === destination.index) return;
    const next = columns.map((column) => ({ ...column, players: [...column.players] }));
    const sourceColumn = next.find((column) => column.id === source.droppableId);
    const destinationColumn = next.find((column) => column.id === destination.droppableId);
    if (!sourceColumn || !destinationColumn) return;
    const [player] = sourceColumn.players.splice(source.index, 1);
    if (!player || String(player.id) !== draggableId) return;
    destinationColumn.players.splice(destination.index, 0, player);
    setColumns(next);
    onAssignmentsChange(assignmentsFor(next));
    setAnnouncement(`${player.name} moved to ${destinationColumn.title}, position ${destination.index + 1}. Validate the board before continuing.`);
  }

  return (
    <section className={styles.boardSection} aria-labelledby="movement-court-board-heading">
      <div className={styles.boardHeadingRow}>
        <div>
          <h4 id="movement-court-board-heading" className={styles.boardHeading}>Next-round court board</h4>
          <p className={styles.instructions}>Drag cards within a court, between courts, or to/from Bench. Green moved up; red moved down.</p>
        </div>
        <div className={styles.legend} aria-label="Movement color legend">
          <span className={styles.legendUp}>↑ Up</span>
          <span className={styles.legendDown}>↓ Down</span>
          <span className={styles.legendStay}>• Same court</span>
        </div>
      </div>
      <p className={styles.srOnly} aria-live="polite">{announcement}</p>
      <DragDropContext onDragEnd={handleDragEnd}>
        <div className={styles.board} data-testid="league-movement-court-board">
          {columns.map((column) => {
            const countIsValid = column.courtNumber == null || [4, 5].includes(column.players.length);
            const rating = averageRating(column.players);
            return (
              <section className={`${styles.column} ${column.courtNumber == null ? styles.benchColumn : ""}`} key={column.id} data-testid={`movement-column-${column.id}`}>
                <header className={styles.columnHeader}>
                  <div>
                    <strong>{column.title}</strong>
                    <span className={`${styles.playerCount} ${countIsValid ? "" : styles.invalidCount}`}>{column.players.length} player{column.players.length === 1 ? "" : "s"}</span>
                  </div>
                  {rating ? <small>Avg JUPR {rating}</small> : null}
                </header>
                <Droppable droppableId={column.id} isDropDisabled={disabled}>
                  {(provided, snapshot) => (
                    <div
                      ref={provided.innerRef}
                      {...provided.droppableProps}
                      className={`${styles.dropZone} ${snapshot.isDraggingOver ? styles.draggingOver : ""}`}
                      aria-label={`${column.title} player cards`}
                    >
                      {column.players.map((player, index) => {
                        const targetSlot = column.courtNumber == null ? null : index + 1;
                        const meta = movementMeta(player, column.courtNumber, targetSlot);
                        const overridden = column.courtNumber !== player.suggestedCourt || targetSlot !== player.suggestedSlot;
                        return (
                          <Draggable draggableId={String(player.id)} index={index} key={player.id} isDragDisabled={disabled}>
                            {(dragProvided, dragSnapshot) => (
                              <article
                                ref={dragProvided.innerRef}
                                {...dragProvided.draggableProps}
                                {...dragProvided.dragHandleProps}
                                className={`${styles.playerCard} ${styles[meta.direction]} ${dragSnapshot.isDragging ? styles.dragging : ""}`}
                                data-testid={`movement-player-${player.id}`}
                                data-movement-direction={meta.direction}
                                aria-label={`${player.name}. ${meta.label}. Position ${index + 1} in ${column.title}.`}
                              >
                                <div className={styles.playerTopLine}>
                                  <strong>{player.name}</strong>
                                  <span className={styles.dragHandle} aria-hidden="true">⠿</span>
                                </div>
                                <div className={styles.movementLine}><span aria-hidden="true">{meta.symbol}</span> {meta.label}</div>
                                <div className={styles.metrics}>
                                  {player.wins != null ? <span><b>{player.wins}</b> wins</span> : <span>New round player</span>}
                                  {player.differential != null ? <span><b>{signed(player.differential)}</b> diff</span> : null}
                                  {player.rating != null ? <span><b>{player.rating >= 100 ? player.rating.toFixed(0) : player.rating.toFixed(2)}</b> JUPR</span> : null}
                                </div>
                                {overridden ? <span className={styles.overrideBadge}><span>Manual board change</span><small>Planned: {player.suggestedCourt == null ? "Bench" : `Court ${player.suggestedCourt}${player.suggestedSlot ? ` · slot ${player.suggestedSlot}` : ""}`}</small></span> : null}
                              </article>
                            )}
                          </Draggable>
                        );
                      })}
                      {provided.placeholder}
                      {!column.players.length ? <p className={styles.emptyColumn}>Drop a player here</p> : null}
                    </div>
                  )}
                </Droppable>
                {!countIsValid ? <p className={styles.capacityWarning} role="alert">Court requires 4 or 5 players before validation.</p> : null}
              </section>
            );
          })}
        </div>
      </DragDropContext>
    </section>
  );
}
