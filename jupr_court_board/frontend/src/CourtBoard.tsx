// /workspaces/JUPR/jupr_court_board/frontend/src/CourtBoard.tsx
import React from "react";
import {
  DragDropContext,
  Draggable,
  Droppable,
  DropResult,
  DraggableProvided,
  DraggableStateSnapshot,
  DroppableProvided,
  DroppableStateSnapshot,
} from "react-beautiful-dnd";

export type PlayerCard = {
  player_id: string; // must be globally unique across all courts + bench
  name: string;
  rating?: number; // display value (e.g., JUPR)
};

export type Court = {
  court_id: string; // e.g. "Court 1", "Court 2", "Bench"
  players: PlayerCard[];
};

export type CourtsPayload = {
  courts: Court[];
};

const BENCH_ID = "Bench";
const TARGET_ON_COURT = 4;

function isBenchCourt(courtId: string) {
  return courtId.trim().toLowerCase() === BENCH_ID.toLowerCase();
}

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

type Props = {
  courts: Court[];
  onChange: (nextCourts: Court[]) => void;
};

export default function CourtBoard({ courts, onChange }: Props) {
  // Ensure Bench exists and is normalized to a single list.
  const normalizedCourts: Court[] = React.useMemo(() => {
    const benchPlayers = courts
      .filter((c) => isBenchCourt(c.court_id))
      .flatMap((c) => c.players);

    const playableCourts = courts
      .filter((c) => !isBenchCourt(c.court_id))
      .map((c) => ({ ...c, players: [...c.players] }));

    return [...playableCourts, { court_id: BENCH_ID, players: benchPlayers }];
  }, [courts]);

  const renderCourtHeader = (court: Court, isBench: boolean) => {
    const isWarn = !isBench && court.players.length !== TARGET_ON_COURT;

    return (
      <div className="cb-court-header">
        <div className="cb-court-title">{isBench ? BENCH_ID : court.court_id}</div>

        {!isBench ? (
          <div
            className={`cb-court-count ${isWarn ? "cb-court-count-warn" : ""}`}
            title={
              isWarn
                ? `Court has ${court.players.length} players (target ${TARGET_ON_COURT}).`
                : `Court has ${TARGET_ON_COURT} players.`
            }
          >
            {court.players.length}/{TARGET_ON_COURT}
            {isWarn ? <span className="cb-warn-pill">!</span> : null}
          </div>
        ) : (
          <div className="cb-court-count" title="Bench players will not be scheduled.">
            {court.players.length} waiting
          </div>
        )}
      </div>
    );
  };

  const renderDroppableCourt = (court: Court, isBench: boolean) => (
    <Droppable droppableId={court.court_id}>
      {(provided: DroppableProvided, snapshot: DroppableStateSnapshot) => (
        <div
          className={`cb-court-body ${snapshot.isDraggingOver ? "cb-court-body-over" : ""}`}
          ref={provided.innerRef}
          {...provided.droppableProps}
        >
          {court.players.map((p, idx) => (
            <Draggable draggableId={p.player_id} index={idx} key={p.player_id}>
              {(dragProvided: DraggableProvided, dragSnapshot: DraggableStateSnapshot) => (
                <div
                  className={`cb-card ${dragSnapshot.isDragging ? "cb-card-dragging" : ""}`}
                  ref={dragProvided.innerRef}
                  {...dragProvided.draggableProps}
                  {...dragProvided.dragHandleProps}
                >
                  <div className="cb-card-name">{p.name}</div>
                  <div className="cb-card-meta">
                    <span className="cb-card-id">{p.player_id}</span>
                    {typeof p.rating === "number" ? (
                      <span className="cb-card-rating">{p.rating.toFixed(3)}</span>
                    ) : null}
                  </div>
                </div>
              )}
            </Draggable>
          ))}

          {provided.placeholder}

          {court.players.length === 0 ? (
            <div className="cb-empty">{isBench ? "Drop players here" : "Drop players here."}</div>
          ) : null}
        </div>
      )}
    </Droppable>
  );

  const renderBench = (benchCourt: Court) => (
    <div className="cb-court cb-court-bench" key={BENCH_ID}>
      {renderCourtHeader(benchCourt, true)}
      {renderDroppableCourt(benchCourt, true)}
    </div>
  );

  const playableCourts = normalizedCourts.filter((court) => !isBenchCourt(court.court_id));
  const benchCourt = normalizedCourts.find((court) => isBenchCourt(court.court_id)) ?? {
    court_id: BENCH_ID,
    players: [],
  };

  const onDragEnd = (result: DropResult) => {
    const { source, destination } = result;
    if (!destination) return;

    const srcCourtId = source.droppableId;
    const dstCourtId = destination.droppableId;

    // Clone courts/players for safe mutation
    const nextCourts = normalizedCourts.map((c) => ({ ...c, players: [...c.players] }));

    const srcIdx = nextCourts.findIndex((c) => c.court_id === srcCourtId);
    const dstIdx = nextCourts.findIndex((c) => c.court_id === dstCourtId);
    const benchIdx = nextCourts.findIndex((c) => c.court_id === BENCH_ID);
    if (srcIdx < 0 || dstIdx < 0 || benchIdx < 0) return;

    // Reorder within the same court/bench
    if (srcCourtId === dstCourtId) {
      const court = nextCourts[srcIdx];
      const from = source.index;
      const to = destination.index;

      const arr = Array.from(court.players);
      const [moved] = arr.splice(from, 1);
      if (!moved) return;

      arr.splice(to, 0, moved);
      court.players = arr;

      onChange(nextCourts);
      return;
    }

    // Remove dragged player from source list
    const srcCourt = nextCourts[srcIdx];
    const [dragged] = srcCourt.players.splice(source.index, 1);
    if (!dragged) return;

    const dstCourt = nextCourts[dstIdx];

    // Dropping into Bench: simple insert at index
    if (dstCourtId === BENCH_ID) {
      const insertAt = clamp(destination.index, 0, dstCourt.players.length);
      dstCourt.players.splice(insertAt, 0, dragged);
      onChange(nextCourts);
      return;
    }

    // Dropping into a normal court (NO auto-bench, allow any size)
    const insertAt = clamp(destination.index, 0, dstCourt.players.length);
    dstCourt.players.splice(insertAt, 0, dragged);
    onChange(nextCourts);
  };

  return (
    <div className="cb-board">
      <DragDropContext onDragEnd={onDragEnd}>
        {playableCourts.map((court) => (
          <div className="cb-court" key={court.court_id}>
            {renderCourtHeader(court, false)}
            {renderDroppableCourt(court, false)}
          </div>
        ))}
        {renderBench(benchCourt)}
      </DragDropContext>
    </div>
  );
}
