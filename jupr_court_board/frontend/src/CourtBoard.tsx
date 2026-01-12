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
const MAX_ON_COURT = 4;

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

type Props = {
  courts: Court[];
  onChange: (nextCourts: Court[]) => void;
};

export default function CourtBoard({ courts, onChange }: Props) {
  // Ensure Bench exists (so logic never breaks)
  const normalizedCourts: Court[] = React.useMemo(() => {
    const hasBench = courts.some((c) => c.court_id === BENCH_ID);
    if (hasBench) return courts;
    return [...courts, { court_id: BENCH_ID, players: [] }];
  }, [courts]);

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
    const benchCourt = nextCourts[benchIdx];

    // Dropping into Bench: simple insert at index
    if (dstCourtId === BENCH_ID) {
      const insertAt = clamp(destination.index, 0, dstCourt.players.length);
      dstCourt.players.splice(insertAt, 0, dragged);
      onChange(nextCourts);
      return;
    }

    // Dropping into a normal court
    const insertAtRaw = destination.index;

    // If destination court has room (<4): normal insert
    if (dstCourt.players.length < MAX_ON_COURT) {
      const insertAt = clamp(insertAtRaw, 0, dstCourt.players.length);
      dstCourt.players.splice(insertAt, 0, dragged);
      onChange(nextCourts);
      return;
    }

    // Destination court is full (=4): REPLACE-TO-BENCH behavior.
    // Drop "onto" a card => that card gets benched, dragged player takes its slot.
    // If dropped past the end (index >= 4), replace last slot (index 3).
    const replaceIndex = clamp(insertAtRaw, 0, MAX_ON_COURT - 1);

    const [benched] = dstCourt.players.splice(replaceIndex, 1);
    dstCourt.players.splice(replaceIndex, 0, dragged);

    if (benched) {
      // Append to bench (change to unshift if you want newest on top)
      benchCourt.players.push(benched);
    }

    onChange(nextCourts);
  };

  return (
    <div className="cb-board">
      <DragDropContext onDragEnd={onDragEnd}>
        {normalizedCourts.map((court) => {
          const isBench = court.court_id === BENCH_ID;

          return (
            <div className={`cb-court ${isBench ? "cb-court-bench" : ""}`} key={court.court_id}>
              <div className="cb-court-header">
                <div className="cb-court-title">{court.court_id}</div>
                {!isBench ? (
                  <div className="cb-court-count">
                    {court.players.length}/{MAX_ON_COURT}
                  </div>
                ) : (
                  <div className="cb-court-count">{court.players.length} waiting</div>
                )}
              </div>

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
                      <div className="cb-empty">
                        {isBench ? "Drop players here to sit out." : "Drop players here."}
                      </div>
                    ) : null}
                  </div>
                )}
              </Droppable>
            </div>
          );
        })}
      </DragDropContext>
    </div>
  );
}
