type Match = {
  status?: string;
  scoreA?: number | null;
  scoreB?: number | null;
  courtLabel?: string;
  reservedCourtLabel?: string;
  hubCompletedAt?: string;
  hubStartedAt?: string;
};

export type HubAutoAdvanceEvent = {
  roundRobinHubId?: string;
  hubAutoAdvancedFromRound?: number;
  hubAssignedMatchIds?: string[];
  generatorKind?: string;
  eventFormat?: string;
  status?: string;
  playoff?: unknown;
  currentRoundNumber?: number;
  rounds?: Array<{ number: number; status: string; stage?: string; hubStartedAt?: string; matches?: Match[]; courts?: Array<{ matches?: Match[] }> }>;
};

/** The hub explicitly prepares its visible horizon; older sessions expose only the current round. */
export function hubPreparedRoundLimit(event?: { currentRoundNumber?: number; hubPreparedThroughRound?: number } | null): number {
  const current = event?.currentRoundNumber || 1;
  const prepared = event?.hubPreparedThroughRound;
  return typeof prepared === "number" && Number.isInteger(prepared) && prepared >= current ? prepared : current;
}

/** Presentation only; the server rechecks the boundary and live court claims. */
export function hubAutoAdvanceBoundary(event?: HubAutoAdvanceEvent | null): number | null {
  if (!event?.roundRobinHubId || event.generatorKind !== "round_robin" || event.status !== "active"
    || event.playoff || event.eventFormat === "moneyball_expectations" || event.hubAssignedMatchIds?.length) return null;
  const boundary = event.hubAutoAdvancedFromRound;
  const current = event.currentRoundNumber;
  if (!Number.isInteger(boundary) || !boundary || boundary < 1 || current !== boundary + 1) return null;
  const rounds = event.rounds || [];
  if (!rounds.some(row => row.number === boundary) || !rounds.some(row => row.number === current && row.status === "active")) return null;
  for (const row of rounds) {
    if (row.number <= boundary) {
      if (!["saved", "played", "skipped"].includes(row.status)) return null;
      continue;
    }
    if (row.status !== (row.number === current ? "active" : "preview") || row.stage === "playoff" || row.hubStartedAt) return null;
    const matches = row.matches?.length ? row.matches : (row.courts || []).flatMap(court => court.matches || []);
    if (matches.some(match => match.scoreA != null || match.scoreB != null || (match.status && match.status !== "scheduled")
      || match.courtLabel || match.reservedCourtLabel || match.hubCompletedAt || match.hubStartedAt)) return null;
  }
  return boundary;
}
