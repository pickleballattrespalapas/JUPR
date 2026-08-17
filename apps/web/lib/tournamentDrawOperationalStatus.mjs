const INACTIVE_DRAW_STATUSES = new Set([
  "archived",
  "cancelled",
  "canceled",
  "deleted",
  "disabled",
  "inactive",
  "void",
  "voided"
]);

function persistedDrawStatus(draw) {
  return String(draw.status || "").trim().toLowerCase();
}

export function isInactiveTournamentDraw(draw) {
  return INACTIVE_DRAW_STATUSES.has(persistedDrawStatus(draw));
}

export function drawOperationalStatus(draw, lifecycleDraw) {
  const persistedStatus = persistedDrawStatus(draw);
  if (INACTIVE_DRAW_STATUSES.has(persistedStatus)) {
    const normalized = persistedStatus.replace(/_/g, " ");
    return `${normalized.charAt(0).toUpperCase()}${normalized.slice(1)}`;
  }
  if (!lifecycleDraw) return "Status unavailable";

  const games = Number(lifecycleDraw.counts.games);
  const finalizedGames = Number(lifecycleDraw.counts.finalized_games);
  const openGames = Number(lifecycleDraw.counts.open_games);
  const publishedGames = Number(lifecycleDraw.counts.published_games || 0);
  const duplicatePublications = Number(
    lifecycleDraw.counts.duplicate_publications
      || lifecycleDraw.counts.duplicate_official_links
      || 0
  );
  const mismatchedOfficialMatches = Number(lifecycleDraw.counts.mismatched_official_matches || 0);
  const missingPublicationEvidence = Number(
    lifecycleDraw.counts.official_matches_without_publication_evidence || 0
  );
  const liveOperations = String(lifecycleDraw.states.live_operations || "");
  const officialPublish = String(lifecycleDraw.states.official_publish || "");
  const validCounts = [games, finalizedGames, openGames, publishedGames].every(
    (value) => Number.isInteger(value) && value >= 0
  )
    && finalizedGames + openGames === games
    && publishedGames <= finalizedGames;
  if (!validCounts) return "Status unavailable";

  const publicationNeedsRecovery = duplicatePublications > 0
    || mismatchedOfficialMatches > 0
    || missingPublicationEvidence > 0
    || (publishedGames > 0 && publishedGames < games);
  if (publicationNeedsRecovery) {
    return `Publish recovery needed · ${publishedGames} of ${games} official`;
  }
  if (officialPublish === "complete" && games > 0 && publishedGames === games) {
    const matchWord = publishedGames === 1 ? "match" : "matches";
    return `Published · ${publishedGames} official ${matchWord}`;
  }
  if (games === 0) return "No games scheduled";
  if (liveOperations === "not_started" && finalizedGames === 0) {
    const gameWord = games === 1 ? "game" : "games";
    return `Not started · ${games} ${gameWord}`;
  }
  if (liveOperations === "in_progress" && finalizedGames > 0 && openGames > 0) {
    return `In progress · ${finalizedGames} of ${games} scored`;
  }
  if (liveOperations === "complete" && openGames === 0 && finalizedGames === games) {
    return `Scores complete · ${finalizedGames} of ${games} scored`;
  }
  return "Status unavailable";
}
