export function swapRosterPositions(rosterOrder, firstParticipantId, secondParticipantId) {
  const firstId = String(firstParticipantId || "").trim();
  const secondId = String(secondParticipantId || "").trim();
  if (!firstId || !secondId) {
    throw new Error("Choose two players to swap.");
  }
  if (firstId === secondId) {
    throw new Error("Choose two different players to swap.");
  }

  const firstIndex = rosterOrder.indexOf(firstId);
  const secondIndex = rosterOrder.indexOf(secondId);
  if (firstIndex < 0 || secondIndex < 0) {
    throw new Error("Both players must be in the current roster.");
  }

  const swapped = [...rosterOrder];
  [swapped[firstIndex], swapped[secondIndex]] = [
    swapped[secondIndex],
    swapped[firstIndex]
  ];
  return swapped;
}
