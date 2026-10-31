/** Pool editor values are deliberately independent from registration divisions. */
export function seededPoolConfiguration(teams, poolCount = 2) {
  if (!Number.isInteger(poolCount) || poolCount < 2 || poolCount > Math.floor(teams.length / 2)) {
    throw new Error("Each pool needs at least two teams.");
  }
  const ordered = [...teams].sort((left, right) =>
    (Number(left.seed) || Number(left.team_number) || 0) - (Number(right.seed) || Number(right.team_number) || 0)
    || String(left.id).localeCompare(String(right.id)));
  const pools = Array.from({ length: poolCount }, (_, index) => ({
    id: String.fromCharCode(65 + index), name: `Pool ${String.fromCharCode(65 + index)}`, team_ids: [], advance_count: 2
  }));
  ordered.forEach((team, index) => {
    const position = index % poolCount;
    const pool = Math.floor(index / poolCount) % 2 ? poolCount - position - 1 : position;
    pools[pool].team_ids.push(String(team.id));
  });
  return { version: 1, pools };
}

export function inspectPoolConfiguration(configuration, teams) {
  if (configuration == null) return { ok: true, errors: [], poolMatches: [], totalMatches: teams.length * (teams.length - 1) / 2, totalQualifiers: 0, uneven: false };
  const errors = [];
  const pools = configuration.pools || [];
  const ids = teams.map(team => String(team.id || ""));
  const assigned = pools.flatMap(pool => pool.team_ids || []);
  if (configuration.version !== 1) errors.push("Reload this pool configuration before editing it.");
  if (pools.length < 2 || pools.length > 8) errors.push("Choose between two and eight pools.");
  if (teams.length < 4 || teams.length > 16 || ids.some(id => !id) || new Set(ids).size !== ids.length) errors.push("Save a draw with 4–16 distinct teams first.");
  if (new Set(pools.map(pool => pool.id)).size !== pools.length) errors.push("Each pool needs a distinct identifier.");
  if (pools.some(pool => !String(pool.name || "").trim())) errors.push("Give each pool a name.");
  if (new Set(pools.map(pool => String(pool.name || "").trim().toLowerCase())).size !== pools.length) errors.push("Use a different name for each pool.");
  if (new Set(assigned).size !== assigned.length) errors.push("A team can belong to only one pool.");
  if (assigned.some(id => !ids.includes(String(id))) || ids.some(id => !assigned.includes(id))) errors.push("Assign every saved team to exactly one pool.");
  for (const pool of pools) {
    const size = (pool.team_ids || []).length;
    if (size < 2) errors.push(`${pool.name || "Each pool"} needs at least two teams.`);
    if (!Number.isInteger(pool.advance_count) || pool.advance_count < 0 || pool.advance_count > size) errors.push(`${pool.name || "Each pool"}: choose an advancement count from 0 to ${size}.`);
  }
  const totalQualifiers = pools.reduce((sum, pool) => sum + (Number(pool.advance_count) || 0), 0);
  if (totalQualifiers < 2 || totalQualifiers > 16) errors.push("Select a total of 2–16 playoff teams across the pools.");
  const poolMatches = pools.map(pool => ({ poolId: pool.id, matches: pool.team_ids.length * (pool.team_ids.length - 1) / 2, gamesPerTeam: pool.team_ids.length - 1 }));
  return { ok: errors.length === 0, errors, poolMatches, totalMatches: poolMatches.reduce((sum, pool) => sum + pool.matches, 0), totalQualifiers, uneven: new Set(pools.map(pool => pool.team_ids.length)).size > 1 };
}

export function moveTeamToPool(configuration, teamId, poolId) {
  return { ...configuration, pools: configuration.pools.map(pool => ({
    ...pool,
    team_ids: [...pool.team_ids.filter(id => id !== teamId), ...(pool.id === poolId ? [teamId] : [])]
  })) };
}
