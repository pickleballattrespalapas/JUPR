/** Search fragments independently so first/last names work in either order. */
export function playerSearchKey(value: string): string {
  return value.normalize("NFKD").replace(/\p{M}/gu, "").toLocaleLowerCase().replace(/\s+/g, " ").trim();
}

export function matchesPlayerSearch(name: string, query: string): boolean {
  const haystack = playerSearchKey(name);
  return playerSearchKey(query).split(" ").every(part => haystack.includes(part));
}

export async function fetchAllPublicPlayers<T>(url: string, signal: AbortSignal): Promise<T[]> {
  const rows: T[] = [];
  for (;;) {
    const pageUrl = new URL(url);
    pageUrl.searchParams.set("limit", "1000");
    pageUrl.searchParams.set("offset", String(rows.length));
    const response = await fetch(pageUrl.toString(), { cache: "no-store", signal });
    if (!response.ok) throw new Error("Player search is temporarily unavailable.");
    const page = await response.json() as { players?: T[]; pagination?: { has_more?: boolean } };
    const players = page.players || [];
    rows.push(...players);
    if (!page.pagination?.has_more || !players.length) return rows;
  }
}
