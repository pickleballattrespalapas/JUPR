"use client";

import SearchablePlayerSelect from "@/components/SearchablePlayerSelect";

import { useRouter } from "next/navigation";
import type { LeagueResultsPlayerOption } from "@/lib/api";

type Props = {
  baseHref: string;
  players: LeagueResultsPlayerOption[];
  selectedPlayerId: string | number | null;
};

const inputStyle = {
  width: "100%",
  boxSizing: "border-box" as const,
  padding: "0.65rem",
  border: "1px solid #cbd5e1",
  borderRadius: "10px",
  background: "white",
  font: "inherit"
};

export default function LeaguePlayerSelect({ baseHref, players, selectedPlayerId }: Props) {
  const router = useRouter();

  return (
    <label>
      <strong>Player</strong><br />
      <SearchablePlayerSelect
        aria-label="Player summary"
        value={selectedPlayerId == null ? "" : String(selectedPlayerId)}
        onValueChange={playerValue => {
          const playerId = playerValue;
          router.push(playerId ? `${baseHref}?player=${encodeURIComponent(playerId)}` : baseHref);
        }}
        style={inputStyle}
      >
        <option value="">Choose a player</option>
        {players.map((player) => (
          <option key={String(player.player_id)} value={String(player.player_id)}>
            {player.player_name}
          </option>
        ))}
      </SearchablePlayerSelect>
    </label>
  );
}
