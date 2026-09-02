"use client";

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
      <select
        aria-label="Player summary"
        value={selectedPlayerId == null ? "" : String(selectedPlayerId)}
        onChange={(event) => {
          const playerId = event.target.value;
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
      </select>
    </label>
  );
}
