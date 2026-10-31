"use client";

import { useState } from "react";
import SearchablePlayerSelect from "@/components/SearchablePlayerSelect";
import { ConfirmAction } from "@/components/ConfirmAction";
import type { ActionCompletion } from "@/components/interaction";
import type { AdminTournamentOpsPlayer, AdminTournamentUnlinkedRegistration } from "@/lib/adminTournamentApi";

type Props = {
  registrations: AdminTournamentUnlinkedRegistration[];
  players: AdminTournamentOpsPlayer[];
  disabled: boolean;
  onLink: (registration: AdminTournamentUnlinkedRegistration, playerId: number, confirmation: string) => Promise<ActionCompletion>;
};

function PlayerLinkRow({ registration, players, disabled, onLink }: Omit<Props, "registrations"> & { registration: AdminTournamentUnlinkedRegistration }) {
  const normalize = (name: string) => name.trim().replace(/\s+/g, " ").toLocaleLowerCase();
  const matches = players.filter(player => normalize(player.name) === normalize(registration.display_name));
  const [selectedId, setSelectedId] = useState(matches.length === 1 ? String(matches[0].id) : "");
  const selected = players.find(player => String(player.id) === selectedId);
  return <li style={{ padding: "0.85rem 0", borderBottom: "1px solid #e2e8f0", listStyle: "none" }}>
    <strong>{registration.display_name}</strong>
    <p style={{ margin: "0.35rem 0", color: "#475569" }}>{matches.length === 1
      ? "Matching name found. Confirm this is the right person."
      : matches.length > 1 ? "More than one profile has this name. Review the profile before choosing."
        : "Search the club roster for this player’s existing profile."}</p>
    <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem", alignItems: "end" }}>
      <label style={{ flex: "1 1 260px" }}>PCS player profile
        <SearchablePlayerSelect aria-label={`Profile for ${registration.display_name}`} value={selectedId} onValueChange={setSelectedId} disabled={disabled}>
          <option value="">Choose a player…</option>
          {players.map(player => <option key={player.id} value={player.id}>{player.name} · #{player.id}{player.active === false ? " · inactive" : ""}</option>)}
        </SearchablePlayerSelect>
      </label>
      <ConfirmAction triggerLabel={`Link ${registration.display_name}`} title="Link this registration to a player profile?"
        description={`Link ${registration.display_name} to ${selected?.name || "the selected player"} (profile #${selectedId}). Their confirmed partnership and event entries will be kept.`}
        confirmLabel="Yes, link player" confirmationText="LINK PLAYER" disabled={disabled || !selected || !registration.updated_at}
        onConfirm={confirmation => onLink(registration, Number(selectedId), confirmation)} />
    </div>
  </li>;
}

export default function TournamentPlayerLinkReview(props: Props) {
  if (!props.registrations.length) return null;
  return <section aria-label="Link player profiles" style={{ border: "1px solid #f59e0b", borderRadius: 12, padding: "1rem", marginBottom: "1rem", background: "#fffbeb" }}>
    <h3 style={{ marginTop: 0 }}>Link player profiles</h3>
    <p>{props.registrations.length} confirmed {props.registrations.length === 1 ? "registration needs" : "registrations need"} a player profile before import. Review and link each player below.</p>
    <ul style={{ padding: 0, marginBottom: 0 }}>
      {props.registrations.map(registration => <PlayerLinkRow key={`${registration.id}:${registration.updated_at}`} registration={registration} players={props.players} disabled={props.disabled} onLink={props.onLink} />)}
    </ul>
    <p style={{ color: "#475569", marginBottom: 0 }}>If no profile exists, create the player in Players, then reload this draw to select them.</p>
  </section>;
}
