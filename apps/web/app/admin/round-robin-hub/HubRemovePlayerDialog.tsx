"use client";

import { useState } from "react";
import { InteractionDialog } from "@/components/interaction";
import type { HubGroup, HubGroupPlayer } from "@/lib/adminRoundRobinHubApi";
import styles from "./RoundRobinHubPanel.module.css";

type Props = {
  groups: HubGroup[]; busy: boolean; disabled: boolean; pending: boolean; error: string;
  onClose: () => void; onRetry: () => void;
  onSubmit: (sessionKey: string, participantId: string) => void;
  onSubmitMany?: (sessionKey: string, participantIds: string[]) => void;
};

function groupUnavailableReason(group: HubGroup): string {
  return group.can_remove_player ? "" : group.remove_player_unavailable_reason || "Players cannot be removed from this round robin here.";
}

function playerUnavailableReason(group: HubGroup, player: HubGroupPlayer): string {
  if (player.can_remove === true) return "";
  if (player.can_remove === false) return player.remove_unavailable_reason || "This player has already left or is already scheduled to leave this round robin.";
  const nextEligibleRound = Math.max(group.current_round_number + 1, player.active_from_round || 1);
  const availableLater = player.inactive_from_round == null || player.inactive_from_round > nextEligibleRound;
  return player.active || availableLater ? "" : "This player has already left this round robin.";
}

export default function HubRemovePlayerDialog({ groups, busy, disabled, pending, error, onClose, onRetry, onSubmit, onSubmitMany }: Props) {
  const [destination, setDestination] = useState<{ sessionKey: string; title: string } | null>(null);
  const [selection, setSelection] = useState<{ participantId: string; label: string } | null>(null);
  const [selections, setSelections] = useState<{ participantId: string; label: string }[]>([]);
  const frozen = busy || disabled || pending;
  const selectedGroup = groups.find(group => group.session_key === destination?.sessionKey);
  const selectedPlayer = selectedGroup?.players?.find(player => player.participant_id === selection?.participantId);
  const groupError = destination ? selectedGroup ? groupUnavailableReason(selectedGroup) : "This round robin is no longer in the event. Choose another round robin." : "";
  const playerError = selection && selectedGroup ? selectedPlayer ? playerUnavailableReason(selectedGroup, selectedPlayer) : "This player is no longer in this round robin. Choose another player." : "";
  const batchPlayers = selections.map(item => selectedGroup?.players?.find(player => player.participant_id === item.participantId));
  const batchError = selections.map((item, index) => {
    const player = batchPlayers[index];
    return player && selectedGroup ? playerUnavailableReason(selectedGroup, player) : `${item.label} is no longer in this round robin.`;
  }).find(Boolean) || "";
  const blocked = frozen || !selectedGroup || Boolean(groupError)
    || (onSubmitMany ? !selections.length || Boolean(batchError) : !selectedPlayer || Boolean(playerError));
  const destinations = [...groups].sort((left, right) => left.title.localeCompare(right.title, undefined, { numeric: true, sensitivity: "base" }));
  const players = [...(selectedGroup?.players || [])].sort((left, right) => left.name.localeCompare(right.name, undefined, { numeric: true, sensitivity: "base" }));
  const close = () => { if (!busy && !pending) onClose(); };

  return <InteractionDialog open phase={busy ? "working" : "ready"} title={onSubmitMany ? "Remove players" : "Remove player"} description={<p>{onSubmitMany ? "Choose everyone leaving this round robin. Their results will be kept." : "Choose the round robin and player to remove."}</p>}
    dirty={Boolean(destination || selection || selections.length)} onRequestClose={close}
    actions={<><button type="button" className={styles.button} disabled={busy || pending} onClick={close}>Cancel</button><button type="submit" form="hub-remove-player-form" className={styles.primary} disabled={blocked}>{onSubmitMany ? `Remove ${selections.length || "selected"} player${selections.length === 1 ? "" : "s"}` : "Remove from this round robin"}</button></>}>
    <form id="hub-remove-player-form" onSubmit={event => {
      event.preventDefault();
      if (blocked || !selectedGroup) return;
      if (onSubmitMany) onSubmitMany(selectedGroup.session_key, selections.map(item => item.participantId));
      else if (selectedPlayer) onSubmit(selectedGroup.session_key, selectedPlayer.participant_id);
    }}>
      <label className={styles.field}>Round robin
        <select aria-label="Round robin" data-autofocus required value={destination?.sessionKey || ""} disabled={frozen}
          aria-invalid={Boolean(groupError)} aria-describedby={groupError ? "hub-remove-player-group-error" : undefined}
          onChange={event => {
            if (frozen) return;
            const group = groups.find(item => item.session_key === event.target.value);
            setDestination(group ? { sessionKey: group.session_key, title: group.title } : null);
            setSelection(null);
            setSelections([]);
          }}>
          <option value="">Choose a round robin</option>
          {destination && !selectedGroup ? <option value={destination.sessionKey} disabled>{destination.title} — No longer available</option> : null}
          {destinations.map(group => {
            const reason = groupUnavailableReason(group);
            return <option key={group.session_key} value={group.session_key} disabled={Boolean(reason)}>{group.title}{reason ? ` — ${reason}` : ""}</option>;
          })}
        </select>
      </label>
      {groupError ? <p id="hub-remove-player-group-error" role="alert">{groupError}</p> : null}
      {onSubmitMany ? <fieldset disabled={frozen || !selectedGroup || Boolean(groupError)}>
        <legend>Players leaving</legend>
        {players.map(player => {
          const selected = selections.some(item => item.participantId === player.participant_id);
          const reason = selectedGroup ? playerUnavailableReason(selectedGroup, player) : "";
          return <label key={player.participant_id} className={styles.field}>
            <span><input type="checkbox" aria-label={`Remove ${player.name}`} checked={selected} disabled={Boolean(reason) && !selected}
              onChange={event => {
                if (frozen || !selectedGroup || groupError) return;
                const checked = event.target.checked;
                setSelections(previous => {
                  if (!checked) return previous.filter(item => item.participantId !== player.participant_id);
                  return previous.some(item => item.participantId === player.participant_id) ? previous
                    : [...previous, { participantId: player.participant_id, label: player.name }];
                });
              }} /> {player.name}{reason ? " — Already leaving or departed" : ` — From round ${player.remove_effective_round ?? selectedGroup?.current_round_number}${player.remove_affected_match_ids?.length ? ` · ${player.remove_affected_match_ids.length} planned game${player.remove_affected_match_ids.length === 1 ? "" : "s"} may change` : ""}`}</span>
          </label>;
        })}
      </fieldset> : <label className={styles.field}>Player
        <select aria-label="Player" required value={selection?.participantId || ""} disabled={frozen || !selectedGroup || Boolean(groupError)}
          aria-invalid={Boolean(playerError)} aria-describedby={playerError ? "hub-remove-player-person-error" : undefined}
          onChange={event => {
            if (frozen) return;
            const player = selectedGroup?.players?.find(item => item.participant_id === event.target.value);
            setSelection(player ? { participantId: player.participant_id, label: player.name } : null);
          }}>
          <option value="">Choose a player</option>
          {selection && !selectedPlayer ? <option value={selection.participantId} disabled>{selection.label} — No longer available</option> : null}
          {players.map(player => {
            const reason = selectedGroup ? playerUnavailableReason(selectedGroup, player) : "";
            return <option key={player.participant_id} value={player.participant_id} disabled={Boolean(reason)}>{player.name}{reason ? " — Already leaving or departed" : ""}</option>;
          })}
        </select>
      </label>}
      {(onSubmitMany ? batchError : playerError) ? <p id="hub-remove-player-person-error" role="alert">{onSubmitMany ? batchError : playerError}</p> : null}
      {!groups.length ? <p className={styles.small}>Add a round robin to this event before removing a player.</p> : selectedGroup && !players.length ? <p className={styles.small}>There are no players in this round robin.</p> : null}
      {onSubmitMany && selections.length > 0 && !groupError && !batchError ? <p className={styles.small}>All selected departures save together. Each player finishes any committed game before leaving. Future games are repaired for the remaining players, and saved results are kept.</p> : null}
      {!onSubmitMany && selectedPlayer && !groupError && !playerError ? <p className={styles.small}>
        Departure takes effect from round {selectedPlayer.remove_effective_round ?? selectedGroup?.current_round_number}. Announced, assigned, and reserved lineups keep their players. Saved results are kept.
      </p> : null}
      {error ? <p role="alert">{error}</p> : null}
      {pending ? <button type="button" className={styles.button} disabled={busy} onClick={onRetry}>Retry saved action</button> : null}
    </form>
  </InteractionDialog>;
}
