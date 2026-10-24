"use client";

import { useState } from "react";
import { InteractionDialog } from "@/components/interaction";
import type { HubGroup } from "@/lib/adminRoundRobinHubApi";
import styles from "./RoundRobinHubPanel.module.css";

type Props = {
  groups: HubGroup[]; busy: boolean; disabled: boolean; pending: boolean; error: string;
  onClose: () => void; onRetry: () => void;
  onSubmit: (source: string, participant: string, destination: string) => void;
};

export default function HubMovePlayerDialog({ groups, busy, disabled, pending, error, onClose, onRetry, onSubmit }: Props) {
  const [sourceKey, setSourceKey] = useState("");
  const [participantId, setParticipantId] = useState("");
  const [destinationKey, setDestinationKey] = useState("");
  const source = groups.find(group => group.session_key === sourceKey);
  const player = source?.players?.find(item => item.participant_id === participantId);
  const destination = groups.find(group => group.session_key === destinationKey);
  const frozen = busy || disabled || pending;
  const reason = sourceKey && !source ? "The source round robin is no longer available." : source && !source.can_remove_player ? source.remove_player_unavailable_reason || "This round robin cannot move players." : participantId && !player ? "This player is no longer in the source round robin." : player?.can_remove === false ? player.remove_unavailable_reason || "This player has already left." : player && (player.on_court || player.reserved || player.current_court || player.reserved_court) ? "Finish this player’s committed match, mark it not played, or cancel its reservation before moving them." : destinationKey && !destination ? "The destination round robin is no longer available." : destinationKey && sourceKey === destinationKey ? "Choose a different destination round robin." : destination && !destination.can_add_player ? destination.add_player_unavailable_reason || "This round robin cannot add players." : destination && player && destination.players?.some(member => member.identity === player.identity && !member.can_rejoin) ? "This player is already in the destination round robin." : "";
  const blocked = frozen || !source || !player || !destination || Boolean(reason);
  const close = () => { if (!busy && !pending) onClose(); };
  return <InteractionDialog open phase={busy ? "working" : "ready"} title="Move player" description={<p>Move one player between round robins in this event.</p>} dirty={Boolean(sourceKey || participantId || destinationKey)} onRequestClose={close}
    actions={<><button type="button" className={styles.button} disabled={busy || pending} onClick={close}>Cancel</button><button type="submit" form="hub-move-player-form" className={styles.primary} disabled={blocked}>Move player to round robin</button></>}>
    <form id="hub-move-player-form" onSubmit={event => { event.preventDefault(); if (!blocked) onSubmit(sourceKey, participantId, destinationKey); }}>
      <label className={styles.field}>From round robin<select aria-label="From round robin" data-autofocus value={sourceKey} disabled={frozen} onChange={event => { setSourceKey(event.target.value); setParticipantId(""); }}><option value="">Choose a round robin</option>{groups.map(group => <option key={group.session_key} value={group.session_key} disabled={!group.can_remove_player}>{group.title}</option>)}</select></label>
      <label className={styles.field}>Player to move<select aria-label="Player to move" value={participantId} disabled={frozen || !source} onChange={event => setParticipantId(event.target.value)}><option value="">Choose a player</option>{source?.players?.map(member => <option key={member.participant_id} value={member.participant_id} disabled={member.can_remove === false}>{member.name}{member.can_remove === false ? " · Departed" : ""}</option>)}</select></label>
      <label className={styles.field}>To round robin<select aria-label="To round robin" value={destinationKey} disabled={frozen} onChange={event => setDestinationKey(event.target.value)}><option value="">Choose a destination</option>{groups.map(group => <option key={group.session_key} value={group.session_key} disabled={group.session_key === sourceKey || !group.can_add_player}>{group.title}</option>)}</select></label>
      {destination ? <p className={styles.small}>Joins {destination.title} from round {destination.add_player_effective_round ?? destination.current_round_number}. Their source membership ends in the same action. Saved results and announced lineups are preserved.</p> : null}
      {reason ? <p role="alert">{reason}</p> : null}{error ? <p role="alert">{error}</p> : null}
      {pending ? <button type="button" className={styles.button} disabled={busy} onClick={onRetry}>Retry saved action</button> : null}
    </form>
  </InteractionDialog>;
}
