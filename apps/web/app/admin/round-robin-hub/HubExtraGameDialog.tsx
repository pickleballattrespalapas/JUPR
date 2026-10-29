"use client";

import { useState } from "react";
import { InteractionDialog } from "@/components/interaction";
import type { HubGroup } from "@/lib/adminRoundRobinHubApi";
import styles from "./RoundRobinHubPanel.module.css";

export type ExtraGameParticipantRef = { session_key: string; participant_id: string };
export type ExtraGamePayload = {
  side_a: ExtraGameParticipantRef[]; side_b: ExtraGameParticipantRef[]; standings_policy: "exhibition";
};
type Props = {
  groups: HubGroup[]; busy: boolean; disabled: boolean; pending: boolean; error: string;
  onClose: () => void; onRetry: () => void; onSubmit: (payload: ExtraGamePayload) => void;
};

export default function HubExtraGameDialog({ groups, busy, disabled, pending, error, onClose, onRetry, onSubmit }: Props) {
  const [chosen, setChosen] = useState(["", "", "", ""]);
  const choices = groups.filter(group => group.status !== "completed" && !group.results_locked &&
    group.event_format === "standard" && !("supplemental" in group && group.supplemental))
    .flatMap(group => (group.players || []).filter(player => player.active).map(player => ({
      value: JSON.stringify([group.session_key, player.participant_id]),
      ref: { session_key: group.session_key, participant_id: player.participant_id },
      identity: player.identity, label: `${player.display_name || player.name} · ${group.title}`,
      unavailable: player.paused || player.on_court || player.reserved || Boolean(player.current_court || player.reserved_court),
    })));
  const selected = chosen.map(value => choices.find(choice => choice.value === value));
  const frozen = busy || disabled || pending;
  const duplicate = new Set(selected.filter(Boolean).map(choice => choice!.identity)).size < selected.filter(Boolean).length;
  const changed = chosen.some((value, index) => Boolean(value) && !selected[index]);
  const unavailable = selected.some(choice => choice?.unavailable);
  const ready = selected.every(Boolean) && !duplicate && !changed && !unavailable;
  const reason = duplicate ? "Choose four different people." : changed ? "A selected player is no longer active. Choose another player." : unavailable ? "Finish or release the selected players' courts and reservations first." : "";
  const close = () => { if (!busy && !pending) onClose(); };
  return <InteractionDialog open phase={busy ? "working" : "ready"} title="Extra game"
    description={<p>Choose two teams from the active divisions. Players keep their division membership.</p>}
    dirty={chosen.some(Boolean)} onRequestClose={close}
    actions={<><button type="button" className={styles.button} disabled={busy || pending} onClick={close}>Cancel</button>
      <button type="submit" form="hub-extra-game-form" className={styles.primary} disabled={frozen || !ready}>Create extra game</button></>}>
    <form id="hub-extra-game-form" onSubmit={event => { event.preventDefault(); if (!frozen && ready) onSubmit({
      side_a: selected.slice(0, 2).map(choice => choice!.ref), side_b: selected.slice(2).map(choice => choice!.ref), standings_policy: "exhibition",
    }); }}>
      <p><strong>Exhibition only.</strong> This game has its own result. It does not count toward division standings, ratings, or official results.</p>
      {chosen.map((value, index) => <label key={index} className={styles.field}>Team {index < 2 ? "A" : "B"} · player {index % 2 + 1}
        <select aria-label={`Extra game team ${index < 2 ? "A" : "B"} player ${index % 2 + 1}`} value={value} disabled={frozen}
          data-autofocus={index === 0 || undefined} onChange={event => setChosen(previous => previous.map((item, position) => position === index ? event.target.value : item))}>
          <option value="">Choose a player</option>
          {value && !selected[index] ? <option value={value} disabled>Player no longer available</option> : null}
          {choices.map(choice => <option key={choice.value} value={choice.value} disabled={choice.unavailable || selected.some((other, position) => position !== index && other?.identity === choice.identity)}>
            {choice.label}{choice.unavailable ? " · Playing, reserved, or paused" : ""}</option>)}
        </select>
      </label>)}
      <p className={styles.small}>Players must finish their current division games before joining an extra game. Court availability is checked again when the game is assigned.</p>
      {reason ? <p role="alert">{reason}</p> : null}{error ? <p role="alert">{error}</p> : null}
      {pending ? <button type="button" className={styles.button} disabled={busy} onClick={onRetry}>Retry saved action</button> : null}
    </form>
  </InteractionDialog>;
}
