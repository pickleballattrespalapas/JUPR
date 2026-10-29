"use client";

import { useState } from "react";
import { InteractionDialog } from "@/components/interaction";
import type { HubGroup } from "@/lib/adminRoundRobinHubApi";
import styles from "./RoundRobinHubPanel.module.css";

type Props = {
  groups: HubGroup[]; busy: boolean; disabled: boolean; pending: boolean; error: string;
  onClose: () => void; onRetry: () => void; onSubmit: (sessionKey: string, finishAfterRound?: number) => void;
};

function unavailableReason(group: HubGroup): string {
  if (group.status === "completed") return "This round robin has already finished.";
  if (group.finish_requested) return `This round robin is already finishing${group.finish_after_round ? ` after round ${group.finish_after_round}` : ""}.`;
  if (group.can_finish && (typeof group.finish_pending_matches !== "number" || (group.finish_pending_matches > 0 && !group.finish_boundary_round))) return "Refresh the hub to review which matches need to finish.";
  return group.can_finish ? "" : group.finish_unavailable_reason || "This round robin cannot be finished here right now.";
}

export default function HubFinishDialog({ groups, busy, disabled, pending, error, onClose, onRetry, onSubmit }: Props) {
  const [selection, setSelection] = useState<{ sessionKey: string; title: string } | null>(null);
  const [chosenRound, setChosenRound] = useState<number | null>(null);
  const selectedGroup = groups.find(group => group.session_key === selection?.sessionKey);
  const selectableGroups = groups.filter(group => group.status !== "completed");
  const selectionError = selection ? selectedGroup ? unavailableReason(selectedGroup) : "This round robin is no longer in the event. Choose another round robin." : "";
  const frozen = busy || disabled || pending;
  const choices = selectedGroup?.finish_round_options;
  const boundary = chosenRound ?? selectedGroup?.finish_boundary_round;
  const selectedChoice = choices?.find(option => option.round_number === boundary);
  const staleChoice = Boolean(choices?.length && boundary !== undefined && !selectedChoice);
  const blocked = frozen || !selectedGroup || Boolean(selectionError) || staleChoice;
  const close = () => { if (!busy && !pending) onClose(); };
  const pendingMatches = selectedChoice?.pending_matches ?? selectedGroup?.finish_pending_matches;

  return <InteractionDialog open phase={busy ? "working" : "ready"} title="Finish round robin"
    description={<p>Choose which round robin to finish. Other round robins can keep playing.</p>}
    dirty={Boolean(selection)} onRequestClose={close}
    actions={<><button type="button" className={styles.button} disabled={busy || pending} onClick={close}>Keep playing</button><button type="submit" form="hub-finish-group-form" className={styles.primary} disabled={blocked}>Finish round robin</button></>}>
    <form id="hub-finish-group-form" onSubmit={event => {
      event.preventDefault();
      if (!blocked && selectedGroup) onSubmit(selectedGroup.session_key, choices?.length ? boundary ?? undefined : undefined);
    }}>
      <label className={styles.field}>Round robin to finish
        <select aria-label="Round robin to finish" data-autofocus required value={selection?.sessionKey || ""} disabled={frozen}
          aria-invalid={Boolean(selectionError)} aria-describedby={selectionError ? "hub-finish-group-error" : undefined}
          onChange={event => {
            if (frozen) return;
            const group = selectableGroups.find(item => item.session_key === event.target.value);
            setSelection(group ? { sessionKey: group.session_key, title: group.title } : null);
            setChosenRound(null);
          }}>
          <option value="">Choose a round robin</option>
          {selection && (!selectedGroup || selectedGroup.status === "completed") ? <option value={selection.sessionKey} disabled>{selection.title} — No longer active</option> : null}
          {selectableGroups.map(group => {
            const reason = unavailableReason(group);
            return <option key={group.session_key} value={group.session_key} disabled={Boolean(reason)}>{group.title}{reason ? ` — ${reason}` : ""}</option>;
          })}
        </select>
      </label>
      {selectedGroup && !selectionError && choices?.length ? <label className={styles.field}>Last round of the night
        <select aria-label="Last round of the night" value={boundary ?? ""} disabled={frozen} onChange={event => setChosenRound(Number(event.target.value))}>
          {staleChoice ? <option value={boundary ?? ""} disabled>Round {boundary} — no longer available</option> : null}
          {choices.map(option => <option key={option.round_number} value={option.round_number}>{option.round_number === 0 ? "Finish now" : `Finish after Round ${option.round_number}`} · {option.pending_matches} {option.pending_matches === 1 ? "result" : "results"} remaining</option>)}
        </select>
      </label> : null}
      {staleChoice ? <p role="alert">A later round has started or the schedule changed. Review the available last rounds before finishing.</p> : null}
      {selectionError ? <p id="hub-finish-group-error" role="alert">{selectionError}</p> : null}
      {selectedGroup && !selectionError ? <div className={styles.notice} style={{ marginTop: "1rem" }}>
        {pendingMatches === 0 ? <p>This round robin will finish now. No matches need to finish{boundary ? ` through round ${boundary}` : " before it closes"}.</p>
          : <><p><strong>Finish after round {boundary}.</strong> {pendingMatches} {pendingMatches === 1 ? "match still needs" : "matches still need"} to finish through that round.</p><p>Playing, reserved, and waiting matches through round {boundary} stay in the court queue. The round robin finishes automatically when they are complete.</p></>}
        <p>No scores are discarded. No more rounds will be generated. Later unstarted rounds will be left unplayed. Standings show missing results and can include recorded games provisionally; final results wait until all remaining games are resolved.</p>
      </div> : null}
      {!selectableGroups.length ? <p>All round robins in this event have finished.</p> : null}
      <p className={styles.small}>When the last round robin finishes, the event closes automatically. Results and sharing remain available in Standings.</p>
      {error ? <p role="alert">{error}</p> : null}
      {pending ? <button type="button" className={styles.button} disabled={busy} onClick={onRetry}>Retry saved action</button> : null}
    </form>
  </InteractionDialog>;
}
