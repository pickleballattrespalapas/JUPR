"use client";

import { useEffect, useState } from "react";
import PlayerSearchInput from "@/components/PlayerSearchInput";
import { InteractionDialog } from "@/components/interaction";
import { fetchAllPublicPlayers } from "@/lib/playerSearch";
import { hubPlayerLabel, type HubGroup, type HubGroupPlayer, type HubMatch } from "@/lib/adminRoundRobinHubApi";
import styles from "./RoundRobinHubPanel.module.css";

type Props = {
  apiBase: string; clubId: string; match: HubMatch; group?: HubGroup;
  stale: boolean; available: boolean; busy: boolean; disabled: boolean; pending: boolean; error: string;
  onClose: () => void; onRetry: () => void; onReview: () => void;
  onSubmit: (payload: Record<string, unknown>) => void;
};

export default function HubEditPlayersDialog({ apiBase, clubId, match, group, stale, available,
  busy, disabled, pending, error, onClose, onRetry, onReview, onSubmit }: Props) {
  const [target, setTarget] = useState("");
  const [replacement, setReplacement] = useState("");
  const [name, setName] = useState("");
  const [profileId, setProfileId] = useState<number | null>(null);
  const [directory, setDirectory] = useState<{ id: string | number; name: string }[]>([]);
  const [loading, setLoading] = useState(true);
  const [directoryError, setDirectoryError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    void fetchAllPublicPlayers<{ id: string | number; name: string }>(
      `${apiBase.replace(/\/$/, "")}/clubs/${encodeURIComponent(clubId.replace(/_/g, "-"))}/players?status=all&sort=name`, controller.signal
    ).then(players => { if (!controller.signal.aborted) setDirectory(players); })
      .catch(() => { if (!controller.signal.aborted) setDirectoryError("Club search is unavailable. You can still enter a guest name."); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [apiBase, clubId]);

  const lineup = [...match.side_a, ...match.side_b];
  const roster = [...(group?.players || [])].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" }));
  function unavailableReason(player: HubGroupPlayer): string {
    if (player.participant_id === target) return "Already selected";
    if (player.paused) return "Paused — resume first";
    if (!lineup.some(member => member.id === player.participant_id) && (player.on_court || player.reserved || player.current_court || player.reserved_court)) return "On another court or reserved";
    const number = match.round_number;
    if ((player.active_from_round || 1) > number || (player.inactive_from_round != null && player.inactive_from_round <= number)
      || player.inactive_rounds?.includes(number) || (!player.active && number === group?.current_round_number)) return "Not active in this round";
    return "";
  }
  const adding = replacement === "new";
  const selected = roster.find(player => player.participant_id === replacement);
  const duplicate = adding && roster.some(player => profileId ? player.profile_id === profileId : player.name.trim().toLocaleLowerCase() === name.trim().toLocaleLowerCase());
  const frozen = busy || pending || disabled;
  const valid = available && !stale && lineup.some(player => player.id === target)
    && (adding ? Boolean(name.trim()) && !duplicate && Boolean(group?.can_add_player) : Boolean(selected && !unavailableReason(selected)));
  const sameMatch = lineup.some(player => player.id === replacement);
  return <InteractionDialog open title="Edit players" phase={busy ? "working" : "ready"}
    description={<p>{group?.title} · Round {match.round_number}</p>}
    dirty={Boolean(target || replacement || name)} onRequestClose={() => { if (!busy && !pending) onClose(); }}
    actions={<><button type="button" className={styles.button} disabled={busy || pending} onClick={onClose}>Cancel</button>
      <button type="submit" form="hub-edit-players-form" className={styles.primary} disabled={frozen || !valid}>Save players</button></>}>
    <div className={styles.scoreGrid}><div><strong>Side A</strong><p>{match.side_a.map(hubPlayerLabel).join(" / ")}</p></div><div><strong>Side B</strong><p>{match.side_b.map(hubPlayerLabel).join(" / ")}</p></div></div>
    <form id="hub-edit-players-form" onSubmit={event => {
      event.preventDefault();
      if (!frozen && valid) onSubmit({ participant_id: target, ...(adding ? { name: name.trim(), player_id: profileId } : { replacement_participant_id: replacement }) });
    }}>
      <label className={styles.field}>Player to replace or swap
        <select aria-label="Player to replace or swap" data-autofocus value={target} disabled={frozen || stale || !available}
          onChange={event => { setTarget(event.target.value); setReplacement(""); setName(""); setProfileId(null); }}>
          <option value="">Choose a player in this match</option>
          {lineup.map(player => <option key={player.id} value={player.id}>{hubPlayerLabel(player)}</option>)}
        </select>
      </label>
      <label className={styles.field}>Replacement
        <select aria-label="Replacement" value={replacement} disabled={frozen || stale || !available || !target}
          onChange={event => { setReplacement(event.target.value); setName(""); setProfileId(null); }}>
          <option value="">Choose a replacement</option>
          <option value="new" disabled={!group?.can_add_player}>Add a new player{!group?.can_add_player ? " — unavailable" : ""}</option>
          <optgroup label="Round robin players">{roster.map(player => {
            const reason = unavailableReason(player);
            const status = reason || (lineup.some(member => member.id === player.participant_id) ? "Swap positions" : player.on_bye ? "On a bye" : "In rotation");
            return <option key={player.participant_id} value={player.participant_id} disabled={Boolean(reason)}>{hubPlayerLabel(player)} · {status}</option>;
          })}</optgroup>
        </select>
      </label>
      {adding ? <><label className={styles.field}>New player name or club profile
        <PlayerSearchInput aria-label="New player name or club profile" value={name} maxLength={120} disabled={frozen || stale}
          options={directory.filter(player => Number.isSafeInteger(Number(player.id)) && Number(player.id) > 0).map(player => ({ value: String(player.id), label: player.name }))}
          loading={loading} error={directoryError} onTextChange={value => { setName(value); setProfileId(null); }}
          onPick={option => { setName(option.label); setProfileId(Number(option.value)); }} />
      </label><p className={styles.small}>{profileId ? "Club profile selected." : "Choose a club profile or enter a guest name."} The new player joins this match and future rotations.</p>
        {duplicate ? <p role="alert">This player is already on the roster. Choose them from Round robin players.</p> : null}
        {directoryError ? <p className={styles.small}>{directoryError}</p> : null}</> : null}
      <p className={styles.small}>{sameMatch ? "These two players swap positions in this match." : "A player on a bye takes this place, and the original player stays available. If the replacement has another waiting game in this round, the two players swap games."}</p>
      {!available ? <p role="alert">This match is no longer available for editing. Close this dialog and review the court board.</p> : stale ? <div className={styles.notice}><p>The hub changed. Review the current lineup; your replacement selection and name are preserved.</p><button type="button" className={styles.button} disabled={frozen} onClick={onReview}>Reload current players</button></div> : null}
      {error ? <p role="alert">{error}</p> : null}
      {pending ? <button type="button" className={styles.button} disabled={busy} onClick={onRetry}>Retry saved action</button> : null}
    </form>
  </InteractionDialog>;
}
