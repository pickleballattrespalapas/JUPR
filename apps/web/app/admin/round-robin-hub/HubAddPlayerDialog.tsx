"use client";

import { useEffect, useState } from "react";
import PlayerSearchInput from "@/components/PlayerSearchInput";
import { InteractionDialog } from "@/components/interaction";
import { fetchAllPublicPlayers } from "@/lib/playerSearch";
import type { HubGroup } from "@/lib/adminRoundRobinHubApi";
import styles from "./RoundRobinHubPanel.module.css";

type DirectoryPlayer = { id: number | string; name: string };
type Props = {
  apiBase: string; clubId: string; groups: HubGroup[]; busy: boolean; disabled: boolean; pending: boolean; error: string;
  onClose: () => void; onRetry: () => void;
  onSubmit: (sessionKey: string, name: string, playerId: number | null) => void;
};

function destinationUnavailableReason(group: HubGroup, playerId: number | null): string {
  if (!group.can_add_player) return group.add_player_unavailable_reason || "This round robin cannot add players here.";
  if (playerId && group.players?.some(member => (member.profile_id === playerId || member.identity === `player:${playerId}`) && !member.can_rejoin)) {
    return "This player is already in this round robin.";
  }
  return "";
}

export default function HubAddPlayerDialog({ apiBase, clubId, groups, busy, disabled, pending, error, onClose, onRetry, onSubmit }: Props) {
  const [name, setName] = useState("");
  const [playerId, setPlayerId] = useState<number | null>(null);
  const [sessionKey, setSessionKey] = useState("");
  const [directory, setDirectory] = useState<DirectoryPlayer[]>([]);
  const [directoryLoading, setDirectoryLoading] = useState(true);
  const [directoryError, setDirectoryError] = useState("");
  const publicClubSlug = clubId.replace(/_/g, "-");

  useEffect(() => {
    const controller = new AbortController();
    void fetchAllPublicPlayers<DirectoryPlayer>(
      `${apiBase.replace(/\/$/, "")}/clubs/${encodeURIComponent(publicClubSlug)}/players?status=all&sort=name`, controller.signal
    ).then(players => {
      if (!controller.signal.aborted) setDirectory(players);
    }).catch(() => {
      if (!controller.signal.aborted) setDirectoryError("Club player search is unavailable. You can still add a guest by name.");
    }).finally(() => {
      if (!controller.signal.aborted) setDirectoryLoading(false);
    });
    return () => controller.abort();
  }, [apiBase, publicClubSlug]);

  const frozen = disabled || busy || pending;
  const selectedGroup = groups.find(group => group.session_key === sessionKey);
  const destinationError = sessionKey ? selectedGroup ? destinationUnavailableReason(selectedGroup, playerId) : "This round robin is no longer in the event. Choose another round robin." : "";
  const blocked = frozen || !selectedGroup || Boolean(destinationError);
  const options = directory.filter(player => Number.isSafeInteger(Number(player.id)) && Number(player.id) > 0).map(player => ({
    value: String(player.id), label: player.name
  }));
  const destinations = [...groups].sort((left, right) => left.title.localeCompare(right.title, undefined, { numeric: true, sensitivity: "base" }));

  return <InteractionDialog open phase={busy ? "working" : "ready"} title="Add player" description={<p>Enter a player, then choose their round robin.</p>}
    dirty={Boolean(name.trim() || sessionKey)} onRequestClose={() => { if (!busy && !pending) onClose(); }}
    actions={<><button type="button" className={styles.button} disabled={busy || pending} onClick={onClose}>Cancel</button><button type="submit" form="hub-add-player-form" className={styles.primary} disabled={blocked || !name.trim()}>Add player to round robin</button></>}>
    <form id="hub-add-player-form" onSubmit={event => { event.preventDefault(); if (!blocked && name.trim()) onSubmit(sessionKey, name.trim(), playerId); }}>
      <label className={styles.field}>Player name or club profile
        <PlayerSearchInput aria-label="Player name or club profile" data-autofocus maxLength={120} value={name} disabled={frozen}
          options={options} loading={directoryLoading} error={directoryError}
          onTextChange={value => { setName(value); setPlayerId(null); }}
          onPick={option => { setName(option.label); setPlayerId(Number(option.value)); }} />
      </label>
      <p className={styles.small}>{playerId ? "Selected club profile. This player shares their identity and pause status across the event." : "Choose a club profile from the suggestions, or enter a guest name."}</p>
      <label className={styles.field}>Round robin
        <select aria-label="Round robin" required value={sessionKey} disabled={frozen || !name.trim()} aria-invalid={Boolean(destinationError)} aria-describedby={destinationError ? "hub-add-player-destination-error" : undefined} onChange={event => setSessionKey(event.target.value)}>
          <option value="">Choose a round robin</option>
          {sessionKey && !selectedGroup ? <option value={sessionKey} disabled>Selected round robin is no longer available</option> : null}
          {destinations.map(group => {
            const reason = destinationUnavailableReason(group, playerId);
            return <option key={group.session_key} value={group.session_key} disabled={Boolean(reason)}>{group.title}{reason ? ` — ${reason}` : ""}</option>;
          })}
        </select>
      </label>
      {destinationError ? <p id="hub-add-player-destination-error" role="alert">{destinationError}</p> : null}
      {!groups.length ? <p className={styles.small}>Add a round robin to this event before adding a player.</p> : null}
      <p className={styles.small}>Players join available matches. Games already assigned keep their players.</p>
      {directoryError ? <p className={styles.small}>{directoryError}</p> : null}
      {error ? <p role="alert">{error}</p> : null}
      {pending ? <button type="button" className={styles.button} disabled={busy} onClick={onRetry}>Retry saved action</button> : null}
    </form>
  </InteractionDialog>;
}
