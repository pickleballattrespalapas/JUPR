"use client";

import { useEffect, useState } from "react";
import PlayerSearchInput from "@/components/PlayerSearchInput";
import { InteractionDialog } from "@/components/interaction";
import { fetchAllPublicPlayers } from "@/lib/playerSearch";
import type { HubGroup } from "@/lib/adminRoundRobinHubApi";
import styles from "./RoundRobinHubPanel.module.css";

type DirectoryPlayer = { id: number | string; name: string };
type Props = {
  apiBase: string; clubId: string; group: HubGroup; busy: boolean; disabled: boolean; pending: boolean; error: string;
  onClose: () => void; onRetry: () => void;
  onSubmit: (name: string, playerId: number | null) => void;
};

export default function HubAddPlayerDialog({ apiBase, clubId, group, busy, disabled, pending, error, onClose, onRetry, onSubmit }: Props) {
  const [name, setName] = useState("");
  const [playerId, setPlayerId] = useState<number | null>(null);
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

  const blocked = disabled || !group.can_add_player;
  const options = directory.filter(player => Number.isSafeInteger(Number(player.id)) && Number(player.id) > 0).map(player => ({
    value: String(player.id), label: player.name,
    disabled: group.players?.some(member => (member.profile_id === Number(player.id) || member.identity === `player:${Number(player.id)}`) && !member.can_rejoin) || false
  }));

  return <InteractionDialog open phase={busy ? "working" : "ready"} title="Add player" description={<p>{group.title}</p>}
    dirty={Boolean(name.trim())} onRequestClose={() => { if (!busy) onClose(); }}
    actions={<><button type="button" className={styles.button} disabled={busy} onClick={onClose}>Cancel</button><button type="submit" form="hub-add-player-form" className={styles.primary} disabled={blocked || !name.trim()}>Add player to round robin</button></>}>
    <form id="hub-add-player-form" onSubmit={event => { event.preventDefault(); if (!blocked && name.trim()) onSubmit(name.trim(), playerId); }}>
      <label className={styles.field}>Player name or club profile
        <PlayerSearchInput aria-label="Player name or club profile" data-autofocus maxLength={120} value={name} disabled={blocked}
          options={options} loading={directoryLoading} error={directoryError}
          onTextChange={value => { setName(value); setPlayerId(null); }}
          onPick={option => { setName(option.label); setPlayerId(Number(option.value)); }} />
      </label>
      <p className={styles.small}>{playerId ? "Selected club profile. This player shares their identity and pause status across the event." : "Choose a club profile from the suggestions, or enter a guest name."}</p>
      <p className={styles.small}>Players join available matches. Games already assigned keep their players.</p>
      {!group.can_add_player ? <p role="alert">{group.add_player_unavailable_reason || "This round robin cannot add players here. Open its management controls to review the roster."}</p> : null}
      {directoryError ? <p className={styles.small}>{directoryError}</p> : null}
      {error ? <p role="alert">{error}</p> : null}
      {pending ? <button type="button" className={styles.button} disabled={busy} onClick={onRetry}>Retry saved action</button> : null}
    </form>
  </InteractionDialog>;
}
