"use client";

import { useEffect, useMemo, useState } from "react";
import StartingRatingReview from "@/components/StartingRatingReview";

type Props = {
  apiBase: string;
  clubId: string;
  accessToken: string;
  playerIds: number[];
  playFormat: string;
  requestedRatings?: Record<string, string>;
  disabled?: boolean;
  onReady: (ready: boolean) => void;
  onConfirmed?: () => void;
};

/** Keep mixed sessions gated until both independent rating histories are checked. */
export default function GeneratorRatingReview({ apiBase, clubId, accessToken, playerIds, playFormat, requestedRatings, disabled, onReady, onConfirmed }: Props) {
  const formats = useMemo<Array<"singles" | "doubles">>(() => playFormat === "doubles_singles" ? ["doubles", "singles"] : [playFormat === "singles" ? "singles" : "doubles"], [playFormat]);
  const context = JSON.stringify([apiBase, clubId, accessToken, [...new Set(playerIds)].sort((a, b) => a - b), formats, requestedRatings || {}]);
  const [checks, setChecks] = useState<Record<string, { context: string; ready: boolean }>>({});
  const ready = playerIds.length > 0 && formats.every(format => checks[format]?.context === context && checks[format]?.ready);
  useEffect(() => { onReady(ready); }, [ready, onReady]);
  return <div style={{ display: "grid", gap: "0.75rem" }}>{formats.map(format => <StartingRatingReview
    key={format + context} apiBase={apiBase} clubId={clubId} accessToken={accessToken}
    playerIds={playerIds} matchFormat={format} requestedRatings={requestedRatings} disabled={disabled}
    title={`${format === "doubles" ? "Doubles" : "Singles"} starting ratings`}
    onReady={value => setChecks(current => current[format]?.context === context && current[format]?.ready === value ? current : { ...current, [format]: { context, ready: value } })}
    onConfirmed={() => onConfirmed?.()}
  />)}</div>;
}
