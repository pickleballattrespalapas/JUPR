"use client";

import { useState } from "react";
import SearchablePlayerSelect from "@/components/SearchablePlayerSelect";
import type { CompetitionDocument, CompetitionPlayer } from "@/lib/interclubCompetition";
import { SubstitutionReview, gameLocation, replacementOptions } from "@/lib/interclubSubstitutions";
import styles from "./competition.module.css";

export default function SubstitutionRepair({ document, change, eligibilityProblem, options, players, disabled, onRepair, onShow }: {
  document: CompetitionDocument; change: SubstitutionReview; eligibilityProblem: string; options: CompetitionPlayer[];
  players: Map<string, CompetitionPlayer>; disabled: boolean; onRepair: (change: SubstitutionReview) => void; onShow: () => void;
}) {
  const [replacement, setReplacement] = useState("");
  const choices = eligibilityProblem ? replacementOptions(document, change.row, change.side, change.outgoing, options, players) : [];
  return <div className={styles.substitutionIssue}>
    <p><strong>{players.get(change.incoming)?.name || "Substitute"} replaces {players.get(change.outgoing)?.name || "injured player"}</strong><br />{gameLocation(document, change.row)}</p>
    {eligibilityProblem ? <>
      <p>{eligibilityProblem}</p>
      <label>Eligible replacement<SearchablePlayerSelect aria-label="Eligible replacement" value={replacement} disabled={disabled} onValueChange={setReplacement}>
        <option value="">Choose a replacement</option>{choices.map(player => <option key={player.entry_id} value={player.entry_id}>{player.name}</option>)}
      </SearchablePlayerSelect></label>
    </> : <p>{change.returningGames.length ? `The injured player is still listed in ${change.returningGames.length} later games. Carry the replacement forward to fix them together.` : "Confirm this as an injury substitution; no written explanation is needed."}</p>}
    <div className={styles.toolbar}>
      <button className={styles.primary} disabled={disabled || !!eligibilityProblem && !choices.some(player => player.entry_id === replacement)} onClick={() => onRepair(eligibilityProblem ? { ...change, incoming: replacement, previousIncoming: change.incoming } : change)}>{eligibilityProblem ? "Apply replacement to remaining games" : change.returningGames.length ? "Carry substitute forward" : "Confirm injury substitution"}</button>
      <button onClick={onShow}>Show this substitution</button>
    </div>
  </div>;
}
