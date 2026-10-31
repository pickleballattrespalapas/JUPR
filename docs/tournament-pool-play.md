# Tournament pool play: operator guide

Pools belong to one playing draw. Registration stays in the existing division, and the pools feed one shared playoff bracket and one division podium.

## Set up the draw

1. Select the tournament, then open **Live Operations → Draw setup & recovery**.
2. Create or select the draw for the registration division. Import or enter its teams once, review their player names and seeds, and **Save teams** before assigning pools.
3. Find **Split into pools** and click **Split into two pools**. Use **Number of pools** if you need more than two.
4. Review the **Team assignments** table. Each team has a **Pool** selector. Change these selectors to put each team in the intended pool. Edit **Pool name** as needed.
5. Set **Teams advancing** separately for each pool. This establishes the recommended qualifiers, subject to your final playoff review.
6. Check the displayed team counts, matches per team, total pool matches and total playoff teams. Resolve any validation messages.
7. Click **Save pools** and confirm. Then use **Round-robin schedule → Generate games**.

**Distribute by seed** uses the saved team seed order in a snake pattern. It replaces the draft assignments, so review them afterward. **Reset to saved pools** discards unsaved pool edits. Changing the number of pools also redistributes teams.

The draw supports 4–16 teams, split into 2–8 pools with at least two teams in each. Every team must appear exactly once. A pool may advance zero through all its teams, with 2–16 recommended playoff teams in total.

## Examples and guaranteed pool matches

| Division | Teams | Pool setup | Advance per pool | Pool matches per team | Total pool matches |
|---|---:|---|---:|---:|---:|
| Mixed 3.5 | 8 | Two pools of four | 2 | 3 | 12 |
| Mixed 3.0 | 6 | Two pools of three | 2 | 2 | 6 |

Pool play is a **single round-robin**: each team plays every other team in its own pool once. Three-team pools have one rotating bye per team over three rounds. Double round-robin is not part of this control.

With two qualifiers from each of two pools, the recommended semifinals are **Pool A first vs Pool B second**, and **Pool B first vs Pool A second**. Semifinal winners play for gold; losers play for bronze. This adds four matches per division, giving totals of **16 matches for the eight-team example** and **10 for the six-team example**.

Every team receives the scheduled pool matches; playoff matches are additional for qualifiers. A three-team pool therefore guarantees only two scheduled matches per team. Forfeits or withdrawals can reduce actual play. These are match counts, regardless of whether an individual match uses one game or best of three.

## Run pool play and review standings

Open **Live Operations → Day workspace**, choose the correct **Tournament day**, and complete the usual readiness/check-in and activation steps. Run matches through **Eligible queue**, **Court board** and **Enter score**. Round-robin match labels identify their pool.

In **Draws & progression**, expand **Pool standings & advancement** on the draw card. Each pool shows its own wins, losses, points for, points against, differential and projected qualifiers. Qualifiers remain provisional until all pool matches are finalized.

Standings use wins, tied-team head-to-head, point differential, points scored, then original team number. Review the displayed tie-break explanations before confirming advancement. Uneven pools are allowed; teams are ranked within their pool, rather than comparing raw wins across pools of different sizes.

## Confirm or override the playoffs

1. Once every pool match is finalized and the draw is ready, click **Review playoff setup**.
2. Review each pool's standings and recommended qualifiers.
3. Check **Playoff structure** and each **Seed** selector. You can change the total playoff size, replace an advancing team with another eligible team, or reorder seeds.
4. If you change the recommended size, teams or seed order, complete **Reason for playoff size, qualifier or seed override**. For example: “Pool A runner-up withdrew; the next team advances.”
5. Review **Scoring by playoff round** and the **Bracket preview**, including byes and cross-pool matchups. **Reset to recommended pool qualifiers** restores the available recommended structure and selections.
6. Click **Generate reviewed playoffs** when the complete bracket is correct. Continue scoring and use the existing **Podium & awards** and publication workflow for the division.

A pool with too few eligible qualifiers leaves missing places for explicit review; it does not silently borrow a team from another pool. The operator may choose a smaller available playoff or fill places with eligible teams, recording the reason. A two-team playoff is a final only. In a three-team playoff, the semifinal loser receives bronze without a separate bronze match.

## Edit boundaries

- Save or reset unsaved pool changes before generating games.
- Pool assignments and saved advancement counts lock once games exist. Final playoff review still permits the documented overrides.
- Before games exist, use **Remove pool split → Save pools** before replacing or importing the team list. Save the revised teams, recreate the split, and review it again.
- Do not move teams between pools after play starts. Use the normal score correction, non-played outcome and substitution workflows as appropriate.
- Finalize corrections before generating playoffs. A changed authoritative result or draw state invalidates an open playoff review; reopen it and inspect the refreshed information.
