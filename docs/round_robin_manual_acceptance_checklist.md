# Round-Robin Generator manual acceptance checklist

Use the exact staging candidate and deployment named in the acceptance handoff. Begin manual testing only after that handoff confirms the focused domain/API/component checks, full Next.js production build, exact Vercel identity, and exact Fly identity.

## PRR-12 — Scored flow

- [ ] Create a scored Round Robin.
- [ ] Confirm ranking can be set to Total wins, Total points, or Point differential.
- [ ] Start the session and enter all current-round scores.
- [ ] Save scores and confirm the individual Round Results are shown.
- [ ] Select **View standings and continue**.
- [ ] Confirm the full cumulative Standings are shown before any next-round action.
- [ ] Confirm Standings include rank, games played, wins, losses, points for,
      points against, and differential.
- [ ] Continue from Standings and confirm the next round opens.
- [ ] Confirm every scored round page retains a Standings link.

## PRR-13 — Unscored flow

- [ ] Create an unscored Round Robin.
- [ ] Confirm ranking controls disappear during setup.
- [ ] Confirm schedule preview, CSV, and PDF remain available.
- [ ] Start the session and confirm no score fields, results table, or standings
      are shown.
- [ ] Select **Round Played** once and confirm the next round opens directly.
- [ ] Refresh and use browser Back; confirm the played round and current round
      remain correct.
- [ ] Confirm **Skip round** remains distinct from **Round Played**.
- [ ] Confirm adaptive add, remove, reorder, and substitute actions still work.
- [ ] In staff mode, confirm official publishing is unavailable.
- [ ] Open a view-only public link and confirm organizer controls are absent.

## PRR-14 — Final-round completion

- [ ] Finish the final scored round through Round Results and Full Standings.
- [ ] Select **Finish session** from Standings and confirm a clear Session complete
      state with final standings preserved.
- [ ] Finish the final unscored round with **Round Played** and confirm a clear
      Session complete state without competitive standings.
- [ ] Confirm refresh and Back preserve the completed session.

## PRR-15 — Doubles + Singles Mix

Run this section with at least two valid court combinations, including one setup
that produces byes, before marking it complete.

- [ ] Select **Doubles + Singles Mix** during Round-Robin setup.
- [ ] Select the total number of players, then choose at least one doubles court
      and one singles court.
- [ ] Confirm the selected courts fit the player count and the setup summary shows
      doubles courts, singles courts, players active each round, byes per round,
      and the automatically balanced round count.
- [ ] Preview the schedule and confirm every round has the requested number of
      doubles games and singles games, with no player appearing twice in one round.
- [ ] Confirm CSV and PDF exports label every game as Singles or Doubles.
- [ ] Across the full preview, confirm singles games, doubles games, and byes are
      distributed as evenly as mathematically possible.
- [ ] Confirm repeated doubles partners and repeated singles opponents are avoided
      until the available unique combinations require a repeat.
- [ ] Start a scored mixed session and confirm Singles and Doubles labels remain
      visible during score entry, Round Results, cumulative Standings, and final
      completion.
- [ ] In staff mode, publish the scored results and confirm singles games use the
      singles match path while doubles games use the doubles match path.
- [ ] Start an unscored mixed session and confirm **Round Played** advances directly
      with no score fields, Round Results, or competitive Standings.
- [ ] Add, remove, reorder, and substitute players after a round; confirm completed
      history remains unchanged and future mixed rounds rebalance safely.
- [ ] Confirm public and staff setup behavior is aligned.
- [ ] Open a public view-only session and confirm organizer controls remain hidden.
