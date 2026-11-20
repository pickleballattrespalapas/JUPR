# Two-group growth audit: October 27, 2026

Tested production commit `b17c0003d01e591c31dfc65249764af0729a9858` with two
synthetic doubles round robins starting at 4 and 6 players and growing to 25 each.
The current release **fails the requirement to avoid unnecessary consecutive
byes**. No production sessions or results were changed.

| Scenario | Shared courts | Longest bye streak, 4 → 25 | Longest bye streak, 6 → 25 | Scored games |
| --- | ---: | ---: | ---: | ---: |
| Rapid arrivals before the first games finish | 10 | 17 | 16 | 207 |
| Gradual arrivals while games finish | 10 | 15 | 15 | 201 |
| Rapid arrivals with one requested initial round | 10 | 16 | 15 | 186 |
| Gradual arrivals, fewer courts, slower second group | 4 | 17 | 15 | 214 |
| Gradual arrivals plus pause/return, slower second group | 10 | 11 | 5 | 169 |

These are bye streaks in **completed rounds**, not counts of waiting court
assignments. All 50 players eventually started a game in each simulation.
Simulated game durations are seeded at 7–13 minutes, multiplied by 1.8 for the
slower group where specified. Recorded minute values are synthetic evidence,
not predictions of real event duration.

All runs preserved group membership, saved scores, and started lineups. They
also verified unique court commitments and assignment of the highest-priority
eligible game. Across the five runs: 2,377 Hub commands and 977 scored games.
Every command serializes and reloads the resulting state.

## Cause and minimal reproduction

`mutate_hub_roster` retains unaffected games in every prepared round. When a
new player cannot fit into those games, the player can inherit byes across
many future rounds. `_prepare_rolling_rounds` eventually supplies a game much
later in the schedule. Its teammates' earlier obligations then delay that game.
The standalone generator's new bye priority does not rebalance those retained
Hub plans.

Start two RRs at 4 and 6, attach both to a Hub, fill their courts, and add players
until both reach 9 before entering a result. The ninth player appears on byes
in both rounds 1 and 2, even though no round-2 game has started. This also occurs
through the Hub service's commit path, using its transactional RPC double.

The repair needs to replan eligible future games after roster changes, using
the generator's bye and matchup priorities, while preserving current announced
games, court assignments, reservations, and recorded results. It must not merely
erase the displayed byes or move the waiting player into a distant future round.

## Reproduction

From the repository root, with the existing Python test dependencies:

```bash
PYTHONPATH=. python -m pytest -q tests/adversarial_hub/test_growth_bye_fairness.py
PYTHONPATH=. python tests/adversarial_hub/growth_audit.py --scenario burst --output /tmp/hub-growth-burst.json
```

Other scenario names: `gradual`, `single_preview`, `limited_courts`, `pause_return`.
The CLI exits nonzero when it observes repeated byes. The three new regression
tests intentionally fail on the tested release. All 90 existing tests in
`test_generator_perfect_rotations.py`, `test_generator_bye_fairness.py`, and
`test_generator_attendance_fairness.py` passed; they did not cover this growing
Hub schedule interaction.

Machine-readable results, including each synthetic player's bye count and first
game wait, are in `evidence/growth-20261027.json`. This audit exercises the real
Hub and generator domain code and one service commit path; it is not a live
database, browser, authentication, or production load test.
