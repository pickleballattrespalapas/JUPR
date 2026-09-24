# Badge reactivation staging release

Implements the settled questionnaire decisions in docs/badge_reactivation_decisions.md.

- Restore 12 automatic badge types and three repeatable admin community awards.
- Retire League Champion, League Runner-Up, League Third Place, and generic Podium from the public catalog. Preserve historical trophies and the existing league awards/tournament podium flows.
- Add club-admin season settings for Battle Tested, Consistency, Steady Hand, and Mr. Reliable. No dates are invented.
- Display direct earning requirements and consistent scopes/repeatability.

## Verification before staging merge

- 314 focused Python/API, badge-history, deployment-contract, and staging-write checks passed.
- Next production build and frontend component suite passed.
- Community form recovery test passed: response loss, refresh, exact retry, and a separate contribution.
- Real staging database transactional tests passed, including authorization, retry idempotency, repeated contributions, season overlap, revision conflicts, date locking, and stale award evidence. All rehearsal data rolled back.
- Migration applied only to staging project sijpxjxvdtrehmqvirfi.
- Before/after existing awards: 1,121 rows; complete-row digest unchanged at cd5f7c8c5e361bcf1c7614581aea004c.
- Staging badge definitions increased from 37 to 52 because 15 restored definitions were absent. The existing staging fixture is included in those 52.
- No seasons, community awards, or historical backfill were created by the migration.
- New functions are service-role-only. New tables have RLS enabled with no client grants or policies intentionally; the advisor's informational no-policy notices reflect this server-only access model.

The canonical staging deployment must still report ready_for_manual_testing for the exact merged SHA before handoff.

## Historical estimate — no awards applied

This estimate uses the earlier production after-repair export, not a fresh production read. Run scripts.preview_badge_reactivation against a new export before any separately authorized production activation. Existing award identities, including revoked identities, are excluded from additions.

Source snapshot SHA-256: 2a4953c2dbe48d7929e2f5f0816489af9bd1ada2bbecb9dbfb3b7e5ccf48a643

| Automatic badge | Potential new awards |
| --- | ---: |
| above_expectations | 134 |
| battle_tested | 0 |
| breakthrough | 14 |
| clutch_performer | 40 |
| consistency | 0 |
| dominant_run | 9 |
| high_output | 11 |
| mr_reliable | 0 |
| nemesis_found | 78 |
| rivalry_streak | 26 |
| rivalry_win | 324 |
| settled_the_score | 35 |

Total: 671 potential awards across 138 players. No community awards are inferred. The three restored seasonal badges have zero candidates because this export contains no admin-defined seasons. Actual season choices will change that result.

The estimate is not a reviewed repair plan. Automatic workers evaluate history and may discover these unissued achievements when processing results. A production activation review must account for that behavior, preserve staff revocations, and use fresh data.

## Staging acceptance

Open /admin/badges while signed in as a club admin. Award a community badge for one selected qualifying action, contribution date, and note. A separate contribution may earn another award. Add the club's chosen season name and dates; saving starts a badge check. Dates become fixed after a season award is issued, but the name remains editable.

Open /clubs/tres-palapas/badge-codex to see earning requirements and configured seasons. The four retired duplicates are absent. Historical profile trophies remain visible.
