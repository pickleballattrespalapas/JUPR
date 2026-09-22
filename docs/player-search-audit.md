# Player search audit — September 22, 2026

Production promotion: the approved release applies the selectors below to every currently deployed surface, including tournament player and partner suggestions. Staging-only interclub, generator-review, notification, and badge-management modules remain on staging. Production keeps its existing navigation and leaderboard filters.

PCS player selectors use suggestions while typing a first- or last-name fragment. `SearchablePlayerSelect` preserves each caller's eligible options, selected IDs, disabled states, and callbacks. It never creates or links a profile based on a partial name. `PlayerSearchInput` supplies keyboard, pointer, touch, duplicate-name, loading, and empty-state behavior. Public directory and leaderboard requests are debounced, abort stale requests, and retain club and view filters.

## Updated selectors

| Area | Surfaces |
| --- | --- |
| Public discovery | Player directory, leaderboard, league player summaries, Match Explorer |
| Registration and updates | Team-league player and partner, verified player updates |
| Match entry and correction | Singles/doubles/round-robin Match Uploader, score entry, Match Log replacement, canonical audit |
| Players | Player editor, social identity linking, duplicate source and keeper |
| Leagues and live play | Team creation, captains, roster additions, substitute pool, lineups, league live arrival/replacement, public/admin generator swaps/removals, live substitutions, team-match rotation |
| Tournament operations | Team player assignments, imported-results mapping, combined-rating registration choice, captain/member/invitation choices, tiebreak player |
| Interclub | Bulk-add ambiguous matches, competition lineup choices |
| Recognition and updates | Badge award/tie winner, badge diagnostics, league award overrides, recap player override, subscription selection, generator submission profile mapping |

Team-league signup and generator review no longer have a separate search box restricting another player field. Match Uploader uses explicit visible suggestions instead of browser-dependent datalists. Public selection widgets and generator setup page through the directory; the directory backend uses ordered, club-scoped database paging. Match Explorer can preview a selection beyond the first directory page.

## Existing live searches retained

- Tournament new/edit registration and partner lookup already show debounced partial-name profile suggestions.
- Round-robin and ladder generator setup already show addable players while typing; their search now handles accents/name fragments and complete roster paging.
- Interclub public signup, regular/late club-player search, player pool, bulk add, meet registration, and recipient lists already refresh their matching results while typing.
- League roster and tournament check-in/editor/registrant/broadcast lists already filter their displayed results on each keystroke. Their action and multi-selection behavior is retained.

New-profile creation, renaming an existing profile, guest-only name entry, and multiline roster imports are data-entry fields, not profile searches. Other entity selectors (team, division, match, club) keep their existing controls. Archived Streamlit upload copies are not the deployed PCS frontend.

## Verification

- Behavioral component checks: partial/accented/reordered names, distinct duplicate IDs, keyboard, explicit selection, required selection, disabled options, multi-select/removal, action-picker reset, form IDs, stale responses, club changes, failure recovery, and roster pagination.
- Existing component suite, API contracts, directory/Match Explorer service and surface checks, Next build/type check, and diff check.
- Staging browser verification and deployment attestation are recorded in the pull request/deployment evidence.
