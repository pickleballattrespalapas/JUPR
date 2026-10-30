# Starting estimates and CourtReserve rosters

CourtReserve's self-reported skill level is a requested estimate. It never
replaces the calculated PCS JUPR of a player with rated match history.

## Operator workflow

1. In the league roster, daily League Live setup, or generator setup, choose
   **Import CourtReserve roster**.
2. Upload an exported CSV/TSV report, or copy and paste its cells from Numbers
   or Excel with the **Full Name** and **Enter Your Skill Level** headings.
   Native Numbers/XLSX files and screenshots are not imported directly.
3. Check the matched PCS profiles. Resolve missing or ambiguous names and
   skip unwanted rows explicitly. The importer never creates a player or
   silently matches a similar name. Create genuinely new profiles using the
   existing player-creation control before importing them.
4. Review starting estimates only for players without rated history.
   An imported value different from the stored estimate is highlighted for
   these players. A range such as `3.75–4.0` requires a numeric decision.
5. Confirm the estimates, then use the reviewed names for the roster or day.
   League roster import adds missing memberships; it does not remove existing
   members. Daily setup selects the imported attendees.

Players with rated history keep their PCS rating even when their requested
level is rounded to a half point. Unrated games do not establish a rating.
Singles and doubles histories are evaluated separately. This release can
change doubles starting estimates; singles confirmation preserves the
existing immutable singles replay baseline.

## Confirmation and integrity

Confirmation records who reviewed an estimate and when, plus the requested
value considered. A previously confirmed, unchanged estimate does not need
another confirmation. A new, different request for an unplayed player reopens
review. Changing a seed updates its unused doubles league seeds together;
played history cannot be reset through this workflow.

The backend checks scored, non-deleted rated matches as well as historical
aggregate and league counters. Zero counters cannot hide existing match
evidence. A database fingerprint protects confirmation from changes since
preview. Seed changes, confirmations, and their audit records commit together.
Public clients cannot call the confirmation RPC. Staff authentication and
player-management permission are required at the API boundary.

Rated setup and publication enforce the review on the server. Unrated play
does not require review. Existing session score entry and completed-operation
recovery stay available; newly introduced rated participants need review.

## Release and verification

Apply `20261030183226_starting_rating_review.sql` to the staging project before
deploying the API and web changes. Production migration/deployment is a
separate authorized release under `AGENTS.md`.

The component suites `courtreserve-roster-import.cjs` and
`starting-rating-review.cjs` cover import parsing, identity resolution,
established-rating protection, explicit range decisions, stale responses,
and uncertain confirmation recovery. Service, route, and SQL regression
tests cover authorization, history evidence, atomic seed changes, and
concurrency conflicts.
