# Southern BCS interclub league: rules and operating specification

Decision record consolidated September 19, 2026 from Joe's guided questionnaire
and the Southern BCS Interclub League Development Roadmap (revision 0.1), with
Joe's September 21 clarification that players may play up.

**Status: approved product direction; the current working branch implements the
competition, eligibility, rating, printing and publication workflows described
below.** Release and end-to-end staging verification remain separate gates. This
document does not certify a deployment or production readiness.

This is the current rules specification. The historical implementation notes in
[pcs_interclub_implementation.md](pcs_interclub_implementation.md) explain earlier
increments and must not override the decisions below. An explicit later user
decision takes precedence over the original roadmap or an operational default.

## 1. Scope and terminology

- A **season** belongs to an organizing club and includes participating clubs,
  skill levels, regular-season meets, and championship finals.
- A **meet** is one gathering at one host venue. Different meets may run at
  different venues on the same date; one club cannot be scheduled at overlapping
  meets.
- A **skill level** means a rating category, such as 3.5 or 4.0. It does not mean
  men's, women's, or mixed doubles.
- A **club matchup** is the contest between two clubs at one skill level.
- In a regular-season club matchup there are two **pairings**: men's and women's
  doubles at a gender-doubles meet, or Mixed A and Mixed B at a mixed-doubles meet.
- Each regular-season pairing plays **three games**. All three are played even
  when one side wins the first two, except a recorded injury, forfeit, or weather
  disposition.
- A championship club matchup uses four individual doubles games and, if
  necessary, one rotating-singles tiebreak. It does not use the regular-season
  two-pairing, three-games-per-pairing structure.

The original Southern BCS plan targets eight clubs, two simultaneous venues,
January–March play, and a three-hour regular-season meet. The original core
levels were 3.5 and 4.0, with optional 4.5 and no 5.0/Open competition. Actual
clubs, advertised levels, dates, host assignments, and court inventory are
organizer configuration, not hardcoded platform limits or confirmed bookings.
Adding a platform-supported level does not silently add it to this season.

## 2. Clubs, accounts, and data boundaries

- A club accepts the season invitation before registering its players or entering
  a meet. Account access, season acceptance, interest registration, and a meet
  lineup are distinct actions with distinct statuses.
- Only staff have accounts. Players use account-free signup and private response
  links; they are not required to create a PCS account.
- Each club controls its own player directory and rosters. Players and ratings
  remain separate club records; entering the league does not transfer records
  between clubs.
- A player represents **one club for the season**. A player who has competed for
  one club cannot transfer to another club in that season.
- The organizer receives submitted player names, ratings, and eligibility facts,
  not participating clubs' contact lists or entire player directories.
- A staff login may hold assignments in multiple clubs. Every action must remain
  bound to the selected club and the staff member's current authorized scope.

Do not implement the representation rule by merging club directories or creating
a new universal player account. Preserve source club/player identities in the
league's submitted records. Any identity ambiguity requiring human review must
remain visible rather than silently admitting an apparent transfer.

## 3. Season interest pool and meet registration

1. The commissioner sets one season registration opening and closing time for
   all clubs. After accepting, each club shares its own registration link;
   individual clubs cannot open or close season registration. The closing time
   must be no later than the first meet starts.
2. Players join that club's season interest pool at the beginning of the season.
   Joining expresses interest; it does not commit the player to every meet or
   guarantee selection.
3. Ordinary signup and administrator bulk additions for verbal or email
   commitments use the commissioner-set registration window. After it closes,
   a club administrator may **request a late player** by selecting an existing
   active club player and explaining the request. This remains pending until
   the commissioner approves it, even when the season has not started yet.
   Joining during an open window after the season starts also requires late
   approval. Approval takes effect when given; it cannot admit a player into
   an already closed meet cutoff. A club administrator cannot bypass approval
   by directly placing a new player on a meet roster.
4. Meet availability, deadlines, lineups and competition settings are locked
   until season registration closes. After it closes, the club invites pool
   members to report availability before each meet. Joe's
   intended timing is roughly one or two weeks ahead; exact dates and response
   deadlines are configurable.
5. The club administrator selects a meet-specific lineup from the approved pool.
   Availability responses and selected lineups remain distinct.
6. After registration closes, the commissioner can add meets and adjust
   upcoming meet dates from **Meet schedule** in the season workspace. Dates
   use the season timezone and must remain within the season without club or
   host scheduling conflicts. Changing a meet time preserves roster choices
   but clears prior availability answers and closes availability collection;
   clubs reopen it and invite players to reconfirm with fresh links. Changes
   do not send messages automatically. Started or scored meets use the
   weather/replay workflow. Republish to update an existing public schedule.

Club administrators choose their participating skill levels **separately for each
meet**. There is no season-long commitment to field a team at every level. The
full team at each entered level has **two men and two women**. A regular-season
club may instead declare an unavailable pairing and submit the two eligible
players for the pairing it can field, using the missing-pairing rules below.
Finals and qualifying MLP matchups require the full four-player lineup. A player
may play up in a higher division, but cannot enter a numeric division at or
above that division's upper rating limit.

Participation fees are handled **outside PCS** by the clubs. This signup flow
does not require online payment or introduce a payment-processing gate.

Email content and recipients must be reviewable before sending. Staging remains
`dry_run`: show usable previews and test links without sending email or requiring
SMTP. Production email activation is not authorized by a staging build request.

## 4. Rating eligibility and roster deadlines

### Confirmed rating rules

- Seed the player's initial interclub league rating from the club they represent.
- Thereafter **the current interclub league rating** determines eligible divisions;
  the current home-club rating and the season's initial seed do not override it.
- Players may play up. A 2.9 player may enter 3.0, 3.5, 4.0 or any higher offered
  division. Numeric divisions have no lower rating bound; they exclude ratings
  at or above the division label plus 0.5.
- As a league rating changes, the player may become eligible or ineligible for
  lower divisions. Reaching 4.0 makes a player ineligible for 3.5; dropping below
  4.0 permits 3.5 again. The player may still choose a higher eligible division.
- Open divisions, including legacy `Open` and `4.5/Open` labels, accept any valid
  positive rating and have no upper ceiling. A missing, zero, negative or
  non-finite rating is not eligible.
- For a specific meet, freeze each player's eligibility rating at **that meet's
  roster deadline**, using the approved rating history available at the cutoff.
  Later rating changes do not silently invalidate that meet's lineup.
- A rescheduled matchup uses a **new roster deadline** and each player's rating
  at that new deadline.
- The starting league rating remains an auditable seed, not a season-long
  eligibility lock.

Upper limits are recorded in the season rules. Use unambiguous bounds, for
example 3.5 means `0 < rating < 4.0`, rather than relying on a rounded display
string to determine eligibility. The exact exclusive upper bound governs even
when legacy metadata stores `max_rating: 3.999`. Existing preferred divisions
remain the player's choices; broader eligibility does not rewrite them.
Any organizer-approved exception must
reference the exact roster revision, player, rule, and approving administrator.

### Roster changes

Club administrators can make eligible substitutions after the deadline and before
play begins, with eligibility evaluated against that meet's rating cutoff. The
history of submitted lineups and exceptions must be retained. A newly entered
late team uses the explicit organizer exception workflow, not a silent deadline
bypass.

After the meet begins, substitutions are allowed **only for injury, between
games**. The replacement must be in the approved season pool, eligible for the
skill level, and preserve the required team composition. No routine tactical
changes are allowed after play starts. Never rewrite the players attached to
games that have already been played.

For rescheduled unfinished matchups, a club may select a **new eligible lineup**
from its approved pool. Players who completed matchups on the original date stay
attached to those historical results.

## 5. Regular-season scheduling and print package

### Participating clubs at each skill level

Schedule a round robin among the clubs that entered that level at that meet:

| Clubs entered at a level | Club opponents per player | Games per player | Scheduling consequence |
| --- | --- | --- | --- |
| 2 | 1 | 3 | One club matchup, two pairings |
| 3 | 2 | 6 | Three club matchups; one club rests in each rotation |
| 4 | 3 | 9 | Six club matchups across three rotations |

This is the explicit exception to the initial roadmap's universal four-club,
nine-game, no-planned-bench assumption. A club that never entered a level is not
automatically recorded as forfeiting against the clubs that did. A host does not
gain a fictitious team in a level it did not enter.

For four clubs, the standard rotations are:

| Rotation | Club matchup 1 | Club matchup 2 |
| --- | --- | --- |
| 1 | Host vs. Visitor 1 | Visitor 2 vs. Visitor 3 |
| 2 | Host vs. Visitor 2 | Visitor 1 vs. Visitor 3 |
| 3 | Host vs. Visitor 3 | Visitor 1 vs. Visitor 2 |

When the host does not enter a particular level, the same round-robin principle
applies to the actual entrants at that level. Label schedules by actual club
names, not mandatory host participation.

Each pairing plays three games to **11, side-out scoring, win by two, no cap**.
A normal completed pairing is won by taking at least two of its three games.
At gender meets each player stays in the men's or women's pairing; at mixed
meets the two submitted mixed pairings are used. Preserve the original plan's
blind submission of opposing mixed lineups until both are locked.

Court and time planning must use actual entries. Two pairings require two courts
to play concurrently for each club matchup. Four clubs and two skill levels
require eight courts for all scheduled pairings to run concurrently; adding a
third level requires twelve. Insufficient courts must produce a clear scheduling
warning or an organizer-reviewed staggered schedule, never overlapping court or
player assignments. The three-hour goal is a planning target, not a score cap.

### Paper-first operating workflow

The meet must be operable **entirely with pencil and paper**. Live online scoring
is not a requirement to run the competition.

Generate a printable package containing:

- The meet name, date, host, format, skill levels, and lineup/schedule revision.
- A court map and rotation schedule with actual club and player names.
- Eligible lineups and printable matchup sheets with space for all three game
  scores per pairing.
- Injury, replacement, forfeit, delay, cancellation, and replay annotations.
- The identities of players who actually played each game and its actual play
  date, especially when a meet spans original and rescheduled dates.
- Space for both sides to verify scores, plus a consolidated meet results sheet.

Enter or import the complete set of **official meet scores together afterward**.
Draft entry may be saved and corrected, but partial drafts are not official meet
results and must not trigger ratings. The supported workflow must make it clear
which games are complete, forfeited, retired, voided for replay, or unplayed.
Official submission can occur later than the scheduled meet date.

The original plan requires two-sided score verification and audited corrections.
Operational default: use the paper sheet as the two-sided verification record,
allow authorized meet staff to transcribe the whole meet, and have the organizer
review it before rating approval. This default does not require every club to
complete an additional online approval ceremony.

## 6. Regular-season standings and Club Cup

### Normal club-matchup points

| Pairing results for Club A | Club A points | Club B points |
| --- | --- | --- |
| Wins both pairings | 3 | 0 |
| Wins one and loses one | 1 | 1 |
| Loses both pairings | 0 | 3 |

Standings use **accumulated points**, not points per meet or a participation
average. Playing more meets gives more opportunities to earn points.

Within each skill level, rank clubs by:

1. Standings points.
2. Pairings won.
3. Games won.
4. Point differential: points scored minus points conceded.
5. Head-to-head.

Do not insert game differential or a different ordering from the older roadmap.
The public view and printed standings must explain the actual rule used.

### Club Cup

- Award skill-level champions and a separate **overall Club Cup**.
- The Cup sums standings points from **all participation categories**, not only
  3.5 and 4.0. Entering more categories intentionally creates more scoring
  opportunities; do not normalize for categories entered.
- Include regular-season totals plus championship-final bonuses: **6 points to
  each skill-level final winner and 3 to its runner-up**.
- If clubs remain completely tied after championship points and the statistical
  tiebreakers, declare **joint Club Cup champions**. No extra Cup contest.

Keep championship bonuses separate from regular-season standings; a final does
not retroactively change who qualified for that final.

## 7. Injuries and unplayed forfeits

### Injury during a game

- If an injury prevents completion, the injured player's side concedes **only
  the interrupted game**.
- Record the actual score when play stopped. That actual score contributes to
  point differential even if the injured side had been ahead.
- Record the official game winner independently of which side had more points
  at stoppage.
- An eligible injury replacement may play the remaining games in that pairing.
- Completed earlier games remain official with their actual participants.
- **Injury-retired games do not affect individual ratings.**

### Missing pairing

- If a club cannot field one of its two pairings, forfeit **only that pairing's
  three games**. The other pairing still plays.
- Those games count as three wins/losses and the corresponding pairing result.
- Exclude unplayed forfeits from point differential and all player rating
  calculations. Do not synthesize 11–0 scores or invented participants.
- Weather cancellations are not forfeits; use the separate rules below.

Operational default for a predeclared partial regular-season lineup: retain the
eligible available pairing and record the missing pairing explicitly. Do not
require fictional players to create its three forfeits. At gender-doubles meets
the available two players are both women or both men; at mixed meets the
available pairing has one woman and one man.

If both clubs are missing the **same pairing**, the operational default is a
double forfeit: three game losses for each club, no game wins, no pairing win,
no point differential, no player rating effects and no weather-draw point.
The winner of the other played pairing receives three standings points and its
opponent receives zero. If the clubs are missing **opposite pairings**, each
wins the other's forfeited pairing and receives one standings point. These
double-forfeit handling details are implementation defaults, not separately
confirmed questionnaire answers.

## 8. Weather, delays, cancellation, and rescheduling

### Temporary delay

Resume the matchup where it stopped, including the interrupted game's score.
Preserve its lineup and apply the normal injury substitution rule. A temporary
pause does not automatically authorize a new tactical lineup.

### Meet canceled and rescheduled

- Completed three-game pairings remain official.
- An unfinished three-game pairing starts over: **replay all three games** and
  replace its earlier scores. The earlier attempt must be auditable as voided,
  not counted a second time in standings, ratings, or participation.
- A club can select a new approved-pool lineup for the replay.
- Use a new roster deadline and the interclub ratings at that deadline.
- Retain actual play dates and players for both the original completed pairings
  and the later replayed pairings.

### Meet cannot be rescheduled

Use the available official results:

- Decide a pairing from its **completed games**, even if only one completed.
  A 1–0 completed-game lead is a pairing win; a 2–0 lead is also a win.
- One completed game won by each club is a **pairing draw**.
- An interrupted game that never finishes is not a completed game and does not
  contribute game statistics or ratings. Do not convert its partial score into
  an injury retirement or weather forfeit.
- If a pairing has no completed games, record it as **unplayed**, with no invented
  game wins, game scores, point differential, or rating effects.
- An **entirely unplayed canceled club matchup earns 1 standings point per club**;
  it creates no game statistics or ratings.
- A club winning one pairing and drawing the other earns **3 points**; its
  opponent earns **1 point** for losing one and drawing the other.

Conservative operational defaults for the remaining combinations, explicitly
distinct from the user-confirmed examples above:

| Pairing outcomes for Club A | Club A points | Club B points | Game records |
| --- | --- | --- | --- |
| Draw + draw | 1 | 1 | Only completed games |
| Win + unplayed | 3 | 1 | Only completed games |
| Loss + unplayed | 1 | 3 | Only completed games |
| Draw + unplayed | 1 | 1 | Only completed games |
| Unplayed + unplayed | 1 | 1 | No games; explicitly confirmed rule |

For standings allocation only, the unplayed pairing uses the neutral treatment
of a drawn pairing. Preserve the distinction in the result status; do not count
an unplayed pairing as games played or an on-court appearance. Any later change
to these operational defaults must be an explicit rules revision.

## 9. Championship qualification and format

### Qualification

- The **top two clubs at each skill level** advance directly to its final.
  There are no semifinals.
- A club must have played **at least one regular-season meet at that skill
  level**, in addition to finishing in the top two. There is no higher minimum
  attendance threshold.
- Every championship player must have played **at least one regular-season meet
  for that club**. It may have been at any skill level.
- The player must remain eligible for the selected championship division using
  their interclub rating frozen at the championship roster deadline. Players may
  play up here too; numeric divisions retain their exclusive upper rating limit.
- Pool registration or availability alone is not a playing appearance.

Operational default: an appearance requires actual play in an official retained
game, not merely a submitted roster or an unplayed forfeit. A weather-voided
attempt cannot silently satisfy qualification. Track appearances from actual
game participants, including eligible injury replacements.

### Standings ties affecting qualification

After every statistical tiebreaker is exhausted:

- A tie across the qualification boundary, such as **second versus third**, is
  resolved by an additional **full MLP-style matchup**, using the championship
  format below.
- A first/second tie needs no additional contest because both clubs qualify.
- A fourth/fifth tie needs no additional contest because neither qualifies.
- For a tie involving three or more clubs across the boundary, stop automatic
  qualification and require organizer-scheduled full MLP qualifying playoffs.
  Do not silently choose a qualifier, draw an unapproved bye, or apply random
  ordering as a sporting tiebreaker.

Operational defaults for a rare multi-club cutoff tie:

- The organizer schedules a full MLP-style round robin among the tied clubs;
  every pair must complete its matchup before that round can resolve places.
- Rank that playoff round by matchup wins, then doubles games won, then doubles
  point differential. Rotating singles decides its team matchup but does not
  add doubles game statistics or individual rating effects.
- Fill any clearly resolved qualifying places. If a smaller tied group still
  crosses the remaining cutoff, that group plays another full MLP round robin
  (or a single matchup for two remaining clubs). Record the next round rather
  than choosing a name, ID, random order, or unapproved bye.
- These extra contests decide qualification only and award no additional
  regular-season standings or Club Cup points.

These multi-club procedures are implementation defaults, not additional answers
from the guided questionnaire. The organizer controls playoff dates and venues;
the program requires complete official playoff results before assigning the
remaining championship places.

### Final at each skill level

Each club fields **two men and two women** and plays:

1. Women's doubles.
2. Men's doubles.
3. One mixed-doubles game.
4. A second mixed-doubles game.

Each doubles game is **one game to 11, side-out scoring, win by two, no cap**.
These are individual game results, not three-game pairings. Preserve all four
scheduled games; an implementation must not silently stop at 3–0.

If the four doubles games finish 2–2, play a rotating-singles team tiebreak:

- Rally scoring to **21**, win by two, no cap.
- Below 4.0: **skinny singles**. At 4.0 and higher: **full-court singles**.
- Both clubs rotate to their next player **after every four rallies**, cycling
  through all four players in a fixed repeated order.
- The tiebreak decides only the team matchup. It creates **no individual league
  or club rating changes**.

The same full format is used for an additional qualifying matchup. Preserve
distinct championship/qualifier result types so they cannot be mistaken for
regular-season standings pairings or earn bonuses twice.

## 10. Approval, publication, ratings, and corrections

- Only **fully completed official games** affect individual ratings. Exclude
  unplayed forfeits, injury retirements, unfinished weather games, voided replay
  attempts, and rotating-singles tiebreaks.
- Normal eligible completed doubles games contribute separately to the player's
  interclub league rating and the overall rating at the club represented.
- Update ratings only when an organizer administrator approves the **exact
  official meet result revision**. Submission or public publication alone is
  not rating approval.
- League and club rating calculations are separate streams; do not copy league
  deltas blindly into club ratings or update unrelated clubs.
- Preserve chronological actual play order, including local club games occurring
  between interclub games and original/rescheduled meet dates. Upload time is
  not the sporting event time.
- Corrections automatically recalculate affected league and represented-club
  rating histories. Retain original/corrected result and approval audit records.
- Approval, retry, replay, and repeated imports must not double-count a game.
  Show pending/failed/completed status rather than claiming success while one
  rating stream is incomplete.
- Reject stale approvals or corrections that reference superseded revisions.

Host and organizer administrators may assign authorized meet staff within existing
club staff limits. Operators act only within their assigned scope. Organizer
administrators alone change league rules, approve eligibility exceptions and
rating updates, or correct published interclub results. No separate captain
account role is introduced.

Public pages show the whole league's standings, schedule, and official results,
with links from participating clubs. Unpublished score drafts and private player
response links must not appear in public responses or printouts intended for
general distribution.

## 11. Configuration and remaining operational discretion

These are setup/operating choices, not reasons to reopen the settled competition
rules or delay development of their controls:

- Actual participating clubs, dates, venues, courts, skill bands, gender/mixed
  format per regular meet, deadlines, travel notices, and backup dates.
- Local collection of fees, venue instructions, conduct terms, and any
  organizer-supplied acknowledgment text. PCS does not invent legal terms.
- An organizer-reviewed schedule when entry counts or courts differ from the
  standard plan, and resolution of a skill level with fewer than two entrants.
  Do not generate a fictitious opponent, forfeit, final, or bonus.
- Dates and venues for rare multi-club qualifying playoffs, using the complete
  MLP round-robin default above before naming finalists.
- Paper layout details, score transcription/import format, and supporting audit
  evidence. Implement useful defaults without claiming the user selected every
  UI detail.

## 12. Superseded assumptions

| Earlier assumption | Current rule |
| --- | --- |
| Fixed season roster/team | Approved season interest pool plus a different eligible lineup for each meet |
| Players can join any later meet automatically | Beginning-of-season pool; late additions require organizer approval |
| Initial rating fixes division eligibility all season | Current interclub rating, frozen at each meet's roster deadline |
| A player must match a division's lower and upper rating bounds | Players may play up; numeric divisions enforce only an exclusive upper limit, and current ratings govern access to lower divisions |
| Clubs commit to levels for the entire season | Clubs choose entered levels separately for each meet |
| Every level always has four clubs and nine games | Two/three/four entrants give three/six/nine games per player |
| Season standings use wins, game difference, then point difference | Club-matchup points, pairings won, games won, point differential, head-to-head |
| Cup emphasizes only core 3.5/4.0 | Sum all participation categories, plus final winner/runner-up bonuses |
| Final uses ordinary three-game pairing results | Four MLP-style doubles games plus rotating singles when tied 2–2 |
| All completed-looking score rows are rated | Only retained official completed games, after exact-revision organizer approval |
| Live online court/scoring tools required to operate | Complete pencil-and-paper package and one official meet submission afterward |

## 13. Staging acceptance checklist

The following are required tests, not claims that the implementation passes them:

- [ ] One organizer and multiple participant clubs complete acceptance → season
  interest registration → late-addition approval → availability → meet selection.
- [ ] Cross-club directory, roster, result, print, correction, and rating-write
  boundaries reject unauthorized requests; stale club tabs and revoked staff
  cannot continue writing.
- [ ] Clubs independently enter skill levels per meet; two/three/four-club
  fixtures create the right opponents, rotations, courts, and 3/6/9 player games.
- [ ] Play-up choices, ratings below/at/above each upper limit, Open divisions,
  invalid ratings, later movement, original cutoff, and
  new rescheduled cutoff produce correct eligibility without rewriting history.
- [ ] The printed package alone supports a complete meet, including actual game
  participants, score verification, substitutions, non-play, and weather notes.
- [ ] Whole-meet entry/import detects missing/duplicate scores and stale roster
  revisions; draft saves do not publish results or update ratings.
- [ ] Standings fixtures exercise 3/1/0 normal outcomes, 3/1 win-plus-draw,
  two draws, unplayed weather cases, forfeits, and the exact tiebreak order.
- [ ] Injury retirement retains the stopped score for differential, records the
  conceded game, rates no retired game, and preserves completed-game identities.
- [ ] A predeclared two-player regular lineup can play its available pairing;
  same-pairing double forfeits and opposite missing pairings award the documented
  statistics and points without fictional players or numeric scores.
- [ ] Delay resumes; rescheduled unfinished pairings fully replay with new
  lineups; canceled-without-reschedule uses only official completed game results.
- [ ] Entirely unplayed canceled matchups give one point per club and zero game
  statistics, rating effects, or participation credit.
- [ ] Skill-level top-two finals and player appearances are correct; unresolved
  qualifying ties block automatic finalist assignment and can be resolved by
  organizer-scheduled MLP play.
- [ ] Four-game finals, 2–2 rotating-singles resolution, four-rally rotation,
  no-cap finish, 6/3 Cup bonuses, and joint Cup champions calculate correctly.
- [ ] Exact-revision rating approval updates only the league and represented
  clubs; duplicate/retry/correction and interleaved local-game replay are safe.
- [ ] Public pages explain standings and show only official results; paper
  printouts contain no unrelated club contacts or private response tokens.
- [ ] Publication is bound to the reviewed official result revisions, season,
  club names and schedule. Concurrent changes require a new preview before
  publishing; the previously published snapshot stays visible until replacement.
- [ ] End-to-end staging rehearsal includes multiple hosts, weather/replay,
  injury/forfeit, a qualifying tiebreak, finals, ratings, and corrected results.
- [ ] All staging email remains dry-run and production remains untouched.

Before declaring staging ready, bind test and deployment evidence to the exact
staging commit and its successful staging handoff artifact, as required by
`AGENTS.md`.

## 14. Current implementation notes

The working implementation includes meet operations for preparing pairings,
printing court sheets, saving complete score drafts, host submission, organizer
approval, corrections, weather decisions, replay lineups, championship and
qualification meets. Stable game identities and saved revisions keep replayed
and corrected results from being counted twice. Healthy new-date lineups are
validated against their new deadline separately from retained earlier pairings.

Season interest approval and represented-club identity are enforced separately
from the meet lineup. Actual game participants determine appearances; rating
eligibility is snapshotted at the applicable roster deadline. Without a global
player account or universal person identifier, duplicate identity detection uses
normalized signup name and email. Deliberately different identity details across
clubs still require organizer review.

Approved completed doubles games feed separate interclub and represented-club
rating streams. Reconciliation replays intervening club games chronologically
from immutable starting ratings, retains historical rating generations for
deadline eligibility, and persists pending/failed repairs for retry. It does not
insert opposing-club players into a club's local player directory.

The public website uses a separate reviewed publication snapshot. It omits
private pool contacts, response links, injury notes, internal job errors and
unapproved score drafts. Legacy published snapshots remain readable under their
original scoring description and are not inferred into the new scoring model.

The canonical data contract and operational multiway qualification procedure
are documented in [interclub_competition_contract.md](interclub_competition_contract.md).
