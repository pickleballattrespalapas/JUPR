# JUPR Badge Requirements (Explicit Unlock Rules)

This file defines the exact, player-facing requirements to unlock each badge.

## Global rules (apply to ALL badge logic unless a badge says otherwise)

1) Only recorded matches with valid scores count.
   - A match must have score_t1 + score_t2 > 0.
   - Matches are filtered by the app (invalid/voided/popups excluded).

2) “League” means the match’s `league` field (defaults to "OVERALL" if empty).

3) “Week” is ISO week (YYYY-W##). “Month” is YYYY-MM. “Season” is derived from match timestamps.

4) If a badge is stackable:
   - You can earn it multiple times (as determined by its context: match/week/month/league/season/opponent).

5) Important reality check:
   - The Streamlit app awards badges through the badge engine registry/evaluators via
     `jupr_app/domain/gamification/ensure_badges.py`.
   - Some catalog badges exist but are NOT currently awarded automatically (they are placeholders or require inputs we don’t store yet).
   - Some catalog badges are marked inactive (`is_active = false`) and are intentionally not unlockable.

---

# Participation & Habit Loop

## participant — Participant (non-stackable)
Unlock: Play 1 recorded match (lifetime).

## dedicated_participant_50 — Dedicated Participant (non-stackable)
Unlock: Play 50 recorded matches (lifetime).

## lifetime_participant_200 — Lifetime Participant (non-stackable)
Unlock: Play 200 recorded matches (lifetime).

## first_win — First Win (non-stackable)
Unlock: Record your first win (lifetime).
- Any win in any league.

## weekly_regular — Weekly Regular (non-stackable)
Unlock: In the SAME league, appear in 4+ consecutive ISO weeks with ≥1 match per week.
- You don’t need 4 weeks in a row on the calendar, you need 4 consecutive ISO week keys.
- Missing a week breaks the streak.

## iron_week — Iron Week (stackable)
Unlock: In the SAME league + SAME ISO week, play 5+ recorded matches.
- Earned once per (league, week).

## marathon_month — Marathon Month (stackable)
Unlock: In the SAME league + SAME month, play 40+ recorded matches.
- Earned once per (league, month).

---

# Skill Growth & Momentum

## level_up — Level Up (stackable)
Unlock: In a league standings table (`df_leagues`), have rating ≥ one of these milestones:
- 1400, 1600, 1800, 2000
Earned once per (league, milestone).

## rocket_start — Rocket Start (non-stackable)
Unlock: In the SAME league, win 4+ of your first 5 matches in that league.
- Only awards if you have at least 5 matches recorded in that league.

## most_improved_monthly — Most Improved (stackable)
Unlock: For each (league, month), the player with the highest total positive Elo change that month.
- Elo delta is summed across that month’s matches.
- If the top monthly Elo sum is ≤ 0, nobody earns it for that (league, month).

## mountain_climber — Mountain Climber (stackable)
Unlock: In a league standings table (`df_leagues`), improve rank vs starting rank by at least:
- 5 places, 10 places, or 20 places
Earned once per (league, tier).
- Rank is based on sorting by rating descending.
- Starting rank is based on sorting by starting_rating descending.
- “Rank delta” = start_rank − current_rank (positive is improvement).

## hot_streak — Hot Streak (stackable)
Unlock: In the SAME league, hit a win streak of exactly:
- 5 wins in a row
- 10 wins in a row
- 20 wins in a row
You earn the badge at the exact match where the streak reaches that tier.
Earned once per (league, tier, streak-ending match).

## bounce_back — Bounce Back (stackable)
Unlock: Win a match immediately after losing your previous match (lifetime timeline order).
Earned once per “bounce-back” win match.

## breakthrough — Breakthrough (non-stackable)
Status: NOT CURRENTLY AWARDED by the automated badge job.
(Defined in the catalog, but there is no evaluator wired into the badge engine registry.)
Recommendation: keep it active only after you implement awarding logic.

## above_expectations — Above Expectations (stackable)
Status: NOT CURRENTLY AWARDED by the automated badge job.
(Defined in the catalog, but not awarded by the badge engine.)

---

# Clutch & Pressure

## ice_in_veins — Ice in Veins (non-stackable)
Unlock: Your first “clutch upset” win:
- You WIN
- Final margin is 2 points or fewer (|margin| ≤ 2)
- Your expected win probability is ≤ 0.40

## clutch_performer — Clutch Performer (non-stackable)
Status: NOT CURRENTLY AWARDED by the automated badge job.
(Defined in the catalog, but not awarded by the badge engine.)

---

# Dominance & Quality

## pickle_perfection — Pickle Perfection (stackable)
Unlock: Win a match where the opponent scores 0 points.
- points_against == 0
Earned once per match.

## blowout_artist — Blowout Artist (stackable)
Unlock: Win a match by 8+ points.
- margin ≥ 8
Earned once per match.

## untouchable — Untouchable (stackable)
Unlock: Get to an 8-win streak (lifetime timeline order).
- Every win after streak ≥ 8 will award another instance (it is stackable in code).
Earned with context_id including the match that ended the streak window.

## clean_sweep_week — Clean Sweep Week (stackable)
Unlock: In the SAME league + SAME ISO week:
- play at least 3 matches, AND
- win ALL of them
Earned once per (league, week).

## high_roller — High Roller (stackable)
Unlock: Win a match where:
- points_for ≥ 15 AND
- margin ≥ 6
Earned once per match.

## dominant_run — Dominant Run (stackable)
Status: NOT CURRENTLY AWARDED by the automated badge job.

## high_output — High Output (stackable)
Status: NOT CURRENTLY AWARDED by the automated badge job.

---

# Versatility & Social Graph

## social_butterfly — Social Butterfly (non-stackable)
Unlock: Have 20+ unique partners across your recorded matches (lifetime).
- Only matches where partner_id exists count.

## network_builder — Network Builder (non-stackable)
Unlock: Have 50+ unique partners across your recorded matches (lifetime).
- Only matches where partner_id exists count.

## draft_master — Draft Master (stackable)
Unlock: In the SAME month:
- WIN matches with 5+ unique partners that month
Earned once per month where you meet the requirement.

## swiss_army_knife — Swiss Army Knife (non-stackable)
Unlock: In the SAME season:
- WIN matches in 3+ distinct leagues
Earned once per season where you meet the requirement.

---

# Prestige / Rarity

## giant_slayer — Giant Slayer (stackable)
Unlock: Win a match where the highest-rated opponent in the match is at least:
- 1800, 2000, or 2200
Earned once per match per tier that you satisfy.

## david_vs_goliath — David vs Goliath (stackable)
Unlock: Win a match where your expected win probability is ≤ 0.25.
Earned once per match.

## upset_champion — Upset Champion (stackable)
Unlock: For each (league, month), the winning match with the LOWEST expected win probability.
- Both winners (doubles team) earn it for that match.
Earned once per (league, month, match).

## legendary_upset — Legendary Upset (stackable)
Unlock: Win a match where your expected win probability is ≤ 0.15.
Earned once per match.

---

# Rivalry & Nemesis

## nemesis_found — Nemesis Found (non-stackable)
Unlock: Against a specific opponent:
- you have played 6+ games vs them, AND
- your win rate vs them is ≤ 40%
Earned once per opponent who qualifies.

## rivalry_win — Rivalry Win (stackable)
Unlock: If an opponent is already your “nemesis” (as defined above),
then any WIN against that opponent earns Rivalry Win.
Earned once per match win vs a nemesis.

## rivalry_streak — Rivalry Streak (stackable)
Unlock: Against the SAME opponent:
- win 3 matches in a row vs them
Earned once per opponent per streak hit.

## settled_the_score — Settled the Score (stackable)
Unlock: Against the SAME opponent, when you bring your record to exactly even
after being down.
- In practice: a win that makes your wins == your losses vs that opponent,
  after previously having more losses than wins.
Earned once per opponent per “evening” event.

---

# Consistency & Reliability

## battle_tested — Battle Tested (stackable)
Status: NOT CURRENTLY AWARDED by the automated badge job.

## consistency — Consistency (stackable)
Status: NOT CURRENTLY AWARDED by the automated badge job.

## steady_hand — Steady Hand (non-stackable in catalog; awarded once per season in rules)
Unlock: In the SAME season:
- play 20+ matches, AND
- win rate ≥ 60%
Earned once per season where you meet the requirement.

## mr_reliable — Mr. Reliable (inactive)
Status: INACTIVE (not unlockable).

---

# Meta / Prestige

## league_champion — League Champion (inactive)
Status: INACTIVE (not unlockable).

## league_runner_up — League Runner-Up (inactive)
Status: INACTIVE (not unlockable).

## league_third_place — League Third Place (inactive)
Status: INACTIVE (not unlockable).

## podium — Podium (inactive)
Status: INACTIVE (not unlockable).

## hall_of_fame_night — Hall of Fame Night (stackable)
Unlock: In a given league, win a match whose abs Elo delta is in the TOP 5% for that league.
- Computed per league using the 95th percentile of abs_elo_delta.
Earned once per match that qualifies.

---

# Tournament Podium

These are NOT awarded by the normal match-based badge job.
They are awarded from tournament podium results (requires tournament tables).

## tournament_champion — Tournament Champion (non-stackable)
Unlock: Your team finishes 1st place in a tournament podium table.

## tournament_runner_up — Tournament Runner-Up (non-stackable)
Unlock: Your team finishes 2nd place in a tournament podium table.

## tournament_third_place — Tournament Third Place (non-stackable)
Unlock: Your team finishes 3rd place in a tournament podium table.

---

# Top Performer Awards

These are defined and active in the catalog, but NOT currently awarded by the automated badge job.

## top_performer_highest_rating — Top Performer: Highest Rating (stackable)
Status: NOT CURRENTLY AWARDED.

## top_performer_most_improved — Top Performer: Most Improved (stackable)
Status: NOT CURRENTLY AWARDED.

## top_performer_best_win_pct — Top Performer: Best Win % (stackable)
Status: NOT CURRENTLY AWARDED.

## top_performer_most_wins — Top Performer: Most Wins (stackable)
Status: NOT CURRENTLY AWARDED.

---

# Sportsmanship & Community (inactive placeholders)

## good_sport — Good Sport (inactive)
Status: INACTIVE (not unlockable).

## community_builder — Community Builder (inactive)
Status: INACTIVE (not unlockable).

## mentor — Mentor (inactive)
Status: INACTIVE (not unlockable).
