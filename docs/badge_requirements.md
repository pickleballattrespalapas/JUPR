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
   - Some catalog badges are tracked or seasonal and are not awarded by the live badge job.
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
Unlock: In a league standings table (`df_leagues`), have JUPR rating ≥ one of these milestones:
- 1400, 1600, 1800, 2000
Earned once per (league, milestone).

## rocket_start — Rocket Start (non-stackable)
Unlock: In the SAME league, win 4+ of your first 5 matches in that league.
- Only awards if you have at least 5 matches recorded in that league.

## most_improved_monthly — Most Improved (stackable)
Unlock: For each (league, month), the player with the highest total positive JUPR rating change that month.
- Rating change is summed across that month’s matches.
- If the top monthly rating change sum is ≤ 0, nobody earns it for that (league, month).

## mountain_climber — Mountain Climber (stackable)
Unlock: In a league standings table (`df_leagues`), improve rank vs starting rank by at least:
- 5 places, 10 places, or 20 places
Earned once per (league, tier).
- Rank is based on sorting by JUPR rating descending.
- Starting rank is based on sorting by starting JUPR rating descending.
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
Unlock: Requirements TBD.

## above_expectations — Above Expectations (stackable)
Unlock: Requirements TBD.

---

# Clutch & Pressure

## ice_in_veins — Ice in Veins (non-stackable)
Unlock: Your first “clutch upset” win:
- You WIN
- Final margin is 2 points or fewer (|margin| ≤ 2)
- Your pre-match win chance is ≤ 0.40 (expected win %)

## clutch_performer — Clutch Performer (non-stackable)
Unlock: Requirements TBD.

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
Unlock: Requirements TBD.

## high_output — High Output (stackable)
Unlock: Requirements TBD.

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
Unlock: Win a match where your pre-match win chance is ≤ 0.25.
Earned once per match.

## upset_champion — Upset Champion (stackable)
Unlock: For each (league, month), the winning match with the LOWEST pre-match win chance.
- Both winners (doubles team) earn it for that match.
Earned once per (league, month, match).

## legendary_upset — Legendary Upset (stackable)
Unlock: Win a match where your pre-match win chance is ≤ 0.15.
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
Unlock: Requirements TBD.

## consistency — Consistency (stackable)
Unlock: Requirements TBD.

## steady_hand — Steady Hand (non-stackable in catalog; awarded once per season in rules)
Unlock: In the SAME season:
- play 20+ matches, AND
- win rate ≥ 60%
Earned once per season where you meet the requirement.

## mr_reliable — Mr. Reliable (inactive)
Unlock: Requirements TBD.

---

# Meta / Prestige

## league_champion — League Champion (inactive)
Unlock: Awarded on league close to the final 1st-place finisher.

## league_runner_up — League Runner-Up (inactive)
Unlock: Awarded on league close to the final 2nd-place finisher.

## league_third_place — League Third Place (inactive)
Unlock: Awarded on league close to the final 3rd-place finisher.

## podium — Podium (inactive)
Unlock: Awarded on league close to any top-3 finisher.

## hall_of_fame_night — Hall of Fame Night (stackable)
Unlock: In a given league, win a match whose rating swing (absolute rating change) is in the TOP 5% for that league.
- Computed per league using the 95th percentile of absolute rating change.
Earned once per match that qualifies.

---

# Tournament Podium

These are awarded from tournament podium results (requires tournament tables).

## tournament_champion — Tournament Champion (non-stackable)
Unlock: Your team finishes 1st place in a tournament podium table.

## tournament_runner_up — Tournament Runner-Up (non-stackable)
Unlock: Your team finishes 2nd place in a tournament podium table.

## tournament_third_place — Tournament Third Place (non-stackable)
Unlock: Your team finishes 3rd place in a tournament podium table.

---

# Top Performer Awards

These are awarded on league close from final standings.

## top_performer_highest_rating — Top Performer: Highest Rating (stackable)
Unlock: Awarded on league close to the highest final JUPR rating.

## top_performer_most_improved — Top Performer: Most Improved (stackable)
Unlock: Awarded on league close to the largest net JUPR rating gain.

## top_performer_best_win_pct — Top Performer: Best Win % (stackable)
Unlock: Awarded on league close to the best final win percentage.

## top_performer_most_wins — Top Performer: Most Wins (stackable)
Unlock: Awarded on league close to the most total wins.

---

# Sportsmanship & Community (inactive placeholders)

## good_sport — Good Sport (inactive)
Unlock: Awarded manually for outstanding sportsmanship.

## community_builder — Community Builder (inactive)
Unlock: Awarded manually for meaningful community impact.

## mentor — Mentor (inactive)
Unlock: Awarded manually for mentorship contributions.
