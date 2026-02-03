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
Unlock: Play **1 recorded match** (lifetime).

## dedicated_participant_50 — Dedicated Participant (non-stackable)
Unlock: Play **50 recorded matches** (lifetime).

## lifetime_participant_200 — Lifetime Participant (non-stackable)
Unlock: Play **200 recorded matches** (lifetime).

## first_win — First Win (non-stackable)
Unlock: Record your **first win** (lifetime). Any league counts.

## weekly_regular — Weekly Regular (non-stackable)
Unlock: In the **same league**, play **≥1 match** in **4 consecutive ISO weeks** (Mon–Sun). Missing a week breaks the streak.

## iron_week — Iron Week (stackable)
Unlock: Play in **3+ different leagues** in a single **ISO week** (Mon–Sun).

## marathon_month — Marathon Month (stackable)
Unlock: In the **same league**, play **40+ matches** in a single **calendar month**. Earned once per league per month.

---

# Skill Growth & Momentum

## level_up — Level Up (stackable)
Unlock: Awarded the first time reaching a **JUPR rating** milestone: **3.0 / 3.5 / 4.0 / 4.5 / 5.0**.

## rocket_start — Rocket Start (non-stackable)
Unlock: In the **same league**, win **4+ of your first 5 matches** (requires 5 recorded matches in that league).

## most_improved_monthly — Most Improved (stackable)
Unlock: **Monthly (per league):** Highest total **positive JUPR rating gain** across matches that month. If nobody finishes net‑positive, no award.

## mountain_climber — Mountain Climber (stackable)
Unlock: In a league, improve your **standing** by **5 / 10 / 20 places** from your starting rank. Standings are ordered by **JUPR rating**.

## hot_streak — Hot Streak (stackable)
Unlock: In the **same league**, reach a win streak of **5 / 10 / 20** consecutive wins (milestones).

## bounce_back — Bounce Back (stackable)
Unlock: Win your **next recorded match** after a loss (based on match history order).

## breakthrough — Breakthrough (non-stackable)
Unlock: **Milestones:** the **first time** you reach a new JUPR rating milestone of **3.25 / 3.75 / 4.25 / 4.75**. Each milestone is earned **once** (lifetime).

## above_expectations — Above Expectations (stackable)
Unlock: Win a match when your **pre‑match win chance is 40% or less** (requires win‑probability; if unavailable, no awards).

---

# Clutch & Pressure

## ice_in_veins — Ice in Veins (non-stackable)
Unlock: Your first **clutch upset**: win by **2 points or fewer** when your **pre‑match win chance** is **40% or less**.

## clutch_performer — Clutch Performer (non-stackable)
Unlock: Record **5 clutch wins** (wins by **2 points or fewer**).

---

# Dominance & Quality

## pickle_perfection — Pickle Perfection (stackable)
Unlock: Win a match **without conceding a point** (opponent scores **0**).

## blowout_artist — Blowout Artist (stackable)
Unlock: Win a match by **8+ points**.

## untouchable — Untouchable (stackable)
Unlock: Reach a **20+ match** win streak (lifetime). Earn it again for each additional consecutive win beyond 20.

## clean_sweep_week — Clean Sweep Week (stackable)
Unlock: In one **ISO week** (Mon–Sun), play **at least one match in 2+ different leagues** and **win them all**.

## high_roller — High Roller (stackable)
Unlock: Record **100+ lifetime match wins**.

## dominant_run — Dominant Run (stackable)
Unlock: In the **same league**, reach win‑streak milestones of **5 / 10 / 20** consecutive wins. Earn each milestone **once per league**.

## high_output — High Output (stackable)
Unlock: In an **ISO week** (Mon–Sun), win a match scoring **11+** while the opponent scores **7 or fewer**. Earn **once per week**.

---

# Versatility & Social Graph

## social_butterfly — Social Butterfly (non-stackable)
Unlock: Play with **20+ different partners** (lifetime). Doubles only; a recorded partner is required.

## network_builder — Network Builder (non-stackable)
Unlock: Play with **50+ different partners** (lifetime). Doubles only; a recorded partner is required.

## draft_master — Draft Master (stackable)
Unlock: In a single **ISO week** (Mon–Sun), record wins with **5+ different partners**.

## swiss_army_knife — Swiss Army Knife (non-stackable)
Unlock: In the same season, record wins in **3+ different leagues**.

---

# Prestige / Rarity

## giant_slayer — Giant Slayer (stackable)
Unlock: Win a match where the highest‑rated opponent **is >5.0+**.

## david_vs_goliath — David vs Goliath (stackable)
Unlock: Win a match when your **pre‑match win chance** is **25% or less**.

## upset_champion — Upset Champion (stackable)
Unlock: **Monthly (per league):** the win with the **lowest pre‑match win chance**. All winners on the team receive it.

## legendary_upset — Legendary Upset (stackable)
Unlock: Win a match when your **pre‑match win chance** is **15% or less**.

---

# Rivalry & Nemesis

## nemesis_found — Nemesis Found (non-stackable)
Unlock: Against an opponent you’ve faced **6+ times**, your head‑to‑head win rate is **40% or less**.

## rivalry_win — Rivalry Win (stackable)
Unlock: Beat your **nemesis** (as defined in *Nemesis Found*).

## rivalry_streak — Rivalry Streak (stackable)
Unlock: Against your **nemesis**, win **3 matches in a row**.

## settled_the_score — Settled the Score (stackable)
Unlock: Against your **nemesis**, win a match that makes your head‑to‑head record **exactly even**.

---

# Consistency & Reliability

## battle_tested — Battle Tested (stackable)
Unlock: In the same season, complete **50+ matches** (no forfeits / invalid matches).

## consistency — Consistency (stackable)
Unlock: In the same season, play **≥1 match** in **6 consecutive ISO weeks** (missing a week breaks the streak).

## steady_hand — Steady Hand (non-stackable in catalog; awarded once per season in rules)
Unlock: In the same season, play **20+ matches** and maintain a win rate of **60%+**.

## mr_reliable — Mr. Reliable (inactive)
Unlock: In the same season, play **30+ matches** and finish with a **70%+ win rate**.

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
Unlock: In a league, win a match with a **top‑5% rating swing** for that league (by **absolute JUPR rating change**).

---

# Tournament Podium

These are awarded from tournament podium results (requires tournament tables).

## tournament_champion — Tournament Champion (non-stackable)
Unlock: Finish **1st** on a tournament podium.

## tournament_runner_up — Tournament Runner-Up (non-stackable)
Unlock: Finish **2nd** on a tournament podium.

## tournament_third_place — Tournament Third Place (non-stackable)
Unlock: Finish **3rd** on a tournament podium.

---

# Top Performer Awards

These are awarded on league close from final standings.

## top_performer_highest_rating — Top Performer: Highest Rating (stackable)
Unlock: **Seasonal (league close):** finish with the highest **JUPR rating** in the league.

## top_performer_most_improved — Top Performer: Most Improved (stackable)
Unlock: **Seasonal (league close):** largest **JUPR rating gain** from season start to season end.

## top_performer_best_win_pct — Top Performer: Best Win % (stackable)
Unlock: **Seasonal (league close):** finish with the best **win percentage** in the league.

## top_performer_most_wins — Top Performer: Most Wins (stackable)
Unlock: **Seasonal (league close):** finish with the most **wins** in the league.

---

# Sportsmanship & Community (inactive placeholders)

## good_sport — Good Sport (inactive)
Unlock: Awarded manually for outstanding sportsmanship.

## community_builder — Community Builder (inactive)
Unlock: Awarded manually for meaningful community impact.

## mentor — Mentor (inactive)
Unlock: Awarded manually for mentorship contributions.
