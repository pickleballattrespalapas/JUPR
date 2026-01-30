# Badge Unlock Requirements (Player-Facing Truth)

This file is the **explicit unlock criteria** shown to players.
Lore/hints remain “documentary voice”; this file is the **hard truth**.

General rules:
- Only matches with valid scores count (score total > 0).
- Awards are computed from match facts and league context.
- If a badge is marked inactive in `badge_catalog.py` it should not be shown as unlockable.
- Unlock logic is implemented in `evaluators.py`; any legacy references to `badge_rules.py` are deprecated.

---

## Participation & Habit Loop

### participant — Participant
Unlock: Play 1 recorded match (any league).
Source: participation thresholds in `gamification/participation.py`.

### dedicated_participant_50 — Dedicated Participant
Unlock: Play 50 recorded matches (lifetime).
Source: participation thresholds in `gamification/participation.py`.

### lifetime_participant_200 — Lifetime Participant
Unlock: Play 200 recorded matches (lifetime).
Source: participation thresholds in `gamification/participation.py`.

### first_win — First Win
Unlock: Record your first win (any league).
Source: `evaluate_first_win` in `evaluators.py`.

### weekly_regular — Weekly Regular
Unlock: In the same league, appear in 4+ consecutive ISO weeks with ≥1 match per week.
Source: `evaluate_weekly_regular` in `evaluators.py`.

### iron_week — Iron Week
Unlock: In the same league and ISO week, play 5+ recorded matches.
Source: `evaluate_iron_week` in `evaluators.py`.

### marathon_month — Marathon Month
Unlock: In the same league and month, play 40+ recorded matches.
Source: `evaluate_marathon_month` in `evaluators.py`.

---

## Skill Growth & Momentum

### level_up — Level Up
Unlock: Rating milestone (1400/1600/1800/2000) in league standings.
Source: `evaluate_level_up` in `evaluators.py`.

### rocket_start — Rocket Start
Unlock: Win 4+ of your first 5 matches in the same league.
Source: `evaluate_rocket_start` in `evaluators.py`.

### most_improved_monthly — Most Improved
Unlock: Top monthly Elo delta per league (positive total only).
Source: `evaluate_most_improved_monthly` in `evaluators.py`.

### mountain_climber — Mountain Climber
Unlock: Improve rank vs starting rank by 5/10/20 places.
Source: `evaluate_mountain_climber` in `evaluators.py`.

### hot_streak — Hot Streak
Unlock: Win streak hits 5, 10, or 20 in the same league (stackable per tier).
Source: `evaluate_hot_streak` in `evaluators.py`.

### bounce_back — Bounce Back
Unlock: Win immediately after a loss.
Source: `evaluate_bounce_back` in `evaluators.py`.

### breakthrough — Breakthrough
Unlock: Not currently implemented (see evaluator placeholder).
TODO: Extract exact condition from implementation.

### above_expectations — Above Expectations
Unlock: Not currently implemented (see evaluator placeholder).
TODO: Extract exact condition from implementation.

---

## Clutch & Pressure

### ice_in_veins — Ice in Veins
Unlock: (defined in `badge_rules.py` via `_award_ice_in_veins`).  
TODO: Extract exact condition from `_award_ice_in_veins`.

### clutch_performer — Clutch Performer
Unlock: (defined in `badge_rules.py` via `_award_clutch_performer` or similar).  
TODO: Extract exact condition from implementation.

---

## Dominance & Quality

### pickle_perfection — Pickle Perfection
Unlock: (defined in `badge_rules.py` via `_award_pickle_perfection`).  
TODO: Extract exact condition from `_award_pickle_perfection`.

### blowout_artist — Blowout Artist
Unlock: (defined in `badge_rules.py` via `_award_blowout_artist`).  
TODO: Extract exact condition from `_award_blowout_artist`.

### untouchable — Untouchable
Unlock: (defined in `badge_rules.py` via `_award_untouchable`).  
TODO: Extract exact condition from `_award_untouchable`.

### clean_sweep_week — Clean Sweep Week
Unlock: (defined in `badge_rules.py` via `_award_clean_sweep_week`).  
TODO: Extract exact condition from `_award_clean_sweep_week`.

### high_roller — High Roller
Unlock: (defined in `badge_rules.py` via `_award_high_roller`).  
TODO: Extract exact condition from `_award_high_roller`.

### dominant_run — Dominant Run
Unlock: (defined in `badge_rules.py` via `_award_dominant_run` or similar).  
TODO: Extract exact condition from implementation.

### high_output — High Output
Unlock: (defined in `badge_rules.py` via `_award_high_output` or similar).  
TODO: Extract exact condition from implementation.

---

## Versatility & Social Graph

### social_butterfly — Social Butterfly
Unlock: (defined in `badge_rules.py` via `_award_social_graph`).  
TODO: Extract exact condition from `_award_social_graph`.

### network_builder — Network Builder
Unlock: (defined in `badge_rules.py` via `_award_social_graph`).  
TODO: Extract exact condition from `_award_social_graph`.

### draft_master — Draft Master
Unlock: (defined in `badge_rules.py` via `_award_draft_master`).  
TODO: Extract exact condition from `_award_draft_master`.

### swiss_army_knife — Swiss Army Knife
Unlock: (defined in `badge_rules.py` via `_award_swiss_army_knife`).  
TODO: Extract exact condition from `_award_swiss_army_knife`.

---

## Prestige / Rarity

### giant_slayer — Giant Slayer
Unlock: (defined in `badge_rules.py` via `_award_giant_slayer`).  
TODO: Extract exact condition from `_award_giant_slayer`.

### david_vs_goliath — David vs Goliath
Unlock: (defined in `badge_rules.py` via `_award_david_vs_goliath`).  
TODO: Extract exact condition from `_award_david_vs_goliath`.

### upset_champion — Upset Champion
Unlock: (defined in `badge_rules.py` via `_award_upset_champion`).  
TODO: Extract exact condition from `_award_upset_champion`.

### legendary_upset — Legendary Upset
Unlock: (defined in `badge_rules.py` via `_award_legendary_upset`).  
TODO: Extract exact condition from `_award_legendary_upset`.

---

## Rivalry & Nemesis

### nemesis_found — Nemesis Found
Unlock: (defined in `badge_rules.py` via `_award_rivalries`).  
TODO: Extract exact condition from `_award_rivalries`.

### rivalry_win — Rivalry Win
Unlock: (defined in `badge_rules.py` via `_award_rivalries`).  
TODO: Extract exact condition from `_award_rivalries`.

### rivalry_streak — Rivalry Streak
Unlock: (defined in `badge_rules.py` via `_award_rivalries`).  
TODO: Extract exact condition from `_award_rivalries`.

### settled_the_score — Settled the Score
Unlock: (defined in `badge_rules.py` via `_award_rivalries`).  
TODO: Extract exact condition from `_award_rivalries`.

---

## Consistency & Reliability

### battle_tested — Battle Tested
Unlock: (defined in `badge_rules.py` via `_award_battle_tested` or similar).  
TODO: Extract exact condition from implementation.

### consistency — Consistency
Unlock: (defined in `badge_rules.py` via `_award_consistency` or similar).  
TODO: Extract exact condition from implementation.

### steady_hand — Steady Hand
Unlock: (defined in `badge_rules.py` via `_award_steady_hand`).  
TODO: Extract exact condition from `_award_steady_hand`.

---

## Tournament Podium

### tournament_champion — Tournament Champion
Unlock: Awarded by tournament podium results (badge engine evaluator).
TODO: Define the tournament-results source of truth + awarding job.

### tournament_runner_up — Tournament Runner-Up
TODO: Same as above.

### tournament_third_place — Tournament Third Place
TODO: Same as above.

---

## Top Performer Awards (League)

### top_performer_highest_rating — Top Performer: Highest Rating
TODO: Define “highest rating” snapshot logic at league end, and awarding trigger.

### top_performer_most_improved — Top Performer: Most Improved
TODO: Define rating delta window + league end logic.

### top_performer_best_win_pct — Top Performer: Best Win %
TODO: Define min-games threshold + league end logic.

### top_performer_most_wins — Top Performer: Most Wins
TODO: Define league end logic + ties.

---

## Special

### hall_of_fame_night — Hall of Fame Night
Unlock: (defined in `badge_rules.py` via `_award_hall_of_fame_night`).  
TODO: Extract exact condition from `_award_hall_of_fame_night`.

---

## Inactive Badges (Not Unlockable)

These are defined in `badge_catalog.py` but currently `is_active=False` and should not be shown as unlockable:
- mr_reliable
- league_champion
- league_runner_up
- league_third_place
- podium
- good_sport
- community_builder
- mentor
Source: `badge_catalog.py` entries show `is_active=False`. :contentReference[oaicite:12]{index=12}
