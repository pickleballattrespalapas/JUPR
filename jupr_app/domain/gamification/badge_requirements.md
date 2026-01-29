# Badge Unlock Requirements (Player-Facing Truth)

This file is the **explicit unlock criteria** shown to players.
Lore/hints remain “documentary voice”; this file is the **hard truth**.

General rules:
- Only matches with valid scores count (score total > 0).
- Awards are computed from match facts and league context.
- If a badge is marked inactive in `badge_catalog.py` it should not be shown as unlockable.

---

## Participation & Habit Loop

### participant — Participant
Unlock: Play 1 recorded match (any league).
Source: participation thresholds in `badges_participation.py`. :contentReference[oaicite:7]{index=7}

### dedicated_participant_50 — Dedicated Participant
Unlock: Play 50 recorded matches (lifetime).
Source: participation thresholds in `badges_participation.py`. :contentReference[oaicite:8]{index=8}

### lifetime_participant_200 — Lifetime Participant
Unlock: Play 200 recorded matches (lifetime).
Source: participation thresholds in `badges_participation.py`. :contentReference[oaicite:9]{index=9}

### first_win — First Win
Unlock: Record your first win (any league).
Source: `_award_first_win` in `badge_rules.py`. :contentReference[oaicite:10]{index=10}

### weekly_regular — Weekly Regular
Unlock: In the same league, appear in 4+ consecutive ISO weeks with ≥1 match per week.
Source: `_award_weekly_regular` in `badge_rules.py`. :contentReference[oaicite:11]{index=11}

### iron_week — Iron Week
Unlock: (defined in `badge_rules.py` via `_award_iron_week`).  
TODO: Extract exact threshold from `_award_iron_week` implementation.

### marathon_month — Marathon Month
Unlock: (defined in `badge_rules.py` via `_award_marathon_month`).  
TODO: Extract exact threshold from `_award_marathon_month` implementation.

---

## Skill Growth & Momentum

### level_up — Level Up
Unlock: (defined in `badge_rules.py` via `_award_level_up`).  
TODO: Extract exact condition from `_award_level_up` implementation.

### rocket_start — Rocket Start
Unlock: (defined in `badge_rules.py` via `_award_rocket_start`).  
TODO: Extract exact condition from `_award_rocket_start` implementation.

### most_improved_monthly — Most Improved
Unlock: (defined in `badge_rules.py` via `_award_most_improved`).  
TODO: Extract exact condition from `_award_most_improved` implementation.

### mountain_climber — Mountain Climber
Unlock: (defined in `badge_rules.py` via `_award_mountain_climber`).  
TODO: Extract exact condition from `_award_mountain_climber` implementation.

### hot_streak — Hot Streak
Unlock: (defined in `badge_rules.py` via `_award_hot_streaks`).  
TODO: Extract exact tier thresholds from `_award_hot_streaks` implementation.

### bounce_back — Bounce Back
Unlock: (defined in `badge_rules.py` via `_award_bounce_back`).  
TODO: Extract exact condition from `_award_bounce_back` implementation.

### breakthrough — Breakthrough
Unlock: (defined in `badge_rules.py` via `_award_breakthrough` or similar).  
TODO: Extract exact condition from implementation.

### above_expectations — Above Expectations
Unlock: (defined in `badge_rules.py` via `_award_above_expectations` or similar).  
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
Unlock: Awarded by tournament results ingestion (not currently in `badge_rules.py`).
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
