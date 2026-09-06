-- Presentation only. Ordered after the existing tournament catalog seed.
-- Preserve availability, prestige, award history, icons, and earning triggers.
begin;
set local lock_timeout = '5s';
set local statement_timeout = '30s';

update public.badges as badge
set category = copy.category, lore = copy.requirement, hint = copy.requirement
from (values
    ('participant', 'Participation', 'Play 1 recorded match (lifetime).'),
    ('dedicated_participant_50', 'Participation', 'Play 50 recorded matches (lifetime).'),
    ('lifetime_participant_200', 'Participation', 'Play 200 recorded matches (lifetime).'),
    ('first_win', 'Participation', 'Record your first win (lifetime). Any league counts.'),
    ('weekly_regular', 'Participation', 'In the same league, play ≥1 match in 4 consecutive ISO weeks (Mon–Sun). Missing a week breaks the streak.'),
    ('iron_week', 'Participation', 'Play in 3+ different leagues in a single ISO week (Mon–Sun).'),
    ('marathon_month', 'Participation', 'In the same league, play 40+ matches in a single calendar month. Earned once per league per month.'),
    ('level_up', 'Improvement', 'Reach a league JUPR rating of 3.0, 3.5, 4.0, 4.5, or 5.0. Earn each milestone once; a later rating drop does not remove it.'),
    ('rocket_start', 'Improvement', 'In the same league, win 4+ of your first 5 matches (requires 5 recorded matches in that league).'),
    ('most_improved_monthly', 'Improvement', 'Monthly (per league): Finish the month with the largest net JUPR rating gain across matches in that league. If nobody finishes net‑positive, no award.'),
    ('mountain_climber', 'Improvement', 'In a league, improve your standing by 5 / 10 / 20 places from your starting rank. Standings are ordered by JUPR rating.'),
    ('hot_streak', 'Match Achievements', 'In the same league, reach a win streak of 5 / 10 / 20 consecutive wins (milestones).'),
    ('bounce_back', 'Improvement', 'Win your next recorded match after a loss (based on match history order).'),
    ('breakthrough', 'Improvement', 'In a league, play 10+ matches and move from outside the top 25 / top 10 (starting rank) into the current top 25 / top 10 (by JUPR rating). Earn each tier separately.'),
    ('above_expectations', 'Match Achievements', 'Win with expected win chance ≤ 40% and a 4+ point margin. If rating deltas are available, the match must be at or above the 75th percentile of absolute rating delta within that league.'),
    ('ice_in_veins', 'Match Achievements', 'Your first clutch upset: win by 2 points or fewer when your pre‑match win chance is 40% or less.'),
    ('clutch_performer', 'Match Achievements', 'Record 5 clutch wins (wins by 2 points or fewer).'),
    ('pickle_perfection', 'Match Achievements', 'Win a match without conceding a point (opponent scores 0).'),
    ('blowout_artist', 'Match Achievements', 'Win a match by 8+ points.'),
    ('untouchable', 'Match Achievements', 'Reach a 20+ match win streak (lifetime). Earn it again for each additional consecutive win beyond 20.'),
    ('clean_sweep_week', 'Match Achievements', 'In one ISO week (Mon–Sun), play at least one match in 2+ different leagues and win them all.'),
    ('high_roller', 'Match Achievements', 'Record 100+ lifetime match wins.'),
    ('dominant_run', 'Match Achievements', 'Across all leagues, reach a 10+ match win streak with average win margin ≥ 5 across the current streak. Earn it again for each additional win while the streak and average margin stay at or above the threshold.'),
    ('high_output', 'Match Achievements', 'Across all leagues, in your last 25 matches, record 20+ wins with average margin ≥ 4.'),
    ('social_butterfly', 'Partnerships', 'Play with 20+ different partners (lifetime). Doubles only; a recorded partner is required.'),
    ('network_builder', 'Partnerships', 'Play with 50+ different partners (lifetime). Doubles only; a recorded partner is required.'),
    ('draft_master', 'Partnerships', 'In a single ISO week (Mon–Sun), record wins with 5+ different partners.'),
    ('swiss_army_knife', 'Participation', 'In one calendar year, record wins in 3+ different leagues.'),
    ('giant_slayer', 'Match Achievements', 'Win a match where the highest‑rated opponent is above 5.0.'),
    ('david_vs_goliath', 'Match Achievements', 'Win a match when your pre‑match win chance is 25% or less.'),
    ('upset_champion', 'Match Achievements', 'Monthly (per league): the win with the lowest pre‑match win chance. All winners on the team receive it.'),
    ('legendary_upset', 'Match Achievements', 'Win a match when your pre‑match win chance is 15% or less.'),
    ('nemesis_found', 'Match Achievements', 'Against an opponent you’ve faced 6+ times, your head‑to‑head win rate is 40% or less.'),
    ('rivalry_win', 'Match Achievements', 'Beat your nemesis (as defined in Nemesis Found).'),
    ('rivalry_streak', 'Match Achievements', 'Against your nemesis, win 3 matches in a row.'),
    ('settled_the_score', 'Match Achievements', 'Against your nemesis, win a match that makes your head‑to‑head record exactly even.'),
    ('battle_tested', 'Participation', 'In the same season, complete 50+ matches (no forfeits / invalid matches).'),
    ('consistency', 'Participation', 'In the same season, play ≥1 match in 6 consecutive ISO weeks (missing a week breaks the streak).'),
    ('steady_hand', 'Match Achievements', 'In the same season, play 20+ matches and maintain a win rate of 60%+.'),
    ('mr_reliable', 'Match Achievements', 'In the same season, play 30+ matches and finish with a 70%+ win rate.'),
    ('league_champion', 'Trophies', 'Awarded on league close to the final 1st-place finisher.'),
    ('league_runner_up', 'Trophies', 'Awarded on league close to the final 2nd-place finisher.'),
    ('league_third_place', 'Trophies', 'Awarded on league close to the final 3rd-place finisher.'),
    ('tournament_champion', 'Trophies', 'Finish 1st on a tournament podium.'),
    ('tournament_runner_up', 'Trophies', 'Finish 2nd on a tournament podium.'),
    ('tournament_third_place', 'Trophies', 'Finish 3rd on a tournament podium.'),
    ('top_performer_highest_rating', 'Trophies', 'Seasonal (league close): finish with the highest JUPR rating in the league.'),
    ('top_performer_most_improved', 'Trophies', 'Seasonal (league close): largest JUPR rating gain from season start to season end.'),
    ('top_performer_best_win_pct', 'Trophies', 'Seasonal (league close): finish with the best win percentage in the league.'),
    ('top_performer_most_wins', 'Trophies', 'Seasonal (league close): finish with the most wins in the league.'),
    ('podium', 'Trophies', 'Awarded on league close to any top-3 finisher.'),
    ('hall_of_fame_night', 'Match Achievements', 'In a league, win a match with a top‑5% rating swing for that league (by absolute JUPR rating change).'),
    ('good_sport', 'Partnerships', 'Awarded manually for outstanding sportsmanship.'),
    ('community_builder', 'Partnerships', 'Awarded manually for meaningful community impact.'),
    ('mentor', 'Partnerships', 'Awarded manually for mentorship contributions.')
) as copy(badge_id, category, requirement)
where badge.badge_id = copy.badge_id;

-- Older deployments may retain mirrored presentation columns on this table.
do $compatibility$
declare presentation_field text;
begin
  foreach presentation_field in array array['category', 'lore', 'hint'] loop
    if exists (select 1 from information_schema.columns c
               where c.table_schema = 'public' and c.table_name = 'badges'
               and c.column_name = presentation_field || '_v2') then
      execute format('update public.badges set %I = %I', presentation_field || '_v2', presentation_field);
    end if;
  end loop;
end
$compatibility$;
commit;
