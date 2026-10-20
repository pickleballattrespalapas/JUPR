-- READ ONLY: preview missing/stale game activity for one explicitly reviewed club.
-- Replace REVIEWED_CLUB_ID before use. This file contains no data mutation.
-- Run against staging first. Production access/repair requires separate approval.
-- No ratings, rated counters, singles baselines, identities or signups are changed.
-- Review manual/merged exclusions and future-dated evidence before a separately
-- authorized, transaction-safe repair. Re-read every proposed row at repair time.
with parameters as (
  select 'REVIEWED_CLUB_ID'::text as club_id, now() as as_of
), recorded_games as (
  select participant.player_id, max(m.date) as recorded_last_game_at,
         count(*) filter (where m.rating_scope = 'unrated') as unrated_games,
         count(*) as scored_games
    from public.matches m
    join parameters p on p.club_id = m.club_id
    cross join lateral (
      select distinct player_id
        from unnest(array[m.t1_p1,m.t1_p2,m.t2_p1,m.t2_p2]) player_id
       where player_id is not null
    ) participant
   where m.deleted_at is null
     and coalesce(m.score_t1,0) + coalesce(m.score_t2,0) > 0
   group by participant.player_id
), candidates as (
  select p.id as player_id,p.club_id,p.last_game_at,p.active,p.inactive_at,
         g.recorded_last_game_at,g.scored_games,g.unrated_games,parameters.as_of,
         strpos(lower(coalesce(p.name,'')), '(merged into ') > 0 as merged_profile,
         (p.active is false and p.inactive_at is null)
           or coalesce(p.inactive_at > g.recorded_last_game_at,false) as preserved_exclusion,
         g.recorded_last_game_at > parameters.as_of - interval '14 days'
           and g.recorded_last_game_at <= parameters.as_of as recent_game
    from public.players p
    join parameters on parameters.club_id=p.club_id
    join recorded_games g on g.player_id=p.id
   where p.last_game_at is null or p.last_game_at < g.recorded_last_game_at
)
select player_id,club_id,scored_games,unrated_games,
       last_game_at as current_last_game_at,recorded_last_game_at as proposed_last_game_at,
       active as current_active,inactive_at as current_inactive_at,
       case when merged_profile then false
            when preserved_exclusion then active else recent_game end as proposed_active,
       case when merged_profile or preserved_exclusion then inactive_at
            when recent_game then null else as_of end as proposed_inactive_at,
       case when merged_profile then 'review merged identity; do not restore visibility'
            when preserved_exclusion then 'preserve explicit exclusion; review timestamp only'
            when recorded_last_game_at > as_of then 'review future-dated match'
            when recent_game then 'restore recent recorded-game activity'
            else 'record historical game; remain inactive' end as review_reason
  from candidates
 order by player_id;
