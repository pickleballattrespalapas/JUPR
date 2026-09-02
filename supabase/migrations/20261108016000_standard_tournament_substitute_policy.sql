-- Standard tournament check-in cannot atomically replace registration, draw,
-- game, live-court, and rating identities. Keep the independently guarded
-- four-player-team roster workflow intact, but make every standard event
-- explicitly ineligible for substitute configuration.

update public.tournament_event_options as event
   set team_allow_substitutes = false,
       updated_at = pg_catalog.clock_timestamp()
 where event.team_allow_substitutes is true
   and upper(coalesce(event.competition_format, 'STANDARD'))
       <> 'FOUR_PLAYER_TEAM';

alter table public.tournament_event_options
  drop constraint if exists
    tournament_event_options_substitutes_four_player_only_v1;

alter table public.tournament_event_options
  add constraint tournament_event_options_substitutes_four_player_only_v1
  check (
    upper(coalesce(competition_format, 'STANDARD')) = 'FOUR_PLAYER_TEAM'
    or team_allow_substitutes is false
  ) not valid;

alter table public.tournament_event_options
  validate constraint
    tournament_event_options_substitutes_four_player_only_v1;

comment on constraint
  tournament_event_options_substitutes_four_player_only_v1
  on public.tournament_event_options is
  'Only four-player-team events may enable guarded between-match roster replacement. Standard tournament check-in never assigns substitutes.';
