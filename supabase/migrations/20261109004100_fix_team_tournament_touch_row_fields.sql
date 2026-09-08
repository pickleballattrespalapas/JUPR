-- Rating reviews and verifications share this trigger with draw-backed rows,
-- but have no draw_id. PL/pgSQL resolves record fields before evaluating AND;
-- enter the table-specific branch before referring to its fields.
create or replace function public.touch_team_tournament_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
  new.updated_at := clock_timestamp();
  if tg_table_name in (
    'tournament_four_player_teams',
    'tournament_team_matchups',
    'tournament_four_player_podium'
  ) then
    if new.draw_id is not null then
      update public.tournament_event_draws
         set updated_at = clock_timestamp()
       where id = new.draw_id;
    end if;
  elsif tg_table_name = 'tournament_team_match_games' then
    update public.tournament_event_draws draw
       set updated_at = clock_timestamp()
      from public.tournament_team_matchups matchup
     where matchup.id = new.matchup_id
       and draw.id = matchup.draw_id;
  end if;
  return new;
end;
$$;

revoke all on function public.touch_team_tournament_updated_at()
  from public, anon, authenticated;
grant execute on function public.touch_team_tournament_updated_at()
  to service_role;
