-- The draw-version trigger is shared by tournament_teams, tournament_games,
-- and tournament_podium. Keep game-only OLD/NEW field access inside the
-- tournament_games branch so PostgreSQL never resolves game columns against a
-- team or podium trigger rowtype.

create or replace function public.touch_tournament_draw_version_from_child()
returns trigger
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_old_draw_id uuid;
  v_new_draw_id uuid;
  v_team_structural_change boolean := false;
  v_game_derivation_change boolean := false;
begin
  if tg_op in ('UPDATE', 'DELETE') then
    v_old_draw_id := old.draw_id;
  end if;
  if tg_op in ('INSERT', 'UPDATE') then
    v_new_draw_id := new.draw_id;
  end if;
  update public.tournament_event_draws
     set updated_at = clock_timestamp()
   where id in (v_old_draw_id, v_new_draw_id);

  if tg_table_name = 'tournament_teams' then
    v_team_structural_change := tg_op <> 'UPDATE';
    if tg_op = 'UPDATE' then
      v_team_structural_change := old.id is distinct from new.id
        or old.tournament_id is distinct from new.tournament_id
        or old.draw_id is distinct from new.draw_id
        or old.registration_day_id is distinct from new.registration_day_id
        or old.event_option_id is distinct from new.event_option_id
        or old.team_number is distinct from new.team_number
        or old.player1_id is distinct from new.player1_id
        or old.player2_id is distinct from new.player2_id;
    end if;
  end if;
  if v_team_structural_change
     and coalesce(current_setting('jupr.tournament_results_import_structural_write', true), 'off') <> 'on'
     and exists (
      select 1 from public.tournament_games g
       where g.draw_id in (v_old_draw_id, v_new_draw_id)
    ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_HAS_GAMES';
  end if;

  if tg_table_name = 'tournament_games' then
    v_game_derivation_change := tg_op <> 'UPDATE';
    if tg_op = 'UPDATE' then
      v_game_derivation_change := old.stage is distinct from new.stage
        or old.team_a_id is distinct from new.team_a_id
        or old.team_b_id is distinct from new.team_b_id
        or old.team_a_source is distinct from new.team_a_source
        or old.team_b_source is distinct from new.team_b_source
        or old.score_a is distinct from new.score_a
        or old.score_b is distinct from new.score_b
        or old.winner_team_id is distinct from new.winner_team_id
        or old.loser_team_id is distinct from new.loser_team_id
        or old.finalized_at is distinct from new.finalized_at;
    end if;
  if v_game_derivation_change and exists (
      select 1 from public.tournament_podium p
       where p.draw_id in (v_old_draw_id, v_new_draw_id)
    ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_SCORE_PODIUM_LOCK';
  end if;
  if v_game_derivation_change and exists (
      select 1
        from public.matches m
        join public.tournament_games published_game on published_game.id = m.tournament_game_id
       where published_game.draw_id in (v_old_draw_id, v_new_draw_id)
    ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_SCORE_PUBLISHED_LOCK';
  end if;
  if v_game_derivation_change
     and (
       (tg_op in ('UPDATE', 'DELETE') and upper(coalesce(old.stage, '')) = 'ROUND_ROBIN')
       or (tg_op in ('INSERT', 'UPDATE') and upper(coalesce(new.stage, '')) = 'ROUND_ROBIN')
     )
     and exists (
       select 1 from public.tournament_games playoff
        where playoff.draw_id in (v_old_draw_id, v_new_draw_id)
          and upper(coalesce(playoff.stage, '')) = 'PLAYOFF'
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DOWNSTREAM_SCORE_LOCK';
  end if;
  end if;
  if tg_table_name = 'tournament_podium' and exists (
    select 1
      from public.tournament_event_draws d
      join public.tournaments t on t.id = d.tournament_id
      join public.player_badges badge
        on badge.club_id = t.club_id
       and badge.context_type = 'tournament'
       and badge.context_id::text like d.tournament_id::text || ':draw:' || d.id::text || ':podium:%'
     where d.id in (v_old_draw_id, v_new_draw_id)
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_PODIUM_ALREADY_AWARDED';
  end if;
  if tg_op = 'DELETE' then
    return old;
  end if;
  return new;
end;
$$;

revoke all on function public.touch_tournament_draw_version_from_child()
  from public, anon, authenticated;
grant execute on function public.touch_tournament_draw_version_from_child()
  to service_role;
