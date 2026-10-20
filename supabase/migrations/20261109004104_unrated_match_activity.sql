-- Record social/unrated participation atomically with its match receipt.
-- This is compatible with older writers that omit activity-only projections.
-- Keep the existing wrapper chain, ACLs, CAS checks and transaction unchanged.
-- Deploy before app code that includes unrated-only player projections.
begin;

do $activity$
declare
  routine_oid oid;
  definition text;
  occurrences integer;
  old_text constant text := $old$  if exists (
    with rated_match_players as (
      select match_player.player_id
        from pg_catalog.jsonb_to_recordset(v_match_rows) as match_row(
          t1_p1 bigint,
          t1_p2 bigint,
          t2_p1 bigint,
          t2_p2 bigint,
          rating_scope text
        )
        cross join lateral pg_catalog.unnest(
          case
            when v_match_format = 'singles'
              then array[match_row.t1_p1, match_row.t2_p1]
            else array[
              match_row.t1_p1,
              match_row.t1_p2,
              match_row.t2_p1,
              match_row.t2_p2
            ]
          end
        ) as match_player(player_id)
       where coalesce(match_row.rating_scope, '') <> 'unrated'
    ),
    planned_players as (
      select update_row.player_id
        from pg_catalog.jsonb_to_recordset(v_player_updates) as update_row(
          player_id bigint
        )
    )
    (
      select player_id from rated_match_players
      except
      select player_id from planned_players
    )
    union all
    (
      select player_id from planned_players
      except
      select player_id from rated_match_players
    )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_DIRECT_MATCH_PLAYER_PLAN_INVALID: rated match players and player projections differ.';
  end if;$old$;
  new_text constant text := $new$  -- All rated participants still require a projection. Older app versions may
  -- omit unrated-only projections; newer versions may supply activity only.
  if exists (
    with match_players as (
      select match_player.player_id,
             pg_catalog.bool_or(coalesce(match_row.rating_scope, '') <> 'unrated') as is_rated
        from pg_catalog.jsonb_to_recordset(v_match_rows) as match_row(
          t1_p1 bigint, t1_p2 bigint, t2_p1 bigint, t2_p2 bigint, rating_scope text
        )
        cross join lateral pg_catalog.unnest(
          case when v_match_format = 'singles'
            then array[match_row.t1_p1, match_row.t2_p1]
            else array[match_row.t1_p1, match_row.t1_p2, match_row.t2_p1, match_row.t2_p2]
          end
        ) as match_player(player_id)
       group by match_player.player_id
    ),
    planned_players as (
      select update_row.player_id, update_row.expected, update_row.after
        from pg_catalog.jsonb_to_recordset(v_player_updates) as update_row(
          player_id bigint, expected jsonb, after jsonb
        )
    )
    select 1
      from match_players
      full join planned_players using (player_id)
     where (match_players.is_rated and planned_players.player_id is null)
        or match_players.player_id is null
        or (
          not match_players.is_rated
          and planned_players.player_id is not null
          and exists (
            select 1
              from pg_catalog.unnest(
                case when v_match_format = 'singles'
                  then array['singles_rating', 'singles_wins', 'singles_losses',
                             'singles_matches_played', 'singles_last_game_at']
                  else array['rating', 'wins', 'losses', 'matches_played']
                end
              ) as rating_field(name)
             where (planned_players.expected -> rating_field.name)
                   is distinct from (planned_players.after -> rating_field.name)
          )
        )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_DIRECT_MATCH_PLAYER_PLAN_INVALID: rated projections are required; unrated-only participants may update activity but not ratings.';
  end if;$new$;
begin
  routine_oid := pg_catalog.to_regprocedure(
    'public.admin_apply_direct_match_entry_atomic_v1_base_20260727(uuid,text,text,text,text,text,text,text,jsonb,jsonb,jsonb,jsonb,jsonb,jsonb)'
  );
  if routine_oid is null then
    raise exception 'Unrated activity requires the existing direct-match atomic base routine';
  end if;
  definition := pg_catalog.pg_get_functiondef(routine_oid);
  occurrences := (length(definition) - length(replace(definition, old_text, ''))) / length(old_text);
  if occurrences = 0 and strpos(definition, new_text) > 0 then
    return; -- Safe to verify or reapply an already installed patch.
  end if;
  if occurrences <> 1 then
    raise exception 'Unrated activity expected one player-plan guard, found %', occurrences;
  end if;
  execute replace(definition, old_text, new_text);
end
$activity$;

commit;
