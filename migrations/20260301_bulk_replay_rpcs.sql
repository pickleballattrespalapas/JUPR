create or replace function public.bulk_update_match_snapshots(rows jsonb)
returns integer
language plpgsql
as $$
declare
  updated_count integer;
begin
  with data as (
    select *
    from jsonb_to_recordset(rows) as x(
      id bigint,
      club_id text,
      elo_delta numeric,
      t1_p1_r numeric,
      t1_p2_r numeric,
      t2_p1_r numeric,
      t2_p2_r numeric,
      t1_p1_r_end numeric,
      t1_p2_r_end numeric,
      t2_p1_r_end numeric,
      t2_p2_r_end numeric
    )
  ),
  upd as (
    update public.matches m
    set
      elo_delta    = d.elo_delta,
      t1_p1_r      = d.t1_p1_r,
      t1_p2_r      = d.t1_p2_r,
      t2_p1_r      = d.t2_p1_r,
      t2_p2_r      = d.t2_p2_r,
      t1_p1_r_end  = d.t1_p1_r_end,
      t1_p2_r_end  = d.t1_p2_r_end,
      t2_p1_r_end  = d.t2_p1_r_end,
      t2_p2_r_end  = d.t2_p2_r_end
    from data d
    where m.club_id = d.club_id
      and m.id      = d.id
    returning 1
  )
  select count(*) into updated_count from upd;

  return updated_count;
end;
$$;

create or replace function public.bulk_update_players_stats(rows jsonb)
returns integer
language plpgsql
as $$
declare
  updated_count integer;
begin
  with data as (
    select *
    from jsonb_to_recordset(rows) as x(
      id bigint,
      club_id text,
      rating numeric,
      wins integer,
      losses integer,
      matches_played integer
    )
  ),
  upd as (
    update public.players p
    set
      rating         = d.rating,
      wins           = d.wins,
      losses         = d.losses,
      matches_played = d.matches_played
    from data d
    where p.club_id = d.club_id
      and p.id      = d.id
    returning 1
  )
  select count(*) into updated_count from upd;

  return updated_count;
end;
$$;
