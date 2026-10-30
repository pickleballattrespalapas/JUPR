-- A pre-play estimate is not a calculated rating. Confirmations are private
-- administrative records and never grant a public client permission to edit
-- a rating. All RPCs are SECURITY INVOKER and service-role-only.
create table if not exists public.player_starting_rating_reviews (
  club_id text not null,
  player_id bigint not null,
  match_format text not null check (match_format in ('doubles','singles')),
  rating_elo numeric(10,4) not null check (rating_elo between 400 and 2800),
  requested_rating text check (length(requested_rating) <= 120),
  confirmed_at timestamptz not null default now(),
  confirmed_by text not null,
  primary key (club_id,player_id,match_format),
  foreign key (club_id,player_id) references public.players(club_id,id) on delete cascade
);
alter table public.player_starting_rating_reviews enable row level security;
revoke all on public.player_starting_rating_reviews from public, anon, authenticated;
grant select,insert,update,delete on public.player_starting_rating_reviews to service_role;

create or replace function public.starting_rating_review_state_v1(
  p_club_id text, p_player_id bigint, p_match_format text
) returns jsonb
language plpgsql stable security invoker
set search_path = pg_catalog, public, pg_temp
as $function$
declare
  v_player public.players%rowtype;
  v_review jsonb;
  v_leagues jsonb;
  v_count integer;
  v_counter integer;
  v_league_counter integer;
  v_state jsonb;
begin
  if p_match_format is null or p_match_format not in ('singles','doubles') then
    raise exception 'STARTING_REVIEW_FORMAT_INVALID';
  end if;
  select * into v_player from public.players
    where club_id=p_club_id and id=p_player_id;
  if not found or position('(merged into ' in lower(v_player.name)) > 0 then
    raise exception 'STARTING_REVIEW_PLAYER_NOT_FOUND';
  end if;

  select count(*)::integer into v_count from public.matches m
   where m.club_id=p_club_id and p_player_id in (m.t1_p1,m.t1_p2,m.t2_p1,m.t2_p2)
     and m.deleted_at is null
     and lower(coalesce(m.rating_scope,'')) <> 'unrated'
     and m.score_t1 >= 0 and m.score_t2 >= 0 and m.score_t1 <> m.score_t2
     and m.score_t1+m.score_t2 > 0
     and case when p_match_format='singles' then
       m.t1_p1 is not null and m.t2_p1 is not null and m.t1_p1<>m.t2_p1
       and m.t1_p2 is null and m.t2_p2 is null
     else
       m.t1_p1 is not null and m.t1_p2 is not null
       and m.t2_p1 is not null and m.t2_p2 is not null
       and m.t1_p1<>m.t1_p2 and m.t1_p1<>m.t2_p1 and m.t1_p1<>m.t2_p2
       and m.t1_p2<>m.t2_p1 and m.t1_p2<>m.t2_p2 and m.t2_p1<>m.t2_p2
       and lower(coalesce(m.match_format,'')) <> 'singles'
     end;

  select coalesce(jsonb_agg(to_jsonb(l) order by l.id),'[]'::jsonb),
         coalesce(max(greatest(coalesce(l.matches_played,0),coalesce(l.wins,0)+coalesce(l.losses,0))),0)
    into v_leagues,v_league_counter
    from public.league_ratings l
   where l.club_id=p_club_id and l.player_id=p_player_id
     and coalesce((select lower(nullif(btrim(meta.match_format),''))
          from public.leagues_metadata meta
         where meta.club_id=l.club_id and meta.league_name=l.league_name limit 1),'doubles')=p_match_format;
  select to_jsonb(r) into v_review
    from public.player_starting_rating_reviews r
   where r.club_id=p_club_id and r.player_id=p_player_id and r.match_format=p_match_format;

  -- Counters are additional conservative evidence for imported legacy or
  -- interclub results. Zero counters never override actual scored match rows.
  v_counter := case when p_match_format='singles' then greatest(
    coalesce(v_player.singles_matches_played,0),
    coalesce(v_player.singles_wins,0)+coalesce(v_player.singles_losses,0),
    coalesce((v_player.singles_replay_baseline->>'matches_played')::integer,0),
    coalesce((v_player.singles_replay_baseline->>'wins')::integer,0)+coalesce((v_player.singles_replay_baseline->>'losses')::integer,0)
  ) else greatest(coalesce(v_player.matches_played,0),coalesce(v_player.wins,0)+coalesce(v_player.losses,0)) end;
  v_state := jsonb_build_object(
    'player',jsonb_build_object(
      'id',v_player.id,'club_id',v_player.club_id,'name',v_player.name,
      'rating',case when p_match_format='singles' then coalesce(v_player.singles_rating,(v_player.singles_replay_baseline->>'rating')::double precision) else v_player.rating end,
      'starting_rating',case when p_match_format='singles' then (v_player.singles_replay_baseline->>'rating')::numeric else v_player.starting_rating end,
      'matches_played',case when p_match_format='singles' then v_player.singles_matches_played else v_player.matches_played end,
      'wins',case when p_match_format='singles' then v_player.singles_wins else v_player.wins end,
      'losses',case when p_match_format='singles' then v_player.singles_losses else v_player.losses end,
      'singles_baseline',case when p_match_format='singles' then v_player.singles_replay_baseline else null end
    ),
    'rated_matches',greatest(v_count,v_counter,v_league_counter),
    'recorded_rated_matches',v_count,'league_rows',v_leagues,'confirmation',v_review
  );
  return v_state || jsonb_build_object('state_fingerprint',
    encode(sha256(convert_to(v_state::text,'UTF8')),'hex'));
end;
$function$;

create or replace function public.preview_starting_rating_review_v1(
  p_club_id text, p_player_ids bigint[], p_match_format text default 'doubles'
) returns jsonb
language plpgsql stable security invoker
set search_path = pg_catalog, public, pg_temp
as $function$
declare v_rows jsonb;
begin
  if p_match_format is null or p_match_format not in ('doubles','singles')
     or p_player_ids is null or cardinality(p_player_ids)>500
     or exists(select 1 from unnest(p_player_ids) i where i is null or i<=0)
     or cardinality(p_player_ids)<>(select count(distinct i) from unnest(p_player_ids) i) then
    raise exception 'STARTING_REVIEW_INPUT_INVALID';
  end if;
  select coalesce(jsonb_agg(public.starting_rating_review_state_v1(p_club_id,i,p_match_format) order by i),'[]'::jsonb)
    into v_rows from unnest(p_player_ids) i;
  return jsonb_build_object('players',v_rows,'match_format',p_match_format);
end;
$function$;

create or replace function public.confirm_starting_rating_reviews_v1(
  p_club_id text, p_match_format text, p_reviews jsonb,
  p_actor_email text, p_actor_role text, p_source text default 'starting_rating_review'
) returns jsonb
language plpgsql security invoker
set search_path = pg_catalog, public, pg_temp
set lock_timeout = '2s'
set statement_timeout = '15s'
as $function$
declare
  v_item jsonb;
  v_before jsonb;
  v_after jsonb;
  v_rows jsonb := '[]'::jsonb;
  v_player_id bigint;
  v_rating numeric(10,4);
  v_ids bigint[];
  v_count integer := 0;
begin
  if p_actor_role is null or p_actor_role not in ('administrator','club_owner','super_admin','operator')
     or nullif(btrim(p_actor_email),'') is null then
    raise exception 'STARTING_REVIEW_PERMISSION_DENIED';
  end if;
  if p_match_format is null or p_match_format not in ('doubles','singles')
     or jsonb_typeof(p_reviews) is distinct from 'array'
     or jsonb_array_length(p_reviews) not between 1 and 500 then
    raise exception 'STARTING_REVIEW_INPUT_INVALID';
  end if;
  if exists(select 1 from jsonb_array_elements(p_reviews) r
    where jsonb_typeof(r) is distinct from 'object'
       or coalesce(r->>'player_id','') !~ '^[1-9][0-9]*$'
       or jsonb_typeof(r->'rating_elo') is distinct from 'number'
       or coalesce(r->>'expected_state_fingerprint','') !~ '^[0-9a-f]{64}$'
       or length(coalesce(r->>'requested_rating',''))>120) then
    raise exception 'STARTING_REVIEW_INPUT_INVALID';
  end if;
  select array_agg((r->>'player_id')::bigint order by (r->>'player_id')::bigint) into v_ids
    from jsonb_array_elements(p_reviews) r;
  if cardinality(v_ids)<>(select count(distinct i) from unnest(v_ids) i) then
    raise exception 'STARTING_REVIEW_INPUT_INVALID';
  end if;

  -- This short administrative transaction excludes concurrent match inserts
  -- and league/aggregate writes while it rechecks evidence and changes seeds.
  -- A rating publisher that read the earlier seed must subsequently pass its
  -- existing player-state CAS; stale computations cannot overwrite this seed.
  lock table public.matches, public.players, public.league_ratings in share row exclusive mode;
  perform 1 from public.players where club_id=p_club_id and id=any(v_ids) order by id for update;
  if exists(select 1 from public.replay_jobs where club_id=p_club_id and status in ('pending','running')) then
    raise exception 'STARTING_REVIEW_REPLAY_ACTIVE';
  end if;
  for v_item in select r from jsonb_array_elements(p_reviews) r order by (r->>'player_id')::bigint loop
    v_player_id := (v_item->>'player_id')::bigint;
    v_rating := (v_item->>'rating_elo')::numeric;
    if v_rating not between 400 and 2800 then raise exception 'STARTING_REVIEW_RATING_INVALID'; end if;
    v_before := public.starting_rating_review_state_v1(p_club_id,v_player_id,p_match_format);
    if (v_before->>'rated_matches')::integer>0 then raise exception 'STARTING_REVIEW_HISTORY_EXISTS'; end if;
    if v_before->>'state_fingerprint' is distinct from v_item->>'expected_state_fingerprint' then
      raise exception 'STARTING_REVIEW_CONFLICT';
    end if;
    if p_match_format='singles' then
      if (v_before->'player'->>'starting_rating')::numeric is distinct from v_rating
         or (v_before->'player'->>'rating')::numeric is distinct from v_rating then
        raise exception 'STARTING_REVIEW_SINGLES_IMMUTABLE';
      end if;
    else
      update public.players set rating=v_rating,starting_rating=v_rating
       where club_id=p_club_id and id=v_player_id;
      update public.league_ratings l set rating=v_rating,starting_rating=v_rating
       where l.club_id=p_club_id and l.player_id=v_player_id
         and coalesce(l.matches_played,0)=0 and coalesce(l.wins,0)=0 and coalesce(l.losses,0)=0
         and coalesce((select lower(nullif(btrim(meta.match_format),'')) from public.leagues_metadata meta
              where meta.club_id=l.club_id and meta.league_name=l.league_name limit 1),'doubles')='doubles';
    end if;
    insert into public.player_starting_rating_reviews(club_id,player_id,match_format,rating_elo,requested_rating,confirmed_at,confirmed_by)
    values(p_club_id,v_player_id,p_match_format,v_rating,nullif(btrim(v_item->>'requested_rating'),''),now(),p_actor_email)
    on conflict(club_id,player_id,match_format) do update
      set rating_elo=excluded.rating_elo,requested_rating=excluded.requested_rating,
          confirmed_at=excluded.confirmed_at,confirmed_by=excluded.confirmed_by;
    v_after := public.starting_rating_review_state_v1(p_club_id,v_player_id,p_match_format);
    insert into public.admin_activity_log(club_id,actor_email,actor_role,action_type,entity_type,entity_id,before_json,after_json,note,source_page,flagged_for_review)
    values(p_club_id,p_actor_email,p_actor_role,'confirm_starting_rating','player',v_player_id::text,
      v_before,v_after,'Confirmed starting estimate before first rated match.',left(coalesce(p_source,'starting_rating_review'),120),false);
    v_rows := v_rows || jsonb_build_array(v_after);
    v_count := v_count+1;
  end loop;
  return jsonb_build_object('ok',true,'confirmed_count',v_count,'players',v_rows,'match_format',p_match_format);
end;
$function$;

revoke all on function public.starting_rating_review_state_v1(text,bigint,text) from public,anon,authenticated;
revoke all on function public.preview_starting_rating_review_v1(text,bigint[],text) from public,anon,authenticated;
revoke all on function public.confirm_starting_rating_reviews_v1(text,text,jsonb,text,text,text) from public,anon,authenticated;
grant execute on function public.starting_rating_review_state_v1(text,bigint,text) to service_role;
grant execute on function public.preview_starting_rating_review_v1(text,bigint[],text) to service_role;
grant execute on function public.confirm_starting_rating_reviews_v1(text,text,jsonb,text,text,text) to service_role;
comment on table public.player_starting_rating_reviews is 'Private confirmation of estimated ability before the first rated match; calculated ratings are never replaced by imported self-ratings.';
