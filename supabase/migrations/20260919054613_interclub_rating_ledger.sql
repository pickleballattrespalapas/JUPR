begin;

-- Only the trusted API computes these projections. Neither public visitors nor
-- authenticated club clients receive direct access to the rating ledger.
create table public.pcs_interclub_rating_generations (
 id uuid primary key default gen_random_uuid(), committed_at timestamptz not null default clock_timestamp(),
 clubs text[] not null, seasons uuid[] not null, fingerprint text not null,
 algorithm text not null default 'jupr-hybrid-score-share-v1', rated_games integer not null
);
create table public.pcs_interclub_rating_sources (
 batch_id uuid primary key references public.pcs_interclub_competition_batches(id),
 season_id uuid not null references public.pcs_interclub_seasons(id), meet_id uuid not null,
 revision integer not null, document jsonb not null, approved_at timestamptz not null,
 generation_id uuid not null references public.pcs_interclub_rating_generations(id)
);
create table public.pcs_interclub_rating_effects (
 generation_id uuid not null references public.pcs_interclub_rating_generations(id),
 season_id uuid not null, entry_id uuid not null references public.pcs_interclub_entries(id),
 batch_id uuid not null references public.pcs_interclub_competition_batches(id), game_id text not null,
 stream text not null check(stream in ('league','overall')), played_at timestamptz not null,
 ordinal integer not null, before_elo double precision not null, after_elo double precision not null,
 primary key(generation_id,entry_id,game_id,stream)
);
create index pcs_interclub_effect_lookup on public.pcs_interclub_rating_effects(entry_id,stream,generation_id,played_at desc,ordinal desc);
create index pcs_interclub_generation_time on public.pcs_interclub_rating_generations(committed_at desc);
create table public.pcs_interclub_current_ratings (
 entry_id uuid primary key references public.pcs_interclub_entries(id), rating numeric not null,
 generation_id uuid not null references public.pcs_interclub_rating_generations(id)
);
create table public.pcs_interclub_rating_repairs (
 club_id text primary key references public.clubs(id), pending boolean not null default true,
 updated_at timestamptz not null default clock_timestamp(), error text
);

do $$ declare t text; begin
 foreach t in array array['pcs_interclub_rating_generations','pcs_interclub_rating_sources','pcs_interclub_rating_effects','pcs_interclub_current_ratings','pcs_interclub_rating_repairs'] loop
  execute format('alter table public.%I enable row level security',t);
  execute format('revoke all on public.%I from public,anon,authenticated',t);
  execute format('grant all on public.%I to service_role',t);
 end loop;
end $$;

-- Preserve the information that was actually available at a roster deadline.
-- Replaying a correction today cannot rewrite last month's eligibility snapshot.
create or replace function public.pcs_interclub_rating_at(p_season_id uuid,p_entry_id uuid,p_cutoff timestamptz)
returns numeric language sql stable security invoker set search_path=public as $$
 select coalesce((
  select e.after_elo::numeric/400 from public.pcs_interclub_rating_effects e
  where e.entry_id=p_entry_id and e.season_id=p_season_id and e.stream='league'
   and e.played_at<=p_cutoff and e.generation_id=(
    select g.id from public.pcs_interclub_rating_generations g
    where p_season_id=any(g.seasons) and g.committed_at<=p_cutoff
    order by g.committed_at desc,g.id desc limit 1)
  order by e.played_at desc,e.ordinal desc limit 1),
  (select starting_rating from public.pcs_interclub_entries where id=p_entry_id and season_id=p_season_id and entered_at<=p_cutoff))
$$;
revoke all on function public.pcs_interclub_rating_at(uuid,uuid,timestamptz) from public,anon,authenticated;
grant execute on function public.pcs_interclub_rating_at(uuid,uuid,timestamptz) to service_role;

create function public.pcs_interclub_rating_snapshot(p_clubs text[])
returns jsonb language plpgsql security invoker set search_path=public as $$
declare v_clubs text[]; v_payload jsonb;
begin
 -- Seasons connect represented clubs; local games connect all club players.
 -- The closure includes other seasons so a historical correction propagates
 -- through subsequent cross-club games without crossing player ownership.
 with recursive connected(club_id) as (
  select unnest(p_clubs)
  union
  select e2.club_id from connected c join public.pcs_interclub_entries e1 on e1.club_id=c.club_id
   join public.pcs_interclub_entries e2 on e2.season_id=e1.season_id
 ) select array_agg(club_id order by club_id) into v_clubs from connected;
 select jsonb_build_object(
  'clubs',to_jsonb(v_clubs),
  'players',coalesce((select jsonb_agg(jsonb_build_object('id',p.id,'club_id',p.club_id,'starting_rating',p.starting_rating,
    'rating',p.rating,'wins',p.wins,'losses',p.losses,'matches_played',p.matches_played,'last_game_at',p.last_game_at) order by p.club_id,p.id)
   from public.players p where p.club_id=any(v_clubs)),'[]'::jsonb),
  'entries',coalesce((select jsonb_agg(to_jsonb(e) order by e.id) from public.pcs_interclub_entries e where e.club_id=any(v_clubs)),'[]'::jsonb),
  'matches',coalesce((select jsonb_agg(jsonb_build_object('id',m.id,'club_id',m.club_id,'date',m.date,'league',m.league,
    'match_type',m.match_type,'match_format',m.match_format,'t1_p1',m.t1_p1,'t1_p2',m.t1_p2,'t2_p1',m.t2_p1,'t2_p2',m.t2_p2,
    'score_t1',m.score_t1,'score_t2',m.score_t2,'rating_bonus_elo',m.rating_bonus_elo,'deleted_at',m.deleted_at,'rating_scope',m.rating_scope)
    order by m.club_id,m.id) from public.matches m where m.club_id=any(v_clubs)),'[]'::jsonb),
  'sources',coalesce((select jsonb_agg(to_jsonb(s) order by s.batch_id) from public.pcs_interclub_rating_sources s
   where exists(select 1 from public.pcs_interclub_entries e where e.season_id=s.season_id and e.club_id=any(v_clubs))),'[]'::jsonb),
  'approved',coalesce((select jsonb_agg(jsonb_build_object('id',b.id,'season_id',b.season_id,'meet_id',b.meet_id,'revision',b.revision,
   'document',b.document,'approved_at',b.approved_at,'ratings_status',b.ratings_status) order by b.id)
   from public.pcs_interclub_competition_batches b where b.state='approved' and exists(
    select 1 from public.pcs_interclub_entries e where e.season_id=b.season_id and e.club_id=any(v_clubs))),'[]'::jsonb),
  'repairs',coalesce((select jsonb_agg(to_jsonb(r) order by r.club_id) from public.pcs_interclub_rating_repairs r where r.club_id=any(v_clubs)),'[]'::jsonb)
 ) into v_payload;
 return jsonb_build_object('snapshot',v_payload,'fingerprint',md5(v_payload::text));
end $$;
revoke all on function public.pcs_interclub_rating_snapshot(text[]) from public,anon,authenticated;
grant execute on function public.pcs_interclub_rating_snapshot(text[]) to service_role;

create function public.pcs_apply_interclub_rating_projection(p_clubs text[],p_fingerprint text,p_projection jsonb,p_expected_batch uuid default null,p_expected_revision integer default null)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare v_snapshot jsonb; v_clubs text[]; v_generation uuid; v_seasons uuid[]; v_count integer; v_expected integer; v_row jsonb;
begin
 -- Table locks also cover local insert/delete phantoms. A private new advisory
 -- lock alone would not serialize the existing local scoring writers.
 lock table public.players,public.matches,public.pcs_interclub_entries,public.pcs_interclub_competition_batches,
  public.pcs_interclub_rating_sources,public.pcs_interclub_rating_repairs in share row exclusive mode;
 v_snapshot:=public.pcs_interclub_rating_snapshot(p_clubs);
 if v_snapshot->>'fingerprint' is distinct from p_fingerprint then return jsonb_build_object('status','conflict'); end if;
 if p_expected_batch is not null and not exists(select 1 from public.pcs_interclub_competition_batches
  where id=p_expected_batch and revision=p_expected_revision and state='approved') then
  raise exception using errcode='40001',message='The approved result revision changed.';
 end if;
 select array_agg(value) into v_clubs from jsonb_array_elements_text(v_snapshot->'snapshot'->'clubs');
 select array_agg(distinct (value->>'season_id')::uuid) into v_seasons from jsonb_array_elements(v_snapshot->'snapshot'->'entries');
 if jsonb_typeof(p_projection->'players')<>'array' or jsonb_typeof(p_projection->'effects')<>'array' then
  raise exception 'Invalid rating projection'; end if;
 v_expected:=jsonb_array_length(v_snapshot->'snapshot'->'players');
 if jsonb_array_length(p_projection->'players')<>v_expected then raise exception 'Incomplete player projection'; end if;
 insert into public.pcs_interclub_rating_generations(clubs,seasons,fingerprint,rated_games)
 values(v_clubs,coalesce(v_seasons,'{}'),p_fingerprint,(p_projection->>'rated_games')::integer) returning id into v_generation;
 perform set_config('pcs.interclub_projection','on',true);
 update public.players p set rating=x.rating,wins=x.wins,losses=x.losses,matches_played=x.matches_played,last_game_at=x.last_game_at
 from jsonb_to_recordset(p_projection->'players') as x(id bigint,club_id text,rating double precision,wins integer,losses integer,matches_played integer,last_game_at timestamptz)
 where p.id=x.id and p.club_id=x.club_id and p.club_id=any(v_clubs);
 get diagnostics v_count=row_count;
 if v_count<>v_expected then raise exception 'Player projection ownership changed'; end if;
 update public.matches m set elo_delta=x.elo_delta,t1_p1_r=x.t1_p1_r,t1_p2_r=x.t1_p2_r,t2_p1_r=x.t2_p1_r,t2_p2_r=x.t2_p2_r,
  t1_p1_r_end=x.t1_p1_r_end,t1_p2_r_end=x.t1_p2_r_end,t2_p1_r_end=x.t2_p1_r_end,t2_p2_r_end=x.t2_p2_r_end
 from jsonb_to_recordset(p_projection->'matches') as x(id bigint,club_id text,elo_delta double precision,t1_p1_r double precision,t1_p2_r double precision,
  t2_p1_r double precision,t2_p2_r double precision,t1_p1_r_end double precision,t1_p2_r_end double precision,t2_p1_r_end double precision,t2_p2_r_end double precision)
 where m.id=x.id and m.club_id=x.club_id and m.club_id=any(v_clubs);
 get diagnostics v_count=row_count;
 if v_count<>jsonb_array_length(p_projection->'matches') then raise exception 'Local match projection changed'; end if;
 insert into public.pcs_interclub_rating_effects(generation_id,season_id,entry_id,batch_id,game_id,stream,played_at,ordinal,before_elo,after_elo)
 select v_generation,x.season_id,x.entry_id,x.batch_id,x.game_id,x.stream,x.played_at,x.ordinal,x.before_elo,x.after_elo
 from jsonb_to_recordset(p_projection->'effects') as x(season_id uuid,entry_id uuid,batch_id uuid,game_id text,stream text,played_at timestamptz,ordinal integer,before_elo double precision,after_elo double precision);
 insert into public.pcs_interclub_current_ratings(entry_id,rating,generation_id)
 select (value->>'entry_id')::uuid,(value->>'rating')::numeric,v_generation from jsonb_array_elements(p_projection->'league_ratings')
 on conflict(entry_id) do update set rating=excluded.rating,generation_id=excluded.generation_id;
 for v_row in select value from jsonb_array_elements(p_projection->'sources') loop
  insert into public.pcs_interclub_rating_sources(batch_id,season_id,meet_id,revision,document,approved_at,generation_id)
  values((v_row->>'batch_id')::uuid,(v_row->>'season_id')::uuid,(v_row->>'meet_id')::uuid,(v_row->>'revision')::integer,
   v_row->'document',(v_row->>'approved_at')::timestamptz,v_generation)
  on conflict(batch_id) do update set revision=excluded.revision,document=excluded.document,approved_at=excluded.approved_at,generation_id=excluded.generation_id;
  update public.pcs_interclub_competition_batches set ratings_status='completed',ratings_error=null
   where id=(v_row->>'batch_id')::uuid and revision=(v_row->>'revision')::integer and state='approved';
 end loop;
 update public.pcs_interclub_rating_repairs set pending=false,error=null,updated_at=clock_timestamp() where club_id=any(v_clubs);
 perform set_config('pcs.interclub_projection','off',true);
 return jsonb_build_object('status','completed','rated_games',(p_projection->>'rated_games')::integer,'generation_id',v_generation);
end $$;
revoke all on function public.pcs_apply_interclub_rating_projection(text[],text,jsonb,uuid,integer) from public,anon,authenticated;
grant execute on function public.pcs_apply_interclub_rating_projection(text[],text,jsonb,uuid,integer) to service_role;

create function public.pcs_fail_interclub_ratings(p_batch_id uuid,p_revision integer,p_error text)
returns jsonb language plpgsql security invoker set search_path=public as $$
begin
 update public.pcs_interclub_competition_batches set ratings_status='failed',ratings_error=left(p_error,300)
 where id=p_batch_id and revision=p_revision and state='approved' and ratings_status<>'completed';
 return jsonb_build_object('status','failed');
end $$;
revoke all on function public.pcs_fail_interclub_ratings(uuid,integer,text) from public,anon,authenticated;
grant execute on function public.pcs_fail_interclub_ratings(uuid,integer,text) to service_role;

create function public.pcs_interclub_queue_local_rating_repair()
returns trigger language plpgsql security invoker set search_path=public as $$
declare v_club text; v_clubs text[];
begin
 if current_setting('pcs.interclub_projection',true)='on' then return coalesce(new,old); end if;
 if tg_op='UPDATE' and (to_jsonb(new)-array['elo_delta','t1_p1_r','t1_p2_r','t2_p1_r','t2_p2_r','t1_p1_r_end','t1_p2_r_end','t2_p1_r_end','t2_p2_r_end'])
  is not distinct from (to_jsonb(old)-array['elo_delta','t1_p1_r','t1_p2_r','t2_p1_r','t2_p2_r','t1_p1_r_end','t1_p2_r_end','t2_p1_r_end','t2_p2_r_end']) then return new; end if;
 v_club:=coalesce(new.club_id,old.club_id);
 if exists(select 1 from public.pcs_interclub_rating_sources s join public.pcs_interclub_entries e on e.season_id=s.season_id where e.club_id=v_club) then
  with recursive connected(club_id) as (
   select v_club union select e2.club_id from connected c join public.pcs_interclub_entries e1 on e1.club_id=c.club_id
    join public.pcs_interclub_entries e2 on e2.season_id=e1.season_id
  ) select array_agg(club_id order by club_id) into v_clubs from connected;
  insert into public.pcs_interclub_rating_repairs(club_id,pending) select unnest(v_clubs),true
   on conflict(club_id) do update set pending=true,error=null,updated_at=clock_timestamp();
  update public.pcs_interclub_competition_batches b set ratings_status='pending',ratings_error=null
   where b.state='approved' and exists(select 1 from public.pcs_interclub_entries e where e.season_id=b.season_id and e.club_id=any(v_clubs));
 end if;
 return coalesce(new,old);
end $$;
revoke all on function public.pcs_interclub_queue_local_rating_repair() from public,anon,authenticated;
grant execute on function public.pcs_interclub_queue_local_rating_repair() to service_role;
create trigger pcs_interclub_local_rating_repair after insert or update or delete on public.matches
 for each row execute function public.pcs_interclub_queue_local_rating_repair();
create trigger pcs_interclub_player_rating_repair after update of rating,starting_rating,wins,losses,matches_played on public.players
 for each row execute function public.pcs_interclub_queue_local_rating_repair();

create function public.pcs_fail_interclub_club_ratings(p_club_id text,p_error text)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare v_clubs text[];
begin
 with recursive connected(club_id) as (
  select p_club_id union select e2.club_id from connected c join public.pcs_interclub_entries e1 on e1.club_id=c.club_id
   join public.pcs_interclub_entries e2 on e2.season_id=e1.season_id
 ) select array_agg(club_id order by club_id) into v_clubs from connected;
 insert into public.pcs_interclub_rating_repairs(club_id,pending,error) select unnest(v_clubs),true,left(p_error,300)
 on conflict(club_id) do update set pending=true,error=excluded.error,updated_at=clock_timestamp();
 update public.pcs_interclub_competition_batches b set ratings_status='failed',ratings_error=left(p_error,300)
 where b.state='approved' and exists(select 1 from public.pcs_interclub_entries e where e.season_id=b.season_id and e.club_id=any(v_clubs));
 return jsonb_build_object('status','failed');
end $$;
revoke all on function public.pcs_fail_interclub_club_ratings(text,text) from public,anon,authenticated;
grant execute on function public.pcs_fail_interclub_club_ratings(text,text) to service_role;
commit;
