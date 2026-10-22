-- Check-in intake is private. Only the narrow FastAPI endpoint may submit names.
-- Review decisions live in hub state so roster changes and review share its CAS.
create table if not exists public.round_robin_hub_checkins (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  hub_id uuid not null,
  session_key text not null check (length(session_key) between 1 and 160),
  name text not null check (length(btrim(name)) between 2 and 120),
  name_key text not null check (length(btrim(name_key)) between 2 and 360),
  created_at timestamptz not null default clock_timestamp(),
  foreign key (club_id, hub_id) references public.round_robin_hubs(club_id, id) on delete cascade,
  unique (hub_id, session_key, name_key)
);
create index if not exists round_robin_hub_checkins_hub_created_idx
  on public.round_robin_hub_checkins(club_id, hub_id, created_at);
alter table public.round_robin_hub_checkins enable row level security;
revoke all on public.round_robin_hub_checkins from public, anon, authenticated;
grant select, insert, delete on public.round_robin_hub_checkins to service_role;

create or replace function public.submit_round_robin_hub_checkin(
  p_hub_id uuid, p_session_key text, p_name text, p_name_key text
) returns jsonb language plpgsql security invoker set search_path = public, pg_temp as $$
declare
  hub public.round_robin_hubs%rowtype;
  live public.live_sessions%rowtype;
  total_count integer;
  recent_count integer;
begin
  if p_hub_id is null or length(coalesce(p_session_key, '')) not between 1 and 160
     or length(btrim(coalesce(p_name, ''))) not between 2 and 120
     or length(btrim(coalesce(p_name_key, ''))) not between 2 and 360 then
    raise exception using errcode = '22023', message = 'Invalid check-in.';
  end if;
  -- Serializes intake and event closure without changing the court-board version.
  select * into hub from public.round_robin_hubs where id = p_hub_id for update;
  if not found then
    raise exception using errcode = 'P0002', message = 'Check-in event not found.';
  end if;
  -- A lost response can be retried even after staff close the event.
  if exists (select 1 from public.round_robin_hub_checkins
             where hub_id=p_hub_id and session_key=p_session_key and name_key=p_name_key) then
    return jsonb_build_object('ok', true);
  end if;
  if hub.status <> 'active' or hub.state->'registration_open' is distinct from 'true'::jsonb then
    raise exception using errcode = '55000', message = 'Check-in is closed.';
  end if;
  if not exists (select 1 from public.round_robin_hub_sessions
                 where club_id=hub.club_id and hub_id=p_hub_id and session_key=p_session_key) then
    raise exception using errcode = '22023', message = 'Choose a round robin in this event.';
  end if;
  select * into live from public.live_sessions
    where club_id=hub.club_id and session_key=p_session_key for share;
  if not found or live.status <> 'active'
     or live.state->>'mode' not in ('admin_play_generator', 'public_play_generator')
     or live.state #>> '{page_state,event,generatorKind}' is distinct from 'round_robin'
     or live.state #>> '{page_state,event,status}' is distinct from 'active'
     or nullif(live.state #>> '{page_state,event,hubFinishAfterRound}', '') is not null
     or coalesce(live.state->'generator_submission', 'null'::jsonb) not in ('null'::jsonb, '{}'::jsonb)
     or nullif(live.state #>> '{official_publish,published_at}', '') is not null
     or coalesce(live.state #> '{official_publish,published_match_ids}', 'null'::jsonb) not in ('null'::jsonb, '[]'::jsonb) then
    raise exception using errcode = '55000', message = 'This round robin is not accepting check-ins.';
  end if;
  select count(*), count(*) filter (where created_at > clock_timestamp() - interval '1 minute')
    into total_count, recent_count from public.round_robin_hub_checkins
    where club_id=hub.club_id and hub_id=p_hub_id;
  if total_count >= 1000 or recent_count >= 120 then
    raise exception using errcode = 'P0003', message = 'Check-in intake limit reached.';
  end if;
  insert into public.round_robin_hub_checkins(club_id, hub_id, session_key, name, name_key)
    values(hub.club_id, p_hub_id, p_session_key, p_name, p_name_key);
  return jsonb_build_object('ok', true);
end;
$$;
revoke all on function public.submit_round_robin_hub_checkin(uuid,text,text,text) from public, anon, authenticated;
grant execute on function public.submit_round_robin_hub_checkin(uuid,text,text,text) to service_role;
