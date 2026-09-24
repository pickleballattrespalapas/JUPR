begin;

-- These are personal inbox choices. They never approve, delete, or otherwise
-- alter a submitted result, registration, player request, or other club record.
create table public.admin_notification_preferences (
  club_id text not null references public.clubs(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  category_key text not null check(category_key ~ '^[a-z][a-z0-9_]{0,79}$'),
  enabled boolean not null default true,
  updated_at timestamptz not null default now(),
  primary key(club_id,user_id,category_key)
);

create table public.admin_notification_states (
  club_id text not null references public.clubs(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  notification_key text not null check(notification_key ~ '^[a-f0-9]{64}$'),
  category_key text not null check(category_key ~ '^[a-z][a-z0-9_]{0,79}$'),
  kind text not null check(kind in ('action','activity')),
  state text not null check(state in ('new','flagged','cleared')),
  snapshot jsonb not null check(jsonb_typeof(snapshot)='object' and
    snapshot ?& array['key','category','kind','source_id','source_version','title','description','href','occurred_at'] and
    snapshot - array['key','category','kind','source_id','source_version','title','description','href','occurred_at'] = '{}'::jsonb and
    snapshot->>'key'=notification_key and snapshot->>'category'=category_key and snapshot->>'kind'=kind),
  occurred_at timestamptz not null,
  updated_at timestamptz not null default now(),
  primary key(club_id,user_id,notification_key)
);

alter table public.admin_notification_preferences enable row level security;
alter table public.admin_notification_states enable row level security;
revoke all on public.admin_notification_preferences,public.admin_notification_states from public,anon,authenticated;
grant select,insert,update,delete on public.admin_notification_preferences,public.admin_notification_states to service_role;

-- Cancellation audit snapshots contain contacts and notes. Return only the
-- event identity, timestamp, safe display name, and its club-owned tournament.
create function public.pcs_admin_tournament_notification_source(
  p_club_id text,p_kind text,p_since timestamptz,p_limit integer default 201,p_source_id text default null
) returns table(id text,tournament_id text,tournament_name text,display_name text,
                occurred_at timestamptz,source_version text,total_count bigint)
language plpgsql stable security invoker set search_path = '' as $$
begin
  if p_club_id is null or length(p_club_id)=0 or p_since is null or p_kind is null or p_kind not in ('registration','cancellation') then
    raise exception 'Choose a club and notification source' using errcode='22023';
  end if;
  if p_kind='registration' then
    return query
      select r.id,t.id::text,t.name,r.display_name,r.submitted_at,r.submitted_at::text,count(*) over()
      from public.tournament_registrations r join public.tournaments t on r.tournament_id=t.id::text
      where t.club_id=p_club_id and r.submitted_at>=p_since
        and lower(coalesce(r.status,'')) not in ('cancelled','withdrawn')
        and (p_source_id is null or r.id=p_source_id)
      order by r.submitted_at desc,r.id
      limit least(greatest(coalesce(p_limit,201),1),201);
  else
    return query
      select c.id::text,t.id::text,t.name,c.before_json#>>'{registration,display_name}',c.cancelled_at,c.cancelled_at::text,count(*) over()
      from private.tournament_registration_cancellations c join public.tournaments t on c.tournament_id=t.id
      where t.club_id=p_club_id and c.cancelled_at>=p_since
        and (p_source_id is null or c.id::text=p_source_id)
      order by c.cancelled_at desc,c.id
      limit least(greatest(coalesce(p_limit,201),1),201);
  end if;
end;
$$;
revoke all on function public.pcs_admin_tournament_notification_source(text,text,timestamptz,integer,text) from public,anon,authenticated;
grant execute on function public.pcs_admin_tournament_notification_source(text,text,timestamptz,integer,text) to service_role;

commit;
