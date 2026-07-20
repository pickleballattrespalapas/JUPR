-- Communications parity requires optimistic concurrency for recap, subscription,
-- digest, and outbox operator workflows. These tables remain server-only: browser
-- clients authenticate with Supabase Auth, then FastAPI uses the service role.

-- Weekly Recap predates the canonical Supabase migration directory in older
-- installations. Canonicalize its minimum schema here so a fresh migration run
-- receives the same server-only object as an upgraded installation.
create table if not exists public.weekly_recaps (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  week_start date not null,
  week_end date not null,
  status text not null default 'draft',
  generated_json jsonb not null default '{}'::jsonb,
  edits_json jsonb not null default '{}'::jsonb,
  final_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  published_at timestamptz null,
  published_by text null,
  unique (club_id, week_start)
);

create index if not exists weekly_recaps_club_week_idx
  on public.weekly_recaps (club_id, week_start desc);

create index if not exists weekly_recaps_club_status_week_idx
  on public.weekly_recaps (club_id, status, week_start desc);

alter table if exists public.weekly_recaps
  add column if not exists row_version bigint not null default 1;

alter table if exists public.player_profile_update_subscriptions
  add column if not exists row_version bigint not null default 1,
  add column if not exists replacement_operation_key uuid null,
  add column if not exists replacement_of_subscription_id uuid null
    references public.player_profile_update_subscriptions(id) on delete set null,
  add column if not exists superseded_by_subscription_id uuid null
    references public.player_profile_update_subscriptions(id) on delete set null;

alter table if exists public.player_weekly_profile_digests
  add column if not exists row_version bigint not null default 1;

-- The foundation uniqueness omitted week_end even though the repository uses
-- an exact date window as its conflict target. Replace it before the first
-- four-column upsert is attempted.
alter table if exists public.player_weekly_profile_digests
  drop constraint if exists player_weekly_profile_digests_club_id_player_id_week_start_key;
alter table if exists public.player_weekly_profile_digests
  drop constraint if exists player_weekly_profile_digests_club_player_window_key;
alter table if exists public.player_weekly_profile_digests
  add constraint player_weekly_profile_digests_club_player_window_key
  unique (club_id, player_id, week_start, week_end);

alter table if exists public.player_profile_update_outbox
  add column if not exists row_version bigint not null default 1,
  add column if not exists queue_operation_key uuid null,
  add column if not exists attempt_count integer not null default 0,
  add column if not exists last_attempt_at timestamptz null,
  add column if not exists last_attempt_by text null,
  add column if not exists delivery_mode text null,
  add column if not exists delivery_attempt_id uuid null,
  add column if not exists digest_snapshot_json jsonb not null default '{}'::jsonb;

-- A subscription may receive distinct windows that share a start date. The
-- repository and idempotency model both identify a row by the full window.
alter table if exists public.player_profile_update_outbox
  drop constraint if exists player_profile_update_outbox_subscription_id_week_start_key;
alter table if exists public.player_profile_update_outbox
  drop constraint if exists player_profile_update_outbox_subscription_window_key;

-- Repeat generations are intentionally retained as distinct history rows.
-- Request-level deduplication is scoped to queue_operation_key below.
create index if not exists player_profile_update_outbox_subscription_window_idx
  on public.player_profile_update_outbox (subscription_id, week_start, week_end);

alter table if exists public.player_profile_update_outbox
  drop constraint if exists player_profile_update_outbox_status_check;

alter table if exists public.player_profile_update_outbox
  add constraint player_profile_update_outbox_status_check
  check (send_status in ('pending', 'sending', 'sent', 'skipped', 'error'));

create unique index if not exists player_profile_update_subscriptions_replacement_operation_uidx
  on public.player_profile_update_subscriptions (replacement_operation_key)
  where replacement_operation_key is not null;

create unique index if not exists player_profile_update_outbox_queue_operation_uidx
  on public.player_profile_update_outbox
    (queue_operation_key, subscription_id, week_start, week_end)
  where queue_operation_key is not null;

create index if not exists player_profile_update_outbox_club_status_created_idx
  on public.player_profile_update_outbox (club_id, send_status, created_at desc);

create table if not exists public.communications_admin_operations (
  operation_key uuid primary key,
  club_id text not null,
  operation_type text not null,
  request_json jsonb not null,
  status text not null default 'started',
  result_json jsonb null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  completed_at timestamptz null,
  constraint communications_admin_operations_status_check
    check (status in ('started', 'completed', 'failed'))
);

create index if not exists communications_admin_operations_club_created_idx
  on public.communications_admin_operations (club_id, created_at desc);

create or replace function public.bump_communications_row_version()
returns trigger
language plpgsql
security invoker
set search_path = pg_catalog, public
as $$
begin
  new.row_version := coalesce(old.row_version, 0) + 1;
  new.updated_at := now();
  return new;
end;
$$;

do $install_version_triggers$
declare
  table_name text;
begin
  foreach table_name in array array[
    'weekly_recaps',
    'player_profile_update_subscriptions',
    'player_weekly_profile_digests',
    'player_profile_update_outbox'
  ]
  loop
    execute format(
      'drop trigger if exists %I on public.%I',
      table_name || '_bump_row_version',
      table_name
    );
    execute format(
      'create trigger %I before update on public.%I for each row execute function public.bump_communications_row_version()',
      table_name || '_bump_row_version',
      table_name
    );
  end loop;
end
$install_version_triggers$;

-- Replacement must be one transaction: there must never be a committed window
-- where the old active subscriber is removed but the replacement insert failed.
create or replace function public.replace_verified_update_subscription(
  p_club_id text,
  p_old_subscription_id uuid,
  p_new_email text,
  p_new_email_normalized text,
  p_new_request_note text,
  p_verified_by text,
  p_admin_note text,
  p_expected_row_version bigint,
  p_operation_key uuid
)
returns jsonb
language plpgsql
security invoker
set search_path = pg_catalog, public
as $$
declare
  old_row public.player_profile_update_subscriptions%rowtype;
  new_row public.player_profile_update_subscriptions%rowtype;
begin
  if p_operation_key is null then
    raise exception 'replacement operation_key is required';
  end if;

  select * into new_row
    from public.player_profile_update_subscriptions
   where replacement_operation_key = p_operation_key
   limit 1;
  if found then
    if new_row.club_id is distinct from p_club_id
       or new_row.replacement_of_subscription_id is distinct from p_old_subscription_id
       or new_row.email_normalized is distinct from p_new_email_normalized
       or coalesce(new_row.request_note, '') is distinct from coalesce(nullif(btrim(p_new_request_note), ''), '')
       or coalesce(new_row.admin_note, '') is distinct from coalesce(nullif(btrim(p_admin_note), ''), '')
       or coalesce(new_row.verified_by, '') is distinct from coalesce(p_verified_by, '') then
      raise exception 'replacement operation_key was already used for another subscription';
    end if;
    return to_jsonb(new_row);
  end if;

  select * into old_row
    from public.player_profile_update_subscriptions
   where id = p_old_subscription_id
     and club_id = p_club_id
   for update;

  if not found then
    raise exception 'active subscription not found';
  end if;
  if old_row.row_version <> p_expected_row_version then
    raise exception 'stale subscription state; reload before replacing';
  end if;
  if old_row.request_status <> 'active' then
    raise exception 'stale subscription state; subscriber is no longer active';
  end if;

  update public.player_profile_update_subscriptions
     set request_status = 'unsubscribed',
         unsubscribed_at = now(),
         admin_note = coalesce(nullif(btrim(p_admin_note), ''), old_row.admin_note)
   where id = old_row.id;

  insert into public.player_profile_update_subscriptions (
    club_id,
    player_id,
    email,
    email_normalized,
    request_status,
    request_note,
    admin_note,
    verified_by,
    verified_at,
    preferences_json,
    replacement_operation_key,
    replacement_of_subscription_id
  ) values (
    old_row.club_id,
    old_row.player_id,
    p_new_email,
    p_new_email_normalized,
    'active',
    nullif(btrim(p_new_request_note), ''),
    nullif(btrim(p_admin_note), ''),
    p_verified_by,
    now(),
    old_row.preferences_json,
    p_operation_key,
    old_row.id
  )
  returning * into new_row;

  update public.player_profile_update_subscriptions
     set superseded_by_subscription_id = new_row.id
   where id = old_row.id;

  return to_jsonb(new_row);
end;
$$;

-- Explicit exposure is required by current Supabase Data API defaults. The
-- service role is the only Data API role allowed to reach these objects.
do $communications_security$
declare
  table_name text;
begin
  foreach table_name in array array[
    'weekly_recaps',
    'player_profile_update_subscriptions',
    'player_weekly_profile_digests',
    'player_profile_update_outbox',
    'communications_admin_operations'
  ]
  loop
    execute format('alter table public.%I enable row level security', table_name);
    execute format('revoke all on table public.%I from public, anon, authenticated', table_name);
    execute format('grant all privileges on table public.%I to service_role', table_name);
  end loop;
end
$communications_security$;

revoke execute on function public.bump_communications_row_version()
  from public, anon, authenticated;
grant execute on function public.bump_communications_row_version()
  to service_role;

revoke execute on function public.replace_verified_update_subscription(
  text, uuid, text, text, text, text, text, bigint, uuid
) from public, anon, authenticated;
grant execute on function public.replace_verified_update_subscription(
  text, uuid, text, text, text, text, text, bigint, uuid
) to service_role;

notify pgrst, 'reload schema';
