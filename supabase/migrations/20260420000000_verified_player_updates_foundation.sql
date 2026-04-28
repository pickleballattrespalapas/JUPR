-- Backfill foundation for verified player update tables into Supabase-managed migrations.
-- Root migrations existed under migrations/ but Supabase applies only supabase/migrations/.

create table if not exists public.player_profile_update_subscriptions (
    id uuid primary key default gen_random_uuid(),
    club_id text not null,
    player_id bigint not null,
    email text not null,
    email_normalized text not null,
    request_status text not null default 'pending_admin_review',
    request_note text null,
    admin_note text null,
    verified_at timestamptz null,
    verified_by text null,
    unsubscribed_at timestamptz null,
    preferences_json jsonb not null default '{"frequency":"weekly","send_only_if_changed":true}'::jsonb,
    last_digest_week_start date null,
    unsubscribe_token text null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table if exists public.player_profile_update_subscriptions add column if not exists id uuid default gen_random_uuid();
alter table if exists public.player_profile_update_subscriptions add column if not exists club_id text;
alter table if exists public.player_profile_update_subscriptions add column if not exists player_id bigint;
alter table if exists public.player_profile_update_subscriptions add column if not exists email text;
alter table if exists public.player_profile_update_subscriptions add column if not exists email_normalized text;
alter table if exists public.player_profile_update_subscriptions add column if not exists request_status text default 'pending_admin_review';
alter table if exists public.player_profile_update_subscriptions add column if not exists request_note text;
alter table if exists public.player_profile_update_subscriptions add column if not exists admin_note text;
alter table if exists public.player_profile_update_subscriptions add column if not exists verified_at timestamptz;
alter table if exists public.player_profile_update_subscriptions add column if not exists verified_by text;
alter table if exists public.player_profile_update_subscriptions add column if not exists unsubscribed_at timestamptz;
alter table if exists public.player_profile_update_subscriptions add column if not exists preferences_json jsonb default '{"frequency":"weekly","send_only_if_changed":true}'::jsonb;
alter table if exists public.player_profile_update_subscriptions add column if not exists last_digest_week_start date;
alter table if exists public.player_profile_update_subscriptions add column if not exists unsubscribe_token text;
alter table if exists public.player_profile_update_subscriptions add column if not exists created_at timestamptz default now();
alter table if exists public.player_profile_update_subscriptions add column if not exists updated_at timestamptz default now();

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'player_profile_update_subscriptions_pkey'
          and conrelid = 'public.player_profile_update_subscriptions'::regclass
    ) then
        alter table public.player_profile_update_subscriptions
            add constraint player_profile_update_subscriptions_pkey primary key (id);
    end if;
end $$;

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'player_profile_update_subscriptions_status_check'
          and conrelid = 'public.player_profile_update_subscriptions'::regclass
    ) then
        alter table public.player_profile_update_subscriptions
            add constraint player_profile_update_subscriptions_status_check
            check (request_status in ('pending_admin_review', 'active', 'rejected', 'unsubscribed'));
    end if;
end $$;

create index if not exists player_profile_update_subscriptions_club_player_idx
    on public.player_profile_update_subscriptions (club_id, player_id);

create index if not exists player_profile_update_subscriptions_status_created_idx
    on public.player_profile_update_subscriptions (request_status, created_at desc);

create index if not exists player_profile_update_subscriptions_email_norm_idx
    on public.player_profile_update_subscriptions (email_normalized);

create unique index if not exists player_profile_update_subscriptions_open_uidx
    on public.player_profile_update_subscriptions (club_id, player_id)
    where request_status in ('pending_admin_review', 'active');

create unique index if not exists player_profile_update_subscriptions_unsubscribe_token_key
    on public.player_profile_update_subscriptions (unsubscribe_token)
    where unsubscribe_token is not null;

create table if not exists public.player_weekly_profile_digests (
    id uuid primary key default gen_random_uuid(),
    club_id text not null,
    player_id bigint not null,
    week_start date not null,
    week_end date not null,
    generated_json jsonb not null default '{}'::jsonb,
    final_json jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint player_weekly_profile_digests_club_player_window_key
        unique (club_id, player_id, week_start, week_end)
);

alter table if exists public.player_weekly_profile_digests add column if not exists id uuid default gen_random_uuid();
alter table if exists public.player_weekly_profile_digests add column if not exists club_id text;
alter table if exists public.player_weekly_profile_digests add column if not exists player_id bigint;
alter table if exists public.player_weekly_profile_digests add column if not exists week_start date;
alter table if exists public.player_weekly_profile_digests add column if not exists week_end date;
alter table if exists public.player_weekly_profile_digests add column if not exists generated_json jsonb default '{}'::jsonb;
alter table if exists public.player_weekly_profile_digests add column if not exists final_json jsonb default '{}'::jsonb;
alter table if exists public.player_weekly_profile_digests add column if not exists created_at timestamptz default now();
alter table if exists public.player_weekly_profile_digests add column if not exists updated_at timestamptz default now();

alter table if exists public.player_weekly_profile_digests
    drop constraint if exists player_weekly_profile_digests_club_id_player_id_week_start_key;

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'player_weekly_profile_digests_club_player_window_key'
          and conrelid = 'public.player_weekly_profile_digests'::regclass
    ) then
        alter table public.player_weekly_profile_digests
            add constraint player_weekly_profile_digests_club_player_window_key
            unique (club_id, player_id, week_start, week_end);
    end if;
end $$;

create table if not exists public.player_profile_update_outbox (
    id uuid primary key default gen_random_uuid(),
    subscription_id uuid not null references public.player_profile_update_subscriptions(id) on delete cascade,
    club_id text not null,
    player_id bigint not null,
    week_start date not null,
    week_end date not null,
    email text not null,
    send_status text not null default 'pending',
    provider_message_id text null,
    sent_at timestamptz null,
    error_text text null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint player_profile_update_outbox_status_check
        check (send_status in ('pending', 'sent', 'skipped', 'error'))
);

alter table if exists public.player_profile_update_outbox add column if not exists id uuid default gen_random_uuid();
alter table if exists public.player_profile_update_outbox add column if not exists subscription_id uuid;
alter table if exists public.player_profile_update_outbox add column if not exists club_id text;
alter table if exists public.player_profile_update_outbox add column if not exists player_id bigint;
alter table if exists public.player_profile_update_outbox add column if not exists week_start date;
alter table if exists public.player_profile_update_outbox add column if not exists week_end date;
alter table if exists public.player_profile_update_outbox add column if not exists email text;
alter table if exists public.player_profile_update_outbox add column if not exists send_status text default 'pending';
alter table if exists public.player_profile_update_outbox add column if not exists provider_message_id text;
alter table if exists public.player_profile_update_outbox add column if not exists sent_at timestamptz;
alter table if exists public.player_profile_update_outbox add column if not exists error_text text;
alter table if exists public.player_profile_update_outbox add column if not exists created_at timestamptz default now();
alter table if exists public.player_profile_update_outbox add column if not exists updated_at timestamptz default now();

alter table if exists public.player_profile_update_outbox
    drop constraint if exists player_profile_update_outbox_subscription_window_key;
alter table if exists public.player_profile_update_outbox
    drop constraint if exists player_profile_update_outbox_subscription_id_week_start_key;

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'player_profile_update_outbox_subscription_id_fkey'
          and conrelid = 'public.player_profile_update_outbox'::regclass
    ) then
        alter table public.player_profile_update_outbox
            add constraint player_profile_update_outbox_subscription_id_fkey
            foreign key (subscription_id)
            references public.player_profile_update_subscriptions(id)
            on delete cascade;
    end if;
end $$;

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'player_profile_update_outbox_status_check'
          and conrelid = 'public.player_profile_update_outbox'::regclass
    ) then
        alter table public.player_profile_update_outbox
            add constraint player_profile_update_outbox_status_check
            check (send_status in ('pending', 'sent', 'skipped', 'error'));
    end if;
end $$;

drop index if exists public.player_profile_update_outbox_subscription_week_idx;
create index if not exists player_profile_update_outbox_subscription_week_window_idx
    on public.player_profile_update_outbox (subscription_id, week_start, week_end);

create or replace function public.set_updated_at_timestamp()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

drop trigger if exists player_profile_update_subscriptions_set_updated_at on public.player_profile_update_subscriptions;
create trigger player_profile_update_subscriptions_set_updated_at
before update on public.player_profile_update_subscriptions
for each row
execute function public.set_updated_at_timestamp();

drop trigger if exists player_weekly_profile_digests_set_updated_at on public.player_weekly_profile_digests;
create trigger player_weekly_profile_digests_set_updated_at
before update on public.player_weekly_profile_digests
for each row
execute function public.set_updated_at_timestamp();

drop trigger if exists player_profile_update_outbox_set_updated_at on public.player_profile_update_outbox;
create trigger player_profile_update_outbox_set_updated_at
before update on public.player_profile_update_outbox
for each row
execute function public.set_updated_at_timestamp();
