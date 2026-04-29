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
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint player_profile_update_subscriptions_status_check
        check (request_status in ('pending_admin_review', 'active', 'rejected', 'unsubscribed'))
);

create index if not exists player_profile_update_subscriptions_club_player_idx
    on public.player_profile_update_subscriptions (club_id, player_id);

create index if not exists player_profile_update_subscriptions_status_created_idx
    on public.player_profile_update_subscriptions (request_status, created_at desc);

create index if not exists player_profile_update_subscriptions_email_norm_idx
    on public.player_profile_update_subscriptions (email_normalized);

create unique index if not exists player_profile_update_subscriptions_open_uidx
    on public.player_profile_update_subscriptions (club_id, player_id)
    where request_status in ('pending_admin_review', 'active');

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
    unique (club_id, player_id, week_start)
);

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
        check (send_status in ('pending', 'sent', 'skipped', 'error')),
    unique (subscription_id, week_start)
);

create index if not exists player_profile_update_outbox_subscription_week_idx
    on public.player_profile_update_outbox (subscription_id, week_start);

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
