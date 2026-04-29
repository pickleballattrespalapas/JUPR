alter table if exists public.player_profile_update_subscriptions
    add column if not exists unsubscribe_token text;

update public.player_profile_update_subscriptions
set unsubscribe_token = md5(random()::text || clock_timestamp()::text || coalesce(id::text, ''))
where unsubscribe_token is null;

create unique index if not exists player_profile_update_subscriptions_unsubscribe_token_key
    on public.player_profile_update_subscriptions (unsubscribe_token)
    where unsubscribe_token is not null;
