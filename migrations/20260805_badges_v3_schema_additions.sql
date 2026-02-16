alter table badges_v2
    add column if not exists status text check (status in ('draft', 'published', 'archived')) default 'draft',
    add column if not exists is_locked boolean default false,
    add column if not exists award_count integer default 0,
    add column if not exists published_at timestamp null,
    add column if not exists archived_at timestamp null,
    add column if not exists club_id text null,
    add column if not exists is_system_badge boolean default false,
    add column if not exists created_by_admin_id text null;

create table if not exists badge_fact_registry (
    fact_key text primary key,
    description text not null,
    data_type text check (data_type in ('numeric', 'boolean')) not null,
    allowed_scope text check (allowed_scope in ('overall', 'league', 'event')) not null
);

create table if not exists badge_rule_conditions (
    id uuid primary key default gen_random_uuid(),
    badge_id uuid references badges_v2(id),
    fact_key text not null references badge_fact_registry(fact_key),
    operator text check (operator in ('>=', '>', '=', '<=', '<', 'is')) not null,
    value_numeric numeric null,
    value_boolean boolean null,
    created_at timestamp with time zone default now()
);
