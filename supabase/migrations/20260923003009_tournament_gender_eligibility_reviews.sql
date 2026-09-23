-- Eligibility decisions stay private to the authenticated admin API.
create table if not exists public.tournament_gender_eligibility_reviews (
    fingerprint text primary key check (fingerprint ~ '^[a-f0-9]{64}$'),
    tournament_id uuid not null references public.tournaments(id) on delete cascade,
    event_option_id text not null references public.tournament_event_options(id) on delete cascade,
    decision text not null check (decision in ('APPROVED', 'DECLINED')),
    reviewed_by text not null,
    reviewed_at timestamptz not null default now()
);
create index if not exists tournament_gender_eligibility_reviews_tournament_idx
    on public.tournament_gender_eligibility_reviews(tournament_id);
alter table public.tournament_gender_eligibility_reviews enable row level security;
revoke all on public.tournament_gender_eligibility_reviews from public, anon, authenticated;
grant select, insert, update, delete on public.tournament_gender_eligibility_reviews to service_role;
