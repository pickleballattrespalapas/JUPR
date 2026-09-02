-- Follow-up for environments where 20261108014000 was already applied.
-- Recreate the generic projection repair with immutable-review exclusion,
-- same-club source proof, and a durable server-only reconciliation queue.

create table if not exists public.tournament_registration_player_projection_reconciliation (
  projection_kind text not null check (
    projection_kind in (
      'TEAM_MEMBER',
      'TEAM_LINK_PLAYER1',
      'TEAM_LINK_PLAYER2'
    )
  ),
  relationship_id text not null,
  tournament_id uuid not null
    references public.tournaments(id) on delete cascade,
  registration_id text not null
    references public.tournament_registrations(id) on delete cascade,
  selection_id text not null
    references public.tournament_registration_selections(id) on delete cascade,
  registration_player_id integer null
    references public.players(id) on delete set null,
  reason_code text not null check (
    reason_code in (
      'FINALIZED_REGISTRATION_CLOSE_REVIEW',
      'REGISTRATION_PLAYER_CLUB_MISMATCH'
    )
  ),
  state text not null default 'PENDING' check (
    state in ('PENDING', 'RESOLVED', 'DISMISSED')
  ),
  created_at timestamptz not null default pg_catalog.clock_timestamp(),
  updated_at timestamptz not null default pg_catalog.clock_timestamp(),
  primary key (projection_kind, relationship_id)
);

create index if not exists tournament_player_projection_reconciliation_tournament_idx
  on public.tournament_registration_player_projection_reconciliation (
    tournament_id, state
  );
create index if not exists tournament_player_projection_reconciliation_registration_idx
  on public.tournament_registration_player_projection_reconciliation (
    registration_id
  );
create index if not exists tournament_player_projection_reconciliation_selection_idx
  on public.tournament_registration_player_projection_reconciliation (
    selection_id
  );
create index if not exists tournament_player_projection_reconciliation_player_idx
  on public.tournament_registration_player_projection_reconciliation (
    registration_player_id
  );

alter table public.tournament_registration_player_projection_reconciliation
  enable row level security;
alter table public.tournament_registration_player_projection_reconciliation
  force row level security;
revoke all on table public.tournament_registration_player_projection_reconciliation
  from public, anon, authenticated, service_role;
grant select, insert, update
  on table public.tournament_registration_player_projection_reconciliation
  to service_role;

comment on table public.tournament_registration_player_projection_reconciliation is
  'Server-only queue for null tournament team player projections that cannot be safely filled because finalized rating evidence is immutable or the registration player belongs to another club.';

-- Registrations linked above are the canonical player identity for their
-- same-tournament partner-team projections. Repair only live null projections;
-- never rewrite an existing member/link player, even when it disagrees.
-- Finalized relationship reviews are immutable, and cross-club registration
-- player links are not trusted as projection sources; queue both for staff.
with projection_candidates (
  projection_kind,
  relationship_id,
  tournament_id,
  registration_id,
  selection_id,
  trigger_selection_ids,
  registration_player_id
) as (
  select
    'TEAM_MEMBER',
    member.id,
    member.tournament_id,
    member.registration_id,
    member.selection_id,
    array[member.selection_id::text],
    registration.player_id
  from public.tournament_registration_team_members as member
  join public.tournament_registrations as registration
    on registration.id = member.registration_id
   and registration.tournament_id = member.tournament_id::text
  where member.player_id is null
    and member.status = 'ACTIVE'
    and registration.player_id is not null

  union all

  select
    'TEAM_LINK_PLAYER1',
    link.id,
    link.tournament_id,
    link.registration1_id,
    link.selection1_id,
    array[link.selection1_id::text, link.selection2_id::text],
    registration.player_id
  from public.tournament_registration_team_links as link
  join public.tournament_registrations as registration
    on registration.id = link.registration1_id
   and registration.tournament_id = link.tournament_id::text
  where link.player1_id is null
    and link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
    and registration.player_id is not null

  union all

  select
    'TEAM_LINK_PLAYER2',
    link.id,
    link.tournament_id,
    link.registration2_id,
    link.selection2_id,
    array[link.selection1_id::text, link.selection2_id::text],
    registration.player_id
  from public.tournament_registration_team_links as link
  join public.tournament_registrations as registration
    on registration.id = link.registration2_id
   and registration.tournament_id = link.tournament_id::text
  where link.player2_id is null
    and link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
    and registration.player_id is not null
)
insert into public.tournament_registration_player_projection_reconciliation (
  projection_kind,
  relationship_id,
  tournament_id,
  registration_id,
  selection_id,
  registration_player_id,
  reason_code,
  state,
  updated_at
)
select
  candidate.projection_kind,
  candidate.relationship_id,
  candidate.tournament_id,
  candidate.registration_id,
  candidate.selection_id,
  candidate.registration_player_id,
  case
    when player.id is null
      or player.club_id is distinct from tournament.club_id
      then 'REGISTRATION_PLAYER_CLUB_MISMATCH'
    else 'FINALIZED_REGISTRATION_CLOSE_REVIEW'
  end,
  'PENDING',
  pg_catalog.clock_timestamp()
from projection_candidates as candidate
join public.tournaments as tournament
  on tournament.id = candidate.tournament_id
left join public.players as player
  on player.id = candidate.registration_player_id
where player.id is null
   or player.club_id is distinct from tournament.club_id
   or exists (
     select 1
       from public.tournament_rating_eligibility_reviews as review
      where review.selection_id::text = any(candidate.trigger_selection_ids)
        and review.review_phase = 'REGISTRATION_CLOSE'
        and review.finalized_at is not null
   )
on conflict (projection_kind, relationship_id) do update set
  tournament_id = excluded.tournament_id,
  registration_id = excluded.registration_id,
  selection_id = excluded.selection_id,
  registration_player_id = excluded.registration_player_id,
  reason_code = excluded.reason_code,
  state = 'PENDING',
  updated_at = excluded.updated_at;

update public.tournament_registration_team_members as member
   set player_id = registration.player_id
  from public.tournament_registrations as registration
  join public.tournaments as tournament
    on tournament.id::text = registration.tournament_id
  join public.players as player
    on player.id = registration.player_id
   and player.club_id = tournament.club_id
 where member.player_id is null
   and member.status = 'ACTIVE'
   and registration.id = member.registration_id
   and registration.tournament_id = member.tournament_id::text
   and registration.player_id is not null
   and not exists (
     select 1
       from public.tournament_rating_eligibility_reviews as review
      where review.selection_id::text = member.selection_id::text
        and review.review_phase = 'REGISTRATION_CLOSE'
        and review.finalized_at is not null
   );

update public.tournament_registration_team_links as link
   set player1_id = registration.player_id
  from public.tournament_registrations as registration
  join public.tournaments as tournament
    on tournament.id::text = registration.tournament_id
  join public.players as player
    on player.id = registration.player_id
   and player.club_id = tournament.club_id
 where link.player1_id is null
   and link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
   and registration.id = link.registration1_id
   and registration.tournament_id = link.tournament_id::text
   and registration.player_id is not null
   and not exists (
     select 1
       from public.tournament_rating_eligibility_reviews as review
      where review.selection_id::text in (
        link.selection1_id::text,
        link.selection2_id::text
      )
        and review.review_phase = 'REGISTRATION_CLOSE'
        and review.finalized_at is not null
   );

update public.tournament_registration_team_links as link
   set player2_id = registration.player_id
  from public.tournament_registrations as registration
  join public.tournaments as tournament
    on tournament.id::text = registration.tournament_id
  join public.players as player
    on player.id = registration.player_id
   and player.club_id = tournament.club_id
 where link.player2_id is null
   and link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
   and registration.id = link.registration2_id
   and registration.tournament_id = link.tournament_id::text
   and registration.player_id is not null
   and not exists (
     select 1
       from public.tournament_rating_eligibility_reviews as review
      where review.selection_id::text in (
        link.selection1_id::text,
        link.selection2_id::text
      )
        and review.review_phase = 'REGISTRATION_CLOSE'
        and review.finalized_at is not null
   );

notify pgrst, 'reload schema';
