alter table if exists public.admin_role_assignments
    add column if not exists club_id text;

update public.admin_role_assignments
set club_id = 'tres_palapas'
where club_id is null or btrim(club_id) = '';

alter table public.admin_role_assignments
    alter column club_id set not null;

alter table public.admin_role_assignments
    drop constraint if exists admin_role_assignments_email_unique;

alter table public.admin_role_assignments
    add constraint admin_role_assignments_club_email_unique unique (club_id, email);

create index if not exists admin_role_assignments_club_email_idx
    on public.admin_role_assignments (club_id, email);

create index if not exists admin_role_assignments_club_user_id_idx
    on public.admin_role_assignments (club_id, user_id);
