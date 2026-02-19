create table if not exists schema_version (
  version text primary key,
  applied_at timestamptz not null default now()
);

insert into schema_version (version)
values ('rebuild_phase1_alignment')
on conflict do nothing;
