--
-- Schema version lock workflow:
-- 1) Add and apply all schema-changing migrations for the new release.
-- 2) Update REQUIRED_SCHEMA_VERSION in jupr_app/data/schema_preflight.py.
-- 3) Add a new migration that inserts the new version value into schema_version.
-- 4) Deploy app + migrations together so startup validation passes.
--
create table if not exists schema_version (
  version text primary key,
  applied_at timestamptz not null default now()
);

insert into schema_version (version)
values ('rebuild_phase1_alignment')
on conflict (version)
do update set applied_at = now();
