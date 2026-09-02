-- `deleted_at` is the canonical match lifecycle state. A short-lived staging
-- repair added `is_active`, which would create conflicting deletion state.
-- The staging table is empty, so remove only that newly introduced column.

alter table if exists public.matches
  drop column if exists is_active;
