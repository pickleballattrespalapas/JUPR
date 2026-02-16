-- Remove legacy V2 badge rules table after migration to Badge Engine V3.

drop table if exists public.badge_rules_v2;
