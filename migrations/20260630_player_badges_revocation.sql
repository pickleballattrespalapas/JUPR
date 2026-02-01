alter table if exists public.player_badges
    add column if not exists revoked_at timestamptz,
    add column if not exists revoked_by text,
    add column if not exists revoke_reason text;
