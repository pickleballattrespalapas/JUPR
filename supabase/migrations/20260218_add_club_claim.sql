-- Add club_id column to auth.users metadata if needed
-- We will store club_id in user_metadata

-- Create helper function to expose club_id claim
create or replace function public.jwt_claims()
returns jsonb
language sql stable
as $$
  select
    coalesce(
      current_setting('request.jwt.claims', true)::jsonb,
      '{}'::jsonb
    );
$$;

-- Ensure club_id is accessible from JWT:
-- RLS will reference:
-- (public.jwt_claims() ->> 'club_id')

-- NOTE:
-- Actual club_id must be set per user in Supabase dashboard:
-- Auth → Users → user_metadata → { "club_id": "tres_palapas" }
