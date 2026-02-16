-- V3 Badge Engine Schema Migration
-- Idempotent, Supabase-compatible migration

-- 1. Extend badges_v2
ALTER TABLE public.badges_v2
ADD COLUMN IF NOT EXISTS status TEXT CHECK (status IN ('draft','published','archived')) DEFAULT 'draft',
ADD COLUMN IF NOT EXISTS is_locked BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS award_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ NULL,
ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ NULL,
ADD COLUMN IF NOT EXISTS club_id TEXT NULL,
ADD COLUMN IF NOT EXISTS is_system_badge BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS created_by_admin_id TEXT NULL;

-- 2. Badge Fact Registry
CREATE TABLE IF NOT EXISTS public.badge_fact_registry (
  fact_key TEXT PRIMARY KEY,
  description TEXT NOT NULL,
  data_type TEXT CHECK (data_type IN ('numeric','boolean')) NOT NULL,
  allowed_scope TEXT CHECK (allowed_scope IN ('overall','league','event')) NOT NULL
);

-- 3. Badge Rule Conditions (V3)
CREATE TABLE IF NOT EXISTS public.badge_rule_conditions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  badge_id UUID NOT NULL REFERENCES public.badges_v2(id) ON DELETE CASCADE,
  fact_key TEXT NOT NULL REFERENCES public.badge_fact_registry(fact_key),
  operator TEXT CHECK (operator IN ('>=','>','=','<=','<','is')) NOT NULL,
  value_numeric NUMERIC NULL,
  value_boolean BOOLEAN NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- 4. Club Roles Table (if not already present)
CREATE TABLE IF NOT EXISTS public.club_user_roles (
  club_id TEXT NOT NULL,
  user_id UUID NOT NULL,
  role TEXT CHECK (role IN ('admin','coordinator','score_entry')) NOT NULL,
  assigned_by UUID NULL,
  assigned_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (club_id, user_id)
);

-- 5. Indexes
CREATE INDEX IF NOT EXISTS idx_badge_rule_conditions_badge_id
ON public.badge_rule_conditions (badge_id);

CREATE INDEX IF NOT EXISTS idx_badges_v2_club_id
ON public.badges_v2 (club_id);
