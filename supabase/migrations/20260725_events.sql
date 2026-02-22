-- Events table for popup round robins and other club events

CREATE TABLE IF NOT EXISTS public.events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  club_id text NOT NULL,
  name text NOT NULL,
  event_type text NOT NULL DEFAULT 'popup_rr',
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  starts_at timestamptz NULL,
  ends_at timestamptz NULL,
  notes text NULL
);

CREATE INDEX IF NOT EXISTS idx_events_club_active ON public.events (club_id, is_active);
CREATE INDEX IF NOT EXISTS idx_events_club_name ON public.events (club_id, name);
