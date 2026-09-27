-- RakshaNet Pulse 2.0 / SIH26069 central event store
-- Run this once in your PostgreSQL/Supabase SQL editor.
CREATE TABLE IF NOT EXISTS public.pulse_events (
    id TEXT PRIMARY KEY,
    fingerprint TEXT UNIQUE NOT NULL,
    source_type TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_url TEXT,
    source_id TEXT,
    title TEXT,
    content TEXT,
    author TEXT,
    published_at TIMESTAMPTZ,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    city TEXT,
    district TEXT,
    state TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    event_category TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    trust_score INTEGER NOT NULL DEFAULT 0,
    duplicate_of TEXT,
    media JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS pulse_events_published_idx ON public.pulse_events (published_at DESC);
CREATE INDEX IF NOT EXISTS pulse_events_category_idx ON public.pulse_events (event_category);
CREATE INDEX IF NOT EXISTS pulse_events_state_idx ON public.pulse_events (state);
CREATE INDEX IF NOT EXISTS pulse_events_verification_idx ON public.pulse_events (verification_status);
CREATE INDEX IF NOT EXISTS pulse_events_source_type_idx ON public.pulse_events (source_type);
