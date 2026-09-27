-- RakshaNet early-warning persistence reference.
-- server.js also attempts to create these tables automatically.

CREATE TABLE IF NOT EXISTS public.push_subscriptions (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT,
    endpoint TEXT UNIQUE NOT NULL,
    subscription JSONB NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.alert_deliveries (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT,
    alert_key TEXT NOT NULL,
    channel TEXT NOT NULL,
    status TEXT NOT NULL,
    provider TEXT,
    destination TEXT,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alert_deliveries_key_time
ON public.alert_deliveries(alert_key, created_at DESC);
