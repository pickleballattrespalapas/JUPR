from datetime import datetime, timezone

def dt_utc_now() -> datetime:
    """Timezone-aware UTC 'now'."""
    return datetime.now(timezone.utc)

def month_key_utc(dt: datetime) -> str:
    """YYYY-MM month key in UTC (used for Pass usage logic)."""
    d = dt.astimezone(timezone.utc)
    return f"{d.year:04d}-{d.month:02d}"
