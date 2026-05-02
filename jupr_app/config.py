from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SMTPConfig:
    host: str
    port: int
    username: str
    password: str
    from_email: str
    from_name: str
    reply_to: str
    use_tls: bool


def get_env_or_default(name: str, default: str = "") -> str:
    value = str(os.getenv(name, "")).strip()
    if value:
        return value
    return str(default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    value = get_env_or_default(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def get_public_base_url(default: str = "http://localhost:8501") -> str:
    for env_name in ("JUPR_PUBLIC_BASE_URL", "PUBLIC_BASE_URL"):
        value = get_env_or_default(env_name)
        if value:
            return value.rstrip("/")
    return str(default).strip().rstrip("/")


def get_smtp_config() -> SMTPConfig:
    host = get_env_or_default("SMTP_HOST")
    port_raw = get_env_or_default("SMTP_PORT")
    username = get_env_or_default("SMTP_USERNAME")
    password = get_env_or_default("SMTP_PASSWORD")
    from_email = get_env_or_default("SMTP_FROM_EMAIL")
    from_name = get_env_or_default("SMTP_FROM_NAME", "JUPR Notifications")
    reply_to = get_env_or_default("SMTP_REPLY_TO", "joe@juprleagues.com")
    use_tls = _env_bool("SMTP_USE_TLS", default=True)

    missing = [
        key
        for key, value in [
            ("SMTP_HOST", host),
            ("SMTP_PORT", port_raw),
            ("SMTP_USERNAME", username),
            ("SMTP_PASSWORD", password),
            ("SMTP_FROM_EMAIL", from_email),
        ]
        if not value
    ]
    if missing:
        raise ValueError(f"Missing SMTP configuration: {', '.join(missing)}")

    try:
        port = int(port_raw)
    except Exception as exc:
        raise ValueError("SMTP_PORT must be an integer") from exc

    return SMTPConfig(
        host=host,
        port=port,
        username=username,
        password=password,
        from_email=from_email,
        from_name=from_name,
        reply_to=reply_to,
        use_tls=use_tls,
    )
