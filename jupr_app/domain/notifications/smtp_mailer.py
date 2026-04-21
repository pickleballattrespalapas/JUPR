from __future__ import annotations

import os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import streamlit as st


def _secret_or_env(name: str, default: str = "") -> str:
    env_val = str(os.getenv(name, "")).strip()
    if env_val:
        return env_val

    try:
        secret_val = st.secrets.get(name, default)
    except Exception:
        return str(default).strip()

    if secret_val is None:
        return str(default).strip()
    return str(secret_val).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    val = _secret_or_env(name).lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "y", "on"}


def _read_smtp_config_raw() -> dict:
    host = _secret_or_env("SMTP_HOST")
    port_raw = _secret_or_env("SMTP_PORT")
    username = _secret_or_env("SMTP_USERNAME")
    password = _secret_or_env("SMTP_PASSWORD")
    from_email = _secret_or_env("SMTP_FROM_EMAIL")
    from_name = _secret_or_env("SMTP_FROM_NAME") or "JUPR"
    use_tls = _env_bool("SMTP_USE_TLS", default=True)

    return {
        "host": host,
        "port_raw": port_raw,
        "username": username,
        "password": password,
        "from_email": from_email,
        "from_name": from_name,
        "use_tls": use_tls,
    }


def get_smtp_config_status() -> dict:
    raw = _read_smtp_config_raw()
    missing = [
        key
        for key, value in [
            ("SMTP_HOST", raw.get("host")),
            ("SMTP_PORT", raw.get("port_raw")),
            ("SMTP_USERNAME", raw.get("username")),
            ("SMTP_PASSWORD", raw.get("password")),
            ("SMTP_FROM_EMAIL", raw.get("from_email")),
        ]
        if not value
    ]

    port: int | None = None
    port_error: str | None = None
    if raw.get("port_raw"):
        try:
            port = int(raw["port_raw"])
        except Exception:
            port_error = "SMTP_PORT must be an integer"

    return {
        "ok": len(missing) == 0 and port_error is None,
        "missing": missing,
        "host": raw.get("host"),
        "port": port,
        "from_email": raw.get("from_email"),
        "from_name": raw.get("from_name"),
        "use_tls": bool(raw.get("use_tls", True)),
        "port_error": port_error,
    }


def _smtp_config_from_env() -> dict:
    status = get_smtp_config_status()
    if status["missing"]:
        raise ValueError(f"Missing SMTP configuration: {', '.join(status['missing'])}")
    if status.get("port_error"):
        raise ValueError(str(status["port_error"]))

    raw = _read_smtp_config_raw()
    return {
        "host": status["host"],
        "port": int(status["port"]),
        "username": raw["username"],
        "password": raw["password"],
        "from_email": status["from_email"],
        "from_name": status["from_name"],
        "use_tls": status["use_tls"],
    }


def send_email_with_inline_chart(
    *,
    to_email: str,
    subject: str,
    html_body: str,
    text_body: str,
    chart_png_bytes: bytes | None = None,
    chart_cid: str | None = None,
) -> str:
    cfg = _smtp_config_from_env()

    msg = MIMEMultipart("related")
    msg["Subject"] = str(subject)
    msg["From"] = f"{cfg['from_name']} <{cfg['from_email']}>"
    msg["To"] = str(to_email).strip()

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text_body or "", "plain", "utf-8"))
    alt.attach(MIMEText(html_body or "", "html", "utf-8"))
    msg.attach(alt)

    if chart_png_bytes:
        image_part = MIMEImage(chart_png_bytes, _subtype="png")
        cid = (chart_cid or "player-digest-chart").strip()
        image_part.add_header("Content-ID", f"<{cid}>")
        image_part.add_header("Content-Disposition", "inline", filename="player-digest-chart.png")
        msg.attach(image_part)

    with smtplib.SMTP(cfg["host"], cfg["port"], timeout=30) as server:
        server.ehlo()
        if cfg["use_tls"]:
            server.starttls()
            server.ehlo()
        server.login(cfg["username"], cfg["password"])
        server.sendmail(cfg["from_email"], [msg["To"]], msg.as_string())

    return "smtp"
