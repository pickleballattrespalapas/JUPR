from __future__ import annotations

import os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def _env_bool(name: str, default: bool = False) -> bool:
    val = str(os.getenv(name, "")).strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "y", "on"}


def _smtp_config_from_env() -> dict:
    host = str(os.getenv("SMTP_HOST", "")).strip()
    port_raw = str(os.getenv("SMTP_PORT", "")).strip()
    username = str(os.getenv("SMTP_USERNAME", "")).strip()
    password = str(os.getenv("SMTP_PASSWORD", "")).strip()
    from_email = str(os.getenv("SMTP_FROM_EMAIL", "")).strip()
    from_name = str(os.getenv("SMTP_FROM_NAME", "")).strip() or "JUPR"
    use_tls = _env_bool("SMTP_USE_TLS", default=True)

    missing = [
        name
        for name, value in [
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

    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "from_email": from_email,
        "from_name": from_name,
        "use_tls": use_tls,
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
