from __future__ import annotations

import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from jupr_app.config import SMTPConfig, get_env_or_default, get_smtp_config


def _env_bool(name: str, default: bool = False) -> bool:
    value = get_env_or_default(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def get_smtp_config_status() -> dict:
    missing: list[str] = []
    host = get_env_or_default("SMTP_HOST")
    port_raw = get_env_or_default("SMTP_PORT")
    username = get_env_or_default("SMTP_USERNAME")
    password = get_env_or_default("SMTP_PASSWORD")
    from_email = get_env_or_default("SMTP_FROM_EMAIL")
    from_name = get_env_or_default("SMTP_FROM_NAME", "JUPR Notifications")
    reply_to = get_env_or_default("SMTP_REPLY_TO", "joe@juprleagues.com")
    use_tls = _env_bool("SMTP_USE_TLS", default=True)

    for key, value in (
        ("SMTP_HOST", host),
        ("SMTP_PORT", port_raw),
        ("SMTP_USERNAME", username),
        ("SMTP_PASSWORD", password),
        ("SMTP_FROM_EMAIL", from_email),
    ):
        if not value:
            missing.append(key)

    port: int | None = None
    port_error: str | None = None
    if port_raw:
        try:
            port = int(port_raw)
        except Exception:
            port_error = "SMTP_PORT must be an integer"

    return {
        "ok": len(missing) == 0 and port_error is None,
        "missing": missing,
        "host": host,
        "port": port,
        "from_email": from_email,
        "from_name": from_name,
        "reply_to": reply_to,
        "reply_to_configured": bool(reply_to),
        "use_tls": use_tls,
        "port_error": port_error,
    }


def _smtp_config_dict(smtp_config: SMTPConfig | None = None) -> dict:
    cfg = smtp_config or get_smtp_config()
    return {
        "host": cfg.host,
        "port": cfg.port,
        "username": cfg.username,
        "password": cfg.password,
        "from_email": cfg.from_email,
        "from_name": cfg.from_name,
        "reply_to": cfg.reply_to,
        "use_tls": cfg.use_tls,
    }


def send_email_with_inline_chart(
    *,
    to_email: str,
    subject: str,
    html_body: str,
    text_body: str,
    chart_png_bytes: bytes | None = None,
    chart_cid: str | None = None,
    unsubscribe_url: str | None = None,
    smtp_config: SMTPConfig | None = None,
) -> str:
    cfg = _smtp_config_dict(smtp_config)

    msg = MIMEMultipart("related")
    msg["Subject"] = str(subject)
    msg["From"] = f"{cfg['from_name']} <{cfg['from_email']}>"
    msg["To"] = str(to_email).strip()
    if cfg.get("reply_to"):
        msg["Reply-To"] = str(cfg["reply_to"]).strip()

    normalized_unsubscribe_url = str(unsubscribe_url or "").strip()
    if normalized_unsubscribe_url:
        msg["List-Unsubscribe"] = f"<{normalized_unsubscribe_url}>"
        msg["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"

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
