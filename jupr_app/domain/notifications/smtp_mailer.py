from __future__ import annotations

import logging
import smtplib
import ssl
import time
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid

from jupr_app.config import SMTPConfig, get_env_or_default, get_smtp_config
from jupr_app.domain.notifications.smtp_delivery import smtp_delivery_failure


logger = logging.getLogger(__name__)
SMTP_RETRY_BUDGET_SECONDS = 10


def _send_smtp_message(cfg: dict, recipient: str, payload: str) -> None:
    # A retry reuses the same MIME payload and Message-ID. Never repeat an
    # interrupted DATA submission: the provider might already have accepted it.
    started_at = time.monotonic()
    for attempt in (1, 2):
        server = None
        stage = "connect"
        failure = None
        accepted = False
        try:
            implicit_tls = cfg["port"] == 465
            smtp_class = smtplib.SMTP_SSL if implicit_tls else smtplib.SMTP
            options = {"timeout": 30}
            if implicit_tls:
                options["context"] = ssl.create_default_context()
            server = smtp_class(cfg["host"], cfg["port"], **options)
            stage = "greeting"
            server.ehlo()
            if cfg["use_tls"] and not implicit_tls:
                stage = "starttls"
                server.starttls(context=ssl.create_default_context())
                stage = "greeting"
                server.ehlo()
            stage = "authentication"
            server.login(cfg["username"], cfg["password"])
            stage = "message"
            server.sendmail(cfg["from_email"], [recipient], payload)
            accepted = True
        except Exception as exc:
            failure = smtp_delivery_failure(exc, stage=stage, attempt_count=attempt)
        finally:
            if server is not None:
                # QUIT cannot revoke DATA acceptance or replace the original
                # rejection with a misleading SMTPResponseException.
                try:
                    response = server.quit()
                    if response is not None and response[0] != 221:
                        raise smtplib.SMTPResponseException(*response)
                except (smtplib.SMTPException, OSError) as exc:
                    logger.warning("SMTP cleanup (accepted=%s): %s", accepted, smtp_delivery_failure(exc, stage="quit", attempt_count=attempt))
                finally:
                    try:
                        server.close()
                    except (smtplib.SMTPException, OSError):
                        pass
        if failure is None:
            return
        logger.warning("%s", failure)
        if attempt == 1 and failure.details["retryable"] and time.monotonic() - started_at < SMTP_RETRY_BUDGET_SECONDS:
            time.sleep(1)
            continue
        raise failure from None


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
    message_id: str | None = None,
    inline_png_images: dict[str, bytes] | None = None,
    reply_to: str | None = None,
) -> str:
    cfg = _smtp_config_dict(smtp_config)

    msg = MIMEMultipart("related")
    msg["Subject"] = str(subject)
    msg["From"] = f"{cfg['from_name']} <{cfg['from_email']}>"
    msg["To"] = str(to_email).strip()
    clean_message_id = str(message_id or "").strip().strip("<>")
    sender_domain = str(cfg["from_email"]).rsplit("@", 1)[-1]
    rfc_message_id = f"<{clean_message_id}@{sender_domain}>" if clean_message_id else make_msgid(domain=sender_domain)
    msg["Message-ID"] = rfc_message_id
    msg["Date"] = formatdate(localtime=False, usegmt=True)
    reply_address = str(reply_to or cfg.get("reply_to") or "").strip()
    if reply_address:
        if "\r" in reply_address or "\n" in reply_address:
            raise ValueError("Invalid reply address.")
        msg["Reply-To"] = reply_address

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

    for cid, data in (inline_png_images or {}).items():
        image_part = MIMEImage(data, _subtype="png")
        image_part.add_header("Content-ID", f"<{cid}>")
        image_part.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        msg.attach(image_part)

    _send_smtp_message(cfg, msg["To"], msg.as_string())

    return rfc_message_id
