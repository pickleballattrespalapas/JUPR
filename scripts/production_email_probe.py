"""Verify production SMTP authentication without delivering or queueing mail.

The deploy workflow runs this source on the existing Fly image before changing
its configuration. Keep it compatible with the pre-email production image.
Never print credentials or raw provider responses.
"""
from __future__ import annotations

import json
import os
import smtplib
import ssl
from dataclasses import asdict

from jupr_app.config import get_smtp_config


REQUIRED_SMTP_NAMES = (
    "SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM_EMAIL",
)


def probe_production_email() -> dict:
    if (
        os.getenv("JUPR_ENV") != "production"
        or os.getenv("FLY_APP_NAME") != "juprleagues-api"
        or os.getenv("SUPABASE_URL", "").rstrip("/")
        != "https://dnoockbwfenunhcibwfn.supabase.co"
    ):
        return {"ok": False, "stage": "identity", "error": "Production identity mismatch"}
    missing = [name for name in REQUIRED_SMTP_NAMES if not os.getenv(name, "").strip()]
    if missing:
        return {"ok": False, "stage": "configuration", "missing": missing}
    try:
        cfg = asdict(get_smtp_config())
        if not 1 <= cfg["port"] <= 65535:
            raise ValueError("Invalid port")
        if cfg["port"] != 465 and not cfg["use_tls"]:
            return {"ok": False, "stage": "configuration", "error": "SMTP must use TLS"}
        if "@" not in cfg["from_email"] or any(c in cfg["from_email"] for c in "\r\n"):
            raise ValueError("Invalid sender")
    except Exception as exc:
        return {"ok": False, "stage": "configuration", "error_type": type(exc).__name__}
    implicit_tls = cfg["port"] == 465
    transport = "tls" if implicit_tls else "starttls"
    stage = "connect"
    try:
        smtp_class = smtplib.SMTP_SSL if implicit_tls else smtplib.SMTP
        options = {"timeout": 20}
        if implicit_tls:
            options["context"] = ssl.create_default_context()
        with smtp_class(cfg["host"], cfg["port"], **options) as server:
            server.ehlo()
            if not implicit_tls:
                stage = "tls"
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
            stage = "authentication"
            server.login(cfg["username"], cfg["password"])
        return {"ok": True, "stage": "authenticated", "transport": transport, "messages_sent": 0}
    except Exception as exc:
        result = {"ok": False, "stage": stage, "error_type": type(exc).__name__}
        if isinstance(exc, smtplib.SMTPResponseException):
            result["smtp_code"] = int(exc.smtp_code)
        return result


def main() -> int:
    result = probe_production_email()
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
