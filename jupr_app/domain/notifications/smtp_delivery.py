"""SMTP outcomes safe to persist without provider text, credentials or message bodies."""
from __future__ import annotations

import re
import smtplib
import ssl
from typing import Any


class SMTPDeliveryError(RuntimeError):
    def __init__(self, details: dict[str, Any]):
        self.details = dict(details)
        super().__init__("SMTP delivery " + ", ".join(f"{key}={value}" for key, value in details.items()))


def smtp_delivery_failure(
    exc: Exception, *, stage: str = "unknown", attempt_count: int = 1,
) -> SMTPDeliveryError:
    """Only explicit rejection or an error before submission proves non-acceptance."""
    replies = []
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        replies = list(exc.recipients.values())
    elif isinstance(exc, smtplib.SMTPResponseException):
        replies = [(exc.smtp_code, exc.smtp_error)]
    codes = [int(code) for code, _reply in replies if isinstance(code, int) and 100 <= code <= 599]
    enhanced_status = None
    for _code, reply in replies:
        raw = reply if isinstance(reply, bytes) else str(reply).encode("utf-8")
        match = re.search(rb"(?<![\d.])([245]\.\d{1,3}\.\d{1,3})(?![\d.])", raw[:1024])
        if match:
            enhanced_status = match.group(1).decode("ascii")
            break
    before_submission = stage in {"connect", "greeting", "starttls", "authentication"}
    explicit_rejection = isinstance(exc, (
        smtplib.SMTPSenderRefused, smtplib.SMTPRecipientsRefused, smtplib.SMTPDataError,
    ))
    not_accepted = before_submission or explicit_rejection
    temporary_response = bool(codes) and len(codes) == len(replies) and all(400 <= code < 500 for code in codes)
    # SMTPException itself inherits OSError; a permanent SMTP response is not
    # a transient socket failure merely because it shares that base class.
    temporary_connection = before_submission and (
        isinstance(exc, smtplib.SMTPServerDisconnected)
        or (isinstance(exc, OSError) and not isinstance(exc, (smtplib.SMTPException, ssl.SSLError)))
    )
    details: dict[str, Any] = {
        "stage": stage,
        "error_type": type(exc).__name__,
        "delivery_outcome": "not_accepted" if not_accepted else "unknown",
        "retryable": not_accepted and (temporary_response or temporary_connection),
        "attempt_count": attempt_count,
    }
    if codes:
        details["smtp_codes"] = sorted(set(codes))
    if enhanced_status:
        details["enhanced_status"] = enhanced_status
    return SMTPDeliveryError(details)


def smtp_failure_details(exc: Exception) -> dict[str, Any]:
    return dict(exc.details) if isinstance(exc, SMTPDeliveryError) else smtp_delivery_failure(exc).details
