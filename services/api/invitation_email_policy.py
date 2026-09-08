"""Short-lived, recipient-bound invitation email testing on isolated staging."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re

from jupr_app.config import (
    EMAIL_MODE_DRY_RUN, EMAIL_MODE_LIVE, get_env_or_default, get_jupr_env,
    get_next_web_base_url,
)
from jupr_app.domain.notifications.smtp_mailer import get_smtp_config_status


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config/staging_invitation_email_test.json"
STAGING_WEB = "https://jupr-git-staging-pickleballattrespalapas1.vercel.app"
STAGING_AUTH = "https://sijpxjxvdtrehmqvirfi.supabase.co"


@dataclass(frozen=True)
class InvitationEmailPolicy:
    enabled: bool = False
    test_mode: bool = False
    recipients: frozenset[str] = frozenset()

    def allows(self, email: str) -> bool:
        return self.enabled and (not self.test_mode or email.strip().lower() in self.recipients)

    def public_options(self) -> dict:
        # Never expose addresses, account existence, or an invitation lookup.
        return {
            "email_enabled": self.enabled,
            **({"email_test_mode": True} if self.test_mode else {}),
        }


def _test_config(now: datetime) -> tuple[frozenset[str], dict]:
    status = {"active": False, "recipient_count": 0, "expires_at": None, "reason": "disabled"}
    try:
        config = json.loads(CONFIG_PATH.read_text())
        if config.get("enabled") is False:
            return frozenset(), status
        if config.get("enabled") is not True:
            raise ValueError("Invalid enable flag")
        emails = config["recipients"]
        if not isinstance(emails, list) or not 1 <= len(emails) <= 3:
            raise ValueError("One to three explicit recipients required")
        if any(
            not isinstance(email, str)
            or len(email) > 254
            or not re.fullmatch(r"[^\s@,*<>]+@[^\s@,*<>]+\.[^\s@,*<>]+", email)
            or email.lower().endswith(".invalid")
            for email in emails
        ):
            raise ValueError("Explicit, real email addresses required")
        recipients = frozenset(email.lower() for email in emails)
        approved = datetime.fromisoformat(config["approved_at"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(config["expires_at"].replace("Z", "+00:00"))
        if (
            approved.tzinfo is None
            or expires.tzinfo is None
            or not timedelta() < expires - approved <= timedelta(days=7)
        ):
            raise ValueError("Timezone-aware testing window of at most seven days required")
        status.update(recipient_count=len(recipients), expires_at=expires.isoformat())
        if not approved <= now < expires:
            status["reason"] = "outside_test_window"
            return frozenset(), status
        status.update(active=True, reason="ready")
        return recipients, status
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        status["reason"] = "invalid_configuration"
        return frozenset(), status


def _staging_test(email_mode: str) -> tuple[frozenset[str], dict]:
    recipients, status = _test_config(datetime.now(timezone.utc))
    smtp = get_smtp_config_status()
    status["smtp_ready"] = bool(smtp["ok"] and smtp["use_tls"])
    environment_matches = (
        get_jupr_env() == "staging"
        and email_mode == EMAIL_MODE_DRY_RUN
        and get_env_or_default("FLY_APP_NAME") == "juprleagues-api-staging"
        and get_env_or_default("SUPABASE_URL").rstrip("/") == STAGING_AUTH
        and get_next_web_base_url(default="") == STAGING_WEB
    )
    if status["active"] and not environment_matches:
        status.update(active=False, reason="environment_mismatch")
    elif status["active"] and not status["smtp_ready"]:
        status.update(active=False, reason="smtp_unavailable")
    return (recipients if status["active"] else frozenset()), status


def invitation_email_policy(email_mode: str) -> InvitationEmailPolicy:
    if get_jupr_env() != "staging":
        # The staging test configuration cannot activate email elsewhere.
        return InvitationEmailPolicy(enabled=email_mode == EMAIL_MODE_LIVE)
    recipients, status = _staging_test(email_mode)
    return InvitationEmailPolicy(enabled=status["active"], test_mode=True, recipients=recipients)


def invitation_email_test_status(email_mode: str) -> dict:
    """Non-secret deployment diagnostics; never a sending or account-creation path."""
    return _staging_test(email_mode)[1]
