from __future__ import annotations

import smtplib
import ssl
import socketserver
import threading
from email import message_from_string

import pytest

from jupr_app.config import SMTPConfig
from jupr_app.domain.notifications import smtp_mailer
from jupr_app.domain.notifications.smtp_delivery import SMTPDeliveryError


@pytest.fixture
def transport(monkeypatch):
    plans, sessions, pauses = [], [], []

    class SMTP:
        def __init__(self, host, port, **options):
            self.plan = plans.pop(0) if plans else {}
            self.events = []
            self.options = options
            sessions.append(self)
            self.step("connect")

        def step(self, stage):
            self.events.append(stage)
            if stage in self.plan:
                raise self.plan[stage]

        def ehlo(self):
            self.step("greeting")

        def starttls(self, *, context):
            assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
            self.step("starttls")

        def login(self, *_args):
            self.step("authentication")

        def sendmail(self, sender, recipients, payload):
            self.payload = payload
            self.step("message")

        def quit(self):
            self.step("quit")

        def close(self):
            self.step("close")

    monkeypatch.setattr(smtp_mailer.smtplib, "SMTP", SMTP)
    monkeypatch.setattr(smtp_mailer.smtplib, "SMTP_SSL", SMTP)
    monkeypatch.setattr(smtp_mailer.time, "sleep", pauses.append)
    return plans, sessions, pauses


def send(*, port=587):
    return smtp_mailer.send_email_with_inline_chart(
        to_email="player@example.com", subject="Edit your registration",
        text_body="Secret edit link", html_body="<p>Secret edit link</p>",
        smtp_config=SMTPConfig(
            host="mail.example.com", port=port, username="private-user",
            password="private-password", from_email="mail@example.com",
            from_name="Club", reply_to="organizer@example.com", use_tls=True,
        ),
    )


def test_temporary_data_rejection_retries_once_with_the_same_message_id(transport):
    plans, sessions, pauses = transport
    plans.extend([{"message": smtplib.SMTPDataError(451, b"4.3.0 temporarily unavailable")}, {}])
    message_id = send()
    assert len(sessions) == 2 and pauses == [1]
    assert sessions[0].payload == sessions[1].payload
    assert message_from_string(sessions[-1].payload)["Message-ID"] == message_id


@pytest.mark.parametrize("stage,error", [
    ("connect", ConnectionRefusedError("private host")),
    ("greeting", smtplib.SMTPServerDisconnected("private host")),
    ("starttls", smtplib.SMTPResponseException(454, b"4.7.0 try later")),
    ("authentication", smtplib.SMTPAuthenticationError(454, b"4.7.0 try later")),
    ("message", smtplib.SMTPRecipientsRefused({"player@example.com": (450, b"4.2.0 try later")})),
])
def test_confirmed_transient_failures_are_bounded(transport, stage, error):
    plans, sessions, pauses = transport
    plans.extend([{stage: error}, {stage: error}])
    with pytest.raises(SMTPDeliveryError) as result:
        send()
    assert len(sessions) == 2 and pauses == [1]
    assert result.value.details["delivery_outcome"] == "not_accepted"
    assert result.value.details["attempt_count"] == 2


@pytest.mark.parametrize("stage,error,outcome", [
    ("authentication", smtplib.SMTPAuthenticationError(535, b"5.7.8 bad credentials"), "not_accepted"),
    ("connect", ssl.SSLCertVerificationError("bad certificate"), "not_accepted"),
    ("message", smtplib.SMTPDataError(550, b"5.7.1 rejected"), "not_accepted"),
    ("message", smtplib.SMTPSenderRefused(550, b"rejected", "mail@example.com"), "not_accepted"),
    ("message", TimeoutError("private body"), "unknown"),
    ("message", smtplib.SMTPServerDisconnected("private body"), "unknown"),
    ("message", smtplib.SMTPResponseException(451, b"unknown phase"), "unknown"),
])
def test_permanent_and_uncertain_failures_are_not_automatically_retried(transport, stage, error, outcome):
    plans, sessions, pauses = transport
    plans.append({stage: error})
    with pytest.raises(SMTPDeliveryError) as result:
        send()
    assert len(sessions) == 1 and pauses == []
    assert result.value.details["delivery_outcome"] == outcome
    assert result.value.details["message_id"].endswith("@example.com>")
    if stage == "message":
        assert result.value.details["message_id"] == message_from_string(sessions[0].payload)["Message-ID"]


def test_quit_error_does_not_turn_an_accepted_message_into_a_failure(transport):
    plans, sessions, pauses = transport
    plans.append({"quit": smtplib.SMTPResponseException(451, b"private provider text")})
    assert send().endswith("@example.com>")
    assert len(sessions) == 1 and pauses == []
    assert sessions[0].events[-1] == "close"


def test_quit_error_cannot_replace_the_original_rejection(transport):
    plans, sessions, _pauses = transport
    plans.append({
        "message": smtplib.SMTPDataError(550, b"5.7.1 actual rejection"),
        "quit": smtplib.SMTPResponseException(421, b"cleanup failure"),
    })
    with pytest.raises(SMTPDeliveryError) as result:
        send()
    assert result.value.details["stage"] == "message"
    assert result.value.details["smtp_codes"] == [550]
    assert result.value.details["enhanced_status"] == "5.7.1"
    assert len(sessions) == 1


def test_errors_and_logs_never_include_provider_text_or_secrets(transport, caplog):
    plans, _sessions, _pauses = transport
    plans.append({"message": smtplib.SMTPDataError(
        550, b"5.7.1 player@example.com private-password edit_token=secret-token",
    )})
    with pytest.raises(SMTPDeliveryError) as result:
        send()
    rendered = str(result.value) + caplog.text
    for private in ("player@example.com", "private-password", "secret-token", "private-user"):
        assert private not in rendered
    assert "550" in rendered and "5.7.1" in rendered


def test_implicit_tls_uses_a_verified_context_without_starttls(transport):
    _plans, sessions, _pauses = transport
    send(port=465)
    assert "starttls" not in sessions[0].events
    context = sessions[0].options["context"]
    assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED


def test_slow_failure_does_not_start_another_connection(transport, monkeypatch):
    plans, sessions, pauses = transport
    plans.append({"connect": TimeoutError("slow connection")})
    elapsed = iter([0, 30])
    monkeypatch.setattr(smtp_mailer.time, "monotonic", lambda: next(elapsed))
    with pytest.raises(SMTPDeliveryError):
        send()
    assert len(sessions) == 1 and pauses == []


def test_real_smtp_acceptance_survives_a_nonstandard_quit_response(caplog):
    accepted = []

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.wfile.write(b"220 fixture ready\r\n")
            while line := self.rfile.readline():
                command = line.split(b" ", 1)[0].strip().upper()
                if command == b"EHLO":
                    self.wfile.write(b"250-fixture\r\n250 AUTH PLAIN\r\n")
                elif command == b"AUTH":
                    self.wfile.write(b"235 authenticated\r\n")
                elif command in {b"MAIL", b"RCPT"}:
                    self.wfile.write(b"250 OK\r\n")
                elif command == b"DATA":
                    self.wfile.write(b"354 send message\r\n")
                    message = []
                    while (data := self.rfile.readline()) not in {b".\r\n", b""}:
                        message.append(data)
                    accepted.append(b"".join(message))
                    self.wfile.write(b"250 message accepted\r\n")
                elif command == b"QUIT":
                    self.wfile.write(b"451 4.3.0 cleanup unavailable\r\n")
                    return

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            message_id = smtp_mailer.send_email_with_inline_chart(
                to_email="player@example.com", subject="fixture", html_body="<p>fixture</p>", text_body="fixture",
                smtp_config=SMTPConfig(
                    host="127.0.0.1", port=server.server_address[1], username="fixture", password="fixture",
                    from_email="club@example.com", from_name="Fixture", reply_to="", use_tls=False,
                ),
            )
            assert len(accepted) == 1
            assert message_id.encode() in accepted[0]
            assert "accepted=True" in caplog.text and "smtp_codes=[451]" in caplog.text
        finally:
            server.shutdown()
            thread.join(timeout=2)
