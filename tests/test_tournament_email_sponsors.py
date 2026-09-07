import base64
from email import message_from_string
from io import BytesIO
from types import SimpleNamespace

from PIL import Image
import pytest

from jupr_app.domain.notifications import smtp_mailer
from jupr_app.domain.notifications import tournament_registration_confirmation_email as confirmation
from jupr_app.domain.notifications import tournament_registration_edit_email as edit
from jupr_app.domain.notifications import tournament_registrant_broadcast_email as broadcast
from jupr_app.domain.notifications import tournament_partner_request_email as partner
from jupr_app.domain.notifications import tournament_pairing_interest_email as pairing
from jupr_app.domain.notifications import tournament_team_invitation_email as team
from jupr_app.domain.notifications.tournament_email_sponsors import (
    MAX_TOTAL_LOGO_BYTES, sponsor_inline_images, sponsor_preview_html,
    with_sponsors_html, with_sponsors_text,
)
from jupr_app.services.tournament_email_sponsor_service import (
    load_tournament_email_sponsors, prepare_tournament_email_sponsors,
    tournament_email_sponsor_snapshot,
)
from tests.test_admin_match_log_service import FakeSupabase

CLUB = "tres_palapas"
TOURNAMENT = "00000000-0000-0000-0000-000000000001"
PATH = f"{CLUB}/{TOURNAMENT}/" + "a" * 32 + ".webp"


def image_bytes(kind):
    output = BytesIO()
    Image.new("RGBA", (300, 100), "#234567").save(output, format=kind)
    return output.getvalue()


def sponsors():
    return [
        {"name": "Homes & Land", "tier": "presenting", "level": "Official Real Estate Partner",
         "public_description": "Local knowledge for 25 years.", "website": "https://example.com/?a=1&b=2",
         "logo_png_base64": base64.b64encode(image_bytes("PNG")).decode()},
        {"name": "Equipment Co", "tier": "premier", "public_description": "Official equipment partner"},
        {"name": "Community Co", "tier": "supporting"},
        {"name": "Hidden Sponsor", "tier": "presenting", "is_visible": False},
    ]


def test_title_sponsor_precedes_message_and_supporting_sponsors_follow_it():
    html = with_sponsors_html("<html><body><h1>Baja Classic</h1><p>Your message</p></body></html>", sponsors())
    text = with_sponsors_text("Your message", sponsors())
    assert html.index("Baja Classic") < html.index("Presented by") < html.index("Your message") < html.index("Supporting sponsors") < html.index("Community sponsors")
    assert "Homes &amp; Land" in html
    assert "Local knowledge for 25 years." in html
    assert 'src="cid:tournament-sponsor-0"' in html
    assert 'href="https://example.com/?a=1&amp;b=2"' in html
    assert "https://example.com/?a=1&b=2" in text
    assert "Hidden Sponsor" not in html + text
    assert "cid:" not in sponsor_preview_html(html, sponsors())
    assert "data:image/png;base64," in sponsor_preview_html(html, sponsors())


def test_sponsor_content_is_escaped_and_unsafe_urls_are_not_links():
    rows = [{"name": '<script>alert("x")</script>', "tier": "presenting",
             "website": "javascript:alert(1)", "public_description": '<img src=x onerror="oops">',
             "notes": "Private amount"}]
    html = with_sponsors_html("<body><h1>Event</h1></body>", rows)
    assert "<script>" not in html and "<img src=x" not in html
    assert "javascript:" not in html and "Private amount" not in html
    assert "&lt;script&gt;" in html


def test_no_sponsors_keeps_original_message():
    body = "<body><h1>Event</h1><p>Hello</p></body>"
    assert with_sponsors_html(body, []) == body
    assert with_sponsors_text("Hello", None) == "Hello"


def database():
    db = FakeSupabase({
        "tournaments": [{"id": TOURNAMENT, "club_id": CLUB}],
        "tournament_registration_settings": [{"tournament_id": TOURNAMENT, "sponsors_json": [
            {"name": "Published", "tier": "presenting", "logo_path": PATH, "notes": "PRIVATE"},
            {"name": "Invisible", "is_visible": False},
            {"name": "Wrong club logo", "logo_path": PATH.replace(CLUB, "another_club")},
        ]}],
        "tournament_setup_drafts": [{"tournament_id": TOURNAMENT, "basics": {"sponsors_json": [{"name": "Unpublished"}]}}],
    })
    downloaded = []
    def download(path):
        downloaded.append(path)
        return image_bytes("WEBP")
    db.storage = SimpleNamespace(from_=lambda bucket: SimpleNamespace(download=download))
    return db, downloaded


def test_only_published_visible_sponsors_and_tournament_owned_logos_are_loaded():
    db, downloaded = database()
    rows = load_tournament_email_sponsors(db, club_id=CLUB, tournament_id=TOURNAMENT)
    assert [row["name"] for row in rows] == ["Published", "Wrong club logo"]
    assert downloaded == [PATH]
    assert "PRIVATE" not in str(rows) and "logo_path" not in str(rows) and "Unpublished" not in str(rows)
    assert base64.b64decode(rows[0]["logo_png_base64"]).startswith(b"\x89PNG")
    assert "logo_png_base64" not in rows[1]
    assert load_tournament_email_sponsors(db, club_id="other", tournament_id=TOURNAMENT) == []


def test_logo_outage_keeps_sponsor_name_and_does_not_need_a_signed_url():
    db, _ = database()
    db.storage = SimpleNamespace(from_=lambda bucket: (_ for _ in ()).throw(RuntimeError("storage down")))
    rows = load_tournament_email_sponsors(db, club_id=CLUB, tournament_id=TOURNAMENT)
    assert rows[0]["name"] == "Published"
    assert "logo_png_base64" not in rows[0]


def test_oversized_or_invalid_images_are_omitted():
    db, _ = database()
    snapshot = tournament_email_sponsor_snapshot(db, club_id=CLUB, tournament_id=TOURNAMENT)
    db.storage = SimpleNamespace(from_=lambda bucket: SimpleNamespace(download=lambda path: b"not an image"))
    assert "logo_png_base64" not in prepare_tournament_email_sponsors(db, snapshot)[0]
    invalid = [{"name": "A", "logo_png_base64": "!!!"}, {"name": "B", "logo_png_base64": "a" * 100000}]
    assert sponsor_inline_images(invalid) == {}
    assert sum(map(len, sponsor_inline_images(sponsors() * 20).values())) <= MAX_TOTAL_LOGO_BYTES


COMMON = dict(tournament_name="Baja Classic", division_name="Mixed", requester_name="Alex", target_name="Beth", board_url="https://example.com/board")


@pytest.mark.parametrize("module,html_builder,text_builder,args", [
    (edit, "build_tournament_registration_edit_email_html", "build_tournament_registration_edit_email_text", dict(tournament_name="Baja Classic", registered_email="alex@example.com", edit_url="https://example.com/edit")),
    (broadcast, "build_tournament_registrant_broadcast_email_html", "build_tournament_registrant_broadcast_email_text", dict(tournament_name="Baja Classic", recipient_name="Alex", subject="Court update", message="Hi Alex, meet at 9.", personalize_greeting=False)),
    (team, "build_team_invitation_email_html", "build_team_invitation_email_text", dict(tournament_name="Baja Classic", team_name="Blue", captain_name="Alex", invited_name="Beth", invitation_url="https://example.com/invite")),
    (partner, "build_tournament_partner_request_email_html", "build_tournament_partner_request_email_text", dict(tournament_name="Baja Classic", target_name="Beth", requester_name="Alex", requester_email="alex@example.com", requester_phone="", event_label="Doubles", division_label="Mixed", day_label="Sunday", message="Partner?")),
    (pairing, "build_pairing_interest_html", "build_pairing_interest_text", {**COMMON, "recipient_kind": "player"}),
    (pairing, "build_pairing_status_html", "build_pairing_status_text", {**COMMON, "recipient_kind": "requester", "action": "accepted"}),
])
def test_every_tournament_template_includes_sponsors(module, html_builder, text_builder, args):
    html = getattr(module, html_builder)(**args, email_sponsors=sponsors())
    text = getattr(module, text_builder)(**args, email_sponsors=sponsors())
    assert "Presented by" in html and "Presented by Homes & Land" in text
    assert "Equipment Co" in html and "Community Co" in text


def test_confirmation_includes_sponsors_without_changing_event_or_payment_details():
    vm = confirmation.build_registration_confirmation_view_model(tournament_name="Baja Classic", display_name="Alex", email="alex@example.com")
    vm["email_sponsors"] = sponsors()
    html = confirmation.build_tournament_registration_confirmation_html(vm)
    text = confirmation.build_tournament_registration_confirmation_text(vm)
    assert "Registration confirmed" in html and "Presented by Homes & Land" in text
    assert "Total due:" in html and "Equipment Co" in html


def test_smtp_message_embeds_png_copies_and_keeps_plain_text(monkeypatch):
    sent = []
    class SMTP:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def ehlo(self): pass
        def login(self, *a): pass
        def sendmail(self, sender, recipients, raw): sent.append((recipients, message_from_string(raw)))
    monkeypatch.setattr(smtp_mailer.smtplib, "SMTP", SMTP)
    monkeypatch.setattr(smtp_mailer, "_smtp_config_dict", lambda _: dict(host="fixture", port=25, username="fixture", password="fixture", from_email="organizer@example.com", from_name="Organizers", reply_to="organizer@example.com", use_tls=False))
    monkeypatch.setattr(confirmation, "get_email_mode", lambda: "live")
    vm = confirmation.build_registration_confirmation_view_model(tournament_name="Baja Classic", display_name="Alex", email="alex@example.com")
    result = confirmation.send_tournament_registration_confirmation_email(view_model=vm, email_sponsors=sponsors())
    assert result["status"] == "sent"
    recipients, mime = sent[0]
    assert recipients == ["alex@example.com"]
    images = [part for part in mime.walk() if part.get_content_type() == "image/png"]
    assert len(images) == 1 and images[0]["Content-ID"] == "<tournament-sponsor-0>"
    assert images[0].get_payload(decode=True) == image_bytes("PNG")
    bodies = {part.get_content_type(): part.get_payload(decode=True).decode() for part in mime.walk() if part.get_content_type() in {"text/plain", "text/html"}}
    assert "Presented by Homes & Land" in bodies["text/plain"]
    assert 'src="cid:tournament-sponsor-0"' in bodies["text/html"]
    assert "signed" not in bodies["text/html"] and "PRIVATE" not in str(mime)
