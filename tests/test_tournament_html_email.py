import json
from pathlib import Path

import pytest

from jupr_app.domain.notifications.tournament_email_content import clean_tournament_message, render_tournament_message_html
from jupr_app.domain.notifications.tournament_registrant_broadcast_email import (
    build_tournament_registrant_broadcast_email_html,
    build_tournament_registrant_broadcast_email_text,
    send_tournament_registrant_broadcast_email,
)
from jupr_app.services import admin_tournament_broadcast_service as service
from jupr_app.services.admin_tournament_registration_reporting_service import build_admin_tournament_broadcast_preview
from tests.test_admin_tournament_broadcast_service import fixture, prepare, attempt


@pytest.mark.parametrize("markup", [
    '<script>alert(1)</script>', '<iframe src="https://example.com"></iframe>',
    '<img src="https://example.com/tracker.png">', '<p onclick="alert(1)">Hello</p>',
    '<a href="javascript:alert(1)">Click</a>', '<a href="data:text/html,hello">Click</a>',
    '<p style="background:url(https://example.com)">Hello</p>',
    '<p style="width:expression(alert(1))">Hello</p>',
    '<p style="background: u\\72l(https://example.com)">Hello</p>',
    '<form action="https://example.com">Hello</form>', '<p>Unclosed',
    '<!doctype html><p>Full document</p>',
])
def test_html_rejects_active_content_and_invalid_markup(markup):
    with pytest.raises(ValueError):
        clean_tournament_message(markup, "html")


def test_styled_message_has_readable_text_links_and_no_duplicate_greeting():
    message = '<table role="presentation" style="background-color:#123e37;max-width:620px"><tr><td><h1>Baja Classic</h1><p>Hi everyone,</p><a href="https://example.test/roster?a=1&amp;b=2">Review roster</a></td></tr></table>'
    options = dict(tournament_name="Baja Classic", recipient_name="Alex", subject="Information",
        message=message, message_format="html", personalize_greeting=False,
        email_sponsors=[{"name": "Homes & Land", "tier": "presenting", "public_description": "Sponsor details"}])
    html = build_tournament_registrant_broadcast_email_html(**options)
    text = build_tournament_registrant_broadcast_email_text(**options)
    assert 'background-color:#123e37;max-width:620px' in html
    assert '&lt;table' not in html
    assert html.count('Hi everyone,') == 1
    assert html.count('Homes &amp; Land') == 1
    assert 'Hi everyone,' in text and '<table' not in text
    assert 'Review roster (https://example.test/roster?a=1&b=2)' in text
    assert html.index('Hi everyone,') < html.index('Sponsor details')


def test_html_review_survives_10000_characters_and_reaches_exactly_once_delivery(fixture, monkeypatch):
    message = '<div style="color:#123e37"><h1>Player guide</h1><p>' + 'Long tournament information. ' * 600 + '</p></div>'
    sent = []
    monkeypatch.setattr(service, 'send_tournament_registrant_broadcast_email', lambda **kw: sent.append(kw) or {'status': 'sent'})
    payload = prepare(fixture, message=message, message_format='html')
    preview = build_admin_tournament_broadcast_preview(fixture, club_id='club', tournament_id='tour_1',
        registration_ids=['registration_1'], subject='Update', message=message, message_format='html')
    assert preview['message_format'] == 'html'
    assert preview['preview']['html'].count('Long tournament information.') == 600
    root = service.create_tournament_broadcast(fixture, **payload)
    assert root['message_format'] == 'html' and root['message'] == message
    assert not sent
    attempt(fixture, root['operation_key'])
    attempt(fixture, root['operation_key'])
    assert len(sent) == 1 and sent[0]['message_format'] == 'html'
    assert sent[0]['message'] == message


def test_switching_message_format_invalidates_review(fixture):
    payload = prepare(fixture, message='<p>Hello</p>', message_format='html')
    with pytest.raises(ValueError, match='Preview the email again'):
        service.create_tournament_broadcast(fixture, **{**payload, 'message_format': 'text'})
    assert not fixture.tables.get(service.TABLE)


def test_plain_text_remains_escaped_and_has_original_size_limit():
    rendered = build_tournament_registrant_broadcast_email_html(tournament_name='Baja', recipient_name='Alex',
        subject='Update', message='<strong>Literal code</strong>')
    assert '&lt;strong&gt;Literal code&lt;/strong&gt;' in rendered
    with pytest.raises(ValueError):
        clean_tournament_message('x' * 10001, 'text')
    with pytest.raises(ValueError):
        clean_tournament_message('<p>' + 'x' * 100000 + '</p>', 'html')
    with pytest.raises(ValueError):
        clean_tournament_message('<p>Hello</p>', 'unsupported')


def test_approved_baja_template_is_complete_and_supported():
    source = (Path(__file__).resolve().parents[1] / 'apps/web/lib/preparedTournamentEmails.ts').read_text()
    message, _ = json.JSONDecoder().raw_decode(source.split('    message: ', 1)[1])
    assert 10000 < len(message) < 100000
    html, text = render_tournament_message_html(clean_tournament_message(message, 'html'))
    for expected in ('7:45', '5:00 p.m.', 'November 15', 'Lifetime 48', 'JOOLA HC-40', 'Mexican pesos', 'present to win', 'Center Court', '2:00 p.m.'):
        assert expected in text
    assert '#123e37' in html and 'max-width:620px' in html


def test_mocked_smtp_gets_styled_html_and_readable_fallback(monkeypatch):
    captured = {}
    monkeypatch.setenv('JUPR_ENV', 'test')
    monkeypatch.setenv('JUPR_EMAIL_MODE', 'live')
    def fake_smtp(**kwargs):
        captured.update(kwargs)
        return 'test-provider-id'
    monkeypatch.setattr('jupr_app.domain.notifications.tournament_registrant_broadcast_email.send_email_with_inline_chart', fake_smtp)
    result = send_tournament_registrant_broadcast_email(tournament_name='Baja', recipient_email='fixture@example.test',
        recipient_name='Fixture', subject='Information', message='<p style="color:#123e37">Ready at 8:00.</p>', message_format='html')
    assert result['status'] == 'sent'
    assert '<p style="color:#123e37">Ready at 8:00.</p>' in captured['html_body']
    assert 'Ready at 8:00.' in captured['text_body'] and '<p' not in captured['text_body']
