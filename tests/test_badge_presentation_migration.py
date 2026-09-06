from pathlib import Path
from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.presentation import badge_category, badge_requirement, CATEGORY_ORDER


def test_presentation_overlay_covers_every_badge_without_changing_awards_or_activation():
    sql = Path('supabase/migrations/20261109001000_badge_plain_requirements.sql').read_text()
    for badge in BADGE_DEFINITIONS:
        values = (badge.badge_id, badge_category(badge.badge_id), badge_requirement(badge.badge_id))
        expected = '(' + ', '.join("'" + value.replace("'", "''") + "'" for value in values) + ')'
        assert expected in sql
        assert badge.lore == badge.hint == values[2]
        assert badge.category in CATEGORY_ORDER
        assert values[2] != 'Requirements TBD'
    assert 'player_badges' not in sql
    assert 'set is_active' not in sql and 'set prestige' not in sql
    assert 'insert into' not in sql
