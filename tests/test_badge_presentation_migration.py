from pathlib import Path
import json, re
from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.presentation import badge_category, badge_requirement, CATEGORY_ORDER


def test_presentation_overlay_covers_every_badge_without_changing_awards_or_activation():
    sql = Path('supabase/migrations/20261109003000_badge_plain_requirements.sql').read_text()
    reactivation = Path('supabase/migrations/20261109001100_badge_reactivation_and_admin_seasons.sql').read_text()
    program_sql = Path('supabase/migrations/20261109001200_program_badge_expansion.sql').read_text()
    programs = {row['badge_id']: row for row in json.loads(re.search(r'\$catalog\$(.*?)\$catalog\$', program_sql, re.S).group(1))}
    for badge in BADGE_DEFINITIONS:
        values = (badge.badge_id, badge_category(badge.badge_id), badge_requirement(badge.badge_id))
        expected = '(' + ', '.join("'" + value.replace("'", "''") + "'" for value in values) + ')'
        if badge.badge_id in programs:
            assert programs[badge.badge_id]['category'] == values[1]
            assert programs[badge.badge_id]['lore'] == values[2]
        else:
            assert expected in sql or expected[:-1] + ',' in reactivation
        assert badge.lore == badge.hint == values[2]
        assert badge.category in CATEGORY_ORDER
        assert values[2] != 'Requirements TBD'
    assert 'player_badges' not in sql
    assert 'set is_active' not in sql and 'set prestige' not in sql
    assert 'insert into' not in sql
