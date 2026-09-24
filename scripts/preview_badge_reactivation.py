"""Read-only activation estimate from an existing, complete club export.

Does not connect to a database, invent seasons, or write awards. The output is
an estimate, not a production repair plan; fresh data must be checked at release.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.award_identity import award_key
from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.badge_engine import compute_candidates_for_club
from jupr_app.domain.gamification.reconciliation import fingerprint

RESTORED_AUTOMATIC = {
    'breakthrough', 'above_expectations', 'clutch_performer', 'dominant_run',
    'high_output', 'nemesis_found', 'rivalry_win', 'rivalry_streak',
    'settled_the_score', 'battle_tested', 'consistency', 'mr_reliable',
}


def preview(snapshot: dict, *, club_id: str, as_of: str) -> dict:
    for table, rows in snapshot.items():
        if table != 'badges' and any(row.get('club_id') not in (None, club_id) for row in rows):
            raise ValueError(f'Mixed club export: {table}')
    existing = snapshot['player_badges']
    old_states = {row['badge_id']: row.get('state') for row in snapshot['badges']}
    definitions = [dict(asdict(b), state='frozen' if old_states.get(b.badge_id) == 'frozen' else b.state) for b in BADGE_DEFINITIONS]
    ctx = SimpleNamespace(club_id=club_id, df_players_all=pd.DataFrame(snapshot['players']),
        df_leagues=pd.DataFrame(snapshot['league_ratings']), df_matches=pd.DataFrame(snapshot['matches']),
        df_meta=pd.DataFrame(snapshot['leagues_metadata']), df_badges=pd.DataFrame(definitions),
        df_player_badges=pd.DataFrame(existing), badge_seasons=snapshot.get('badge_seasons', []))
    known = {award_key(row) for row in existing}  # Includes revocations.
    additions = []
    for candidate in compute_candidates_for_club(club_id, ctx=ctx, as_of=pd.Timestamp(as_of).to_pydatetime(), strict=True):
        if candidate.badge_id not in RESTORED_AUTOMATIC:
            continue
        row = asdict(candidate)
        key = award_key(row)
        if key not in known:
            known.add(key)
            additions.append(row)
    counts = Counter(row['badge_id'] for row in additions)
    return {'version': 'badge-reactivation-v1', 'club_id': club_id, 'as_of': as_of,
        'source_snapshot_sha256': fingerprint(snapshot), 'source_award_rows': len(existing),
        'configured_seasons': len(snapshot.get('badge_seasons', [])),
        'new_candidate_count': len(additions), 'players_with_new_candidates': len({r['player_id'] for r in additions}),
        'by_badge': {badge: counts[badge] for badge in sorted(RESTORED_AUTOMATIC)},
        'existing_awards_changed': 0, 'community_awards_inferred': 0,
        'limitation': 'Estimate from the supplied export. No awards applied. Recheck fresh data before production activation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--club-id', required=True)
    parser.add_argument('--as-of', required=True)
    args = parser.parse_args()
    print(json.dumps(preview(json.loads(args.snapshot.read_text()), club_id=args.club_id, as_of=args.as_of), indent=2))
