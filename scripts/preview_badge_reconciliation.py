"""Offline reconciliation preview: no credentials, network, or database writes."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jupr_app.domain.gamification.reconciliation import build_reconciliation_plan, simulate_reconciliation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True, help='JSON snapshot or directory of per-table JSON files')
    parser.add_argument('--club-id', required=True)
    parser.add_argument('--as-of', required=True, help='UTC timestamp used to close periods and verify aggregate eligibility')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.snapshot.is_dir():
        snapshot = {name: json.loads((args.snapshot / f'{name}.json').read_text()) for name in ('players', 'league_ratings', 'leagues_metadata', 'matches', 'badges', 'player_badges')}
    else:
        snapshot = json.loads(args.snapshot.read_text())
    plan = build_reconciliation_plan(snapshot, club_id=args.club_id, as_of=args.as_of)
    simulated = simulate_reconciliation(snapshot, plan)
    rerun = build_reconciliation_plan(simulated, club_id=args.club_id, as_of=args.as_of)
    if rerun['additions'] or rerun['revocations']:
        raise RuntimeError('Reconciliation did not reach an idempotent state')
    args.output.write_text(json.dumps(plan, indent=2, default=str) + '\n')
    print(json.dumps(plan['summary'], indent=2))


if __name__ == '__main__':
    main()
