from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label} match count={count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


engine = ROOT / "jupr_app/domain/adaptive_play_engine.py"
replace_once(
    engine,
    '''def _balanced_group_sizes(player_count: int, court_count: int, play_format: str) -> list[int]:
    minimum = 2 if play_format == "singles" else 4
    if player_count < minimum:
        return []
    if court_count > 0:
        max_courts = min(court_count, max(1, player_count // minimum))
    else:
        target = 4 if play_format == "singles" else 5
        max_courts = max(1, round(player_count / target))
        max_courts = min(max_courts, max(1, player_count // minimum))
    while max_courts > 1 and player_count // max_courts < minimum:
        max_courts -= 1
    sizes = [player_count // max_courts] * max_courts
    for idx in range(player_count % max_courts):
        sizes[idx] += 1
    return sizes
''',
    '''def _balanced_group_sizes(player_count: int, court_count: int, play_format: str) -> list[int]:
    minimum = 2 if play_format == "singles" else 4
    if player_count < minimum:
        return []

    if court_count > 0:
        group_count = min(int(court_count), max(1, player_count // minimum))
    elif play_format == "doubles":
        # Preserve the familiar 4-player / 5-player ladder court model whenever
        # the roster can fit it exactly. Prefer the most simultaneous courts.
        exact: list[tuple[int, int, int]] = []
        for fives in range(0, (player_count // 5) + 1):
            remainder = player_count - (fives * 5)
            if remainder >= 0 and remainder % 4 == 0:
                fours = remainder // 4
                if fours + fives:
                    exact.append((fours + fives, fours, fives))
        if exact:
            _groups, fours, fives = max(exact, key=lambda item: (item[0], item[1]))
            return [5] * fives + [4] * fours
        group_count = min(
            max(1, math.ceil(player_count / 5)),
            max(1, player_count // minimum),
        )
    else:
        # Singles ladder courts are kept at five players or fewer when the
        # roster permits it, while still supporting every roster size.
        group_count = min(
            max(1, math.ceil(player_count / 5)),
            max(1, player_count // minimum),
        )

    while group_count > 1 and player_count // group_count < minimum:
        group_count -= 1
    sizes = [player_count // group_count] * group_count
    for idx in range(player_count % group_count):
        sizes[idx] += 1
    return sorted(sizes, reverse=True)
''',
    "balanced ladder groups",
)
replace_once(
    engine,
    '''    mini_count = max(3, len(ids))
''',
    '''    mini_count = 3 if len(ids) == 4 else max(3, len(ids))
''',
    "four-player ladder game count",
)

tests = ROOT / "tests/test_adaptive_play_engine.py"
with tests.open("a", encoding="utf-8") as handle:
    handle.write(
        '''\n\ndef test_ladder_prefers_four_and_five_player_courts_and_three_games_for_four():\n'''
        '''    event = create_generator_preview(\n'''
        '''        generator_kind="ladder",\n'''
        '''        play_format="doubles",\n'''
        '''        title="Twelve-player Ladder",\n'''
        '''        participant_names=[f"Player {idx}" for idx in range(1, 13)],\n'''
        '''        total_rounds=3,\n'''
        '''        court_count=0,\n'''
        '''    )\n'''
        '''\n'''
        '''    courts = event["rounds"][0]["courts"]\n'''
        '''    assert [court["size"] for court in courts] == [4, 4, 4]\n'''
        '''    assert all(len(court["matches"]) == 3 for court in courts)\n'''
        '''\n'''
        '''    nine = create_generator_preview(\n'''
        '''        generator_kind="ladder",\n'''
        '''        play_format="doubles",\n'''
        '''        title="Nine-player Ladder",\n'''
        '''        participant_names=[f"Player {idx}" for idx in range(1, 10)],\n'''
        '''        total_rounds=2,\n'''
        '''        court_count=0,\n'''
        '''    )\n'''
        '''    assert [court["size"] for court in nine["rounds"][0]["courts"]] == [5, 4]\n'''
        '''    assert [len(court["matches"]) for court in nine["rounds"][0]["courts"]] == [5, 3]\n'''
    )

workspace = ROOT / "apps/web/app/admin/play-generators/GeneratorWorkspace.tsx"
replace_once(
    workspace,
    '''            <small style={{ color: "#64748b" }}>Use 0 to schedule the maximum simultaneous courts.</small>
''',
    '''            <small style={{ color: "#64748b" }}>
              {generatorKind === "ladder"
                ? "Use 0 to create balanced ladder courts automatically."
                : "Use 0 to schedule the maximum simultaneous courts."}
            </small>
''',
    "court-count helper",
)
