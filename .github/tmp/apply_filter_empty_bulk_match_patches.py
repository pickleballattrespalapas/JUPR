from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


panel_path = Path("apps/web/app/admin/match-log/MatchLogApplyPanel.tsx")
panel = panel_path.read_text(encoding="utf-8")
panel = replace_once(
    panel,
    '''        return patch;
      });
      if (generated.every((patch) => patchFields(patch).length === 0)) throw new Error("Choose at least one bulk field change.");''',
    '''        return patch;
      }).filter((patch) => patchFields(patch).length > 0);
      if (!generated.length) throw new Error("Choose at least one bulk field change.");''',
    "filter unchanged selected matches",
)
panel_path.write_text(panel, encoding="utf-8")

layout_path = Path("apps/web/tests/match-uploader-layout.cjs")
layout = layout_path.read_text(encoding="utf-8")
layout = replace_once(
    layout,
    '''assert.match(matchLogPanel, /bulkScoreEdits/, "Bulk editor must stage independent score changes");
assert.match(matchLogApi, /matchIds\\?: string \\| Array<string \\| number>/, "Match Log client must send multiple IDs");''',
    '''assert.match(matchLogPanel, /bulkScoreEdits/, "Bulk editor must stage independent score changes");
assert.match(matchLogPanel, /\\.filter\\(\\(patch\\) => patchFields\\(patch\\)\\.length > 0\\)/, "Bulk editor must omit unchanged selected matches from staged patches");
assert.match(matchLogApi, /matchIds\\?: string \\| Array<string \\| number>/, "Match Log client must send multiple IDs");''',
    "unchanged bulk patch source contract",
)
layout_path.write_text(layout, encoding="utf-8")
