from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


form_path = Path("apps/web/app/admin/match-uploader/MatchUploaderForm.tsx")
form = form_path.read_text(encoding="utf-8")

form = replace_once(
    form,
    '''  useEffect(() => {
    setQuery(selectedName);
  }, [selectedName]);

  useEffect(() => {
    if (!value && exactPlayer && cleanedQuery) {
      onChange(String(exactPlayer.id));
    }
  }, [cleanedQuery, exactPlayer, onChange, value]);''',
    '''  const exactPlayerId = exactPlayer ? String(exactPlayer.id) : "";
  const previousExactPlayerId = useRef(exactPlayerId);

  useEffect(() => {
    setQuery(selectedName);
  }, [selectedName]);

  useEffect(() => {
    const previousId = previousExactPlayerId.current;
    previousExactPlayerId.current = exactPlayerId;
    if (!value && exactPlayerId && cleanedQuery && exactPlayerId !== previousId) {
      onChange(exactPlayerId);
    }
  }, [cleanedQuery, exactPlayerId, onChange, value]);''',
    "exact-player transition selection",
)

form = replace_once(
    form,
    '''  const singlesError = singlesValidationAttempted ? validateSingles(singlesRow) : null;
  const singlesScoreInvalid = Boolean(singlesError && /score|tied/i.test(singlesError));''',
    '''  const singlesError = singlesValidationAttempted ? validateSingles(singlesRow) : null;
  const singlesScoreA = Number(singlesRow.scoreA || 0);
  const singlesScoreB = Number(singlesRow.scoreB || 0);
  const singlesPlayersDuplicate = Boolean(
    singlesRow.playerA
    && singlesRow.playerB
    && singlesRow.playerA === singlesRow.playerB,
  );
  const singlesScoreInvalid = singlesValidationAttempted && (
    !Number.isFinite(singlesScoreA)
    || !Number.isFinite(singlesScoreB)
    || singlesScoreA < 0
    || singlesScoreB < 0
    || singlesScoreA + singlesScoreB <= 0
    || singlesScoreA === singlesScoreB
  );''',
    "field-level singles score validation",
)

form = replace_once(
    form,
    '<SearchablePlayerInput inputId="singles-player-a"',
    '<SearchablePlayerInput key={`singles-player-a-${singlesRow.playerA || "empty"}`} inputId="singles-player-a"',
    "singles player A reset key",
)
form = replace_once(
    form,
    'invalid={singlesValidationAttempted && !singlesRow.playerA}',
    'invalid={singlesValidationAttempted && (!singlesRow.playerA || singlesPlayersDuplicate)}',
    "singles player A validation",
)
form = replace_once(
    form,
    '<SearchablePlayerInput inputId="singles-player-b"',
    '<SearchablePlayerInput key={`singles-player-b-${singlesRow.playerB || "empty"}`} inputId="singles-player-b"',
    "singles player B reset key",
)
form = replace_once(
    form,
    'invalid={singlesValidationAttempted && !singlesRow.playerB}',
    'invalid={singlesValidationAttempted && (!singlesRow.playerB || singlesPlayersDuplicate)}',
    "singles player B validation",
)

form = replace_once(
    form,
    'return /\\b(unable|unavailable|disabled|required|must|cannot|could not|not configured|sign in|error|invalid|select|enter|choose|failed)\\b/i.test(message);',
    'return /\\b(unable|unavailable|disabled|required|must|cannot|could not|not configured|sign in|error|invalid|select|enter|choose|failed|conflict|changed|reload|retry|nothing)\\b/i.test(message);',
    "conflict alert classification",
)

form_path.write_text(form, encoding="utf-8")

layout_path = Path("apps/web/tests/match-uploader-layout.cjs")
layout = layout_path.read_text(encoding="utf-8")
layout = replace_once(
    layout,
    'assert.match(form, /if \\(!value && exactPlayer && cleanedQuery\\)/, "released exact-name players must become selectable without editing text");',
    '''assert.match(form, /const previousExactPlayerId = useRef\\(exactPlayerId\\)/, "exact-name selection must track availability transitions");
assert.match(form, /exactPlayerId !== previousId/, "released exact-name players must become selectable without reselecting cleared values");
assert.match(form, /const singlesScoreInvalid = singlesValidationAttempted &&/, "singles score validation must be field-level");
assert.match(form, /singlesScoreA \\+ singlesScoreB <= 0/, "blank singles scores must be highlighted");
assert.ok(form.includes('key={`singles-player-a-${singlesRow.playerA || "empty"}`}'), "singles Player 1 must remount cleanly after reset");
assert.ok(form.includes('key={`singles-player-b-${singlesRow.playerB || "empty"}`}'), "singles Player 2 must remount cleanly after reset");
assert.match(form, /failed\\|conflict\\|changed\\|reload\\|retry\\|nothing/, "concurrency conflicts must render as errors");''',
    "layout acceptance assertions",
)
layout_path.write_text(layout, encoding="utf-8")

regression_path = Path("tests/test_manual_acceptance_ux_regressions.py")
regression = regression_path.read_text(encoding="utf-8")
regression = replace_once(
    regression,
    '    assert "Review Match Log before retrying" in source\n',
    '''    assert "Review Match Log before retrying" in source
    assert "const singlesScoreInvalid = singlesValidationAttempted &&" in source
    assert "exactPlayerId !== previousId" in source
    assert 'key={`singles-player-a-${singlesRow.playerA || "empty"}`}' in source
    assert 'key={`singles-player-b-${singlesRow.playerB || "empty"}`}' in source
    assert "failed|conflict|changed|reload|retry|nothing" in source
''',
    "manual acceptance regression assertions",
)
regression_path.write_text(regression, encoding="utf-8")
