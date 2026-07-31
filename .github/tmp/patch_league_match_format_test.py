from pathlib import Path

path = Path("tests/test_api_contract_admin_match_uploader_singles.py")
text = path.read_text(encoding="utf-8")
old = '''                "league_name": "Summer Social",\n                "k_factor": 32,\n'''
new = '''                "league_name": "Summer Social",\n                "match_format": "singles",\n                "k_factor": 32,\n'''
if text.count(old) != 1:
    raise SystemExit(f"Expected one singles metadata fixture marker; found {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
