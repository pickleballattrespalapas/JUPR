from pathlib import Path


def test_public_base_url_prefers_env_with_fallback():
    contents = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert 'os.getenv("JUPR_PUBLIC_BASE_URL", "")' in contents
    assert 'PUBLIC_BASE_URL = get_public_base_url()' in contents
    assert 'return PUBLIC_BASE_URL_FALLBACK' in contents
