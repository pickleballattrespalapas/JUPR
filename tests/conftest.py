import sys
from pathlib import Path
import os
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def require_api_dependency(module_name: str):
    """Import optional API test dependency.

    Locally, missing deps skip API contract tests. In CI guard mode,
    missing deps fail loudly so we don't get false confidence from skips.
    """
    try:
        return __import__(module_name)
    except ModuleNotFoundError as exc:
        if os.getenv("JUPR_CI_REQUIRE_API_TESTS") == "1":
            raise pytest.UsageError(
                f"Missing required API test dependency '{module_name}'. "
                "Install root and services/api requirements before running API contract tests."
            ) from exc
        pytest.skip(f"optional API dependency '{module_name}' is not installed", allow_module_level=True)
