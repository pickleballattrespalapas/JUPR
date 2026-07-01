"""JUPR application package."""
from __future__ import annotations

# Install runtime compatibility patches as soon as the package is imported.
# Streamlit imports jupr_app.* before it lazy-loads page modules, so this is a
# reliable place to activate player Trophy Room fixes in deployed environments.
try:
    import sitecustomize as _jupr_runtime_patches  # noqa: F401
except Exception:
    pass
