#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/.streamlit"
cat > "$HOME/.streamlit/config.toml" <<EOT
[server]
port = ${PORT}
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false
EOT

streamlit run app.py
