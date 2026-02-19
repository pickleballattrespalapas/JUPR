#!/usr/bin/env bash
set -e

mkdir -p ~/.streamlit

cat > ~/.streamlit/config.toml <<EOF
[server]
port = $PORT
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false
EOF

exec streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0

