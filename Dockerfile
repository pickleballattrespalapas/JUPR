FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# System dependencies (PDF / WeasyPrint / fonts)
COPY packages.txt /tmp/packages.txt
RUN apt-get update \
    && xargs -r -a /tmp/packages.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* /tmp/packages.txt

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# App code
COPY . /app

# Fail build early if Streamlit component frontend assets are missing.
RUN test -f /app/jupr_court_board/frontend/build/index.html

# Make start script executable
RUN chmod +x /app/start.sh

EXPOSE 8080

CMD ["/bin/bash", "/app/start.sh"]
