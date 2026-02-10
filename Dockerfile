FROM python:3.11-slim

WORKDIR /app

COPY packages.txt /tmp/packages.txt
RUN apt-get update \
    && xargs -r -a /tmp/packages.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* /tmp/packages.txt

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

EXPOSE 8080
ENV PORT=8080

CMD ["/bin/bash", "start.sh"]
