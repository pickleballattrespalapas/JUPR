"""Start a real Uvicorn process with the Docker log config, without a database."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def test_shipped_logging_records_success_denial_failure_without_query_credentials(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    assert "--log-config services/api/logging.json" in (repo / "Dockerfile.api").read_text()
    (tmp_path / "logging_fixture.py").write_text('''
from fastapi import FastAPI, HTTPException
from services.api.middleware import StructuredRequestLoggingMiddleware
app = FastAPI()
app.add_middleware(StructuredRequestLoggingMiddleware)
@app.get('/ok')
def ok(): return {'ok': True}
@app.get('/denied')
def denied(): raise HTTPException(status_code=403, detail='denied')
@app.get('/fail')
def fail(): raise ValueError('fixture failure')
''')
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(repo)]), "JUPR_ENV": "test"}
    logfile = tmp_path / "server.log"
    with logfile.open("w") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "logging_fixture:app", "--host", "127.0.0.1",
             "--port", str(port), "--log-config", str(repo / "services/api/logging.json")],
            cwd=repo, env=env, stdout=output, stderr=subprocess.STDOUT,
        )
        try:
            origin = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 15
            while True:
                try:
                    with urllib.request.urlopen(origin + "/ok", timeout=1):
                        break
                except (urllib.error.URLError, ConnectionError):
                    assert process.poll() is None, logfile.read_text()
                    assert time.monotonic() < deadline, logfile.read_text()
                    time.sleep(0.05)
            for route, status in [("ok", 200), ("denied", 403), ("fail", 500)]:
                request = urllib.request.Request(
                    origin + "/" + route + "?edit_token=DUMMY_EDIT_CAPABILITY&email=PRIVATE_QUERY",
                    headers={"X-Request-ID": "audit-" + route},
                )
                try:
                    response = urllib.request.urlopen(request, timeout=5)
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    assert response.status == status
                    if status != 500:
                        assert response.headers["x-request-id"] == "audit-" + route
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    text = logfile.read_text()
    assert "DUMMY_EDIT_CAPABILITY" not in text
    assert "PRIVATE_QUERY" not in text
    records = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    for route, status in [("ok", 200), ("denied", 403), ("fail", 500)]:
        matching = [r for r in records if r.get("request_id") == "audit-" + route]
        assert len(matching) == 1
        assert matching[0]["path"] == "/" + route
        assert matching[0]["status_code"] == status
        assert matching[0]["elapsed_ms"] >= 0
    failed = next(r for r in records if r.get("request_id") == "audit-fail")
    assert failed["event"] == "request.failed"
    assert failed["exception_type"] == "ValueError"
