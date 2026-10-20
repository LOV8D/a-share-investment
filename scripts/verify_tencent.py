"""Verify a Lighthouse deployment without printing its generated password."""
from __future__ import annotations

import ipaddress
import json
from pathlib import Path
import subprocess
import sys

import httpx


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/verify_tencent.py PUBLIC_IP")
    host = str(ipaddress.ip_address(sys.argv[1]))
    identity = Path.home() / ".ssh/a_share_investment_tencent"
    result = subprocess.run(
        [
            "ssh", "-i", str(identity), "-o", "IdentitiesOnly=yes",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            f"ubuntu@{host}",
            "cat /home/ubuntu/a-share-investment/deployment-credentials.txt",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    credentials = dict(line.split(": ", 1) for line in result.stdout.splitlines() if ": " in line)
    password = credentials["Password"]
    with httpx.Client(auth=(credentials["Username"], password), verify=False, trust_env=False, timeout=30) as client:
        for path, expected in (("/", "A share investment"), ("/static/app.js", "function render"), ("/static/style.css", ".sidebar")):
            response = client.get(f"https://{host}{path}")
            response.raise_for_status()
            assert expected in response.text, path
            print(f"{path}: HTTP {response.status_code}")
        payloads = {}
        for path in ("/api/health", "/api/status", "/api/live-status", "/api/data", "/api/report"):
            response = client.get(f"https://{host}{path}")
            response.raise_for_status()
            payloads[path] = response.json()
            print(f"{path}: HTTP {response.status_code}")
    health = payloads["/api/health"]
    status = payloads["/api/status"]
    live = payloads["/api/live-status"]
    data = payloads["/api/data"]
    report = payloads["/api/report"]
    assert health["status"] == "ok"
    assert status.get("latest_run_id")
    assert data.get("passed") is True, json.dumps(data.get("errors", []), ensure_ascii=False)
    assert not report.get("empty")
    print("latest_run_id:", status["latest_run_id"])
    print("live_signal_date:", live.get("signal_date"))
    print("history_quality_passed:", data["passed"])


if __name__ == "__main__":
    main()
