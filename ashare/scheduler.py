"""Small container scheduler for the cloud deployment."""
from __future__ import annotations

from datetime import datetime, time as clock_time
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from .config import ROOT, load_config
from .pipeline import write_json


def _schedule() -> dict:
    path = ROOT / "runs/paper_schedule.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _should_attempt(now: datetime, schedule: dict, schedule_time: str = "00:00:00", enabled: bool = True) -> bool:
    today = str(now.date())
    due = now.timetz().replace(tzinfo=None) >= clock_time.fromisoformat(schedule_time)
    attempted = schedule.get("last_attempt_date") == today
    return enabled and due and not attempted


def main():
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print("cloud scheduler ready: Asia/Shanghai, daily 00:00, one attempt only", flush=True)
    while not stopped:
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        cfg = load_config()
        schedule = _schedule()
        if _should_attempt(now, schedule, cfg.schedule_time, cfg.daily_schedule_enabled):
            write_json(ROOT / "runs/paper_schedule.json", schedule | {
                "last_attempt_date": str(now.date()),
                "last_result": "started",
            })
            print(f"{now.isoformat()} starting paper pipeline", flush=True)
            result = subprocess.run([sys.executable, "-m", "ashare", "paper"], cwd=ROOT)
            latest = _schedule()
            if result.returncode:
                write_json(ROOT / "runs/paper_schedule.json", latest | {
                    "last_attempt_date": str(now.date()),
                    "last_result": "failed",
                    "error": f"paper command exited with status {result.returncode}",
                })
                print(f"paper pipeline failed ({result.returncode}); no automatic retry today", flush=True)
            elif latest.get("last_result") == "no_new_session":
                print("no newer completed trading session; no automatic retry today", flush=True)
        for _ in range(6):
            if stopped:
                break
            time.sleep(10)


if __name__ == "__main__":
    main()
