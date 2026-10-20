from datetime import datetime
from zoneinfo import ZoneInfo

from ashare.scheduler import _should_attempt


def test_cloud_scheduler_runs_at_midnight_once_only():
    midnight = datetime(2026, 9, 22, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    seeded = {"last_success_signal": "2026-09-18", "last_result": "seeded"}
    failed = {"last_attempt_date": "2026-09-22", "last_result": "failed"}
    success = {"last_attempt_date": "2026-09-22", "last_result": "success"}
    assert _should_attempt(midnight, seeded)
    assert not _should_attempt(midnight, failed)
    assert not _should_attempt(midnight, success)
    assert not _should_attempt(midnight, seeded, enabled=False)
    assert not _should_attempt(datetime(2026, 9, 21, 23, 59, tzinfo=ZoneInfo("Asia/Shanghai")), seeded, "23:59:59")
