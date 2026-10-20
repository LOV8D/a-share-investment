"""Run an isolated cloud collection probe without changing the paper account."""
from __future__ import annotations

import ipaddress
from pathlib import Path
import shlex
import subprocess
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/preflight_cloud_collection.py PUBLIC_IP")
    host = str(ipaddress.ip_address(sys.argv[1]))
    code = (
        "from pathlib import Path; "
        "from ashare.live_data import collect_live; "
        "s=collect_live('2026-09-18',live_root=Path('/app/data/live-preflight'),"
        "include_text=False,progress=lambda m,p:print(f'{p:.0%} {m}',flush=True)); "
        "print('PREFLIGHT_RESULT',s['passed'],s['signal_date'],s['gate']['price_coverage'])"
    )
    remote = (
        "cd /home/ubuntu/a-share-investment && "
        "sudo docker compose run --rm -T app python -c " + shlex.quote(code)
    )
    subprocess.run(
        [
            "ssh", "-i", str(Path.home() / ".ssh/a_share_investment_tencent"),
            "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            f"ubuntu@{host}", remote,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
