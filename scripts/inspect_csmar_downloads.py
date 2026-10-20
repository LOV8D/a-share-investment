"""Inspect CSMAR download archives without changing the original files."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


DEFAULT_PATTERNS = (
    "/Users/watson/Downloads/*回报率*.zip",
    "/Users/watson/Downloads/*资产负债表*.zip",
    "/Users/watson/Downloads/*利润表*.zip",
    "/Users/watson/Downloads/*现金流量表*.zip",
    "/Users/watson/Downloads/*股利分配*.zip",
)


def archives() -> list[Path]:
    found: dict[str, Path] = {}
    for pattern in DEFAULT_PATTERNS:
        for raw in glob.glob(pattern):
            path = Path(raw)
            if " (1).zip" not in path.name:
                found[str(path)] = path
    return sorted(found.values(), key=lambda item: item.name)


def decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample() -> None:
    for path in archives():
        print(f"\nZIP {path.name}")
        with ZipFile(path) as bundle:
            for info in bundle.infolist():
                if not info.filename.lower().endswith(".csv"):
                    continue
                print(f"ENTRY {info.filename} bytes={info.file_size}")
                with bundle.open(info) as source:
                    for _ in range(4):
                        line = decode(source.readline()).rstrip("\r\n")
                        print(line[:1800])


def scan(output: Path) -> None:
    report: dict[str, object] = {"archives": []}
    for path in archives():
        archive_report: dict[str, object] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "entries": [],
        }
        print(f"Scanning {path.name}", flush=True)
        with ZipFile(path) as bundle:
            for info in bundle.infolist():
                if not info.filename.lower().endswith(".csv"):
                    continue
                entry: dict[str, object] = {
                    "name": info.filename,
                    "size_bytes": info.file_size,
                    "rows": 0,
                    "columns": [],
                    "first_data_row": None,
                    "last_data_row": None,
                }
                with bundle.open(info) as binary:
                    text = (decode(raw) for raw in binary)
                    reader = csv.reader(text)
                    try:
                        header = next(reader)
                    except StopIteration:
                        header = []
                    entry["columns"] = header
                    for row in reader:
                        if not row:
                            continue
                        entry["rows"] = int(entry["rows"]) + 1
                        if entry["first_data_row"] is None:
                            entry["first_data_row"] = row
                        entry["last_data_row"] = row
                archive_report["entries"].append(entry)
                print(
                    f"  {info.filename}: {entry['rows']} rows; "
                    f"first={entry['first_data_row'][:3] if entry['first_data_row'] else None}; "
                    f"last={entry['last_data_row'][:3] if entry['last_data_row'] else None}",
                    flush=True,
                )
        report["archives"].append(archive_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sample", "scan"))
    parser.add_argument("--output", type=Path, default=Path("runs/csmar-download-scan.json"))
    args = parser.parse_args()
    if args.mode == "sample":
        sample()
    else:
        scan(args.output)


if __name__ == "__main__":
    main()
