"""Print compact schema and coverage diagnostics for the RESSET ZIP exports."""

from __future__ import annotations

import glob
import io
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


def inspect(path: Path) -> None:
    print(f"\nZIP {path.name}")
    with ZipFile(path) as bundle:
        for name in bundle.namelist():
            lower = name.lower()
            if lower.endswith(".csv"):
                raw = bundle.read(name)
                try:
                    frame = pd.read_csv(io.BytesIO(raw), dtype=str, encoding="utf-8-sig")
                except UnicodeDecodeError:
                    frame = pd.read_csv(io.BytesIO(raw), dtype=str, encoding="gb18030")
                print(f"CSV {name}: rows={len(frame):,} columns={list(frame.columns)}")
                print(frame.head(3).to_string(index=False))
                print(frame.tail(3).to_string(index=False))
                coverage(frame)
            elif lower.endswith((".xlsx", ".xls")):
                raw = bundle.read(name)
                book = pd.ExcelFile(io.BytesIO(raw))
                for sheet in book.sheet_names:
                    frame = pd.read_excel(book, sheet_name=sheet, dtype=str)
                    print(f"XLSX {name}/{sheet}: rows={len(frame):,} columns={list(frame.columns)}")
                    print(frame.head(3).to_string(index=False))
                    print(frame.tail(3).to_string(index=False))
                    coverage(frame)


def coverage(frame: pd.DataFrame) -> None:
    for column in frame.columns:
        label = str(column).lower()
        if any(token in label for token in ("date", "日期", "trddt", "begdt", "enddt")):
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.notna().any():
                print(f"  DATE {column}: {parsed.min().date()} -> {parsed.max().date()} non-null={parsed.notna().sum():,}")
    for column in frame.columns:
        label = str(column).lower()
        if any(token in label for token in ("指数代码", "idxcd", "indexcode")):
            print(f"  INDEX {column}: {frame[column].dropna().unique()[:20].tolist()}")


def main() -> None:
    paths = sorted(Path(p) for p in glob.glob("/Users/watson/Downloads/RESSET 股票_指数*.zip"))
    for path in paths:
        inspect(path)


if __name__ == "__main__":
    main()
