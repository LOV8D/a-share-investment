"""Process the RESSET index exports and the CSMAR corporate-action export.

The downloaded ZIP files are never edited.  Immutable copies and SHA-256
fingerprints are kept under ``data/raw``.  Usable tables are written to
``data/staging/real``.  The historical membership export supplied on
2026-09-20 is intentionally kept as an *incomplete candidate*: its date filter
removed constituents that joined the CSI 300 before 2009, so it must not be
promoted to ``data/inbox`` or used for a backtest.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWNLOADS = Path.home() / "Downloads"
STAGING = ROOT / "data" / "staging" / "real"
CSMAR_STAGING = ROOT / "data" / "staging" / "csmar"


def log(message: str) -> None:
    print(message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def archive(path: Path, vendor: str, batch: str, family: str) -> dict[str, object]:
    destination = ROOT / "data" / "raw" / vendor / "FULL" / batch
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / path.name
    digest = sha256(path)
    if target.exists() and sha256(target) != digest:
        raise ValueError(f"原始归档冲突：{target}")
    if not target.exists():
        log(f"归档 {path.name}")
        shutil.copy2(path, target)
    return {
        "family": family,
        "source_path": str(path),
        "archived_path": str(target),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    }


def find_sources(downloads: Path) -> tuple[list[Path], list[Path], Path]:
    membership = sorted(downloads.glob("RESSET 股票_指数成分*.zip"))
    quotations = sorted(downloads.glob("RESSET 股票_指数行情*.zip"))
    actions = sorted(downloads.glob("*红利分配事件表*.zip"))
    if not membership:
        raise FileNotFoundError("Downloads 中找不到 RESSET 股票_指数成分 ZIP")
    if not quotations:
        raise FileNotFoundError("Downloads 中找不到 RESSET 股票_指数行情 ZIP")
    if len(actions) != 1:
        raise FileNotFoundError(f"应有 1 个红利分配事件表 ZIP，实际找到 {len(actions)} 个")
    return membership, quotations, actions[0]


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(io.BytesIO(raw), dtype=str, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError("无法识别 CSV 编码") from last_error


def read_resset_archives(paths: list[Path], token: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        with ZipFile(path) as bundle:
            for name in bundle.namelist():
                if token not in Path(name).name or "数据词典" in name:
                    continue
                lower = name.lower()
                if lower.endswith(".csv"):
                    frame = read_csv_bytes(bundle.read(name))
                elif lower.endswith(".xlsx"):
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="Workbook contains no default style")
                        frame = pd.read_excel(io.BytesIO(bundle.read(name)), dtype=str)
                else:
                    continue
                frame["_source_zip"] = path.name
                frames.append(frame)
    if not frames:
        raise ValueError(f"RESSET ZIP 中找不到 {token} 数据表")
    return pd.concat(frames, ignore_index=True)


def official_json(path: Path, value_name: str) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("success") or not isinstance(payload.get("data"), list):
        raise ValueError(f"中证指数文件结构异常：{path.name}")
    frame = pd.DataFrame(payload["data"])
    frame["date"] = pd.to_datetime(frame["tradeDate"], format="%Y%m%d", errors="raise")
    frame[value_name] = pd.to_numeric(frame["close"], errors="raise")
    return frame[["date", value_name]].sort_values("date").drop_duplicates("date", keep="last")


def process_benchmark(paths: list[Path], batch: str) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = read_resset_archives(paths, "IDXQTTN")
    required = ["指数代码_IdxCd", "交易日期_TrdDt", "收盘价(元/点)_ClPr"]
    missing = set(required) - set(raw)
    if missing:
        raise ValueError(f"RESSET 指数行情缺字段：{sorted(missing)}")
    raw = raw.loc[raw["指数代码_IdxCd"].astype(str).str.zfill(6).eq("000300")].copy()
    raw["date"] = pd.to_datetime(raw["交易日期_TrdDt"], errors="raise").dt.normalize()
    raw["close"] = pd.to_numeric(raw["收盘价(元/点)_ClPr"], errors="raise")
    duplicate = raw.groupby("date")["close"].agg(["size", "min", "max"])
    conflict_dates = int(((duplicate["max"] - duplicate["min"]).abs() > 1e-8).sum())
    if conflict_dates:
        raise ValueError(f"RESSET 重叠区间有 {conflict_dates} 个日期的收盘值互相冲突")
    price = raw[["date", "close"]].sort_values("date").drop_duplicates("date", keep="last")

    official_root = ROOT / "data" / "raw" / "CSINDEX" / "FULL" / batch
    price_path = official_root / "000300_20090101_20260901.json"
    total_path = official_root / "H00300_20090101_20260901.json"
    if not price_path.exists() or not total_path.exists():
        raise FileNotFoundError("缺少已保存的中证指数价格/全收益 JSON")
    official_price = official_json(price_path, "official_close")
    official_total = official_json(total_path, "total_return_close")
    benchmark = price.merge(official_total, on="date", how="left", validate="one_to_one")
    missing_total = int(benchmark["total_return_close"].isna().sum())
    if missing_total:
        raise ValueError(f"全收益指数缺少 {missing_total} 个 RESSET 交易日")
    comparison = benchmark.merge(official_price, on="date", how="left", validate="one_to_one")
    max_price_difference = float((comparison["close"] - comparison["official_close"]).abs().max())
    if max_price_difference > 0.011:
        raise ValueError(f"RESSET 与中证指数收盘点位最大差异 {max_price_difference:.6f}，超出舍入容差")
    benchmark = benchmark[["date", "close", "total_return_close"]].sort_values("date")
    benchmark.to_parquet(STAGING / "benchmark.parquet", index=False)
    pd.DataFrame({"date": benchmark["date"]}).to_parquet(STAGING / "calendar.parquet", index=False)
    return benchmark, {
        "rows": len(benchmark),
        "start": str(benchmark.date.min().date()),
        "end": str(benchmark.date.max().date()),
        "raw_rows": len(raw),
        "overlap_rows_removed": int(len(raw) - len(benchmark)),
        "overlap_conflict_dates": conflict_dates,
        "total_return_missing_dates": missing_total,
        "max_abs_close_difference_vs_csindex": max_price_difference,
        "price_source": "RESSET 股票_指数行情",
        "total_return_source": "中证指数 H00300 官方接口留档",
    }


def sum_unique(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.unique(np.round(values, 10)).sum()) if len(values) else 0.0


def process_actions(path: Path, symbol_map: dict[str, str]) -> tuple[pd.DataFrame, dict[str, object]]:
    with ZipFile(path) as bundle:
        names = [name for name in bundle.namelist() if Path(name).name == "ER_ProfitSharing.csv"]
        if len(names) != 1:
            raise ValueError(f"红利分配 ZIP 中 ER_ProfitSharing.csv 数量不是 1：{names}")
        raw = read_csv_bytes(bundle.read(names[0]))
    required = [
        "EventID", "Symbol", "Term", "DivdendYear", "DeclareDate", "BonusRatio",
        "ConversionRatio", "DividentBT", "RecordDate", "ExDividendDate",
    ]
    missing = set(required) - set(raw)
    if missing:
        raise ValueError(f"红利分配事件表缺字段：{sorted(missing)}")
    source_rows = len(raw)
    raw["code"] = raw["Symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
    for column in ("DeclareDate", "RecordDate", "ExDividendDate"):
        raw[column] = pd.to_datetime(raw[column], errors="coerce").dt.normalize()
    for column in ("BonusRatio", "ConversionRatio", "DividentBT"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    invalid_date_rows = int(raw["ExDividendDate"].isna().sum())
    unmapped_rows = int((~raw["code"].isin(symbol_map)).sum())
    usable = raw.loc[raw["ExDividendDate"].notna() & raw["code"].isin(symbol_map)].copy()
    usable["symbol"] = usable["code"].map(symbol_map)

    # One plan can appear on multiple rows (cash and shares separately), and an
    # identical share ratio is occasionally repeated.  Deduplicate equal values
    # inside the same plan, then add distinct components and concurrent plans.
    plan_keys = [
        "symbol", "ExDividendDate", "Term", "DivdendYear", "DeclareDate", "RecordDate"
    ]
    plans = (
        usable.groupby(plan_keys, dropna=False)
        .agg(
            bonus_per_10=("BonusRatio", sum_unique),
            conversion_per_10=("ConversionRatio", sum_unique),
            cash_per_10=("DividentBT", sum_unique),
            source_events=("EventID", "nunique"),
        )
        .reset_index()
    )
    grouped = (
        plans.groupby(["ExDividendDate", "symbol"], as_index=False)
        .agg(
            bonus_per_10=("bonus_per_10", "sum"),
            conversion_per_10=("conversion_per_10", "sum"),
            cash_per_10=("cash_per_10", "sum"),
            source_events=("source_events", "sum"),
            source_plans=("Term", "size"),
        )
        .rename(columns={"ExDividendDate": "date"})
    )
    grouped["share_multiplier"] = 1 + (grouped["bonus_per_10"] + grouped["conversion_per_10"]) / 10
    grouped["cash_per_share"] = grouped["cash_per_10"] / 10
    # The export has the preliminary-plan date, not the later implementation
    # announcement timestamp.  Mark the exact terms available at 00:00 on the
    # effective date; this is conservative and sufficient for book adjustment.
    grouped["known_at"] = grouped["date"].dt.tz_localize("Asia/Shanghai")
    invalid_values = grouped.loc[
        grouped["share_multiplier"].le(0) | grouped["cash_per_share"].lt(0)
    ]
    if len(invalid_values):
        raise ValueError(f"聚合后有 {len(invalid_values)} 个公司行动数值不合法")
    actions = grouped[["date", "symbol", "share_multiplier", "cash_per_share", "known_at"]].copy()
    actions = actions.sort_values(["date", "symbol"]).reset_index(drop=True)
    if actions.duplicated(["date", "symbol"]).any():
        raise ValueError("公司行动聚合后仍有重复 date + symbol")
    actions.to_parquet(STAGING / "actions.parquet", index=False)
    plans.to_parquet(STAGING / "actions_plan_detail.parquet", index=False)
    return actions, {
        "source_rows": source_rows,
        "rows": len(actions),
        "symbols": int(actions.symbol.nunique()),
        "start": str(actions.date.min().date()),
        "end": str(actions.date.max().date()),
        "source_rows_without_ex_date": invalid_date_rows,
        "source_rows_not_in_observed_a_share_map": unmapped_rows,
        "duplicate_event_ids": int(raw["EventID"].duplicated().sum()),
        "aggregated_same_symbol_date_groups": int(
            usable.groupby(["symbol", "ExDividendDate"]).size().gt(1).sum()
        ),
        "availability_method": "effective_date_midnight_conservative_fallback",
        "ratio_unit": "source per 10 shares; output multiplier/per-share cash",
    }


def normalize_membership(raw: pd.DataFrame, symbol_map: dict[str, str]) -> pd.DataFrame:
    required = [
        "指数代码_IdxCd", "证券代码_成分_SecuCd_Compo", "证券名称_成分_SecuNm_Compo",
        "开始日期_BegDt", "结束日期_EndDt",
    ]
    missing = set(required) - set(raw)
    if missing:
        raise ValueError(f"RESSET 指数成分缺字段：{sorted(missing)}")
    frame = raw.loc[raw["指数代码_IdxCd"].astype(str).str.zfill(6).eq("000300")].copy()
    frame["code"] = frame["证券代码_成分_SecuCd_Compo"].astype(str).str.extract(r"(\d{6})", expand=False)
    frame["begin_date"] = pd.to_datetime(frame["开始日期_BegDt"], errors="raise").dt.normalize()
    frame["end_date"] = pd.to_datetime(frame["结束日期_EndDt"], errors="coerce").dt.normalize()
    frame["symbol"] = frame["code"].map(symbol_map)
    columns = ["symbol", "code", "证券名称_成分_SecuNm_Compo", "begin_date", "end_date"]
    frame = frame[columns].rename(columns={"证券名称_成分_SecuNm_Compo": "name"})
    return frame.sort_values(["symbol", "begin_date", "end_date"], na_position="last").drop_duplicates().reset_index(drop=True)


def expand_candidate(intervals: pd.DataFrame, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    for row in intervals.dropna(subset=["symbol"]).itertuples(index=False):
        mask = dates >= row.begin_date
        if pd.notna(row.end_date):
            mask &= dates < row.end_date  # RESSET end date is replacement/effective-out date.
        active = dates[mask]
        if len(active):
            pieces.append(pd.DataFrame({"date": active, "symbol": row.symbol}))
    universe = pd.concat(pieces, ignore_index=True).drop_duplicates(["date", "symbol"])
    counts = universe.groupby("date", as_index=False).size().rename(columns={"size": "constituents"})
    universe = universe.merge(counts, on="date", how="left", validate="many_to_one")
    universe["weight"] = 1 / universe["constituents"]
    universe["known_at"] = universe["date"].dt.tz_localize("Asia/Shanghai")
    universe = universe[["date", "symbol", "weight", "known_at"]].sort_values(["date", "symbol"])
    return universe.reset_index(drop=True), counts.sort_values("date").reset_index(drop=True)


def read_official_current(batch: str) -> pd.DataFrame | None:
    root = ROOT / "data" / "raw" / "CSINDEX" / "FULL" / batch
    xls = root / f"000300cons_{batch}.xls"
    if not xls.exists():
        return None
    candidates = [Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")]
    resolved = shutil.which("soffice")
    if resolved:
        candidates.insert(0, Path(resolved))
    executable = next((path for path in candidates if path.exists()), None)
    if executable is None:
        return None
    with tempfile.TemporaryDirectory(prefix="ashare-csi-") as folder:
        result = subprocess.run(
            [str(executable), "--headless", "--convert-to", "csv", "--outdir", folder, str(xls)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        converted = sorted(Path(folder).glob("*.csv"))
        if result.returncode != 0 or not converted:
            return None
        frame = pd.read_csv(converted[0], dtype=str)
    code_column = next((column for column in frame if "成份券代码" in column), None)
    date_column = next((column for column in frame if "日期" in column), None)
    if code_column is None:
        return None
    frame["code"] = frame[code_column].astype(str).str.extract(r"(\d{6})", expand=False)
    frame["date"] = pd.to_datetime(frame[date_column], format="%Y%m%d", errors="coerce") if date_column else pd.NaT
    return frame.dropna(subset=["code"])


def diagnose_open_source(dates: pd.DatetimeIndex) -> dict[str, object] | None:
    path = ROOT / "data" / "raw" / "OPEN_SOURCE" / "FULL" / "20260920" / "index-constitution_csi300_history.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    frame["begin_date"] = pd.to_datetime(frame["opt-in"], errors="coerce")
    frame["end_date"] = pd.to_datetime(frame["opt-out"], errors="coerce")
    pieces = []
    for row in frame.itertuples(index=False):
        mask = dates >= row.begin_date
        if pd.notna(row.end_date):
            mask &= dates < row.end_date
        active = dates[mask]
        if len(active):
            pieces.append(pd.DataFrame({"date": active, "symbol": row.symbol}))
    expanded = pd.concat(pieces, ignore_index=True).drop_duplicates(["date", "symbol"])
    counts = expanded.groupby("date", as_index=False).size().rename(columns={"size": "constituents"})
    counts.to_csv(STAGING / "open_source_fallback_daily_counts.csv", index=False)
    distribution = {str(int(k)): int(v) for k, v in counts.constituents.value_counts().sort_index().items()}
    return {
        "source_rows": len(frame),
        "minimum_daily_count": int(counts.constituents.min()),
        "maximum_daily_count": int(counts.constituents.max()),
        "exact_300_dates": int(counts.constituents.eq(300).sum()),
        "count_distribution": distribution,
        "decision": "quality-control-only; rejected as final licensed universe because many dates have 299 or 301 names",
    }


def process_membership(
    paths: list[Path], symbol_map: dict[str, str], dates: pd.DatetimeIndex, batch: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = read_resset_archives(paths, "IDXCOMPO")
    intervals = normalize_membership(raw, symbol_map)
    unmapped = int(intervals.symbol.isna().sum())
    intervals.to_parquet(STAGING / "membership_intervals_candidate.parquet", index=False)
    universe, counts = expand_candidate(intervals, dates)
    universe.to_parquet(STAGING / "universe_candidate_incomplete.parquet", index=False)
    counts.to_csv(STAGING / "membership_candidate_daily_counts.csv", index=False)

    official = read_official_current(batch)
    current_check: dict[str, object] | None = None
    if official is not None:
        official_codes = set(official.code)
        active_codes = set(intervals.loc[intervals.end_date.isna(), "code"])
        missing_codes = sorted(official_codes - active_codes)
        extras = sorted(active_codes - official_codes)
        official[[column for column in official.columns if not column.startswith("Unnamed")]].to_parquet(
            STAGING / "official_current_constituents.parquet", index=False
        )
        current_check = {
            "official_rows": len(official_codes),
            "resset_active_rows": len(active_codes),
            "resset_is_subset_of_official": active_codes.issubset(official_codes),
            "missing_from_resset": len(missing_codes),
            "extra_in_resset": len(extras),
            "missing_codes": missing_codes,
        }

    first_count = int(counts.iloc[0].constituents)
    last_count = int(counts.iloc[-1].constituents)
    non_300 = counts.loc[counts.constituents.ne(300), ["date", "constituents"]].copy()
    expected_temporary_vacancy = {
        pd.Timestamp("2009-12-29"),
        pd.Timestamp("2009-12-30"),
        pd.Timestamp("2009-12-31"),
    }
    vacancy_dates = set(non_300.date)
    current_matches = bool(
        current_check
        and current_check["official_rows"] == 300
        and current_check["resset_active_rows"] == 300
        and current_check["missing_from_resset"] == 0
        and current_check["extra_in_resset"] == 0
    )
    complete = bool(
        first_count == 300
        and last_count == 300
        and current_matches
        and vacancy_dates == expected_temporary_vacancy
        and non_300.constituents.eq(298).all()
    )
    if complete:
        universe.to_parquet(STAGING / "universe_validated.parquet", index=False)
    return intervals, {
        "raw_rows_across_duplicate_exports": len(raw),
        "unique_interval_rows": len(intervals),
        "symbols": int(intervals.code.nunique()),
        "begin_date_min": str(intervals.begin_date.min().date()),
        "begin_date_max": str(intervals.begin_date.max().date()),
        "active_intervals_without_end_date": int(intervals.end_date.isna().sum()),
        "unmapped_codes": unmapped,
        "candidate_daily_rows": len(universe),
        "first_session": str(counts.iloc[0].date.date()),
        "first_session_constituents": first_count,
        "last_session": str(counts.iloc[-1].date.date()),
        "last_session_constituents": last_count,
        "minimum_daily_count": int(counts.constituents.min()),
        "maximum_daily_count": int(counts.constituents.max()),
        "dates_with_300": int(counts.constituents.eq(300).sum()),
        "complete": bool(complete),
        "non_300_dates": [
            {"date": str(row.date.date()), "constituents": int(row.constituents)}
            for row in non_300.itertuples(index=False)
        ],
        "temporary_vacancy_explanation": (
            "邯郸钢铁(600001)与承德钒钛(600357)于2009-12-29退市；"
            "定期替补在2010-01-04生效，因此三个交易日为298只。"
            if complete
            else None
        ),
        "temporary_vacancy_source": (
            "https://www.sse.com.cn/market/sseindex/diclosure/c/c_20150911_3984987.shtml"
            if complete
            else None
        ),
        "interval_convention": "begin inclusive, end exclusive",
        "candidate_weight": "equal 1/N diagnostic weight; not vendor index weight",
        "official_current_check": current_check,
    }


def write_fees() -> tuple[pd.DataFrame, dict[str, object]]:
    fees = pd.DataFrame(
        [
            ["2005-01-01", "2015-07-31", 0.0010, 0.00002],
            ["2015-08-01", "2022-04-28", 0.0010, 0.00002],
            ["2022-04-29", "2023-08-27", 0.0010, 0.00001],
            ["2023-08-28", "2099-12-31", 0.0005, 0.00001],
        ],
        columns=["start_date", "end_date", "sell_stamp_rate", "transfer_rate"],
    )
    fees[["start_date", "end_date"]] = fees[["start_date", "end_date"]].apply(pd.to_datetime)
    fees.to_parquet(STAGING / "fees.parquet", index=False)
    return fees, {
        "rows": len(fees),
        "stamp_duty": "sell only: 0.10% through 2023-08-27; 0.05% from 2023-08-28",
        "transfer_fee": "0.02 per mille through 2022-04-28; 0.01 per mille from 2022-04-29",
        "pre_2015_note": "single-rate schema uses 0.02 per mille approximation before nationwide unification; commission/minimum fee still applies separately",
    }


def report_markdown(manifest: dict) -> str:
    b = manifest["benchmark"]
    a = manifest["actions"]
    m = manifest["membership"]
    current = m.get("official_current_check") or {}
    if m["complete"]:
        membership_section = f"""## 历史成分已通过

- RESSET 成分文件去重后有 {m['unique_interval_rows']:,} 条区间，覆盖首批成分至当前；首日和末日均为 300 只。
- 当前 300 只与中证指数官方名单完全一致。
- 2009-12-29 至 2009-12-31 为 298 只：邯郸钢铁与承德钒钛退市后，定期替补在 2010-01-04 生效。这是官方规则造成的短暂空缺，不是数据遗漏。
- 已生成 `universe_validated.parquet`，可以进入正式组装。"""
        next_section = """## 下一步

不再需要下载结构化历史数据。下一步组装 `data/inbox`，补齐停牌日和行业，再运行真实结构化回测。"""
    else:
        membership_section = f"""## 不能用于回测的成分候选

- RESSET 成分文件去重后只有 {m['unique_interval_rows']:,} 条区间，开始日期最早是 {m['begin_date_min']}。
- 展开后，{m['first_session']} 只有 **{m['first_session_constituents']}** 只；{m['last_session']} 只有 **{m['last_session_constituents']}** 只，不是完整沪深300。
- 当前无结束日期的 RESSET 成分只有 {m['active_intervals_without_end_date']} 只。与中证指数当前名单核对：官方 {current.get('official_rows', '—')} 只，RESSET 少 {current.get('missing_from_resset', '—')} 只。
- 原因已经定位：下载时把“开始日期”设成 2009 年，数据库因此排除了 2009 年以前已经入选且之后仍在指数中的股票。
- 候选文件保存在 `data/staging/real` 只用于诊断，没有复制到 `data/inbox`，引擎不会误跑。"""
        next_section = """## 唯一还要补的文件

重新导出 RESSET **股票_指数成分**：指数代码 `000300`，开始日期从 `2005-04-08`（指数发布日期/首批成分起始日）开始，或把开始日期条件留空；结束日期选最新；字段全部选择。不要只导出 2009 年以后开始的区间。"""
    return f"""# 新下载数据处理与质量报告

生成时间：{manifest['created_at']}

## 已经可用

- **沪深300基准**：{b['rows']:,} 个交易日，{b['start']} 至 {b['end']}。普通指数来自 RESSET；全收益指数来自中证指数 H00300 官方接口留档。RESSET 与中证指数收盘点位最大绝对差 {b['max_abs_close_difference_vs_csindex']:.6f} 点，仅为显示精度差。
- **公司行动**：{a['rows']:,} 个 `除权除息日 + 股票` 事件，覆盖 {a['start']} 至 {a['end']}。源文件的“每10股”已换成每股现金和股数乘数。
- **交易日历**：跟随 RESSET 指数行情，共 {b['rows']:,} 个日期。
- **历史税费表**：已生成；2009–2015 年过户费因标准表暂不区分沪深市场，使用有明确标记的近似值。

{membership_section}

{next_section}

历史新闻、政策和论坛不下载、不回测。财报实际披露日期目前用法定截止日次日的保守口径，不阻止第一版结构化回测。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, default=DEFAULT_DOWNLOADS)
    parser.add_argument("--batch", default="20260920")
    args = parser.parse_args()

    membership_paths, quotation_paths, action_path = find_sources(args.downloads)
    STAGING.mkdir(parents=True, exist_ok=True)
    provenance: list[dict[str, object]] = []
    for path in membership_paths:
        provenance.append(archive(path, "RESSET", args.batch, "csi300_membership_candidate"))
    for path in quotation_paths:
        provenance.append(archive(path, "RESSET", args.batch, "csi300_price_index"))
    provenance.append(archive(action_path, "CSMAR", args.batch, "corporate_actions"))

    symbol_frame = pd.read_parquet(CSMAR_STAGING / "security_observation_range.parquet")
    symbol_map = dict(zip(symbol_frame["Stkcd"].astype(str).str.zfill(6), symbol_frame["symbol"]))
    csmar_manifest = json.loads((CSMAR_STAGING / "manifest.json").read_text(encoding="utf-8"))

    log("处理沪深300价格指数和全收益指数")
    benchmark, benchmark_stats = process_benchmark(quotation_paths, args.batch)
    log("处理除权除息与送转事件")
    actions, action_stats = process_actions(action_path, symbol_map)
    log("诊断沪深300历史成分完整性")
    intervals, membership_stats = process_membership(
        membership_paths, symbol_map, pd.DatetimeIndex(benchmark.date), args.batch
    )
    log("生成历史税费表")
    fees, fee_stats = write_fees()
    open_source_stats = diagnose_open_source(pd.DatetimeIndex(benchmark.date))

    blocker = (
        "RESSET「股票_指数成分」完整历史：指数 000300；开始日期从 2005-04-08，"
        "或不限制开始日期；结束日期到最新；字段全选。当前文件遗漏 2009 年前已入选的成分。"
    )
    membership_ready = bool(membership_stats["complete"])
    manifest = {
        "batch": args.batch,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source": "CSMAR + RESSET + CSI official index series supplied/approved by user",
        "status": "processed_staging_ready_for_assembly" if membership_ready else "processed_staging_waiting_for_complete_membership",
        "ready_for_backtest": False,
        "ready_for_assembly": membership_ready,
        "historical_text_backtest": False,
        "provenance": provenance,
        "prices": csmar_manifest["prices"],
        "fundamentals": csmar_manifest["fundamentals"],
        "benchmark": benchmark_stats,
        "actions": action_stats,
        "calendar": {
            "rows": len(benchmark),
            "start": benchmark_stats["start"],
            "end": benchmark_stats["end"],
            "future_sessions_after_last_stock_price": int(
                (benchmark.date > pd.Timestamp(csmar_manifest["prices"]["end"])).sum()
            ),
        },
        "fees": fee_stats,
        "membership": membership_stats,
        "open_source_membership_diagnostic": open_source_stats,
        "staged_tables": {
            "prices": {"rows": csmar_manifest["prices"]["written_rows"], "path": "data/staging/csmar/prices_observed.parquet", "status": "ready_for_universe_filter_and_industry_merge"},
            "fundamentals": {"rows": csmar_manifest["fundamentals"]["rows"], "path": "data/staging/csmar/fundamentals_provisional.parquet", "status": "provisional_availability"},
            "benchmark": {"rows": len(benchmark), "path": "data/staging/real/benchmark.parquet", "status": "ready"},
            "actions": {"rows": len(actions), "path": "data/staging/real/actions.parquet", "status": "ready"},
            "calendar": {"rows": len(benchmark), "path": "data/staging/real/calendar.parquet", "status": "ready"},
            "fees": {"rows": len(fees), "path": "data/staging/real/fees.parquet", "status": "ready_with_pre_2015_approximation"},
            "universe": {
                "rows": membership_stats["candidate_daily_rows"],
                "path": "data/staging/real/universe_validated.parquet" if membership_ready else "data/staging/real/universe_candidate_incomplete.parquet",
                "status": "validated_ready_for_assembly" if membership_ready else "blocked_incomplete_do_not_backtest",
            },
        },
        "blocking_inputs": [] if membership_ready else [blocker],
        "validation_upgrade_inputs": [
            "财报真实首次披露时间（最终验收升级项；第一版继续用法定截止日次日）",
            "逐日官方成分权重（当前选股只需要成员关系；未来做指数增强和主动风险约束时再补）",
        ],
        "assumptions": csmar_manifest.get("assumptions", []) + [
            "公司行动源字段为每10股，已按计划内去重后换算成每股现金和股数乘数。",
            "公司行动缺实施公告时间，known_at 保守记为除权除息日 00:00 Asia/Shanghai。",
            "RESSET 成分结束日期按退出生效日处理：开始日包含，结束日不包含。",
            "不使用公开社区成分表替代学校授权数据；该表只做完整性对照。",
        ],
    }
    json_write(STAGING / "manifest.json", manifest)
    (STAGING / "数据组装状态.md").write_text(report_markdown(manifest), encoding="utf-8")
    membership_message = "历史成分已通过，可进入正式组装" if membership_ready else "历史成分仍不完整，未进入正式回测目录"
    log(
        f"完成：基准 {len(benchmark):,} 行，公司行动 {len(actions):,} 行；"
        f"成分首日 {membership_stats['first_session_constituents']} 只、末日 "
        f"{membership_stats['last_session_constituents']} 只，{membership_message}。"
    )


if __name__ == "__main__":
    main()
