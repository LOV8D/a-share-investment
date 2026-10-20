"""Normalize the CSMAR archives currently in Downloads into auditable staging data.

The script never edits the downloaded archives. It copies one immutable copy into
data/raw, streams the large daily-return files into Parquet, and writes a data
quality report. The result intentionally stays in data/staging until the missing
CSI 300 point-in-time tables are supplied.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWNLOADS = Path.home() / "Downloads"
RAW_ROOT = ROOT / "data" / "raw" / "CSMAR" / "FULL"
STAGING_ROOT = ROOT / "data" / "staging" / "csmar"
A_SHARE_MARKETS = {1, 4, 16, 32}
SH_MARKETS = {1, 32}
ST_STATUSES = {2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16}

PRICE_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "amount",
    "adj_factor",
    "upper_limit",
    "lower_limit",
    "suspended",
    "is_st",
    "market_cap",
    "float_market_cap",
    "turnover_rate",
    "industry",
    "min_order",
    "lot_step",
    "listed_days",
    "available_at",
]


def log(message: str) -> None:
    print(message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_match(folder: Path, pattern: str) -> Path:
    matches = [Path(p) for p in glob.glob(str(folder / pattern)) if " (1).zip" not in Path(p).name]
    if not matches:
        raise FileNotFoundError(f"Downloads 中找不到 {pattern}")
    if len(matches) > 1:
        raise ValueError(f"{pattern} 匹配到多个非重复文件：{[p.name for p in matches]}")
    return matches[0]


def locate_sources(downloads: Path) -> dict[str, list[Path]]:
    daily = sorted(Path(p) for p in glob.glob(str(downloads / "*回报率*.zip")))
    daily = [p for p in daily if " (1).zip" not in p.name]
    if not daily:
        raise FileNotFoundError("Downloads 中找不到日个股回报率 ZIP")
    return {
        "daily": daily,
        "balance": [first_match(downloads, "*资产负债表*.zip")],
        "income": [first_match(downloads, "*利润表*.zip")],
        "cashflow": [first_match(downloads, "*现金流量表*.zip")],
        "dividend_indicator": [first_match(downloads, "*股利分配*.zip")],
    }


def archive_copy(sources: dict[str, list[Path]], batch: str) -> tuple[dict[str, list[Path]], list[dict[str, object]]]:
    destination = RAW_ROOT / batch
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, list[Path]] = defaultdict(list)
    provenance: list[dict[str, object]] = []
    for family, paths in sources.items():
        for source in paths:
            digest = sha256(source)
            target = destination / source.name
            if target.exists() and sha256(target) != digest:
                raise ValueError(f"原始归档冲突：{target}")
            if not target.exists():
                log(f"复制原始文件：{source.name}")
                shutil.copy2(source, target)
            copied[family].append(target)
            provenance.append(
                {
                    "family": family,
                    "source_path": str(source),
                    "archived_path": str(target),
                    "size_bytes": source.stat().st_size,
                    "sha256": digest,
                }
            )
    return dict(copied), provenance


def csv_entries(bundle: ZipFile, prefix: str) -> list[str]:
    return [name for name in bundle.namelist() if name.lower().endswith(".csv") and Path(name).name.startswith(prefix)]


def scan_symbols_and_calendar(paths: list[Path], chunk_size: int) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, int]]:
    earliest: dict[str, pd.Timestamp] = {}
    latest: dict[str, pd.Timestamp] = {}
    markets: dict[str, int] = {}
    calendar: set[np.datetime64] = set()
    counters = {"raw_rows": 0, "a_share_rows": 0, "excluded_market_rows": 0, "market_conflicts": 0}
    log("第一遍扫描行情：建立上市起点、交易日历和市场映射")
    for path in paths:
        log(f"  扫描 {path.name}")
        with ZipFile(path) as bundle:
            for name in csv_entries(bundle, "TRD_Dalyr"):
                with bundle.open(name) as source:
                    reader = pd.read_csv(
                        source,
                        usecols=["Stkcd", "Trddt", "Markettype"],
                        dtype={"Stkcd": "string", "Trddt": "string", "Markettype": "Int64"},
                        chunksize=chunk_size,
                    )
                    for chunk in reader:
                        counters["raw_rows"] += len(chunk)
                        market = pd.to_numeric(chunk["Markettype"], errors="coerce")
                        chunk = chunk.loc[market.isin(A_SHARE_MARKETS)].copy()
                        chunk["Markettype"] = market.loc[chunk.index].astype("int16")
                        counters["a_share_rows"] += len(chunk)
                        dates = pd.to_datetime(chunk["Trddt"], errors="coerce")
                        valid = dates.notna() & chunk["Stkcd"].notna()
                        chunk = chunk.loc[valid]
                        dates = dates.loc[valid]
                        calendar.update(dates.to_numpy(dtype="datetime64[ns]"))
                        frame = pd.DataFrame(
                            {
                                "Stkcd": chunk["Stkcd"].astype(str).str.zfill(6),
                                "date": dates.to_numpy(),
                                "market": chunk["Markettype"].to_numpy(),
                            }
                        )
                        summary = frame.groupby("Stkcd", sort=False).agg(first_date=("date", "min"), last_date=("date", "max"), market=("market", "last"))
                        for code, row in summary.iterrows():
                            first = pd.Timestamp(row.first_date)
                            last = pd.Timestamp(row.last_date)
                            if code not in earliest or first < earliest[code]:
                                earliest[code] = first
                            if code not in latest or last > latest[code]:
                                latest[code] = last
                            current = int(row.market)
                            if code in markets and markets[code] != current:
                                counters["market_conflicts"] += 1
                            markets[code] = current
    counters["excluded_market_rows"] = counters["raw_rows"] - counters["a_share_rows"]
    records = []
    for code in sorted(earliest):
        market = markets[code]
        suffix = "SH" if market in SH_MARKETS else "SZ"
        records.append(
            {
                "Stkcd": code,
                "symbol": f"{code}.{suffix}",
                "market_type": market,
                "first_observed_date": earliest[code],
                "last_observed_date": latest[code],
            }
        )
    symbols = pd.DataFrame(records)
    sessions = pd.DatetimeIndex(sorted(calendar), name="date")
    log(f"  A股行情 {counters['a_share_rows']:,} 行，{len(symbols):,} 只证券，{len(sessions):,} 个交易日")
    return symbols, sessions, counters


def read_financial_csv(path: Path, prefix: str, usecols: list[str]) -> pd.DataFrame:
    with ZipFile(path) as bundle:
        entries = csv_entries(bundle, prefix)
        if len(entries) != 1:
            raise ValueError(f"{path.name} 中 {prefix} CSV 数量不是 1：{entries}")
        with bundle.open(entries[0]) as source:
            return pd.read_csv(source, usecols=usecols, dtype={"Stkcd": "string", "Accper": "string", "Typrep": "string"})


def quarter_rows(frame: pd.DataFrame, symbol_map: pd.Series) -> tuple[pd.DataFrame, int]:
    frame["Stkcd"] = frame["Stkcd"].astype("string").str.zfill(6)
    frame["report_period"] = pd.to_datetime(frame["Accper"], errors="coerce")
    quarter_end = frame["report_period"].dt.strftime("%m-%d").isin(["03-31", "06-30", "09-30", "12-31"])
    frame = frame.loc[frame["Typrep"].eq("A") & frame["Stkcd"].isin(symbol_map.index) & quarter_end].copy()
    frame["symbol"] = frame["Stkcd"].map(symbol_map)
    duplicate_rows = int(frame.duplicated(["symbol", "report_period"], keep=False).sum())
    if duplicate_rows:
        frame = frame.sort_values(["symbol", "report_period"]).drop_duplicates(["symbol", "report_period"], keep="last")
    return frame, duplicate_rows


def conservative_available_at(period: pd.Series) -> pd.Series:
    values = []
    for date in pd.to_datetime(period):
        if date.month == 3:
            deadline = pd.Timestamp(year=date.year, month=4, day=30)
        elif date.month == 6:
            deadline = pd.Timestamp(year=date.year, month=8, day=31)
        elif date.month == 9:
            deadline = pd.Timestamp(year=date.year, month=10, day=31)
        else:
            deadline = pd.Timestamp(year=date.year + 1, month=4, day=30)
        # Midnight after the legal deadline avoids assuming an intraday filing time.
        values.append(deadline.tz_localize("Asia/Shanghai") + pd.offsets.Day(1))
    return pd.Series(values, index=period.index, dtype="datetime64[ns, Asia/Shanghai]")


def ttm_from_ytd(frame: pd.DataFrame, source: str) -> pd.Series:
    base = frame[["symbol", "report_period", source]].copy()
    base["year"] = base["report_period"].dt.year
    prior_same = base[["symbol", "report_period", source]].copy()
    prior_same["report_period"] = prior_same["report_period"] + pd.DateOffset(years=1)
    prior_same = prior_same.rename(columns={source: "prior_same"})
    annual = base.loc[base["report_period"].dt.month.eq(12), ["symbol", "year", source]].copy()
    annual["year"] = annual["year"] + 1
    annual = annual.rename(columns={source: "prior_annual"})
    work = base.merge(prior_same, on=["symbol", "report_period"], how="left").merge(annual, on=["symbol", "year"], how="left")
    current = pd.to_numeric(work[source], errors="coerce")
    annual_row = work["report_period"].dt.month.eq(12)
    result = current + pd.to_numeric(work["prior_annual"], errors="coerce") - pd.to_numeric(work["prior_same"], errors="coerce")
    result.loc[annual_row] = current.loc[annual_row]
    result.index = frame.index
    return result


def prior_year_value(frame: pd.DataFrame, source: str) -> pd.Series:
    previous = frame[["symbol", "report_period", source]].copy()
    previous["report_period"] = previous["report_period"] + pd.DateOffset(years=1)
    previous = previous.rename(columns={source: "prior_year_value"})
    merged = frame[["symbol", "report_period"]].merge(previous, on=["symbol", "report_period"], how="left")
    merged.index = frame.index
    return pd.to_numeric(merged["prior_year_value"], errors="coerce")


def process_financials(paths: dict[str, list[Path]], symbols: pd.DataFrame, staging: Path) -> dict[str, object]:
    log("处理三张财务报表：计算 TTM 和基本面因子")
    symbol_map = symbols.set_index("Stkcd")["symbol"]
    balance = read_financial_csv(
        paths["balance"][0],
        "FS_Combas",
        ["Stkcd", "Accper", "Typrep", "A001000000", "A002000000", "A003100000", "A003000000"],
    )
    income = read_financial_csv(
        paths["income"][0],
        "FS_Comins",
        ["Stkcd", "Accper", "Typrep", "B001100000", "B001201000", "B002000000", "B002000101"],
    )
    cashflow = read_financial_csv(
        paths["cashflow"][0],
        "FS_Comscfd",
        ["Stkcd", "Accper", "Typrep", "C001000000"],
    )
    balance, balance_duplicates = quarter_rows(balance, symbol_map)
    income, income_duplicates = quarter_rows(income, symbol_map)
    cashflow, cashflow_duplicates = quarter_rows(cashflow, symbol_map)
    keys = ["symbol", "report_period"]
    selected_balance = balance[keys + ["A001000000", "A002000000", "A003100000", "A003000000"]]
    selected_income = income[keys + ["B001100000", "B001201000", "B002000000", "B002000101"]]
    selected_cashflow = cashflow[keys + ["C001000000"]]
    frame = selected_balance.merge(selected_income, on=keys, how="outer").merge(selected_cashflow, on=keys, how="outer")
    frame = frame.sort_values(keys).reset_index(drop=True)
    numeric = [c for c in frame.columns if c not in keys]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame["net_profit_ytd"] = frame["B002000101"].combine_first(frame["B002000000"])
    frame["book_equity"] = frame["A003100000"].combine_first(frame["A003000000"])
    frame["revenue_ttm"] = ttm_from_ytd(frame, "B001100000")
    frame["operating_cost_ttm"] = ttm_from_ytd(frame, "B001201000")
    frame["net_profit_ttm"] = ttm_from_ytd(frame, "net_profit_ytd")
    frame["operating_cashflow_ttm"] = ttm_from_ytd(frame, "C001000000")
    frame["total_assets"] = frame["A001000000"]
    frame["total_liabilities"] = frame["A002000000"]
    frame["roe"] = frame["net_profit_ttm"] / frame["book_equity"].replace(0, np.nan)
    frame["gross_margin"] = (frame["revenue_ttm"] - frame["operating_cost_ttm"]) / frame["revenue_ttm"].replace(0, np.nan)
    for source, target in [("revenue_ttm", "revenue_growth"), ("net_profit_ttm", "profit_growth")]:
        prior = prior_year_value(frame, source)
        frame[target] = (frame[source] - prior) / prior.abs().replace(0, np.nan)
    frame["available_at"] = conservative_available_at(frame["report_period"])
    output_columns = [
        "symbol",
        "report_period",
        "available_at",
        "revenue_ttm",
        "net_profit_ttm",
        "book_equity",
        "operating_cashflow_ttm",
        "total_assets",
        "total_liabilities",
        "roe",
        "gross_margin",
        "revenue_growth",
        "profit_growth",
    ]
    output = frame[output_columns].sort_values(keys)
    output.to_parquet(staging / "fundamentals_provisional.parquet", index=False, compression="zstd")
    return {
        "rows": len(output),
        "symbols": int(output["symbol"].nunique()),
        "start": str(output["report_period"].min().date()),
        "end": str(output["report_period"].max().date()),
        "duplicate_rows_before_deduplication": balance_duplicates + income_duplicates + cashflow_duplicates,
        "availability_method": "statutory_deadline_plus_one_day_provisional",
        "non_null_rates": {column: float(output[column].notna().mean()) for column in output_columns[3:]},
    }


def process_dividend_indicators(path: Path, symbols: pd.DataFrame, staging: Path) -> dict[str, object]:
    log("处理股利分配指标：保留行业与每股现金股利，不冒充除权除息事件")
    symbol_map = symbols.set_index("Stkcd")["symbol"]
    frame = read_financial_csv(
        path,
        "FI_T11",
        ["Stkcd", "Accper", "Typrep", "Source", "Indcd", "Indnme", "F110101B", "F110201B", "F110301B", "F110701B"],
    )
    frame, duplicate_rows = quarter_rows(frame, symbol_map)
    frame["available_at"] = conservative_available_at(frame["report_period"])
    for column in ["F110101B", "F110201B", "F110301B", "F110701B"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    industry = frame[["symbol", "report_period", "available_at", "Indcd", "Indnme"]].rename(columns={"Indcd": "industry_code", "Indnme": "industry"})
    industry = industry.sort_values(["symbol", "report_period"])
    industry.to_parquet(staging / "industry_provisional.parquet", index=False, compression="zstd")
    dividends = frame[
        ["symbol", "report_period", "available_at", "Source", "F110101B", "F110201B", "F110301B", "F110701B"]
    ].rename(
        columns={
            "Source": "source_type",
            "F110101B": "cash_dividend_pre_tax_per_share",
            "F110201B": "cash_dividend_after_tax_per_share",
            "F110301B": "payout_ratio",
            "F110701B": "dividend_multiple",
        }
    )
    dividends.to_parquet(staging / "dividend_indicators.parquet", index=False, compression="zstd")
    return {
        "rows": len(dividends),
        "symbols": int(dividends["symbol"].nunique()),
        "industry_non_null_rate": float(industry["industry"].notna().mean()),
        "cash_dividend_non_null_rate": float(dividends["cash_dividend_pre_tax_per_share"].notna().mean()),
        "duplicate_rows_before_deduplication": duplicate_rows,
        "usable_as_corporate_actions": False,
    }


def process_prices(paths: list[Path], symbols: pd.DataFrame, staging: Path, chunk_size: int) -> dict[str, object]:
    log("第二遍扫描行情：转换字段、单位和代码并写入 Parquet")
    output_path = staging / "prices_observed.parquet"
    hash_path = staging / ".price_key_hashes.bin"
    if output_path.exists():
        output_path.unlink()
    if hash_path.exists():
        hash_path.unlink()
    first_dates = symbols.set_index("Stkcd")["first_observed_date"]
    writer: pq.ParquetWriter | None = None
    counters: dict[str, int] = defaultdict(int)
    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None
    raw_columns = [
        "Stkcd",
        "Trddt",
        "Opnprc",
        "Hiprc",
        "Loprc",
        "Clsprc",
        "Dnshrtrd",
        "Dnvaltrd",
        "Dsmvosd",
        "Dsmvtll",
        "Adjprcwd",
        "Markettype",
        "Trdsta",
        "Ahshrtrd_D",
        "Ahvaltrd_D",
        "PreClosePrice",
        "LimitDown",
        "LimitUp",
    ]
    numeric_columns = [column for column in raw_columns if column not in {"Stkcd", "Trddt"}]
    try:
        with hash_path.open("ab") as hash_file:
            for path in paths:
                log(f"  转换 {path.name}")
                with ZipFile(path) as bundle:
                    for name in csv_entries(bundle, "TRD_Dalyr"):
                        with bundle.open(name) as source:
                            reader = pd.read_csv(
                                source,
                                usecols=raw_columns,
                                dtype={"Stkcd": "string", "Trddt": "string"},
                                chunksize=chunk_size,
                            )
                            for raw in reader:
                                counters["raw_rows"] += len(raw)
                                raw[numeric_columns] = raw[numeric_columns].apply(pd.to_numeric, errors="coerce")
                                raw = raw.loc[raw["Markettype"].isin(A_SHARE_MARKETS)].copy()
                                counters["a_share_rows"] += len(raw)
                                code = raw["Stkcd"].astype("string").str.zfill(6)
                                date = pd.to_datetime(raw["Trddt"], format="%Y-%m-%d", errors="coerce").astype("datetime64[ns]")
                                suffix = np.where(raw["Markettype"].isin(SH_MARKETS), ".SH", ".SZ")
                                close = raw["Clsprc"]
                                volume = raw["Dnshrtrd"].fillna(0) + raw["Ahshrtrd_D"].fillna(0)
                                amount = raw["Dnvaltrd"].fillna(0) + raw["Ahvaltrd_D"].fillna(0)
                                float_market_cap = raw["Dsmvosd"] * 1000.0
                                market_cap = raw["Dsmvtll"] * 1000.0
                                float_shares = float_market_cap / close.replace(0, np.nan)
                                first = code.map(first_dates)
                                listed_days = (date - pd.to_datetime(first)).dt.days + 1
                                market_type = raw["Markettype"].astype("int16")
                                frame = pd.DataFrame(
                                    {
                                        "date": date,
                                        "symbol": code.astype(str) + suffix,
                                        "open": raw["Opnprc"],
                                        "high": raw["Hiprc"],
                                        "low": raw["Loprc"],
                                        "close": close,
                                        "prev_close": raw["PreClosePrice"],
                                        "volume": volume.astype("float64"),
                                        "amount": amount.astype("float64"),
                                        "adj_factor": raw["Adjprcwd"] / close.replace(0, np.nan),
                                        "upper_limit": raw["LimitUp"],
                                        "lower_limit": raw["LimitDown"],
                                        "suspended": ((volume <= 0) | close.isna()).astype("int8"),
                                        "is_st": raw["Trdsta"].isin(ST_STATUSES).astype("int8"),
                                        "market_cap": market_cap,
                                        "float_market_cap": float_market_cap,
                                        "turnover_rate": volume / float_shares.replace(0, np.nan),
                                        "industry": pd.Series(pd.NA, index=raw.index, dtype="string"),
                                        "min_order": np.where(market_type.eq(32), 200, 100).astype("int16"),
                                        "lot_step": np.where(market_type.eq(32), 1, 100).astype("int16"),
                                        "listed_days": listed_days.astype("Int32"),
                                        "available_at": (date + pd.to_timedelta(20, unit="h")).dt.tz_localize("Asia/Shanghai"),
                                    }
                                )
                                frame = frame[PRICE_COLUMNS].reset_index(drop=True)
                                core_required = [
                                    "date",
                                    "symbol",
                                    "open",
                                    "high",
                                    "low",
                                    "close",
                                    "prev_close",
                                    "volume",
                                    "amount",
                                    "adj_factor",
                                    "market_cap",
                                    "float_market_cap",
                                    "turnover_rate",
                                    "listed_days",
                                    "available_at",
                                ]
                                counters["rows_with_missing_core_fields"] += int(frame[core_required].isna().any(axis=1).sum())
                                counters["rows_without_limit_prices"] += int(frame[["upper_limit", "lower_limit"]].isna().any(axis=1).sum())
                                counters["invalid_ohlc_rows"] += int(
                                    (
                                        (frame["high"] < frame[["open", "close"]].max(axis=1))
                                        | (frame["low"] > frame[["open", "close"]].min(axis=1))
                                        | (frame["low"] <= 0)
                                    ).sum()
                                )
                                keys = pd.util.hash_pandas_object(frame[["date", "symbol"]], index=False).to_numpy(dtype="uint64")
                                keys.tofile(hash_file)
                                table = pa.Table.from_pandas(frame, preserve_index=False)
                                if writer is None:
                                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd", use_dictionary=["symbol"])
                                writer.write_table(table, row_group_size=chunk_size)
                                counters["written_rows"] += len(frame)
                                chunk_min, chunk_max = frame["date"].min(), frame["date"].max()
                                min_date = chunk_min if min_date is None or chunk_min < min_date else min_date
                                max_date = chunk_max if max_date is None or chunk_max > max_date else max_date
    finally:
        if writer is not None:
            writer.close()
    hashes = np.memmap(hash_path, dtype="uint64", mode="r")
    sorted_hashes = np.sort(np.asarray(hashes))
    duplicate_keys = int(np.count_nonzero(sorted_hashes[1:] == sorted_hashes[:-1])) if len(sorted_hashes) else 0
    del hashes, sorted_hashes
    hash_path.unlink(missing_ok=True)
    counters["duplicate_date_symbol_keys"] = duplicate_keys
    log(f"  已写入 {counters['written_rows']:,} 行，重复主键 {duplicate_keys:,} 行")
    return {
        **dict(counters),
        "symbols": len(symbols),
        "start": str(min_date.date()) if min_date is not None else None,
        "end": str(max_date.date()) if max_date is not None else None,
        "output_size_bytes": output_path.stat().st_size,
        "industry_status": "pending_asof_merge_from_industry_provisional",
        "suspension_status": "observed_rows_only_pending_universe_grid",
    }


def write_reports(
    staging: Path,
    batch: str,
    provenance: list[dict[str, object]],
    scan: dict[str, int],
    prices: dict[str, object],
    financials: dict[str, object],
    dividends: dict[str, object],
    sessions: pd.DatetimeIndex,
) -> None:
    manifest = {
        "batch": batch,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source": "CSMAR exports supplied by user",
        "status": "processed_staging_waiting_for_point_in_time_inputs",
        "ready_for_backtest": False,
        "historical_text_backtest": False,
        "provenance": provenance,
        "scan": scan,
        "prices": prices,
        "fundamentals": financials,
        "dividend_indicators": dividends,
        "calendar": {
            "rows": len(sessions),
            "start": str(sessions.min().date()),
            "end": str(sessions.max().date()),
            "future_sessions_included": False,
        },
        "assumptions": [
            "财报可用时间暂按法定披露截止日次日 00:00（Asia/Shanghai）处理，属于保守替代口径。",
            "行情只有实际记录行；待导入历史沪深300成分后，才能补齐成分股停牌日网格。",
            "CSMAR 日流通市值和总市值原单位为千元，已乘以 1000 转为人民币元。",
            "复权因子按考虑现金红利再投资的可比收盘价除以未复权收盘价计算。",
            "科创板最低买入 200 股、递增 1 股；其余板块按 100 股整手暂存。",
            "新闻、政策和论坛不做历史回测；未来模拟盘只使用当时首次获取的文本。",
        ],
        "blocking_inputs": [
            "沪深300逐日历史成分及权重（带当时可知/生效日期）",
            "沪深300价格指数与全收益指数日线",
            "公司行动事件：除权除息日、每股现金、送转股比例、公告日期",
        ],
        "validation_upgrade_inputs": [
            "真实财报首次披露日期（最终基本面回测验收必需；当前先用保守截止日替代）"
        ],
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# CSMAR 数据处理与质量报告

处理批次：`{batch}`  
结论：现有文件已完成字段、代码、单位和日期标准化，并保存在 staging。由于缺少历史沪深300成分、基准指数和可执行公司行动，当前数据集还不能作为最终回测输入。

## 已完成

- A股日行情：{int(prices['written_rows']):,} 行，{int(prices['symbols']):,} 只证券，{prices['start']} 至 {prices['end']}。
- 交易日历：{len(sessions):,} 个已观测交易日。
- 基本面：{int(financials['rows']):,} 个证券-报告期记录，已计算 TTM 收入、利润、经营现金流、ROE、毛利率和同比增长。
- 行业/分红指标：{int(dividends['rows']):,} 行；行业可用于后续风险分组。
- 原 ZIP 已按 SHA-256 留存；Downloads 中的原件未修改。

## 质量检查

- 行情重复 `date + symbol` 主键：{int(prices['duplicate_date_symbol_keys']):,}。
- 行情核心字段有缺失的记录：{int(prices['rows_with_missing_core_fields']):,}。
- 未给出涨跌停价的记录：{int(prices['rows_without_limit_prices']):,}。这类记录常见于上市、复牌或不设价格限制的交易日，保留为空并在执行层单独处理。
- OHLC 关系异常记录：{int(prices['invalid_ohlc_rows']):,}。
- 财务三表合并前重复证券-报告期记录：{int(financials['duplicate_rows_before_deduplication']):,}。
- 财报实际披露日缺失；暂用“法定截止日次日”作为保守可用时间。它不阻止第一版验收，只会降低信号时效。
- 下载的“股利分配”是财务指标表，不含除权除息日，因此没有转换成 `actions`。

## 现在还必须下载的三项

1. 沪深300逐日历史成分及权重。
2. 沪深300价格指数和全收益指数日线。
3. 除权除息事件表：除权除息日、每股现金、送转股比例、公告日期。

财务报告首次实际披露日期暂时不用继续找。第一版用“法定披露截止日次日”的保守口径；以后需要提高基本面信号时效时再补真实披露日。

历史新闻、政策和论坛数据不再要求，也不会进入历史回测。模拟盘阶段再接入当时可获取的公开/授权文本。
"""
    (staging / "数据处理与质量报告.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, default=DEFAULT_DOWNLOADS)
    parser.add_argument("--batch", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--chunk-size", type=int, default=250_000)
    args = parser.parse_args()
    sources = locate_sources(args.downloads)
    paths, provenance = archive_copy(sources, args.batch)
    staging = STAGING_ROOT
    staging.mkdir(parents=True, exist_ok=True)
    symbols, sessions, scan = scan_symbols_and_calendar(paths["daily"], args.chunk_size)
    symbols.to_parquet(staging / "security_observation_range.parquet", index=False, compression="zstd")
    pd.DataFrame({"date": sessions}).to_parquet(staging / "calendar_observed.parquet", index=False, compression="zstd")
    financials = process_financials(paths, symbols, staging)
    dividends = process_dividend_indicators(paths["dividend_indicator"][0], symbols, staging)
    prices = process_prices(paths["daily"], symbols, staging, args.chunk_size)
    write_reports(staging, args.batch, provenance, scan, prices, financials, dividends, sessions)
    log(f"完成：{staging}")


if __name__ == "__main__":
    main()
