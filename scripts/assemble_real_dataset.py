"""Assemble the validated staging tables into the canonical real dataset.

The script is deliberately strict about the CSI 300 membership audit.  Missing
vendor price rows on membership dates are represented as suspended sessions:
OHLC is the last close, volume/amount/turnover are zero, and the record is made
available at 20:00 China time.  This preserves holdings and the 300-name panel
without pretending a trade occurred.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.dataset as pads


ROOT = Path(__file__).resolve().parents[1]
REAL_STAGING = ROOT / "data" / "staging" / "real"
CSMAR_STAGING = ROOT / "data" / "staging" / "csmar"
DEFAULT_OUTPUT = ROOT / "data" / "inbox"


def log(message: str) -> None:
    print(message, flush=True)


def attach_industry(prices: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    history = history.loc[history.industry.notna()].copy()
    history = history.sort_values(["symbol", "available_at", "report_period"]).drop_duplicates(
        ["symbol", "available_at"], keep="last"
    )
    pieces: list[pd.DataFrame] = []
    for symbol, frame in prices.groupby("symbol", sort=False):
        labels = history.loc[history.symbol.eq(symbol), ["available_at", "industry"]].sort_values("available_at")
        work = frame.sort_values("available_at").drop(columns="industry")
        if labels.empty:
            work["industry"] = "未知行业"
        else:
            work = pd.merge_asof(work, labels, on="available_at", direction="backward")
            work["industry"] = work["industry"].fillna("未知行业")
        pieces.append(work)
    result = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    return result, {
        "known_industry_rate_all_price_rows": float(result.industry.ne("未知行业").mean()),
        "unknown_industry_rows": int(result.industry.eq("未知行业").sum()),
        "industry_method": "latest CSMAR report classification available by the 20:00 decision cutoff",
    }


def assemble_prices(universe: pd.DataFrame, calendar: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    source_path = CSMAR_STAGING / "prices_observed.parquet"
    symbols = sorted(universe.symbol.unique())
    dataset = pads.dataset(source_path, format="parquet")
    table = dataset.to_table(filter=pads.field("symbol").isin(symbols))
    observed = table.to_pandas()
    observed = observed.sort_values(["symbol", "date"]).drop_duplicates(["date", "symbol"], keep="last")
    source_rows = len(observed)
    price_end = pd.Timestamp(observed.date.max())
    sessions = pd.DatetimeIndex(calendar.loc[calendar.date.le(price_end), "date"].sort_values().unique())
    first_membership = universe.groupby("symbol").date.min().to_dict()

    pieces: list[pd.DataFrame] = []
    missing_membership_keys = 0
    support_rows_added = 0
    for n, symbol in enumerate(symbols, start=1):
        stock = observed.loc[observed.symbol.eq(symbol)].set_index("date").sort_index()
        start = pd.Timestamp(first_membership[symbol])
        target_dates = sessions[sessions >= start]
        union_index = stock.index.union(target_dates).sort_values()
        work = stock.reindex(union_index)
        originally_missing = work.index.isin(target_dates) & work["symbol"].isna().to_numpy()
        work["symbol"] = symbol

        carry_columns = [
            "close", "adj_factor", "is_st", "market_cap", "float_market_cap",
            "min_order", "lot_step",
        ]
        work[carry_columns] = work[carry_columns].ffill()
        # A missing first target row is unusual but can occur when the security
        # was already suspended at the start of our price sample.  Use the first
        # later observed attributes only for non-predictive book continuity.
        work[carry_columns] = work[carry_columns].bfill()
        work = work.loc[target_dates].copy()
        missing = work.index.isin(union_index[originally_missing])
        missing_series = pd.Series(missing, index=work.index)
        last_close = work["close"]
        for column in ("open", "high", "low", "close", "prev_close"):
            work.loc[missing_series, column] = last_close.loc[missing_series]
        work.loc[missing_series, ["volume", "amount", "turnover_rate"]] = 0.0
        work.loc[missing_series, ["upper_limit", "lower_limit"]] = np.nan
        work.loc[missing_series, "suspended"] = 1
        work.loc[missing_series, "available_at"] = (
            pd.DatetimeIndex(work.index[missing_series]) + pd.to_timedelta(20, unit="h")
        ).tz_localize("Asia/Shanghai")
        first_observed = pd.Timestamp(stock.index.min())
        work["listed_days"] = (pd.DatetimeIndex(work.index) - first_observed).days + 1
        work["industry"] = pd.NA
        work.index.name = "date"
        work = work.reset_index()
        pieces.append(work)

        member_dates = set(universe.loc[universe.symbol.eq(symbol), "date"])
        missing_dates = set(work.loc[work.suspended.eq(1), "date"])
        missing_membership_keys += len(member_dates & missing_dates)
        support_rows_added += int(missing_series.sum())
        if n % 100 == 0:
            log(f"  已组装 {n}/{len(symbols)} 只历史成分")

    prices = pd.concat(pieces, ignore_index=True)
    numeric_complete = [
        "open", "high", "low", "close", "prev_close", "volume", "amount", "adj_factor",
        "market_cap", "float_market_cap", "turnover_rate", "min_order", "lot_step", "listed_days",
    ]
    if prices[numeric_complete].isna().any().any():
        missing = prices[numeric_complete].isna().sum()
        raise ValueError(f"停牌网格补齐后仍有核心数值缺失：{missing[missing.gt(0)].to_dict()}")
    prices["suspended"] = prices["suspended"].fillna(1).astype("int8")
    prices["is_st"] = prices["is_st"].fillna(0).astype("int8")
    prices["min_order"] = prices["min_order"].astype("int32")
    prices["lot_step"] = prices["lot_step"].astype("int32")
    prices["listed_days"] = prices["listed_days"].astype("int32")
    prices, industry_stats = attach_industry(
        prices, pd.read_parquet(CSMAR_STAGING / "industry_provisional.parquet")
    )
    columns = [
        "date", "symbol", "open", "high", "low", "close", "prev_close", "volume", "amount",
        "adj_factor", "upper_limit", "lower_limit", "suspended", "is_st", "market_cap",
        "float_market_cap", "turnover_rate", "industry", "min_order", "lot_step", "listed_days",
        "available_at",
    ]
    prices = prices[columns]
    return prices, {
        "source_observed_rows_for_historical_members": source_rows,
        "rows": len(prices),
        "symbols": len(symbols),
        "start": str(prices.date.min().date()),
        "end": str(prices.date.max().date()),
        "support_suspended_rows_added": support_rows_added,
        "missing_membership_sessions_filled_as_suspended": missing_membership_keys,
        **industry_stats,
    }


def write_empty_optional(output: Path) -> None:
    # Optional tables stay absent until forward testing starts.  The loader
    # supplies their canonical empty schemas, avoiding fake historical text.
    for name in ("documents", "events", "holdings", "auction"):
        for suffix in ("csv", "parquet"):
            path = output / f"{name}.{suffix}"
            if path.exists():
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    stage_manifest = json.loads((REAL_STAGING / "manifest.json").read_text(encoding="utf-8"))
    if not stage_manifest.get("ready_for_assembly") or not stage_manifest["membership"].get("complete"):
        raise ValueError("历史沪深300成分尚未通过 staging 完整性闸门")

    log("读取已验证的沪深300历史成员")
    universe = pd.read_parquet(REAL_STAGING / "universe_validated.parquet")
    calendar = pd.read_parquet(REAL_STAGING / "calendar.parquet")
    benchmark = pd.read_parquet(REAL_STAGING / "benchmark.parquet")
    observed_end = pd.Timestamp(stage_manifest["prices"]["end"])
    universe = universe.loc[universe.date.le(observed_end)].copy()
    # Equal membership weights are intentional: the engine uses weight only as
    # a validated universe marker.  Official index weights are not model inputs.
    universe["weight"] = 1 / universe.groupby("date").symbol.transform("size")

    log("补齐历史成分的停牌日并合并当时可用行业")
    prices, price_stats = assemble_prices(universe, calendar)
    log(f"  输出 {len(prices):,} 行价格，补入 {price_stats['support_suspended_rows_added']:,} 行停牌/持仓支持记录")

    symbols = set(universe.symbol)
    fundamentals = pd.read_parquet(CSMAR_STAGING / "fundamentals_provisional.parquet")
    fundamentals = fundamentals.loc[fundamentals.symbol.isin(symbols)].sort_values(
        ["symbol", "report_period", "available_at"]
    )
    actions = pd.read_parquet(REAL_STAGING / "actions.parquet")
    actions = actions.loc[actions.symbol.isin(symbols) & actions.date.le(observed_end)].sort_values(
        ["date", "symbol"]
    )
    fees = pd.read_parquet(REAL_STAGING / "fees.parquet")

    log("写入 canonical data/inbox")
    tables = {
        "calendar": calendar,
        "prices": prices,
        "universe": universe,
        "fundamentals": fundamentals,
        "benchmark": benchmark,
        "actions": actions,
        "fees": fees,
    }
    for name, frame in tables.items():
        frame.to_parquet(output / f"{name}.parquet", index=False, compression="zstd")
    write_empty_optional(output)

    manifest = {
        "mode": "real",
        "source": "CSMAR stock/fundamental/actions + RESSET CSI300 membership/index + official CSI total-return series",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "initial_cash": 100000,
        "historical_text_backtest": False,
        "ready_for_structured_backtest": True,
        "tables": {name: {"rows": len(frame)} for name, frame in tables.items()},
        "price_assembly": price_stats,
        "universe": {
            "weight_method": "equal 1/N membership marker; not official benchmark weight",
            "daily_count_min": int(universe.groupby("date").size().min()),
            "daily_count_max": int(universe.groupby("date").size().max()),
            "temporary_298_dates": ["2009-12-29", "2009-12-30", "2009-12-31"],
            "temporary_298_reason": "two merger-related delistings before the 2010-01-04 scheduled replacements",
            "source_url": "https://www.sse.com.cn/market/sseindex/diclosure/c/c_20150911_3984987.shtml",
        },
        "fundamentals": {
            "availability": "statutory filing deadline plus one day; conservative provisional fallback",
            "actual_disclosure_date_required_for_first_version": False,
            "actual_disclosure_date_status": "optional timing-accuracy upgrade; the fallback is conservative and avoids look-ahead",
        },
        "corporate_actions": {
            "cash_and_share_distributions": "included",
            "merger_successor_security_conversion": "not present in supplied profit-sharing table; audit after backtest",
        },
        "fees": stage_manifest["fees"],
        "limitations": [
            "Historical news, policy and forum text is intentionally excluded from backtesting.",
            "Rows after a historical member's last vendor quote are kept suspended at the last close for book continuity; any held merger/delisting position must be audited.",
            "The exchange calendar has one session after the last stock-price date, sufficient for the next conditional plan but not ten future sessions.",
        ],
        "staging_manifest": "data/staging/real/manifest.json",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    log("运行正式数据契约校验")
    from ashare.data import load_dataset

    dataset = load_dataset(output, strict=True)
    quality = dataset.quality
    report = {
        "created_at": manifest["created_at"],
        "passed": quality["passed"],
        "errors": quality["errors"],
        "warnings": quality["warnings"],
        "tables": manifest["tables"],
        "price_assembly": price_stats,
    }
    (output / "质量报告.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"正式数据校验通过；警告 {len(quality['warnings'])} 条，错误 0 条")


if __name__ == "__main__":
    main()
