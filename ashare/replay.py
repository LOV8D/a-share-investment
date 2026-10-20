"""Point-in-time replay of the live production strategy over a chosen period."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path

import pandas as pd

from .config import ROOT, Config
from .data import Dataset, cutoff, dataset_hash, load_daily_dataset, read_table, validate
from .features import build_panel
from .models import production_scores
from .pipeline import write_json
from .portfolio import backtest
from .research import performance


def _slice_dataset(folder: Path, start, end, feature_lookback_sessions: int = 320) -> Dataset:
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if start > end:
        raise ValueError("回放开始日期不能晚于结束日期")
    calendar = read_table(folder, "calendar")
    sessions = pd.DatetimeIndex(calendar.loc[calendar.date <= end, "date"].sort_values().unique())
    in_period = sessions[(sessions >= start) & (sessions <= end)]
    if in_period.empty:
        raise ValueError("所选区间没有交易日")
    first_index = sessions.get_loc(in_period[0])
    feature_start = sessions[max(0, first_index - feature_lookback_sessions)]

    all_sessions = pd.DatetimeIndex(calendar.date.sort_values().unique())
    latest = pd.Timestamp(all_sessions.max())
    lookback = int((all_sessions >= feature_start).sum())
    source = load_daily_dataset(folder, lookback_sessions=max(lookback, feature_lookback_sessions + len(in_period) + 5))
    tables = {}
    for name, frame in source.tables.items():
        value = frame.copy()
        if "date" in value:
            value = value[(value.date >= feature_start) & (value.date <= end)].copy()
        tables[name] = value
    tables["holdings"] = tables["holdings"].iloc[:0].copy()
    manifest = dict(source.manifest)
    manifest.pop("decision_as_of", None)
    manifest.update({
        "mode": "point_in_time_replay",
        "replay_start": str(in_period[0].date()),
        "replay_end": str(in_period[-1].date()),
        "feature_start": str(feature_start.date()),
    })
    quality = validate(tables)
    if quality["errors"]:
        raise ValueError("回放数据校验失败：" + "；".join(quality["errors"][:10]))
    return Dataset(Path(folder), tables, manifest, quality)


def _production_scores(panel: pd.DataFrame, start, end, cfg: Config) -> pd.DataFrame:
    scores = panel[(panel.date >= pd.Timestamp(start)) & (panel.date <= pd.Timestamp(end))].copy()
    if scores.empty:
        raise ValueError("所选区间没有可评分样本")
    return production_scores(scores, cfg).sort_values(["date", "rank"])


def run_replay(folder, cfg: Config, start, end, output_root=None) -> dict:
    """Replay the transparent production score and next-open execution rules.

    LLM/text alpha is deliberately zero unless historical events already carry
    real published, first-seen and processed timestamps. The current production
    model has no trained text component, so replay never invents one retroactively.
    """
    requested_start, requested_end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    ds = _slice_dataset(Path(folder), requested_start, requested_end)
    replay_cfg = cfg
    panel = build_panel(ds, replay_cfg)
    scores = _production_scores(panel, requested_start, requested_end, replay_cfg)
    actual_start, actual_end = scores.date.min(), scores.date.max()
    curve, executions, holdings, cash, daily_plans = backtest(scores, ds, replay_cfg, include_plans=True)
    metrics = performance(curve)
    metrics["benchmark_total_return"] = float(curve.benchmark.iloc[-1] - 1)
    filled = executions[executions.filled_shares > 0].copy() if len(executions) else executions
    last_market = ds["prices"][ds["prices"].date == actual_end].set_index("symbol")
    ending_market_value = sum(int(shares) * float(last_market.loc[symbol, "close"]) for symbol, shares in holdings.items())
    event_times = pd.DataFrame()
    if len(ds["events"]):
        event_times = ds["events"].copy()
        event_times["usable_at"] = event_times[["published_at", "first_seen_at", "processed_at"]].max(axis=1)
    run_id = "replay-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = Path(output_root or ROOT / "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    curve.to_csv(run_dir / "equity_curve.csv", index=False)
    executions.to_csv(run_dir / "executions.csv", index=False)
    scores.to_parquet(run_dir / "scores.parquet", index=False)
    write_json(run_dir / "daily_plans.json", daily_plans)
    text_active = bool(scores.text_model_active.any())
    text_weight_used = replay_cfg.text_alpha_weight if text_active else 0.0
    replay_cutoff = cutoff(actual_end, replay_cfg.decision_time)
    usable_events = int((event_times.usable_at <= replay_cutoff).sum()) if len(event_times) else 0
    future_events_excluded = int((event_times.usable_at > replay_cutoff).sum()) if len(event_times) else 0
    report = {
        "run_id": run_id,
        "kind": "point_in_time_replay",
        "mode": "forward_rules_historical_replay",
        "requested_start": str(requested_start.date()),
        "requested_end": str(requested_end.date()),
        "start": str(actual_start.date()),
        "end": str(actual_end.date()),
        "as_of": str(actual_end.date()),
        "initial_cash": replay_cfg.initial_cash,
        "ending_cash": float(cash),
        "ending_market_value": float(ending_market_value),
        "ending_nav": float(cash + ending_market_value),
        "ending_positions": {symbol: int(shares) for symbol, shares in sorted(holdings.items())},
        "filled_orders": int(len(filled)),
        "rejected_or_unfilled_orders": int(len(executions) - len(filled)),
        "metrics": metrics,
        "text": {
            "llm_configured": bool(os.getenv("AI_MODEL") and os.getenv("AI_BASE_URL")),
            "weight_used": text_weight_used,
            "events_usable_by_end": usable_events,
            "future_events_excluded": future_events_excluded,
            "reason": ("仅使用在各信号时点前已发布、已发现且已处理的事件。"
                       if text_active else "没有在历史信号时点前已处理的事件；禁止把今天生成的GLM结果回填到历史。"),
        },
        "point_in_time_policy": "strict",
        "lookahead_controls": {
            "price_and_membership": "每个信号日23:59:59时点校验 available_at/known_at",
            "fundamentals": "仅 as-of 合并 available_at 不晚于信号时点的财报",
            "events": "published_at、first_seen_at、processed_at 三者最大值不晚于信号时点",
            "labels": "生产排序不使用未来收益标签；成交只使用下一交易日开盘",
        },
        "benchmark_note": "2026-08-31之后的公开增量基准为沪深300价格指数代理；不含分红再投资。",
        "worldquant101": {
            "usable": 83,
            "unavailable": 18,
            "unavailable_ids": [48,58,59,63,67,69,70,76,79,80,82,87,89,90,91,93,97,100],
            "reason": "缺少论文要求的sector/industry/subindustry分层历史分类，未做静默替代。",
        },
        "model_card": {
            "algorithm": "核心价值/质量/低风险 50% + WorldQuant 101可用因子组 50%",
            "challengers": "本次短区间回放不使用ML挑战者决定订单",
            "features": 83 + 23,
            "text_features": 5,
            "text_active": text_active,
            "text_weight_used": text_weight_used,
            "train_end": "固定透明规则，不在回放区间拟合权重",
            "train_rows": 0,
            "horizon": replay_cfg.label_horizon,
            "refit_every": replay_cfg.refit_every,
        },
        "factors": [],
        "candidates": [],
        "folds": [],
        "ablation": [{"name": "日度前十生产组合", **metrics}],
        "config": replay_cfg.dict(),
        "input_collection_id": ds.manifest.get("collection_id"),
        "curve": curve.to_dict("records"),
    }
    write_json(run_dir / "report.json", report)
    write_json(run_dir / "config.json", replay_cfg.dict())
    write_json(run_dir / "manifest.json", {
        "run_id": run_id,
        "kind": "point_in_time_replay",
        "created_at": pd.Timestamp.now(tz="Asia/Shanghai"),
        "input_folder": str(Path(folder).resolve()),
        "input_collection_id": ds.manifest.get("collection_id"),
        "input_files_sha256": dataset_hash(folder),
        "requested_period": [str(requested_start.date()), str(requested_end.date())],
        "actual_period": [str(actual_start.date()), str(actual_end.date())],
        "text_weight_used": text_weight_used,
    })
    summary = (
        f"# 模拟盘回放：{report['start']} 至 {report['end']}\n\n"
        f"- 初始资金：¥{replay_cfg.initial_cash:,.2f}\n"
        f"- 期末净值：¥{report['ending_nav']:,.2f}\n"
        f"- 区间收益：{metrics['total_return']:.2%}\n"
        f"- 沪深300价格基准：{metrics['benchmark_total_return']:.2%}\n"
        f"- 最大回撤：{metrics['max_drawdown']:.2%}\n"
        f"- 成交订单：{report['filled_orders']}\n"
        f"- 累计费用：¥{metrics['total_fees']:,.2f}\n\n"
        f"文本/GLM实际权重：{text_weight_used:.0%}。没有把事后抓取或事后处理的文本倒灌进回放。\n"
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return report
