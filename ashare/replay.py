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
from .features import build_cached_panel
from .models import production_scores, xgboost_walk_scores
from .pipeline import write_json
from .portfolio import backtest, make_plan
from .research import performance


def _latest_incremental_base(output_root: Path, requested_start, requested_end, cfg: Config):
    """Return the newest compatible replay that can be extended in place."""
    candidates = []
    for path in output_root.glob("replay-*/report.json"):
        try:
            report = json.loads(path.read_text())
            end = pd.Timestamp(report["end"]).normalize()
            same_rules = all(report.get("config", {}).get(key) == getattr(cfg, key) for key in (
                "top_k", "hold_buffer", "max_weight", "max_sector_weight",
                "cash_reserve", "max_daily_turnover", "max_daily_orders",
                "min_trade_weight", "commission_rate", "min_commission",
                "slippage_bps", "rebalance_frequency", "enable_text_alpha",
                "text_alpha_weight", "worldquant_weight",
            ))
            if (report.get("kind") == "point_in_time_replay"
                    and pd.Timestamp(report["start"]).normalize() == requested_start
                    and end < requested_end
                    and float(report.get("initial_cash", 0)) == float(cfg.initial_cash)
                    and report.get("curve") and report.get("ending_positions") is not None
                    and same_rules
                    and (path.parent / "daily_plans.json").exists()
                    and (path.parent / "executions.csv").exists()
                    and (path.parent / "scores.parquet").exists()):
                candidates.append((end, path.stat().st_mtime, path.parent, report))
        except (KeyError, OSError, TypeError, ValueError):
            continue
    return max(candidates, default=None, key=lambda item: (item[0], item[1]))


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


def run_replay(folder, cfg: Config, start, end, output_root=None, reconstruction=None, incremental=True) -> dict:
    """Replay the transparent production score and next-open execution rules.

    LLM/text alpha is deliberately zero unless historical events already carry
    real published, first-seen and processed timestamps. The current production
    model has no trained text component, so replay never invents one retroactively.
    """
    requested_start, requested_end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    output_root = Path(output_root or ROOT / "runs")
    incremental = _latest_incremental_base(output_root, requested_start, requested_end, cfg) if incremental and reconstruction is None and cfg.structured_model=="prior" else None
    slice_start = incremental[0] if incremental else requested_start
    ds = _slice_dataset(Path(folder), slice_start, requested_end,feature_lookback_sessions=1100 if cfg.structured_model=="xgboost" else 320)
    replay_cfg = cfg
    panel = build_cached_panel(ds, replay_cfg)
    reconstruction_meta={}
    if reconstruction:
        recon=Path(reconstruction)
        signals=pd.read_parquet(recon/"daily_signals.parquet")
        if signals.duplicated(["date","symbol"]).any():raise ValueError("重建事件信号主键重复")
        panel=panel.merge(signals[["date","symbol","reconstructed_text_score"]],on=["date","symbol"],how="left")
        reconstruction_meta=json.loads((recon/"signals_manifest.json").read_text())
    folds=[]
    model_root=output_root/"xgb-models"/("zoo-v1" if cfg.factor_zoo_enabled else "legacy")
    if cfg.structured_model=="xgboost":
        scores,folds=xgboost_walk_scores(panel,replay_cfg,slice_start,requested_end,model_root)
    else:
        scores = _production_scores(panel, slice_start, requested_end, replay_cfg)
    actual_end = scores.date.max()
    if incremental and actual_end > incremental[0]:
        _, _, previous_dir, previous = incremental
        extension_curve, extension_executions, holdings, cash, extension_plans = backtest(
            scores, ds, replay_cfg, include_plans=True,
            initial_holdings=previous["ending_positions"],
            initial_cash=previous["ending_cash"],
            state_date=incremental[0],
            benchmark_scale=float(previous["curve"][-1]["benchmark"]),
        )
        curve = pd.concat([pd.DataFrame(previous["curve"]), extension_curve], ignore_index=True)
        old_executions = pd.read_csv(previous_dir / "executions.csv")
        executions = pd.concat([old_executions, extension_executions], ignore_index=True)
        old_scores = pd.read_parquet(previous_dir / "scores.parquet")
        scores = pd.concat([old_scores, scores[scores.date > incremental[0]]], ignore_index=True)
        daily_plans = json.loads((previous_dir / "daily_plans.json").read_text()) + extension_plans
        actual_start = pd.Timestamp(previous["start"])
        incremental_from = str(incremental[0].date())
        incremental_sessions = int(len(extension_curve))
    else:
        actual_start = scores.date.min()
        curve, executions, holdings, cash, daily_plans = backtest(scores, ds, replay_cfg, include_plans=True)
        incremental_from = None
        incremental_sessions = 0
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
    run_dir = output_root / run_id
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
        "mode": "retrospective_event_reconstruction" if reconstruction else "forward_rules_historical_replay",
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
                       if text_active else "没有在历史信号时点前已处理的事件；禁止把今天生成的AI结果回填到历史。"),
        },
        "point_in_time_policy": "retrospective_publication_filter" if reconstruction else "strict",
        "reconstruction": reconstruction_meta,
        "events": pd.read_parquet(Path(reconstruction)/"events.parquet").tail(100).to_dict("records") if reconstruction else [],
        "incremental_from": incremental_from,
        "incremental_sessions": incremental_sessions,
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
            "algorithm": "XGBoost学习23个核心因子与83个WorldQuant因子" if cfg.structured_model=="xgboost" else "核心价值/质量/低风险 50% + WorldQuant 101可用因子组 50%",
            "challengers": "本次短区间回放不使用ML挑战者决定订单",
            "features": len(folds[-1].get("selected_features",[])) if folds and folds[-1].get("selected_features") else 83+23,
            "text_features": 5,
            "text_active": text_active,
            "text_weight_used": text_weight_used,
            "explanation":f"XGBoost结构化排序与AI事件评分按{1-cfg.text_alpha_weight:.0%}/{cfg.text_alpha_weight:.0%}后融合；历史重建与严格点时记录分别标识。",
            "train_end": folds[-1]["train_end"] if folds else "固定透明规则，不在回放区间拟合权重",
            "train_rows": folds[-1]["rows"] if folds else 0,
            "horizon": replay_cfg.label_horizon,
            "refit_every": replay_cfg.refit_every,
        },
        "factors": [],
        "candidates": [],
        "folds": folds,
        "ablation": [{"name": ("XGBoost + AI历史重建" if reconstruction else "日度前十生产组合"), **metrics}],
        "config": replay_cfg.dict(),
        "input_collection_id": ds.manifest.get("collection_id"),
        "curve": curve.to_dict("records"),
    }
    if reconstruction:
        source_events=pd.read_parquet(Path(reconstruction)/"events.parquet")
        report["events"]=source_events.sort_values("published_at",ascending=False).head(300).to_dict("records")
        report["text"]["reason"]=reconstruction_meta.get("warning")
        report["text"]["events_usable_by_end"]=len(source_events)
        report["lookahead_controls"]["events"]="仅按历史发布时间筛选；真实首次抓取和处理时间保留，属于事后重建"
        report["text"]["future_events_excluded"]=None
        report["text"]["reconstructed"]=True
        report["warnings"]=[reconstruction_meta.get("warning"),"历史新闻/论坛覆盖不完整；无事件股票使用中性50分，此为建模约定。"]
        report["text"]["informative_coverage"]=float(scores.text_score.sub(.5).abs().gt(1e-9).mean())
        report["text"]["stock_days"]=len(scores)
        report["text"]["configured_weight"]=cfg.text_alpha_weight
        report["text"]["daily_weight_used"]=[{"date":str(day.date()),"weight":cfg.text_alpha_weight if frame.text_model_active.all() else None,"informative_stocks":int(frame.text_score.sub(.5).abs().gt(1e-9).sum())} for day,frame in scores.groupby("date")]
        for baseline_path in sorted(output_root.glob("replay-*/report.json"),reverse=True):
            try:
                baseline=json.loads(baseline_path.read_text())
                if (baseline.get("start")==report["start"] and baseline.get("end")==report["end"] and baseline.get("config",{}).get("structured_model")=="xgboost" and baseline.get("text",{}).get("weight_used")==0 and len(baseline.get("ablation",[]))>=3):
                    report["ablation"]=baseline["ablation"]+report["ablation"]
                    break
            except (OSError,ValueError):pass

    if reconstruction:
        # A hypothetical replay account gets its own next-open plan; the
        # actual forward-paper account and latest pointer are not mutated.
        calendar=read_table(folder,"calendar")
        future=calendar.loc[calendar.date>actual_end,"date"]
        if len(future):
            next_day=future.min()
            latest_scores=scores[scores.date==actual_end].sort_values("rank").copy()
            market=ds["prices"][ds["prices"].date==actual_end].copy()
            trailing=ds["prices"].sort_values(["symbol","date"]).groupby("symbol").amount.apply(lambda x:x.tail(20).mean())
            market["adv20"]=market.symbol.map(trailing)
            plan=make_plan(latest_scores,holdings,cash,market,next_day,cfg,ds["fees"])
            plan.update(account_mode="历史事件重建的假设模拟账户",rebalance_due=True,rebalance_frequency="daily",retrospective=True)
            report["plan"]=plan;report["next_session"]=str(next_day.date())
            report["decision_cutoff"]=str(cutoff(actual_end,cfg.decision_time))
            keep=[c for c in latest_scores if not c.startswith("x_")]
            ranked=latest_scores[keep].to_dict("records")
            for row in ranked:
                row["target_weight"]=plan["target_weights"].get(row["symbol"],0.)
                row["explanations"]=[]
                row["minimum_order_value"]=float(row["close"])*float(row.get("min_order",100))
            if cfg.structured_model=="xgboost" and folds:
                from xgboost import XGBRanker,DMatrix
                import numpy as np
                latest_model=XGBRanker();latest_model.load_model(model_root/f"xgb-{folds[-1]['signal_start']}.json")
                columns=latest_model.get_booster().feature_names
                contributions=latest_model.get_booster().predict(DMatrix(latest_scores[columns].fillna(0.)),pred_contribs=True)
                for row,values in zip(ranked,contributions):
                    indices=np.argsort(np.abs(values[:-1]))[-4:][::-1]
                    row["explanations"]=[{"factor":columns[i][2:],"contribution":float(values[i])} for i in indices]
            report["rankings"]=ranked
            write_json(run_dir/"plan.json",plan)
            pd.DataFrame(plan["orders"]).to_csv(run_dir/"orders.csv",index=False)
    if folds and cfg.structured_model=="xgboost":
        importance_path=model_root/f"xgb-{folds[-1]['signal_start']}.importance.json"
        report["feature_importance"]=json.loads(importance_path.read_text())
        write_json(run_dir/"factor_importance.json",report["feature_importance"])
        screen_path=importance_path.with_name(f"xgb-{folds[-1]['signal_start']}.screen.json")
        if screen_path.exists():
            report["factor_screen"]=json.loads(screen_path.read_text())
            report["model_card"]["features"]=report["factor_screen"]["selected_count"]
            report["model_card"]["algorithm"]="XGBoost（264候选因子训练期粗筛后学习组合）"
            report["factor_screen_history"]=[json.loads((model_root/f"xgb-{f['signal_start']}.screen.json").read_text()) for f in folds]
            write_json(run_dir/"factor_screen.json",report["factor_screen"])
    report["data_audit"]={"quarantined_symbols":len({q["symbol"] for q in ds.manifest.get("corporate_action_quarantine",[])}),"repair_source":ds.manifest.get("repair_source")}
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
        "incremental_from": incremental_from,
        "incremental_sessions": incremental_sessions,
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
        f"文本/AI实际权重：{text_weight_used:.0%}。没有把事后抓取或事后处理的文本倒灌进回放。\n"
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return report
