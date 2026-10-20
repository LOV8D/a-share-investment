import importlib.metadata
import json
import platform
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from .config import ROOT
from .data import Dataset,load_dataset,load_daily_dataset,dataset_hash,cutoff
from .features import build_panel
from .models import walk_forward,explanations,fit_model,training_slice,predict
from .portfolio import make_plan,backtest,should_rebalance
from .research import performance,factor_report,factor_candidates

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.bool_,)):return bool(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:return None
    if isinstance(value,(pd.Timestamp,datetime)):return value.isoformat()
    return value

def write_json(path,data):
    Path(path).write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False))

def freeze_research_dataset(ds):
    """Keep every reported historical return inside the registered cutoff."""
    value=ds.manifest.get("historical_research_cutoff")
    if not value:
        return ds
    end=pd.Timestamp(value).normalize()
    tables={name:frame.copy() for name,frame in ds.tables.items()}
    for name in ("prices","universe","benchmark","actions","auction"):
        if len(tables[name]) and "date" in tables[name]:
            tables[name]=tables[name][tables[name].date<=end].copy()
    return Dataset(ds.folder,tables,ds.manifest,ds.quality)

def run_engine(folder,cfg,kind="research",output_root=None,progress=None):
    progress=progress or (lambda *args:None)
    progress("校验数据与可用时间",.03)
    ds=load_dataset(folder) if kind=="research" else load_daily_dataset(folder,cfg.train_window_days+100)
    if kind=="research":
        ds=freeze_research_dataset(ds)
    progress("构建量价与基本面因子",.10)
    panel=build_panel(ds,cfg)
    if panel.empty:raise ValueError("没有截止时刻可用的完整成分数据")
    latest_date=panel.date.max()
    future=ds["calendar"][ds["calendar"].date>latest_date]
    if future.empty:raise ValueError("交易日历缺少下一个交易日")
    next_date=future.date.min()
    decision_cutoff=panel.loc[panel.date==latest_date,"cutoff"].max()
    if ds.manifest.get("mode") != "synthetic" and decision_cutoff>pd.Timestamp.now(tz="Asia/Shanghai"):
        raise ValueError("真实数据的信号截止时刻尚未到达，不能生成收盘后计划")
    run_id=datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir=Path(output_root or ROOT/"runs")/run_id
    run_dir.mkdir(parents=True)
    curves=[];metrics={};factor_rows=[];folds=[];ic_summary={};ablation=[];backtest_audit={}
    if kind=="research":
        scores,folds,model,latest=walk_forward(panel,cfg,lambda msg,p:progress(msg,.20+.35*p))
        progress("逐日模拟次日成交及成本",.62)
        curve,trades,ending_holdings,ending_cash=backtest(scores,ds,cfg)
        metrics=performance(curve)
        ending_market=ds["prices"][ds["prices"].date==ds["prices"].date.max()].set_index("symbol")
        terminal_suspended=sorted(s for s in ending_holdings if s in ending_market.index and bool(ending_market.loc[s,"suspended"]))
        backtest_audit={"ending_cash":ending_cash,"ending_positions":len(ending_holdings),"terminal_suspended_positions":terminal_suspended,"rejected_suspended_orders":int((trades.fill_reason=="停牌/无有效开盘价").sum()) if len(trades) else 0}
        curves=curve.to_dict("records")
        curve.to_csv(run_dir/"equity_curve.csv",index=False)
        trades.to_csv(run_dir/"executions.csv",index=False)
        scores.to_parquet(run_dir/"oos_predictions.parquet",index=False)
        progress("计算样本外 IC、基线及双倍成本检验",.76)
        factor_rows,ics,ic_summary=factor_report(scores,cfg)
        ics.to_csv(run_dir/"daily_ic.csv",index=False)
        ablation.append({"name":"生产组合（核心因子 + WorldQuant 101）",**metrics})
        for name,col,test_cfg in [("Ridge ML 挑战者","ridge_raw",cfg),("LightGBM 挑战者","tree_raw",cfg),("生产组合（双倍成本）","score",replace(cfg,commission_rate=cfg.commission_rate*2,min_commission=cfg.min_commission*2,slippage_bps=cfg.slippage_bps*2))]:
            c,_,_,_=backtest(scores,ds,test_cfg,col)
            m=performance(c)
            ablation.append({"name":name,**m})
    else:
        progress("使用截至今天已成熟的历史标签训练",.40)
        model=fit_model(training_slice(panel,latest_date,cfg),latest_date,cfg)
        latest=predict(model,panel[panel.date==latest_date],cfg)
    progress("生成账户约束下的次日计划",.90)
    holdings={};cash=cfg.initial_cash;account_mode="全现金模拟账户"
    account_path=Path(folder)/"account.json"
    if account_path.exists():
        account=json.loads(account_path.read_text())
        if str(account.get("as_of"))!=str(latest_date.date()):raise ValueError("account.json 的 as_of 必须等于信号日收盘日期")
        cash=float(account["cash"])
        if not np.isfinite(cash) or cash<0:raise ValueError("账户现金不合法")
        holdings={r.symbol:int(r.shares) for r in ds["holdings"].itertuples() if r.shares>0}
        account_mode="引擎自动维护的模拟账户" if account.get("mode")=="paper" else "用户提供的账户快照"
    elif len(ds["holdings"]):
        raise ValueError("有持仓时必须同时提供 account.json：as_of 和 cash；不能假设初始资金仍全额可用")
    market=ds["prices"][ds["prices"].date==latest_date].copy()
    trailing=ds["prices"].sort_values(["symbol","date"]).groupby("symbol").amount.apply(lambda x:x.tail(20).mean())
    market["adv20"]=market.symbol.map(trailing)
    plan=make_plan(latest,holdings,cash,market,next_date,cfg,ds["fees"])
    plan["account_mode"]=account_mode
    plan["rebalance_due"]=should_rebalance(latest_date,next_date,cfg)
    plan["rebalance_frequency"]=cfg.rebalance_frequency
    if not plan["rebalance_due"]:
        plan["preview_orders"]=plan["orders"]
        plan["orders"]=[]
        plan["status"]="MONITOR_ONLY"
        plan["schedule_reason"]="当前配置不是调仓日，只保留评分。"
    # Same-day sellable values must be supplied again to the morning precheck.
    explain=explanations(model,latest)
    latest_columns=["date","symbol","industry","close","score","rank","structured_score","prior_score","worldquant_score","worldquant_score_rank","worldquant_coverage","text_score","ml_challenger_score","ridge_raw","tree_raw","eligible","volatility_20","adv20","event_sentiment","event_risk","event_coverage","missing_fraction","model_train_end","text_model_active"]
    ranked=[]
    for row in latest[latest_columns].to_dict("records"):
        row["explanations"]=explain.get(row["symbol"],[])
        row["target_weight"]=plan["target_weights"].get(row["symbol"],0.)
        row["minimum_order_value"]=float(market.set_index("symbol").loc[row["symbol"],"min_order"])*row["close"]
        ranked.append(row)
    events=ds["events"].copy()
    if len(events):
        events["usable_at"]=events[["published_at","first_seen_at","processed_at"]].max(axis=1)
        events=events[events.usable_at<=decision_cutoff].sort_values("usable_at",ascending=False).head(80)
    warnings=list(ds.quality["warnings"])
    quarantines=ds.manifest.get("corporate_action_quarantine",[])
    if quarantines:
        labels=", ".join(f"{row.get('symbol')}({row.get('date')})" for row in quarantines[:12])
        warnings.append(f"公司行动明细待补：{labels}；相关证券已按上一可信收盘价冻结，不参与买卖，其余股票继续更新。")
    synthetic=ds.manifest.get("mode")=="synthetic"
    if synthetic:warnings.insert(0,"全部价格、证券代码、文本与收益为合成演示，不能用于判断策略收益或下单。")
    if cfg.initial_cash<plan["minimum_capital_for_topk"]:warnings.append(f"当前资金无法在单股上限内持有全部理论前{cfg.top_k}名；按最低申报数量筛选可实现标的。")
    if not plan["rebalance_due"]:warnings.append(plan["schedule_reason"])
    text_active=bool(latest.text_model_active.any())
    if not text_active:warnings.append("当前信号时点没有合格的GLM事件，文本实际权重为0；不会回填未来信息。")
    next_open=cutoff(next_date,"09:30:00")
    if next_open<pd.Timestamp.now(tz="Asia/Shanghai"):
        if plan["orders"]:
            plan["blocked_orders"]=plan["orders"]
            plan["orders"]=[]
        plan["status"]="STALE_BLOCKED"
        warnings.append("计划对应的开盘时刻已过去，订单已自动阻断；请更新到最近收盘数据。")
    warnings.append("日线成交模型无法观察竞价排队和真实冲击；分红现金按除息日计入，未模拟到账延迟和差异化红利税。")
    if backtest_audit.get("terminal_suspended_positions"):
        warnings.append("回测末日仍持有停牌/退市证券，缺少换股映射；该次回测不能通过公司行动审计。")
    report_mode="synthetic" if synthetic else ("forward_paper" if ds.manifest.get("mode")=="forward_paper" else "real_research")
    report={"run_id":run_id,"kind":kind,"mode":report_mode,"as_of":str(latest_date.date()),"decision_cutoff":str(decision_cutoff),"next_session":str(next_date.date()),"historical_research_cutoff":ds.manifest.get("historical_research_cutoff"),"config":cfg.dict(),"metrics":metrics|ic_summary,"curve":curves,"rankings":ranked,"plan":plan,"factors":factor_rows,"candidates":factor_candidates(panel,cfg),"events":events.to_dict("records"),"folds":folds,"ablation":ablation,"backtest_audit":backtest_audit,"quality":ds.quality,"warnings":warnings,"worldquant101":{"usable":83,"unavailable":18,"unavailable_ids":[48,58,59,63,67,69,70,76,79,80,82,87,89,90,91,93,97,100]},"model_card":{"algorithm":"核心价值/质量/低风险 50% + WorldQuant 101可用因子组 50%","challengers":"LightGBM LambdaRank + Ridge（仅观察，不参与订单）","features":len(model.columns),"text_features":len(model.text_columns),"text_active":text_active,"text_weight_used":cfg.text_alpha_weight if text_active else 0.,"train_end":model.trained_through,"train_rows":model.rows,"horizon":cfg.label_horizon,"refit_every":cfg.refit_every,"explanation":"GLM只抽取当时可见事件；结构化分数80%与事件分数20%做后融合。无合格事件时结构化分数权重为100%。TreeSHAP只解释LightGBM挑战者。","validation":"每日收盘评分、下一交易日开盘调仓；成熟标签、时间隔离与滚动样本外挑战者预测","live_ready":False}}
    report=clean(report)
    write_json(run_dir/"report.json",report)
    write_json(run_dir/"plan.json",plan)
    write_json(run_dir/"config.json",cfg.dict())
    order_columns=list((plan["orders"] or plan.get("preview_orders") or [{"signal_date":None,"trade_date":None,"symbol":None,"side":None,"shares":None,"limit_price":None}])[0])
    pd.DataFrame(plan["orders"],columns=order_columns).to_csv(run_dir/"orders.csv",index=False)
    latest[latest_columns].to_csv(run_dir/"rankings.csv",index=False)
    write_json(run_dir/"manifest.json",{"run_id":run_id,"created_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),"dataset":str(Path(folder).resolve()),"data_hashes":dataset_hash(folder),"python":platform.python_version(),"dependencies":{p:importlib.metadata.version(p) for p in ["numpy","pandas","scikit-learn","lightgbm"]},"data_manifest":ds.manifest,"model_train_end":model.trained_through})
    model.ranker.booster_.save_model(str(run_dir/"ranker.txt"))
    if model.text_model is not None:model.text_model.booster_.save_model(str(run_dir/"text_ranker.txt"))
    write_json(run_dir/"ridge.json",{"columns":model.columns,"coefficients":model.ridge.coef_.tolist(),"intercept":float(model.ridge.intercept_)})
    # Atomic pointer update; failed runs never replace the latest good report.
    pointer=run_dir.parent/"latest.json"
    temp=pointer.with_suffix(".tmp");write_json(temp,{"run_id":run_id});temp.replace(pointer)
    progress("完成：评分、回测与次日条件计划已保存",1.)
    return report
