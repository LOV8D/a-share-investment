import numpy as np
import pandas as pd
from .features import PRICE_FEATURES,FUND_FEATURES,TEXT_FEATURES,WORLDQUANT_FEATURES

def performance(curve):
    r=curve.equity.pct_change().fillna(0)
    b=curve.benchmark.pct_change().fillna(0)
    n=max(len(r)-1,1)
    excess=r-b
    ann=float(curve.equity.iloc[-1]**(252/n)-1)
    bench_ann=float(curve.benchmark.iloc[-1]**(252/n)-1)
    drawdown=curve.equity/curve.equity.cummax()-1
    benchmark_drawdown=curve.benchmark/curve.benchmark.cummax()-1
    return {"total_return":float(curve.equity.iloc[-1]-1),"annual_return":ann,"benchmark_annual_return":bench_ann,"annual_excess":ann-bench_ann,"max_drawdown":float(drawdown.min()),"benchmark_max_drawdown":float(benchmark_drawdown.min()),"annualized_volatility":float(r.std()*np.sqrt(252)),"benchmark_annualized_volatility":float(b.std()*np.sqrt(252)),"sharpe":float(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else None,"benchmark_sharpe":float(b.mean()/b.std()*np.sqrt(252)) if b.std()>0 else None,"information_ratio":float(excess.mean()/excess.std()*np.sqrt(252)) if excess.std()>0 else None,"tracking_error":float(excess.std()*np.sqrt(252)),"average_turnover":float(curve.turnover.mean()),"total_fees":float(curve.fees.sum()),"average_exposure":float(curve.exposure.mean()),"sessions":len(curve),"start":curve.date.iloc[0],"end":curve.date.iloc[-1]}

def rank_ic(a,b):
    valid=pd.DataFrame({"a":a,"b":b}).dropna()
    if len(valid)<10 or valid.a.nunique()<2 or valid.b.nunique()<2:return np.nan
    return valid.a.rank().corr(valid.b.rank())

def block_bootstrap_mean(values,seed=42,block=10,iterations=500):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if len(a)<20:return [None,None]
    rng=np.random.default_rng(seed);means=[]
    for _ in range(iterations):
        starts=rng.integers(0,len(a),size=int(np.ceil(len(a)/block)))
        sample=np.concatenate([a[(s+np.arange(block))%len(a)] for s in starts])[:len(a)]
        means.append(sample.mean())
    return [float(v) for v in np.quantile(means,[.025,.975])]

def factor_report(scores, cfg):
    usable=scores[scores.eligible & scores.target.notna()]
    rows=[]
    for factor in PRICE_FEATURES+FUND_FEATURES+WORLDQUANT_FEATURES+TEXT_FEATURES:
        values=[rank_ic(g[f"x_{factor}"],g.target) for _,g in usable.groupby("date")]
        values=np.asarray(values);valid=values[np.isfinite(values)]
        lo,hi=block_bootstrap_mean(valid,cfg.seed,block=max(10,cfg.label_horizon))
        family = "文本事件" if factor in TEXT_FEATURES else "WorldQuant 101" if factor in WORLDQUANT_FEATURES else "基本面" if factor in FUND_FEATURES else "量价"
        rows.append({"factor":factor,"family":family, "rank_ic":float(valid.mean()) if len(valid) else None,"icir":float(valid.mean()/valid.std()) if len(valid) and valid.std()>0 else None,"positive_rate":float((valid>0).mean()) if len(valid) else None,"ci_low":lo,"ci_high":hi,"days":len(valid),"status":"待独立检验"})
    daily=pd.DataFrame([{"date":str(d)[:10],"rank_ic":rank_ic(g.score,g.target),"structured_ic":rank_ic(g.structured_score,g.target)} for d,g in usable.groupby("date")])
    model_ic=float(daily.rank_ic.mean()) if len(daily) else None
    ci=block_bootstrap_mean(daily.rank_ic,cfg.seed,block=max(10,cfg.label_horizon)) if len(daily) else [None,None]
    return rows,daily,{"rank_ic":model_ic,"rank_ic_ci":ci,"note":"按日期分块 bootstrap；因子表为样本外诊断，查看后已消耗该测试集，不能用它反复选参。"}

def factor_candidates(panel,cfg):
    """Deterministic, bounded factor mining on TRAIN data only, no eval/exec of AI code."""
    from .models import training_slice
    from .mining import mine
    return mine(training_slice(panel,panel.date.max(),cfg),cfg,budget=6)
