"""Bounded symbolic factor research. Expressions are parsed, never eval/exec'ed."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .features import PRICE_FEATURES,FUND_FEATURES,TEXT_FEATURES

ALLOWED=["x_"+c for c in PRICE_FEATURES+FUND_FEATURES+TEXT_FEATURES]

def evaluate_formula(expression,frame):
    tree=ast.parse(expression,mode="eval")
    if len(list(ast.walk(tree)))>40:raise ValueError("因子表达式过长")
    def walk(node):
        if isinstance(node,ast.Expression):return walk(node.body)
        if isinstance(node,ast.Name) and node.id in ALLOWED:return frame[node.id]
        if isinstance(node,ast.Constant) and type(node.value) in (int,float) and abs(node.value)<=10:return node.value
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -walk(node.operand)
        if isinstance(node,ast.BinOp):
            a,b=walk(node.left),walk(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
            args=[walk(a) for a in node.args]
            if node.func.id=="abs" and len(args)==1:return np.abs(args[0])
            if node.func.id=="minimum" and len(args)==2:return np.minimum(*args)
            if node.func.id=="maximum" and len(args)==2:return np.maximum(*args)
        raise ValueError("只允许已注册因子、有限常数、加减乘以及abs/minimum/maximum；拒绝代码执行")
    result=walk(tree)
    if not isinstance(result,pd.Series):raise ValueError("因子必须返回每条股票记录的数值")
    return result.replace([np.inf,-np.inf],np.nan)

def default_candidates(budget=32):
    rows=[{"formula":"x_roe + x_earnings_yield","hypothesis":"质量与估值的互补"},{"formula":"x_cashflow_yield - x_accruals","hypothesis":"经营现金流验证盈利质量"},{"formula":"x_momentum_60 - x_volatility_20","hypothesis":"风险调整后的中期动量"},{"formula":"x_reversal_1 - x_amihud_20","hypothesis":"在流动性好的证券中检验短期反转"},{"formula":"x_revenue_growth + x_book_to_price","hypothesis":"兼顾增长与估值"},{"formula":"x_event_sentiment * x_earnings_yield","hypothesis":"价值与新事件的交互"}]
    for a in ["x_momentum_5","x_momentum_20","x_momentum_60","x_reversal_1"]:
        for b in ["x_roe","x_cashflow_yield","x_earnings_yield","x_volatility_20"]:
            for op in ["+","*"]:
                rows.append({"formula":f"{a} {op} {b}","hypothesis":"价格行为与基本面/风险的预注册交互候选"})
    return rows[:max(1,min(budget,100))]

def mine(train,cfg,proposals=None,budget=32):
    from .research import rank_ic,block_bootstrap_mean
    candidates=(proposals or default_candidates(budget))[:budget]
    days=sorted(train.date.unique())
    if len(days)<60:return []
    boundary=pd.Timestamp(days[int(.7*len(days))])
    # Purge labels crossing discovery -> internal validation boundary.
    discovery=(train.date<boundary)&(train.label_end<boundary)
    validation=train.date>=boundary
    results=[]
    for p in candidates:
        values=evaluate_formula(p["formula"],train)
        frame=pd.DataFrame({"date":train.date,"x":values,"y":train.target})
        first=frame[discovery].groupby("date").apply(lambda g:rank_ic(g.x,g.y),include_groups=False).dropna()
        last=frame[validation].groupby("date").apply(lambda g:rank_ic(g.x,g.y),include_groups=False).dropna()
        direction=1 if first.mean()>=0 else -1
        lo,hi=block_bootstrap_mean(last*direction,cfg.seed,block=max(10,cfg.label_horizon),iterations=200)
        results.append({"candidate":hashlib.sha256(p["formula"].encode()).hexdigest()[:10],"formula":p["formula"],"hypothesis":p.get("hypothesis",""),"direction_from_discovery":direction,"discovery_ic":float(first.mean()) if len(first) else None,"validation_ic":float(last.mean()*direction) if len(last) else None,"ci_low":lo,"ci_high":hi,"discovery_end":str(train.loc[discovery,"date"].max())[:10],"validation_start":str(boundary.date()),"max_input_date":str(train.date.max())[:10],"trials":len(candidates),"status":"候选，未自动上线"})
    return results

def propose_with_llm(provider="ollama",model=None,budget=10):
    import os
    import httpx
    model=model or os.getenv("AI_MODEL")
    if not model:raise ValueError("设置 AI_MODEL 后才能生成AI因子提议")
    prompt="为沪深300日频研究提出最多"+str(min(budget,30))+"个有经济解释的因子。只返回JSON对象 candidates 数组，每项formula和hypothesis。仅可用以下变量："+", ".join(ALLOWED)+"。表达式只允许加、减、乘、abs、minimum、maximum以及绝对值<=10的常数，不允许Python属性、索引、导入或读取外部数据。不要声称经过回测。"
    messages=[{"role":"user","content":prompt}]
    if provider=="ollama":
        r=httpx.post(os.getenv("AI_BASE_URL","http://127.0.0.1:11434").rstrip("/")+"/api/chat",json={"model":model,"messages":messages,"stream":False,"format":"json","options":{"temperature":.2}},timeout=180)
        r.raise_for_status();body=r.json()["message"]["content"]
    else:
        base=os.getenv("AI_BASE_URL");key=os.getenv("AI_API_KEY")
        if not base or not key:raise ValueError("需要 AI_BASE_URL 和 AI_API_KEY")
        r=httpx.post(base.rstrip("/")+"/chat/completions",headers={"Authorization":"Bearer "+key},json={"model":model,"messages":messages,"temperature":.2,"response_format":{"type":"json_object"}},timeout=180)
        r.raise_for_status();body=r.json()["choices"][0]["message"]["content"]
    rows=json.loads(body)["candidates"][:min(budget,30)]
    fixture=pd.DataFrame({c:[0.,.1] for c in ALLOWED})
    for row in rows:evaluate_formula(row["formula"],fixture)
    return {"model":model,"prompt_version":"symbolic-proposals-v1","candidates":rows,"status":"未回测的提议"}
