from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from dataclasses import replace
from ashare.config import load_config
from ashare.models import production_scores
from ashare.data import load_daily_dataset
from ashare.portfolio import backtest
from ashare.research import performance
from ashare.pipeline import write_json
p=Path(sys.argv[1]);r=json.loads((p/'report.json').read_text());scores=pd.read_parquet(p/'scores.parquet')
cfg=load_config(ROOT/'runs/research-xgboost-config.json')
ds=load_daily_dataset(ROOT/'data/reconstruction/20260915-20261116/dataset',lookback_sessions=100)
results=[]
for name,c in [('固定因子 · 原日度换仓',replace(cfg,structured_model='prior',hold_buffer=0,max_daily_turnover=2.,enable_text_alpha=False)),('XGBoost · 原日度换仓',replace(cfg,hold_buffer=0,max_daily_turnover=2.,enable_text_alpha=False)),('XGBoost · 持仓缓冲与换手约束',replace(cfg,enable_text_alpha=False))]:
 s=production_scores(scores,c);curve,trades,_,_=backtest(s,ds,c)
 m=performance(curve);m['filled_orders']=int(trades.filled_shares.gt(0).sum());results.append({'name':name,**m});print(name,m['total_return'],m['total_fees'],m['average_exposure'],flush=True)
r['ablation']=results;r['warnings']=r.get('warnings',[])+['AI密钥返回401：Key is blocked；本报告仅为结构化对照，30%事件重建尚未完成。','平均仓位差异显著，请同时比较平均仓位、费用和回撤，不能把相对满仓指数的差额全部解释为选股能力。']
r['text']['configured_weight']=.3;r['text']['status']='blocked_api_key';r['text']['reason']='Paratera密钥被停用，未发布虚假的30%AI事件回测。'
write_json(p/'report.json',r);write_json(p/'framework_comparison.json',results)
