"""Resume historical AI scoring and rebuild the local research report.

Usage: .venv/bin/python scripts/run_ai_reconstruction.py
Public inputs/cache live outside Git. Never logs API credentials.
"""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ashare.config import load_config
from ashare.pipeline import write_json
from dataclasses import replace
import httpx,pandas as pd

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--start',default='2026-09-15');parser.add_argument('--end',default='2026-11-16');a=parser.parse_args()
    # Read only environment assignments, without evaluating shell code.
    for line in (ROOT/'.env').read_text().splitlines():
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line:continue
        k,v=line.removeprefix('export ').split('=',1)
        os.environ[k.strip()]=v.strip().strip('"').strip("'")
    output=ROOT/'data/reconstruction'/f"{a.start.replace('-','')}-{a.end.replace('-','')}"
    status=output/'status.json';output.mkdir(parents=True,exist_ok=True)
    def update(state,message,**extra):write_json(status,{'state':state,'message':message,'updated_at':pd.Timestamp.now(tz='Asia/Shanghai'),**extra})
    try:
        update('checking','检查AI服务')
        response=httpx.post(os.environ['AI_BASE_URL'].rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+os.environ['AI_API_KEY']},json={'model':os.environ['AI_MODEL'],'messages':[{'role':'user','content':'只返回JSON：{"ok":true}'}],'temperature':0,'max_tokens':30},timeout=30)
        if response.status_code==401:
            update('blocked','Paratera返回401：当前AI_API_KEY被停用或无效。恢复本地.env密钥后重新运行本命令。');print('AI密钥被停用；历史材料已保留，没有发布虚假的30%事件回测。');return 2
        response.raise_for_status()
        from ashare.reconstruct_signals import reconstruct
        update('extracting','AI按历史决策日处理原文')
        summary=reconstruct(output,ROOT/'data/live/current',a.start,a.end,workers=8)
        update('backtesting','执行XGBoost70% + AI30%历史重建回测',summary=summary)
        from ashare.replay import run_replay
        cfg=replace(load_config(),structured_model='xgboost',hold_buffer=5,max_daily_turnover=.6,text_alpha_weight=.3)
        report=run_replay(output/'dataset',cfg,a.start,a.end,reconstruction=output,incremental=False)
        update('complete','历史重建回测完成',run_id=report['run_id'],summary=summary)
        print(ROOT/'runs'/report['run_id']);return 0
    except Exception as exc:
        update('failed','未完成：'+type(exc).__name__+'；已成功原文和AI缓存可复用。')
        raise
if __name__=='__main__':sys.exit(main())
