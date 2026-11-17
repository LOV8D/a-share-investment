"""Run the expanded zoo with training-only screening and the saved AI corpus."""
from pathlib import Path
import json,os
from ashare.config import ROOT,load_config
from ashare.replay import run_replay
from ashare.pipeline import write_json


def main():
    # The saved AI reconstruction is reused; this job does not call an LLM.
    folder=ROOT/'data/live/current'
    reconstruction=ROOT/'data/reconstruction/20260915-20261116'
    cfg=load_config(overrides={'factor_zoo_enabled':True,'structured_model':'xgboost'})
    write_json(ROOT/'runs/factor_zoo_job.json',{'state':'running','message':'计算158个新增候选并按训练窗口粗筛'})
    try:
        print('Computing expanded features and training screened XGBoost...',flush=True)
        result=run_replay(folder,cfg,'2026-09-15','2026-11-16',reconstruction=reconstruction,incremental=False)
        write_json(ROOT/'config.json',cfg.dict())
        # Keep the original event audit linked to the currently displayed reconstruction.
        status_path=reconstruction/'status.json';status=json.loads(status_path.read_text())
        status['run_id']=result['run_id'];status['message']='扩展因子池的历史事件重建回测完成'
        write_json(status_path,status)
        write_json(ROOT/'runs/factor_zoo_job.json',{'state':'complete','run_id':result['run_id'],'metrics':result['metrics'],'selected':result['factor_screen']['selected_count']})
        print(result['run_id'],json.dumps(result['metrics'],ensure_ascii=False),flush=True)
    except Exception as exc:
        write_json(ROOT/'runs/factor_zoo_job.json',{'state':'failed','error':type(exc).__name__+': '+str(exc)})
        raise


if __name__=='__main__':main()
