import argparse
import json
from pathlib import Path
from .config import ROOT,load_config

def main():
    parser=argparse.ArgumentParser(description="A share investment · 沪深300量化与AI研究引擎")
    parser.add_argument("--config",default=str(ROOT/"config.json"))
    sub=parser.add_subparsers(dest="command",required=True)
    demo=sub.add_parser("demo");demo.add_argument("--days",type=int,default=560);demo.add_argument("--symbols",type=int,default=300)
    sub.add_parser("templates")
    for command in ["validate","research"]:
        p=sub.add_parser(command);p.add_argument("--data",default=str(ROOT/"data/inbox"));p.add_argument("--capital",type=float);p.add_argument("--text-alpha",action="store_true")
    daily=sub.add_parser("daily");daily.add_argument("--data",default=str(ROOT/"data/live/current"));daily.add_argument("--capital",type=float);daily.add_argument("--text-alpha",action="store_true")
    collect=sub.add_parser("collect-live");collect.add_argument("--as-of");collect.add_argument("--no-text",action="store_true")
    paper=sub.add_parser("paper");paper.add_argument("--as-of");paper.add_argument("--no-text",action="store_true");paper.add_argument("--force",action="store_true");paper.add_argument("--capital",type=float);paper.add_argument("--text-alpha",action="store_true")
    serve=sub.add_parser("serve");serve.add_argument("--host",default="127.0.0.1");serve.add_argument("--port",type=int,default=8765)
    imp=sub.add_parser("import");imp.add_argument("--source",required=True);imp.add_argument("--table",required=True);imp.add_argument("--mapping",required=True);imp.add_argument("--dest",default=str(ROOT/"data/inbox"))
    ext=sub.add_parser("extract");ext.add_argument("--data",default=str(ROOT/"data/inbox"));ext.add_argument("--provider",choices=["ollama","compatible"],default="ollama");ext.add_argument("--model");ext.add_argument("--as-of")
    replay=sub.add_parser("replay");replay.add_argument("--data",default=str(ROOT/"data/live/current"));replay.add_argument("--start",required=True);replay.add_argument("--end",required=True);replay.add_argument("--capital",type=float)
    feed=sub.add_parser("fetch-rss");feed.add_argument("--url",required=True);feed.add_argument("--symbol",required=True);feed.add_argument("--output",required=True);feed.add_argument("--source-type",default="news",choices=["news","policy","filing","forum"])
    pre=sub.add_parser("precheck");pre.add_argument("--run",required=True);pre.add_argument("--data",default=str(ROOT/"data/inbox"))
    miner=sub.add_parser("mine");miner.add_argument("--data",default=str(ROOT/"data/inbox"));miner.add_argument("--proposals");miner.add_argument("--budget",type=int,default=32);miner.add_argument("--output",default=str(ROOT/"runs/factor-experiments.json"))
    propose=sub.add_parser("propose-factors");propose.add_argument("--provider",choices=["ollama","compatible"],default="ollama");propose.add_argument("--model");propose.add_argument("--budget",type=int,default=10);propose.add_argument("--output",default=str(ROOT/"runs/factor-proposals.json"))
    args=parser.parse_args()
    from .data import load_dataset,write_templates,read_table
    cfg=load_config(args.config)
    if args.command=="collect-live":
        from .live_data import collect_live
        result=collect_live(args.as_of,include_text=not args.no_text,cfg=cfg,progress=lambda msg,p:print(f"{p:.0%} {msg}",flush=True))
        print(json.dumps(result,ensure_ascii=False,indent=2,default=str));return
    if args.command=="paper":
        overrides={}
        if args.capital is not None:overrides["initial_cash"]=args.capital
        if args.text_alpha:overrides["enable_text_alpha"]=True
        cfg=load_config(args.config,overrides)
        from .live_data import collect_live,LIVE_ROOT
        from .paper import advance_paper_account,verify_and_finalize_run,paper_job_lock
        from .pipeline import run_engine,write_json
        from datetime import datetime
        from zoneinfo import ZoneInfo
        with paper_job_lock(LIVE_ROOT):
            status=collect_live(args.as_of,include_text=not args.no_text,cfg=cfg,progress=lambda msg,p:print(f"{p*.55:.0%} {msg}",flush=True))
            schedule_path=ROOT/"runs/paper_schedule.json"
            schedule=json.loads(schedule_path.read_text()) if schedule_path.exists() else {}
            today=str(datetime.now(ZoneInfo("Asia/Shanghai")).date())
            if not args.force and schedule.get("last_success_signal")==status["signal_date"]:
                schedule.update(last_attempt_date=today,last_result="no_new_session")
                write_json(schedule_path,schedule)
                print(f"没有新的交易日：最新仍为 {status['signal_date']}，本次不重复生成计划");return
            if not args.no_text and cfg.enable_text_alpha:
                from .events import extract_current_documents
                extract_current_documents(LIVE_ROOT/"current")
            advance_paper_account(LIVE_ROOT/"current",cfg,LIVE_ROOT)
            report=run_engine(LIVE_ROOT/"current",cfg,"daily",progress=lambda msg,p:print(f"{55+p*40:.0f}% {msg}",flush=True))
            report=verify_and_finalize_run(report["run_id"],cfg,LIVE_ROOT)
            from .replay import run_replay
            print(f"96% 刷新 {cfg.paper_replay_start} 起的严格点时模拟盘",flush=True)
            replay=run_replay(LIVE_ROOT/"current",cfg,cfg.paper_replay_start,report["as_of"])
            write_json(schedule_path,{"last_attempt_date":today,"last_success_signal":report["as_of"],"run_id":report["run_id"],"last_result":"success"})
            print(ROOT/"runs"/report["run_id"])
            print(ROOT/"runs"/replay["run_id"]);return
    if args.command=="propose-factors":
        from .mining import propose_with_llm
        from .pipeline import write_json
        write_json(args.output,propose_with_llm(args.provider,args.model,args.budget));print(args.output);return
    if args.command=="mine":
        from .features import build_panel
        from .models import training_slice
        from .mining import mine
        from .pipeline import write_json
        panel=build_panel(load_dataset(args.data),cfg)
        proposals=json.loads(Path(args.proposals).read_text())["candidates"] if args.proposals else None
        results=mine(training_slice(panel,panel.date.max(),cfg),cfg,proposals,max(1,min(args.budget,100)))
        write_json(args.output,{"experiments":results,"trial_count":len(results),"status":"未上线候选；需要外部保留期检验"});print(args.output);return
    if args.command=="replay":
        from .replay import run_replay
        if args.capital is not None:cfg=load_config(args.config,{"initial_cash":args.capital})
        result=run_replay(args.data,cfg,args.start,args.end)
        print(ROOT/"runs"/result["run_id"]);return
    if args.command=="templates":
        write_templates(ROOT/"data/templates");print(ROOT/"data/templates");return
    if args.command=="serve":
        import uvicorn
        uvicorn.run("ashare.web:app",host=args.host,port=args.port);return
    if args.command=="demo":
        from .demo import generate_demo
        from .pipeline import run_engine
        path=generate_demo(ROOT/"data/demo",args.days,args.symbols,cfg.seed)
        report=run_engine(path,cfg,progress=lambda msg,p:print(f"{p:.0%} {msg}",flush=True))
        print("合成演示完成：",report["run_id"]);return
    if args.command=="import":
        from .importer import convert
        print("导入行数：",convert(args.source,args.table,args.mapping,args.dest));return
    if args.command=="extract":
        from .events import extract_documents
        print(json.dumps(extract_documents(read_table(args.data,"documents"),Path(args.data)/"events.csv",args.provider,args.model,args.as_of),ensure_ascii=False));return
    if args.command=="fetch-rss":
        from .events import fetch_rss
        fetch_rss(args.url,args.symbol,args.source_type).to_csv(args.output,index=False);print(args.output);return
    if args.command=="precheck":
        from .portfolio import auction_precheck
        from .pipeline import write_json
        plan=json.loads((ROOT/"runs"/args.run/"plan.json").read_text())
        holdings=read_table(args.data,"holdings")
        result=auction_precheck(plan,read_table(args.data,"auction"),dict(zip(holdings.symbol,holdings.sellable_shares)))
        dest=ROOT/"runs"/args.run/"precheck.json";write_json(dest,result);print(dest);return
    overrides={}
    if args.capital is not None:overrides["initial_cash"]=args.capital
    if args.text_alpha:overrides["enable_text_alpha"]=True
    cfg=load_config(args.config,overrides)
    if args.command=="validate":
        ds=load_dataset(args.data,strict=False);print(json.dumps(ds.quality,ensure_ascii=False,indent=2));raise SystemExit(0 if ds.quality["passed"] else 1)
    from .pipeline import run_engine
    if args.command=="daily" and cfg.enable_text_alpha:
        from .events import extract_current_documents
        extract_current_documents(args.data)
    report=run_engine(args.data,cfg,"research" if args.command=="research" else "daily",progress=lambda msg,p:print(f"{p:.0%} {msg}",flush=True))
    print(ROOT/"runs"/report["run_id"])

if __name__=="__main__":main()
