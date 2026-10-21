import io
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .config import ROOT,load_config,Config
from .schema import SCHEMAS,DESCRIPTIONS,REQUIRED
from .data import load_dataset,write_templates,read_table,cutoff

@asynccontextmanager
async def lifespan(app):
    stop=threading.Event()
    thread=None
    if os.getenv("ASHARE_INTERNAL_SCHEDULER","1") != "0":
        thread=threading.Thread(target=scheduler_loop,args=(stop,),daemon=True)
        thread.start()
    yield
    stop.set()

app=FastAPI(title="A share investment",docs_url="/api/docs",redoc_url=None,lifespan=lifespan)
allowed_hosts=[item.strip() for item in os.getenv("APP_ALLOWED_HOSTS","127.0.0.1,localhost,testserver").split(",") if item.strip()]
app.add_middleware(TrustedHostMiddleware,allowed_hosts=allowed_hosts)
app.mount("/static",StaticFiles(directory=ROOT/"ashare/static"),name="static")
state={"running":False,"message":"就绪","progress":0.,"error":None,"error_at":None}
lock=threading.Lock()

def _lightweight_quality(folder:Path):
    """Build the data-centre inventory without loading the full research panel.

    The assembled real dataset ships with an audited quality report.  Reading
    Parquet footers and CSV headers keeps the always-on web process small; the
    daily worker still performs its own rolling-window validation before it can
    publish a plan.
    """
    report_path=next((folder/name for name in ("质量报告.json","quality_report.json") if (folder/name).exists()),None)
    if report_path is None:return None
    report=json.loads(report_path.read_text())
    counts=report.get("tables",{})
    inventory=[]
    structural_errors=[]
    newest_input_mtime=0.
    for name,columns in SCHEMAS.items():
        path=next((folder/f"{name}.{ext}" for ext in ("parquet","csv") if (folder/f"{name}.{ext}").exists()),None)
        exists=path is not None
        rows=int(counts.get(name,{}).get("rows",0) or 0)
        actual_columns=[]
        if path is not None:
            newest_input_mtime=max(newest_input_mtime,path.stat().st_mtime)
            if path.suffix==".parquet":
                import pyarrow.parquet as pq
                parquet=pq.ParquetFile(path)
                rows=parquet.metadata.num_rows
                actual_columns=parquet.schema_arrow.names
            else:
                actual_columns=list(pd.read_csv(path,nrows=0).columns)
        missing=sorted(set(columns)-set(actual_columns)) if exists else []
        status="missing" if not exists else ("error" if missing else "ready")
        if name in REQUIRED and not exists:structural_errors.append(f"{name}: 缺少必需数据")
        elif missing:structural_errors.append(f"{name}: 缺少字段 {missing}")
        inventory.append({"name":name,"description":DESCRIPTIONS[name],"rows":rows,"required":name in REQUIRED,"status":status,"columns":columns})
    warnings=list(report.get("warnings",[]))
    if newest_input_mtime>report_path.stat().st_mtime+1:
        warnings.append("部分数据文件晚于质量报告；每日任务仍会在发布计划前重新校验滚动窗口")
    errors=list(report.get("errors",[]))+structural_errors
    return {
        "errors":errors,
        "warnings":warnings,
        "inventory":inventory,
        "passed":bool(report.get("passed",not errors)) and not structural_errors,
        "quality_report":{"created_at":report.get("created_at"),"lightweight":True},
        "price_assembly":report.get("price_assembly"),
    }

@app.middleware("http")
async def local_origin(request,call_next):
    if request.method not in ("GET","HEAD","OPTIONS"):
        origin=request.headers.get("origin")
        if origin:
            parsed=urlparse(origin)
            if parsed.netloc != request.headers.get("host") or parsed.scheme not in ("http","https"):
                return JSONResponse({"detail":"仅允许本地同源操作"},status_code=403)
    response=await call_next(request)
    response.headers["X-Content-Type-Options"]="nosniff"
    response.headers["Referrer-Policy"]="no-referrer"
    response.headers["Content-Security-Policy"]="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
    return response

@app.get("/")
def index():return FileResponse(ROOT/"ashare/static/index.html")

@app.get("/api/status")
def status():
    pointer=ROOT/"runs/latest.json"
    latest=json.loads(pointer.read_text())["run_id"] if pointer.exists() else None
    from .live_data import get_live_status
    return state|{"latest_run_id":latest,"live":get_live_status()}

@app.get("/api/live-status")
def live_status():
    from .live_data import get_live_status
    return get_live_status()

@app.get("/api/report")
def report(run_id:str|None=None):
    if run_id is None:
        pointer=ROOT/"runs/latest.json"
        if not pointer.exists():return {"empty":True}
        run_id=json.loads(pointer.read_text())["run_id"]
    if Path(run_id).name!=run_id:raise HTTPException(400,"非法运行编号")
    path=ROOT/"runs"/run_id/"report.json"
    if not path.is_file():raise HTTPException(404,"找不到运行结果")
    return json.loads(path.read_text())


@app.get("/api/intelligence")
def intelligence():
    """Show the forward event ledger independently from signal eligibility.

    A freshly processed event may be visible here before it becomes eligible
    for a portfolio score. The eligibility flag makes that timing explicit.
    """
    folder=ROOT/"data/live/current"
    documents=read_table(folder,"documents")
    events=read_table(folder,"events")
    cutoff_value=None
    current=report()
    if not current.get("empty") and current.get("decision_cutoff"):
        cutoff_value=pd.Timestamp(current["decision_cutoff"])
    if len(events):
        events=events.copy()
        events["usable_at"]=events[["published_at","first_seen_at","processed_at"]].max(axis=1)
        events["signal_eligible"]=True if cutoff_value is None else events.usable_at.le(cutoff_value)
        events=events.sort_values("usable_at",ascending=False).head(100)
        records=json.loads(events.to_json(orient="records",date_format="iso",force_ascii=False))
    else:
        records=[]
    error_path=folder/"events.errors.json"
    try:
        errors=json.loads(error_path.read_text()) if error_path.exists() else []
    except (OSError,ValueError):
        errors=[]
    extracted_documents=int(events.document_id.astype(str).nunique()) if len(events) else 0
    return {
        "events":records,
        "document_count":int(len(documents)),
        "event_count":int(len(events)),
        "pending_documents":max(0,int(len(documents))-extracted_documents),
        "errors":errors[-20:],
        "model":os.getenv("AI_MODEL") or None,
        "decision_cutoff":current.get("decision_cutoff") if isinstance(current,dict) else None,
    }

@app.get("/api/research-report")
def research_report():
    """Read the latest completed real-data backtest without rerunning it on the web host."""
    for path in sorted((ROOT/"runs").glob("*/report.json"),reverse=True):
        try:
            result=json.loads(path.read_text())
        except (OSError,ValueError):
            continue
        if ((result.get("kind")=="research" and result.get("mode")=="real_research")
            or result.get("kind")=="point_in_time_replay") and result.get("curve"):
            return result
    return {"empty":True}

@app.get("/api/runs")
def runs():
    return [{"run_id":p.parent.name} for p in sorted((ROOT/"runs").glob("*/report.json"),reverse=True)][:30]

@app.get("/api/config")
def config():return load_config().dict()

@app.put("/api/config")
async def save_config(request:Request):
    try:
        data=await request.json()
        cfg=Config(**data)
    except (TypeError,ValueError) as e:raise HTTPException(400,str(e))
    from .pipeline import write_json
    write_json(ROOT/"config.json",cfg.dict())
    return cfg.dict()

@app.get("/api/data")
def data(mode:str="real"):
    if mode not in ("real","synthetic"):raise HTTPException(400,"未知模式")
    folder=ROOT/("data/demo" if mode=="synthetic" else "data/inbox")
    try:
        quality=_lightweight_quality(folder) if mode=="real" else None
        if quality is None:quality=load_dataset(folder,strict=False).quality
        staging_paths=[ROOT/"data/staging/real/manifest.json",ROOT/"data/staging/csmar/manifest.json"]
        staging_path=next((path for path in staging_paths if path.exists()),None)
        quality["staging"]=json.loads(staging_path.read_text()) if mode=="real" and staging_path else None
        dataset_manifest=folder/"manifest.json"
        quality["dataset_manifest"]=json.loads(dataset_manifest.read_text()) if dataset_manifest.exists() else None
        if mode=="real":
            from .live_data import get_live_status
            quality["live"]=get_live_status()
            current=ROOT/"data/live/current"
            live_inventory={}
            if current.exists():
                try:
                    import pyarrow.parquet as pq
                    for name in SCHEMAS:
                        path=current/f"{name}.parquet"
                        if path.exists():live_inventory[name]=pq.ParquetFile(path).metadata.num_rows
                except Exception:
                    live_inventory={}
            quality["live_inventory"]=live_inventory
        return quality
    except Exception as e:raise HTTPException(400,f"文件解析失败：{str(e)[:300]}")

@app.get("/api/schema")
def schema():return [{"name":n,"columns":c,"description":DESCRIPTIONS[n],"required":n in REQUIRED} for n,c in SCHEMAS.items()]

@app.post("/api/import/{table}")
async def upload(table:str,request:Request,ext:str="csv"):
    if table not in SCHEMAS or ext not in ("csv","parquet"):raise HTTPException(400,"未知表名或文件格式")
    if state["running"]:raise HTTPException(409,"引擎运行中，完成后再导入")
    raw=await request.body()
    if len(raw)>80_000_000:raise HTTPException(413,"单文件限制80MB，大文件请直接放入 data/inbox")
    try:
        frame=pd.read_csv(io.BytesIO(raw),dtype={"symbol":str}) if ext=="csv" else pd.read_parquet(io.BytesIO(raw))
        missing=set(SCHEMAS[table])-set(frame)
        if missing:raise ValueError("缺少字段："+", ".join(sorted(missing)))
    except Exception as e:raise HTTPException(400,str(e)[:400])
    dest=ROOT/"data/inbox"/f"{table}.{ext}"
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_suffix(".upload");temp.write_bytes(raw);temp.replace(dest)
    alternate=dest.with_suffix(".parquet" if ext=="csv" else ".csv")
    if alternate.exists():alternate.unlink()
    return {"table":table,"rows":len(frame)}

@app.post("/api/run")
async def start_run(request:Request):
    body=await request.json()
    mode=body.get("mode","synthetic");kind=body.get("kind","research")
    if mode not in ("synthetic","real") or kind not in ("daily","research"):raise HTTPException(400,"未知模式")
    if mode=="real" and kind=="research" and os.getenv("ASHARE_INTERNAL_SCHEDULER")=="0":
        raise HTTPException(409,"云端轻量服务器只展示已完成的历史回测；完整回测请在本地运行，避免云端内存不足")
    if mode=="real" and kind=="daily":
        from .live_data import get_live_status
        live=get_live_status()
        pointer=ROOT/"runs/latest.json"
        if live.get("passed") and pointer.exists():
            try:
                latest_id=json.loads(pointer.read_text())["run_id"]
                latest=json.loads((ROOT/"runs"/latest_id/"report.json").read_text())
                if (latest.get("mode")=="forward_paper" and latest.get("as_of")==live.get("signal_date")
                    and latest.get("config")==load_config().dict()):
                    return {"accepted":False,"message":f"{live['signal_date']} 的模拟计划已生成；数据和参数都没有变化，无需重复运行"}
            except (OSError,ValueError,KeyError):
                pass
    if not launch_job(mode,kind):raise HTTPException(409,"已有任务在运行")
    return {"accepted":True}

@app.post("/api/collect")
def start_collection():
    if not launch_collection():raise HTTPException(409,"已有任务在运行")
    return {"accepted":True}

def launch_collection():
    with lock:
        if state["running"]:return False
        cfg=load_config()
        state.update(running=True,progress=0.,error=None,error_at=None,message="准备采集公开增量数据")
    def worker():
        try:
            from .live_data import collect_live
            result=collect_live(cfg=cfg,progress=lambda message,value:state.update(message=message,progress=value))
            state.update(error=None,message=f"采集完成：{result['signal_date']} 收盘",progress=1.)
        except Exception as e:
            state.update(error=str(e)[:700],error_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),message="采集被质量闸门阻断")
        finally:state["running"]=False
    threading.Thread(target=worker,daemon=True).start()
    return True

def launch_job(mode,kind,scheduled_date=None,scheduled_attempt_date=None):
    with lock:
        if state["running"]:return False
        cfg=load_config()
        state.update(running=True,progress=0.,error=None,error_at=None,message="准备数据")
        if scheduled_date:
            from .pipeline import write_json
            write_json(ROOT/"runs/paper_schedule.json",{
                "last_attempt_date":scheduled_attempt_date or str(pd.Timestamp.now(tz="Asia/Shanghai").date()),
                "target_signal":scheduled_date,
                "last_result":"started",
            })
    def worker():
        try:
            from .pipeline import run_engine
            folder=ROOT/("data/demo" if mode=="synthetic" else "data/inbox")
            if mode=="synthetic" and not (folder/"manifest.json").exists():
                from .demo import generate_demo
                generate_demo(folder)
            if mode=="real" and kind=="daily":
                from .live_data import collect_live,LIVE_ROOT
                from .paper import advance_paper_account,verify_and_finalize_run,paper_job_lock
                with paper_job_lock(LIVE_ROOT):
                    collection=collect_live(cfg=cfg,progress=lambda message,value:state.update(message=message,progress=value*.55))
                    if scheduled_date and collection.get("signal_date") != scheduled_date:
                        raise RuntimeError(f"等待 {scheduled_date} 的完整收盘数据；当前最新为 {collection.get('signal_date','未知')}")
                    folder=LIVE_ROOT/"current"
                    if cfg.enable_text_alpha:
                        from .events import extract_current_documents
                        state.update(message="GLM提取当时可见事件",progress=.56)
                        extract_current_documents(folder)
                    state.update(message="更新模拟账户",progress=.57)
                    advance_paper_account(folder,cfg,LIVE_ROOT)
                    result=run_engine(folder,cfg,kind,progress=lambda message,value:state.update(message=message,progress=.60+value*.35))
                    state.update(message="独立行情源复核计划标的",progress=.96)
                    result=verify_and_finalize_run(result["run_id"],cfg,LIVE_ROOT)
                    from .replay import run_replay
                    state.update(message=f"刷新 {cfg.paper_replay_start} 起的严格点时模拟盘",progress=.98)
                    run_replay(folder,cfg,cfg.paper_replay_start,result["as_of"])
            else:
                result=run_engine(folder,cfg,kind,progress=lambda message,value:state.update(message=message,progress=value))
            state.update(run_id=result["run_id"],error=None)
            if scheduled_date:
                from .pipeline import write_json
                write_json(ROOT/"runs/paper_schedule.json",{
                    "last_attempt_date":scheduled_attempt_date or scheduled_date,
                    "target_signal":scheduled_date,
                    "last_success_signal":result["as_of"],
                    "last_result":"success",
                    "run_id":result["run_id"],
                })
        except Exception as e:
            state.update(error=str(e)[:700],error_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),message="运行未完成，请检查数据与参数")
            if scheduled_date:
                from .pipeline import write_json
                write_json(ROOT/"runs/paper_schedule.json",{
                    "last_attempt_date":scheduled_attempt_date or scheduled_date,
                    "target_signal":scheduled_date,
                    "last_result":"failed",
                    "error":str(e)[:300],
                })
        finally:state["running"]=False
    threading.Thread(target=worker,daemon=True).start()
    return True

def scheduled_signal_date(folder,now):
    from .data import read_table
    calendar=read_table(folder,"calendar")
    if calendar.empty:return None
    today=pd.Timestamp(now.tz_convert("Asia/Shanghai").date())
    completed=calendar.loc[calendar.date<today,"date"]
    return None if completed.empty else pd.Timestamp(completed.max()).normalize()

def schedule_ready(folder,cfg,now):
    from .data import cutoff
    if not cfg.daily_schedule_enabled:return False,"自动运行已关闭"
    today=now.tz_convert("Asia/Shanghai").date()
    if now<cutoff(today,cfg.schedule_time):return False,f"等待 {cfg.schedule_time[:5]} 自动运行"
    target=scheduled_signal_date(folder,now)
    if target is None:return False,"等待真实数据与交易日历"
    return True,f"准备采集并校验 {target.date()} 收盘数据"

def scheduler_loop(stop):
    while not stop.is_set():
        try:
            cfg=load_config();now=pd.Timestamp.now(tz="Asia/Shanghai")
            today=str(now.date())
            done=ROOT/"runs/paper_schedule.json"
            schedule=json.loads(done.read_text()) if done.exists() else {}
            last=schedule.get("last_attempt_date")
            if last==today:
                outcome=schedule.get("last_result")
                if outcome=="success":state["schedule_message"]=f"今日自动运行成功（信号日 {schedule.get('last_success_signal','—')}）"
                elif outcome=="failed":state["schedule_message"]="今日自动运行失败；不会自动重试，可检查原因后手动生成计划"
                elif outcome=="started" and state["running"]:state["schedule_message"]="今日自动任务正在运行"
                elif outcome=="started":state["schedule_message"]="今日自动任务曾启动但未完成；不会自动重试，可手动生成计划"
                else:state["schedule_message"]=f"今日没有新的交易日数据（最近信号日 {schedule.get('last_success_signal','—')}）"
            elif not state["running"]:
                if not cfg.daily_schedule_enabled:
                    state["schedule_message"]="产品内置自动运行已关闭"
                else:
                    from .live_data import LIVE_ROOT
                    ready,message=schedule_ready(LIVE_ROOT/"current",cfg,now)
                    state["schedule_message"]=message
                    if ready:
                        from .pipeline import write_json
                        target=str(scheduled_signal_date(LIVE_ROOT/"current",now).date())
                        if schedule.get("last_success_signal")==target:
                            write_json(done,{
                                "last_attempt_date":today,
                                "target_signal":target,
                                "last_success_signal":target,
                                "last_result":"no_new_session",
                            })
                            state["schedule_message"]=f"没有新的交易日数据（最近信号日 {target}）"
                        elif launch_job("real","daily",target,today):
                            state["schedule_message"]=f"正在处理 {target} 收盘数据并生成下一交易日计划"
                    else:
                        if now>=cutoff(now.date(),cfg.schedule_time):
                            target=scheduled_signal_date(LIVE_ROOT/"current",now)
                            from .pipeline import write_json
                            write_json(done,{
                                "last_attempt_date":today,
                                "target_signal":str(target.date()) if target is not None else None,
                                "last_success_signal":schedule.get("last_success_signal"),
                                "last_result":"failed",
                                "error":message,
                            })
                            state["schedule_message"]=message+"；今天不会自动重试，可稍后手动生成计划"
        except Exception as e:
            state["schedule_message"]="自动运行未完成："+str(e)[:180]
        stop.wait(60)

@app.get("/api/download/{run_id}/{filename}")
def download(run_id:str,filename:str):
    allowed={"orders.csv","plan.json","rankings.csv","equity_curve.csv","executions.csv","daily_ic.csv","manifest.json","report.json","config.json"}
    if filename not in allowed or Path(run_id).name!=run_id:raise HTTPException(400,"非法下载路径")
    path=ROOT/"runs"/run_id/filename
    if not path.exists():raise HTTPException(404,"该运行未生成此文件")
    return FileResponse(path,filename=filename)

@app.get("/api/template/{table}")
def template(table:str):
    if table not in SCHEMAS:raise HTTPException(404,"未知数据表")
    write_templates(ROOT/"data/templates")
    return FileResponse(ROOT/"data/templates"/f"{table}.csv",filename=f"{table}.csv")

@app.get("/api/docs/data-request")
def data_request():return FileResponse(ROOT/"docs/剩余必需数据.md",media_type="text/plain; charset=utf-8")

@app.get("/api/health")
def health():return {"status":"ok","broker_execution":False}
