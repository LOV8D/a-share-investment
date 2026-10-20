import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .schema import SCHEMAS

def generate_demo(folder, days=560, n=300, seed=42):
    """Synthetic fixture, NOT historical A-share observations or real ticker recommendations."""
    out=Path(folder);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    all_dates=pd.bdate_range(end="2026-09-24",periods=days+11)
    dates=all_dates[:days]
    sectors=["金融","工业","信息技术","医药","消费","能源","材料","公用事业","通信","可选消费"]
    symbols=[f"{990001+i:06d}.{'SH' if i%2==0 else 'SZ'}" for i in range(n)]
    factor=rng.normal(0,.001,(days,n))
    market=rng.normal(.00015,.007,days)
    prices=[];fund=[];events=[];docs=[];universes=[]
    for j,s in enumerate(symbols):
        prev=float(rng.uniform(4,130))
        latent=rng.normal(0,.2)
        for i,d in enumerate(dates):
            gap=float(np.clip(rng.normal(0,.003),-.08,.08))
            opening=round(prev*(1+gap),2)
            ret=market[i]+factor[i,j]+rng.normal(0,.009)+latent*.00005
            close=round(opening*(1+np.clip(ret,-.09,.09)),2)
            limitup=round(prev*1.1,2);limitdown=round(prev*.9,2)
            close=min(limitup,max(limitdown,close))
            high=min(limitup,round(max(opening,close)*(1+rng.uniform(.001,.012)),2))
            low=max(limitdown,round(min(opening,close)*(1-rng.uniform(.001,.012)),2))
            volume=int(rng.uniform(1e6,15e6)); amount=volume*(opening+close)/2
            shares=1e9*(1+j%10)
            prices.append([d,s,opening,high,low,close,prev,volume,amount,1.,limitup,limitdown,0,0,close*shares,close*shares*.65,volume/(shares*.65),sectors[j%10],200 if j%20==0 else 100,1 if j%20==0 else 100,1000+i,f"{d.date()} 15:30:00+08:00"])
            universes.append([d,s,1/n,f"{d.date()} 08:00:00+08:00"])
            if rng.random()<.012:
                polarity=float(rng.choice([-1,1]))
                title="演示事件：季度经营数据更新" if polarity>0 else "演示事件：需求波动风险"
                body="以下为合成测试文本。公司披露季度经营数据，订单与现金流发生变化，仅用于测试事件因子处理。"
                document_id=f"demo-{j}-{i}"
                published=f"{d.date()} 17:00:00+08:00"
                seen=f"{d.date()} 17:01:00+08:00"
                processed=f"{d.date()} 17:02:00+08:00"
                sha=hashlib.sha256((body+document_id).encode()).hexdigest()
                docs.append([document_id,s,"filing","https://example.invalid/synthetic",published,seen,title,body])
                events.append([document_id,document_id,s,"earnings",polarity,.7,.8,1.,"filing","https://example.invalid/synthetic",published,seen,processed,title,"公司披露季度经营数据，订单与现金流发生变化","synthetic-fixture","fixture-v1",sha])
            prev=close
        for q in pd.date_range(dates[0]-pd.Timedelta(days=200),dates[-1],freq="QE"):
            available=q+pd.Timedelta(days=35)
            assets=shares*18; equity=assets*.5
            profit=equity*(.08+latent*.03+rng.normal(0,.008))
            fund.append([s,q,f"{available.date()} 18:00:00+08:00",assets*.4,profit,equity,profit*rng.uniform(.5,1.4),assets,assets-equity,profit/equity,.3+latent*.1,rng.normal(.08,.1),rng.normal(.06,.15)])
    frames={"prices":pd.DataFrame(prices,columns=SCHEMAS["prices"]),"universe":pd.DataFrame(universes,columns=SCHEMAS["universe"]),"fundamentals":pd.DataFrame(fund,columns=SCHEMAS["fundamentals"]),"events":pd.DataFrame(events,columns=SCHEMAS["events"]),"documents":pd.DataFrame(docs,columns=SCHEMAS["documents"]),"calendar":pd.DataFrame({"date":all_dates}),"benchmark":pd.DataFrame({"date":dates,"close":4000*np.cumprod(1+market),"total_return_close":4000*np.cumprod(1+market+.00008)}),"actions":pd.DataFrame(columns=SCHEMAS["actions"]),"fees":pd.DataFrame([["2005-01-01","2099-12-31",.0005,.00001]],columns=SCHEMAS["fees"])}
    for name,frame in frames.items():frame.to_parquet(out/f"{name}.parquet",index=False)
    (out/"manifest.json").write_text(json.dumps({"mode":"synthetic","source":"seeded simulation; no real securities","seed":seed,"symbols":n,"sessions":days,"calendar":"synthetic weekdays, NOT official holiday calendar","fees":"illustrative constant, NOT historical fee schedule","text":"synthetic labels, NOT LLM outputs","corporate_actions":"none in fixture"},ensure_ascii=False,indent=2))
    return out
