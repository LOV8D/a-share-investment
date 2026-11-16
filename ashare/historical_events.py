"""Auditable historical text reconstruction. Never changes first-seen timestamps."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time
import pandas as pd
import requests
from .config import ROOT
from .data import read_table
from .schema import SCHEMAS

HEADERS={'User-Agent':'AShareResearch/0.3 (personal research)'}
LIST_URL='https://np-anotice-stock.eastmoney.com/api/security/ann'
CONTENT_URL='https://np-cnotice-stock.eastmoney.com/api/content/ann'

def get_json(url, params):
    session=requests.Session();session.trust_env=False
    response=session.get(url,params=params,headers=HEADERS,timeout=(5,15))
    response.raise_for_status()
    return response.json()

def publication_time(item):
    # Date-only disclosures are conservatively usable at the end of that day.
    day=pd.Timestamp(item['notice_date']).normalize()
    value=str(item.get('display_time',''))
    value=re.sub(r':\d{3}$','',value)
    stamp=pd.to_datetime(value,errors='coerce')
    if pd.isna(stamp) or stamp.normalize()!=day:stamp=day+pd.Timedelta(hours=23,minutes=59,seconds=59)
    return stamp.tz_localize('Asia/Shanghai')

def collect_archive(folder,start,end,output,workers=4,progress=print):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    raw=output/'raw';raw.mkdir(exist_ok=True)
    universe=read_table(folder,'universe')
    universe=universe[universe.date.between(pd.Timestamp(start),pd.Timestamp(end))]
    symbols=sorted(universe.symbol.unique()); fetched=pd.Timestamp.now(tz='Asia/Shanghai').isoformat()
    def one(symbol):
        path=raw/f'{symbol}.json'
        if path.exists():return symbol,json.loads(path.read_text()),None
        records=[]
        try:
            params={'stock_list':symbol[:6],'begin_time':start,'end_time':end,'sr':'-1','page_size':100,'page_index':1,'ann_type':'A','client_source':'web','f_node':0,'s_node':0}
            j=get_json(LIST_URL,params); total=int(j['data']['total_hits'])
            records.extend(j['data']['list'])
            for page in range(2,(total+99)//100+1):
                records.extend(get_json(LIST_URL,params|{'page_index':page})['data']['list'])
            path.write_text(json.dumps(records,ensure_ascii=False))
            return symbol,records,None
        except Exception as e:return symbol,records,type(e).__name__
    docs=[];errors=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,s) for s in symbols]
        for i,f in enumerate(as_completed(futures)):
            symbol,items,error=f.result()
            if error:errors.append({'symbol':symbol,'error':error})
            for item in items:
                if not any(x.get('stock_code')==symbol[:6] for x in item.get('codes',[])):continue
                published=publication_time(item)
                if not pd.Timestamp(start)<=published.tz_localize(None).normalize()<=pd.Timestamp(end):continue
                title=item['title'];url=f"https://data.eastmoney.com/notices/detail/{symbol[:6]}/{item['art_code']}.html"
                docs.append(dict(document_id=hashlib.sha256((symbol+url).encode()).hexdigest()[:24],symbol=symbol,source_type='filing',source_url=url,published_at=published.isoformat(),first_seen_at=fetched,title=title,text=title,art_code=item['art_code'],text_scope='title_only',retrieved_at=fetched))
            if (i+1)%25==0:progress(f'历史公告 {i+1}/{len(symbols)}，已取得 {len(docs)} 篇',flush=True)
    frame=pd.DataFrame(docs).drop_duplicates('document_id') if docs else pd.DataFrame(columns=SCHEMAS['documents'])
    frame.to_parquet(output/'documents.parquet',index=False)
    coverage={'start':start,'end':end,'requested_symbols':len(symbols),'successful_symbols':len(symbols)-len(errors),'documents':len(frame),'errors':errors,'retrieved_at':fetched,'scope':'historical_reconstruction','news_forum_status':'历史新闻与论坛尚未证实完整覆盖，不将缺失视为无事件'}
    (output/'coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2))
    return frame

def enrich_archive(output,limit_per_day=20,workers=4,progress=print):
    output=Path(output);frame=pd.read_parquet(output/'documents.parquet')
    if frame.empty:return frame
    frame['day']=pd.to_datetime(frame.published_at,utc=True).dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d')
    material=r'业绩|利润|回购|增持|减持|合同|中标|诉讼|处罚|重组|分红|权益分派|风险|收购|预测|预告'
    frame['priority']=frame.title.str.contains(material).astype(int)
    selected=frame.sort_values(['priority','published_at','document_id'],ascending=[False,False,True]).groupby('day',sort=True).head(limit_per_day).copy()
    cache=output/'content';cache.mkdir(exist_ok=True)
    def one(row):
        p=cache/f"{row['art_code']}.json"
        try:
            if p.exists():j=json.loads(p.read_text())
            else:
                j=get_json(CONTENT_URL,{'art_code':row['art_code'],'client_source':'web','page_index':1});p.write_text(json.dumps(j,ensure_ascii=False))
            data=j.get('data') or {};body=data.get('notice_content') or data.get('content') or ''
            if len(body)>len(row['text']):row['text']=re.sub('<[^>]+>',' ',body);row['text_scope']='fulltext'
            row['text']=row['text'][:22000]
        except Exception as e:row['content_error']=type(e).__name__
        return row
    rows=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,row in enumerate(pool.map(one,selected.to_dict('records'))):
            rows.append(row)
            if (i+1)%50==0:progress(f'公告正文 {i+1}/{len(selected)}',flush=True)
    selected=pd.DataFrame(rows);selected.to_parquet(output/'selected_documents.parquet',index=False)
    return selected

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--start',default='2026-09-15');p.add_argument('--end',required=True);p.add_argument('--data',default=str(ROOT/'data/live/current'));p.add_argument('--output',required=True);p.add_argument('--per-day',type=int,default=20)
    a=p.parse_args();collect_archive(a.data,a.start,a.end,a.output);enrich_archive(a.output,a.per_day)
