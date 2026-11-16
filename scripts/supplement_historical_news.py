"""Best-effort dated news archive; records gaps instead of inventing forum history."""
import hashlib,json,re
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import requests,pandas as pd
from ashare.data import read_table
p=Path('data/reconstruction/20260915-20261116');raw=p/'news_raw';raw.mkdir(exist_ok=True)
u=read_table('data/live/current','universe');symbols=sorted(u.loc[u.date>=pd.Timestamp('2026-09-15'),'symbol'].unique());now=pd.Timestamp.now(tz='Asia/Shanghai').isoformat()
def one(symbol):
 path=raw/f'{symbol}.json'
 try:
  if path.exists():items=json.loads(path.read_text())
  else:
   inner={'uid':'','keyword':symbol[:6],'type':['cmsArticleWebOld'],'client':'web','clientType':'web','clientVersion':'curr','param':{'cmsArticleWebOld':{'searchScope':'default','sort':'time','pageIndex':1,'pageSize':100,'preTag':'','postTag':''}}}
   s=requests.Session();s.trust_env=False;r=s.get('https://search-api-web.eastmoney.com/search/jsonp',params={'cb':'research_callback','param':json.dumps(inner)},headers={'User-Agent':'Mozilla/5.0','Referer':'https://so.eastmoney.com/'},timeout=(4,12));r.raise_for_status();text=r.text;start=text.find('{');end=text.rfind('}');j=json.loads(text[start:end+1]);items=j.get('result',{}).get('cmsArticleWebOld',[]);path.write_text(json.dumps(items,ensure_ascii=False))
  rows=[]
  for item in items:
   t=pd.to_datetime(item.get('date'),errors='coerce')
   if pd.isna(t):continue
   t=t.tz_localize('Asia/Shanghai') if t.tzinfo is None else t
   if not pd.Timestamp('2026-09-15',tz='Asia/Shanghai')<=t<=pd.Timestamp('2026-11-16 23:59:59',tz='Asia/Shanghai'):continue
   title=re.sub('<[^>]+>','',item.get('title',''));body=re.sub('<[^>]+>','',item.get('content',''));url='https://finance.eastmoney.com/a/'+str(item['code'])+'.html'
   rows.append({'document_id':hashlib.sha256((symbol+url).encode()).hexdigest()[:24],'symbol':symbol,'source_type':'news','source_url':url,'published_at':t.isoformat(),'first_seen_at':now,'title':title,'text':title+'\n'+body,'text_scope':'source_excerpt','retrieved_at':now})
  return rows,None
 except Exception as e:return [],{'symbol':symbol,'error':type(e).__name__}
rows=[];errors=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 fs=[pool.submit(one,s) for s in symbols]
 for i,f in enumerate(as_completed(fs)):
  r,e=f.result();rows+=r
  if e:errors.append(e)
  if (i+1)%50==0:print('历史新闻',i+1,'/',len(symbols),'取得',len(rows),'条',flush=True)
d=pd.DataFrame(rows);d.to_parquet(p/'news_documents.parquet',index=False)
if len(d):
 d['day']=pd.to_datetime(d.published_at,utc=True).dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d')
 selected=pd.read_parquet(p/'selected_documents.parquet');selected=pd.concat([selected,d.groupby('day').head(8)],ignore_index=True).drop_duplicates('document_id');selected.to_parquet(p/'selected_documents.parquet',index=False)
(p/'news_coverage.json').write_text(json.dumps({'documents':len(rows),'errors':errors,'limitation':'每只股票最多100条检索摘要，非完整新闻档案；历史论坛全文未取得，不使用当下热度替代历史。'},ensure_ascii=False,indent=2));print('NEWS_DONE',len(rows),flush=True)
