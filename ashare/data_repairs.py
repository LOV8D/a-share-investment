"""Exact-statement fallback and independently sourced corporate actions."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
import requests
from .schema import SCHEMAS
from .data import read_table,load_dataset,validate
from .config import ROOT
from .live_data import merge_table,_save_frame,_write_json,_limit_price

URL='https://datacenter-web.eastmoney.com/api/data/v1/get'

def fetch_report(name,filter_value,cache):
    cache=Path(cache)
    if cache.exists():return json.loads(cache.read_text())
    s=requests.Session();s.trust_env=False;rows=[];page=1
    while True:
        r=s.get(URL,params={'reportName':name,'columns':'ALL','filter':filter_value,'pageSize':500,'pageNumber':page,'source':'WEB','client':'WEB'},timeout=(5,15));r.raise_for_status();j=r.json()
        if not j.get('success'):raise ValueError(str(j.get('message','source returned no data')))
        result=j.get('result') or {};rows+=result.get('data',[])
        if page>=int(result.get('pages',1)):break
        page+=1
    cache.parent.mkdir(parents=True,exist_ok=True);cache.write_text(json.dumps(rows,ensure_ascii=False));return rows

def collect_repairs(folder,output,symbols=None,report_period="2026-09-30"):
    output=Path(output);raw=output/'repair_sources';raw.mkdir(parents=True,exist_ok=True)
    if symbols is None:
        universe=read_table(folder,'universe'); symbols=sorted(universe.loc[universe.date>=pd.Timestamp('2026-09-01'),'symbol'].unique())
    symbols=sorted(symbols)
    chunks=[symbols[i:i+30] for i in range(0,len(symbols),30)]
    period=pd.Timestamp(report_period);current=str(period.date());prior_year=str((period-pd.DateOffset(years=1)).date());annual=f"{period.year-1}-12-31"
    periods=sorted({current,prior_year,annual})
    jobs=[]
    for i,chunk in enumerate(chunks):
        codes=','.join('"'+s[:6]+'"' for s in chunk);f=f'(SECURITY_CODE in ({codes}))'
        for report_day in periods:
            for kind in ['BALANCE','INCOME','CASHFLOW']:
                jobs.append((kind,report_day,i,f+f"(REPORT_DATE='{report_day}')"))
        jobs.append(('ACTIONS','',i,f+f"(EX_DIVIDEND_DATE>='{period.year}-01-01')(EX_DIVIDEND_DATE<='{pd.Timestamp.now(tz='Asia/Shanghai').date()}')"))
    gathered={};errors=[]
    def one(job):
        k,p,i,f=job;name='RPT_SHAREBONUS_DET' if k=='ACTIONS' else 'RPT_DMSK_FN_'+k
        return job,fetch_report(name,f,raw/f'{k}-{p}-{i}.json')
    with ThreadPoolExecutor(max_workers=6) as pool:
        fs=[pool.submit(one,j) for j in jobs]
        for i,ft in enumerate(as_completed(fs)):
            try:
                (k,p,_,_),rows=ft.result();gathered.setdefault((k,p),[]).extend(rows)
            except Exception as e:errors.append(type(e).__name__)
            if (i+1)%20==0:print('备用财务/公司行动',i+1,'/',len(jobs),flush=True)
    keyed={key:{r['SECUCODE']:r for r in rows} for key,rows in gathered.items() if key[0]!='ACTIONS'}
    def row(kind,period,symbol):return keyed.get((kind,period),{}).get(symbol,{})
    def num(r,*keys):
        for key in keys:
            try:
                if r.get(key) is not None:return float(r[key])
            except (ValueError,TypeError):pass
        return np.nan
    frows=[]
    for symbol in symbols:
        b=row('BALANCE',current,symbol);inc=row('INCOME',current,symbol);c=row('CASHFLOW',current,symbol)
        if not (b and inc and c):continue
        def ttm(kind,*keys):
            if period.month==12:return num(row(kind,current,symbol),*keys)
            return num(row(kind,annual,symbol),*keys)+num(row(kind,current,symbol),*keys)-num(row(kind,prior_year,symbol),*keys)
        values={'revenue_ttm':ttm('INCOME','TOTAL_OPERATE_INCOME','OPERATE_INCOME'),'net_profit_ttm':ttm('INCOME','PARENT_NETPROFIT'),'operating_cashflow_ttm':ttm('CASHFLOW','NETCASH_OPERATE'),'book_equity':num(b,'TOTAL_PARENT_EQUITY','TOTAL_EQUITY'),'total_assets':num(b,'TOTAL_ASSETS'),'total_liabilities':num(b,'TOTAL_LIABILITIES')}
        if not all(np.isfinite(list(values.values()))) or min(values['book_equity'],values['total_assets'])<=0:continue
        used_rows=[b,inc,c]+[row(kind,p,symbol) for kind in ['INCOME','CASHFLOW'] for p in [annual,prior_year]]
        timestamps=[pd.Timestamp(r[k]) for r in used_rows for k in ['NOTICE_DATE','UPDATE_DATE'] if r.get(k)]
        if not timestamps:continue
        available=max(timestamps).normalize()+pd.Timedelta(hours=23,minutes=59,seconds=59)
        frows.append({'symbol':symbol,'report_period':period,'available_at':available.tz_localize('Asia/Shanghai'),**values,'roe':np.nan,'gross_margin':np.nan,'revenue_growth':num(inc,'TOTAL_OPERATE_INCOME_YOY')/100,'profit_growth':num(inc,'PARENT_NETPROFIT_YOY')/100})
    actions=[]
    for a in gathered.get(('ACTIONS',''),[]):
        if a.get('ASSIGN_PROGRESS')!='实施分配' or not a.get('EX_DIVIDEND_DATE'):continue
        notice=a.get('NOTICE_DATE') or a.get('PLAN_NOTICE_DATE')
        if not notice:continue
        cash=num(a,'PRETAX_BONUS_RMB');ratio=num(a,'BONUS_IT_RATIO')
        cash=0. if not np.isfinite(cash) else cash/10
        ratio=0. if not np.isfinite(ratio) else ratio/10
        actions.append({'symbol':a['SECUCODE'],'date':pd.Timestamp(a['EX_DIVIDEND_DATE']).normalize(),'share_multiplier':1+ratio,'cash_per_share':cash,'known_at':pd.Timestamp(notice).normalize().tz_localize('Asia/Shanghai')+pd.Timedelta(hours=23,minutes=59,seconds=59)})
    pd.DataFrame(frows,columns=SCHEMAS['fundamentals']).to_parquet(output/'fundamentals_repaired.parquet',index=False)
    pd.DataFrame(actions,columns=SCHEMAS['actions']).drop_duplicates(['date','symbol']).to_parquet(output/'actions_repaired.parquet',index=False)
    summary={'fundamentals':len(frows),'actions':len(actions),'errors':errors,'source':URL,'retrieved_at':pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),'policy':'财务使用资产负债表真实金额，TTM=去年全年+当期累计-去年同期；可用日不早于披露/更新日；分红送转单位每10股转换为每股。'}
    _write_json(output/'repairs_manifest.json',summary);print(summary,flush=True);return summary


def apply_repairs(folder,output):
    """Build a separate research dataset; preserve the live account and old reports."""
    folder=Path(folder);output=Path(output);dest=output/'dataset';dest.mkdir(exist_ok=True)
    ds=load_dataset(folder);tables=ds.tables
    tables['fundamentals']=merge_table(tables['fundamentals'],pd.read_parquet(output/'fundamentals_repaired.parquet'),'fundamentals')
    tables['actions']=merge_table(tables['actions'],pd.read_parquet(output/'actions_repaired.parquet'),'actions')
    sources=[]
    for p in sorted((ROOT/'data/live/raw').glob('*/baostock_prices.parquet')):
        f=pd.read_parquet(p);f['date']=pd.to_datetime(f.date);f['symbol']=f.code.str[3:]+'.'+f.code.str[:2].str.upper();sources.append(f)
    raw=pd.concat(sources).drop_duplicates(['date','symbol'],keep='last').set_index(['date','symbol'])
    market=tables['prices'].sort_values(['symbol','date']).copy();confirmed=0;unresolved=[]
    quarantine={(x['symbol'],pd.Timestamp(x['date'])) for x in ds.manifest.get('corporate_action_quarantine',[])}
    # Restore true raw marks only where the cash/share adjustment explains the
    # vendor's ex-date reference price to within two price ticks.
    for symbol,idx in market.groupby('symbol',sort=False).groups.items():
        indices=list(idx);prior=None;blocked=False
        acts=tables['actions'][tables['actions'].symbol==symbol].set_index('date')
        for index in indices:
            r=market.loc[index];day=r.date
            if day<=pd.Timestamp('2026-08-31'):prior=r;continue
            if prior is None or (day,symbol) not in raw.index:prior=r;continue
            src=raw.loc[(day,symbol)]
            factor=float(prior.adj_factor);shares=float(prior.market_cap)/float(prior.close)
            if day in acts.index:
                a=acts.loc[day]
                reference=(float(prior.close)-float(a.cash_per_share))/float(a.share_multiplier)
                observed=float(src.preclose)
                if abs(reference-observed)<=.021:
                    factor*=float(prior.close)/observed;shares*=float(a.share_multiplier);blocked=False;confirmed+=1
                else:blocked=True
            if (symbol,day) in quarantine and day not in acts.index and not blocked:
                observed=float(src.preclose)
                if abs(observed-float(prior.close))>.021:blocked=True
            if blocked:
                unresolved.append({'symbol':symbol,'date':str(day.date()),'reason':'独立公司行动与除权参考价尚未核对，保留冻结'})
                prior=r;continue
            active=str(src.tradestatus)=='1'
            if active:
                for k in ['open','high','low','close','volume','amount']:market.at[index,k]=float(src[k])
                close=float(src.close);market.at[index,'prev_close']=float(src.preclose)
                market.at[index,'suspended']=0;market.at[index,'adj_factor']=factor
                market.at[index,'market_cap']=close*shares
                turn=float(src.turn)/100
                market.at[index,'turnover_rate']=turn
                if turn>0:market.at[index,'float_market_cap']=close*float(src.volume)/turn
                for direction,col in [(1,'upper_limit'),(-1,'lower_limit')]:market.at[index,col]=_limit_price(float(src.preclose),direction,int(r.is_st),symbol)
            prior=market.loc[index]
    tables['prices']=market.sort_values(['date','symbol']).reset_index(drop=True)
    manifest=ds.manifest|{'source':'repaired research copy','repair_manifest':json.loads((output/'repairs_manifest.json').read_text()),'corporate_action_quarantine':unresolved,'actions_independently_confirmed':confirmed}
    for name,frame in tables.items():_save_frame(dest/f'{name}.parquet',frame)
    _write_json(dest/'manifest.json',manifest)
    tables['prices']=market
    quality=validate(tables);_write_json(dest/'quality.json',quality)
    if quality['errors']:raise ValueError(str(quality['errors'][:8]))
    for name,frame in tables.items():_save_frame(dest/f'{name}.parquet',frame)
    _write_json(dest/'manifest.json',ds.manifest|{'corporate_action_quarantine':unresolved,'repair_source':str(output/'repairs_manifest.json'),'mode':'research_reconstruction'})
    print('修复集完成',confirmed,'个公司行动参考价核对；仍隔离',len({q['symbol'] for q in unresolved}),flush=True)
    return dest
