from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from ashare.config import Config
from ashare.models import production_scores,PRIOR_POSITIVE,PRIOR_NEGATIVE,training_slice,xgboost_columns,fit_xgboost

def test_reconstructed_signals_use_thirty_percent_and_no_silent_xgb_fallback():
    f=pd.DataFrame({'date':pd.to_datetime(['2026-09-15']*3),'worldquant_score':[.1,.5,.9],'event_sentiment':[0.,0.,0.],'event_risk':[0.,0.,0.],'event_disagreement':[0.,0.,0.],'event_coverage':[0.,0.,0.],'reconstructed_text_score':[.2,.5,.9],'structured_prediction':[.8,.1,.5]})
    for c in PRIOR_POSITIVE+PRIOR_NEGATIVE:f['x_'+c]=[.1,.2,.3]
    cfg=replace(Config(),structured_model='xgboost',text_alpha_weight=.3)
    s=production_scores(f,cfg)
    assert np.allclose(s.combined_score,.7*s.structured_score+.3*f.reconstructed_text_score)
    with pytest.raises(ValueanáError if False else ValueError):production_scores(f.drop(columns='structured_prediction'),cfg)

def test_xgboost_trains_only_matured_historical_labels():
    cfg=replace(Config(),min_train_days=40,embargo_days=5,train_window_days=60,structured_model='xgboost',factor_zoo_enabled=False)
    dates=pd.bdate_range('2025-01-01',periods=100)
    f=pd.MultiIndex.from_product([dates,['000001.SZ','600000.SH']],names=['date','symbol']).to_frame(index=False)
    f['label_end']=f.date+pd.Timedelta(days=9);f['label_available_at']=pd.to_datetime(f.label_end).dt.tz_localize('Asia/Shanghai')+pd.Timedelta(hours=18)
    f['eligible']=True;f['target']=np.tile([.01,-.01],len(dates));f['relevance']=np.tile([4,0],len(dates))
    from ashare.features import PRICE_FEATURES,FUND_FEATURES
    for c in PRICE_FEATURES+FUND_FEATURES:f['x_'+c]=np.tile([.1,-.1],len(dates))
    from ashare.worldquant101 import USABLE
    wq=pd.DataFrame({f'x_wq{n:03d}':np.tile([.1,-.1],len(dates)) for n in USABLE})
    f=pd.concat([f,wq],axis=1)
    signal=dates[85];train=training_slice(f,signal,cfg)
    assert train.label_end.max()<signal and train.label_available_at.max()<signal.tz_localize('Asia/Shanghai')
    future=f.copy();future.loc[future.date>=signal,'target']=1e20
    pd.testing.assert_frame_equal(train,training_slice(future,signal,cfg))
    assert len(xgboost_columns(train))==106
    model=fit_xgboost(train,cfg);assert np.isfinite(model.predict(train[xgboost_columns(train)])).all()

def test_exact_statement_ttm_and_per_ten_share_actions(tmp_path,monkeypatch):
    from ashare import data_repairs as repairs
    def fake(name,query,cache):
        if name=='RPT_SHAREBONUS_DET':return [{'SECUCODE':'000001.SZ','ASSIGN_PROGRESS':'实施分配','EX_DIVIDEND_DATE':'2026-11-02','NOTICE_DATE':'2026-10-25','PRETAX_BONUS_RMB':2.,'BONUS_IT_RATIO':1.}]
        period=query.split("REPORT_DATE='")[1].split("'")[0]
        value={'2025-12-31':100.,'2025-09-30':70.,'2026-09-30':90.}[period]
        return [{'SECUCODE':'000001.SZ','NOTICE_DATE':'2026-10-25','UPDATE_DATE':'2026-10-26','TOTAL_OPERATE_INCOME':value,'PARENT_NETPROFIT':value/10,'NETCASH_OPERATE':value/5,'TOTAL_PARENT_EQUITY':200.,'TOTAL_ASSETS':1000.,'TOTAL_LIABILITIES':800.}]
    monkeypatch.setattr(repairs,'fetch_report',fake)
    repairs.collect_repairs(tmp_path,tmp_path,symbols=['000001.SZ'])
    f=pd.read_parquet(tmp_path/'fundamentals_repaired.parquet').iloc[0]
    a=pd.read_parquet(tmp_path/'actions_repaired.parquet').iloc[0]
    assert f.revenue_ttm==120. and f.net_profit_ttm==12. and f.operating_cashflow_ttm==24.
    assert str(f.available_at.date())=='2026-10-26'
    assert a.cash_per_share==.2 and a.share_multiplier==1.1

def test_reconstruction_keeps_real_clocks_and_normalizes_optional_fields(tmp_path,monkeypatch):
    from ashare.reconstruct_signals import reconstruct
    import os
    folder=tmp_path/'market';folder.mkdir();output=tmp_path/'texts';output.mkdir()
    pd.DataFrame({'date':pd.to_datetime(['2026-09-15','2026-09-16'])}).to_parquet(folder/'calendar.parquet',index=False)
    pd.DataFrame({'date':pd.to_datetime(['2026-09-15','2026-09-16']),'symbol':['000001.SZ']*2,'weight':[1.,1.],'known_at':['2026-09-01T00:00:00+08:00']*2}).to_parquet(folder/'universe.parquet',index=False)
    now=pd.Timestamp.now(tz='Asia/Shanghai').isoformat();body='公司签订了一项新的业务合同。'
    doc={'document_id':'test','symbol':'000001.SZ','source_type':'filing','source_url':'https://example.com/a','published_at':'2026-09-16T14:00:00+08:00','first_seen_at':now,'title':'业务合同','text':body,'text_scope':'source_excerpt'}
    pd.DataFrame([doc]).to_parquet(output/'selected_documents.parquet',index=False)
    for k,v in {'AI_MODEL':'test','AI_BASE_URL':'https://example.invalid/v1','AI_API_KEY':'not-a-real-key'}.items():monkeypatch.setenv(k,v)
    class Response:
        status_code=200
        def raise_for_status(self):pass
        def json(self):
            import json
            return {'choices':[{'message':{'content':json.dumps({'events':[{'document_id':'test','category':'contract','sentiment':.5,'materiality':.8,'confidence':.9,'novelty':.9,'evidence':body,'priced_in':{'unknown':True},'surprise':'unknown','horizon_days':3,'risk_flags':'待核验'}]})}}]}
    monkeypatch.setattr('ashare.reconstruct_signals.httpx.post',lambda *a,**k:Response())
    result=reconstruct(output,folder,'2026-09-15','2026-09-16',workers=1)
    signals=pd.read_parquet(output/'daily_signals.parquet');events=pd.read_parquet(output/'events.parquet')
    assert signals.iloc[0].reconstructed_text_score==.5
    assert signals.iloc[1].reconstructed_text_score>.5
    assert events.iloc[0].first_seen_at==now
    assert pd.Timestamp(events.iloc[0].processed_at)>=pd.Timestamp(now)
    assert result['events']==1 and result['text_weight']==.3
    assert pd.isna(events.iloc[0].priced_in)

def test_report_serializes_parquet_array_fields(tmp_path):
    from ashare.pipeline import write_json
    import json
    p=tmp_path/'report.json';write_json(p,{'risk_flags':np.array(['待核验','波动风险']),'scores':np.array([.2,np.nan])})
    r=json.loads(p.read_text());assert r['risk_flags']==['待核验','波动风险'] and r['scores']==[.2,None]
