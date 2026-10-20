import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from ashare.web import app
from ashare.events import extract_one

client=TestClient(app)

def test_api_and_assets():
    assert client.get('/api/health').json()['broker_execution'] is False
    assert client.get('/').status_code==200
    assert client.get('/static/app.js').status_code==200
    assert len(client.get('/api/schema').json())>=10
    assert client.get('/api/config').json()['initial_cash']==1000000
    assert 'live_inventory' in client.get('/api/data?mode=real').json()

def test_api_blocks_external_origin_and_path_traversal():
    assert client.post('/api/run',headers={'origin':'https://evil.example'},json={}).status_code==403
    assert client.get('/api/report',params={'run_id':'../config'}).status_code==400
    assert client.post('/api/import/unknown',content=b'a,b').status_code==400
    assert client.get('/api/download/test/secrets.txt').status_code==400

def test_research_report_is_separate_from_latest_paper_report(tmp_path,monkeypatch):
    import ashare.web as web
    monkeypatch.setattr(web,'ROOT',tmp_path)
    runs=tmp_path/'runs'
    for run_id,result in [
        ('20260922-220000',{'kind':'daily','mode':'forward_paper','curve':[]}),
        ('20260920-210000',{'kind':'research','mode':'real_research','curve':[{'date':'2026-08-31','equity':1.2,'benchmark':1.1}]})
    ]:
        folder=runs/run_id;folder.mkdir(parents=True)
        (folder/'report.json').write_text(json.dumps(result),encoding='utf-8')
    (runs/'latest.json').write_text(json.dumps({'run_id':'20260922-220000'}),encoding='utf-8')
    assert web.report()['kind']=='daily'
    assert web.research_report()['kind']=='research'

def test_cloud_blocks_memory_heavy_research_job(monkeypatch):
    monkeypatch.setenv('ASHARE_INTERNAL_SCHEDULER','0')
    response=client.post('/api/run',json={'kind':'research','mode':'real'})
    assert response.status_code==409
    assert '内存不足' in response.json()['detail']

def test_daily_button_reports_existing_current_plan(tmp_path,monkeypatch):
    import ashare.web as web
    from ashare.config import load_config
    monkeypatch.setattr(web,'ROOT',tmp_path)
    monkeypatch.setattr('ashare.live_data.get_live_status',lambda:{'passed':True,'signal_date':'2026-09-22'})
    folder=tmp_path/'runs/20260922-210000';folder.mkdir(parents=True)
    (folder/'report.json').write_text(json.dumps({'mode':'forward_paper','as_of':'2026-09-22','config':load_config().dict()}),encoding='utf-8')
    (tmp_path/'runs/latest.json').write_text(json.dumps({'run_id':folder.name}),encoding='utf-8')
    response=client.post('/api/run',json={'kind':'daily','mode':'real'})
    assert response.status_code==200
    assert response.json()['accepted'] is False
    assert '无需重复运行' in response.json()['message']

def test_api_missing_real_data_is_explicit():
    response=client.get('/api/data?mode=real')
    assert response.status_code==200
    assert isinstance(response.json()['errors'],list)

def test_real_data_page_uses_audited_report_without_loading_full_panel(tmp_path,monkeypatch):
    import ashare.web as web
    from ashare.schema import SCHEMAS,REQUIRED
    folder=tmp_path/'data/inbox';folder.mkdir(parents=True)
    for name in REQUIRED:
        pd.DataFrame(columns=SCHEMAS[name]).to_csv(folder/f'{name}.csv',index=False)
    report={
        'created_at':'2026-09-21T12:00:00+08:00',
        'passed':True,
        'errors':[],
        'warnings':[],
        'tables':{name:{'rows':1} for name in REQUIRED},
    }
    (folder/'质量报告.json').write_text(json.dumps(report),encoding='utf-8')
    monkeypatch.setattr(web,'ROOT',tmp_path)
    monkeypatch.setattr(web,'load_dataset',lambda *a,**k:(_ for _ in ()).throw(AssertionError('full dataset loaded')))
    monkeypatch.setattr('ashare.live_data.get_live_status',lambda:{})
    result=web.data('real')
    assert result['passed'] is True
    assert result['quality_report']['lightweight'] is True
    assert next(item for item in result['inventory'] if item['name']=='prices')['rows']==1

def test_extractor_requires_real_configuration(monkeypatch):
    monkeypatch.delenv('AI_MODEL',raising=False)
    with pytest.raises(ValueError,match='AI_MODEL'):extract_one({})

def test_extractor_rejects_invented_evidence(monkeypatch):
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'message':{'content':json.dumps({'category':'earnings','sentiment':1,'materiality':.8,'confidence':.9,'novelty':1,'evidence':'这是原文中不存在的所谓证据'})}}
    monkeypatch.setattr('ashare.events.httpx.post',lambda *a,**k:Response())
    d={'text':'公司公告本年度利润增长百分之十。','symbol':'600000.SH','title':'季度报告'}
    with pytest.raises(ValueError,match='证据'):extract_one(d,model='test-model')

def test_extractor_preserves_real_processing_time(monkeypatch):
    body='公司公告本年度利润增长百分之十。'
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'message':{'content':json.dumps({'category':'earnings','sentiment':1,'materiality':.8,'confidence':.9,'novelty':1,'evidence':body})}}
    monkeypatch.setattr('ashare.events.httpx.post',lambda *a,**k:Response())
    d={'text':body,'symbol':'600000.SH','title':'季度报告','document_id':'1','source_type':'filing','source_url':'https://example.com','published_at':'2020-01-01T18:00:00+08:00','first_seen_at':'2020-01-01T18:01:00+08:00'}
    event=extract_one(d,model='test-model')
    assert pd.Timestamp(event['processed_at']).year>=2026
    assert event['evidence']==body and len(event['text_sha256'])==64


def test_extractor_rejects_document_first_seen_after_decision(monkeypatch):
    d={'text':'公司公告本年度利润增长百分之十。','symbol':'600000.SH','title':'季度报告','published_at':'2026-09-10T18:00:00+08:00','first_seen_at':'2026-09-11T08:00:00+08:00'}
    with pytest.raises(ValueError,match='决策时点之后'):
        extract_one(d,model='test-model',decision_cutoff='2026-09-10T20:00:00+08:00')

def test_scheduler_targets_previous_completed_session(tmp_path):
    from ashare.web import schedule_ready,scheduled_signal_date
    from ashare.config import Config
    now=pd.Timestamp('2026-09-11T00:01:00+08:00')
    ready,_=schedule_ready(tmp_path,Config(),now)
    assert not ready
    pd.DataFrame({'date':['2026-09-10','2026-09-11','2026-09-14']}).to_csv(tmp_path/'calendar.csv',index=False)
    assert schedule_ready(tmp_path,Config(),now)[0]
    assert scheduled_signal_date(tmp_path,now)==pd.Timestamp('2026-09-10')
    assert scheduled_signal_date(tmp_path,pd.Timestamp('2026-09-13T00:01:00+08:00'))==pd.Timestamp('2026-09-11')
    later=Config(schedule_time='00:30:00')
    assert not schedule_ready(tmp_path,later,pd.Timestamp('2026-09-11T00:29:00+08:00'))[0]

def test_symbolic_factor_sandbox():
    from ashare.mining import evaluate_formula
    f=pd.DataFrame({'x_roe':[.1,.2],'x_earnings_yield':[.3,-.1]})
    assert evaluate_formula('x_roe + x_earnings_yield',f).tolist()==pytest.approx([.4,.1])
    for expression in ["__import__('os').system('echo unsafe')","x_roe.shift(-1)","open('/tmp/secret')","x_roe[0]","forward_return","x_roe ** 99"]:
        with pytest.raises(ValueError):evaluate_formula(expression,f)

def test_config_rejects_nonfinite():
    from ashare.config import Config
    with pytest.raises(ValueError):Config(initial_cash=float('nan'))
