import json
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from ashare.config import Config
from ashare.data import cutoff,load_dataset,validate
from ashare.demo import generate_demo
from ashare.features import attach_fundamentals,attach_events,build_panel,PRICE_FEATURES,FUND_FEATURES
from ashare.models import training_slice,robust_prior,production_scores
from ashare.worldquant101 import USABLE,UNAVAILABLE
from ashare.portfolio import round_order,trade_fee,make_plan,execute_plan,allocate_weights,auction_precheck,should_rebalance

@pytest.fixture
def cfg():return replace(Config(),initial_cash=100000,top_k=5,max_weight=.2,max_sector_weight=.4,max_daily_turnover=1.,cash_reserve=.03)

@pytest.fixture
def fees():return pd.DataFrame({"start_date":pd.to_datetime(["2010-01-01","2023-08-28"]),"end_date":pd.to_datetime(["2023-08-27","2099-12-31"]),"sell_stamp_rate":[.001,.0005],"transfer_rate":[.00002,.00001]})

@pytest.fixture
def market():
    return pd.DataFrame([{"date":pd.Timestamp("2026-09-10"),"symbol":"600000.SH","open":10.,"close":10.,"upper_limit":11.,"lower_limit":9.,"suspended":0,"industry":"金融","min_order":100,"lot_step":100,"adv20":1e8}])

def order(side="BUY",shares=500):
    return {"symbol":"600000.SH","side":side,"shares":shares,"limit_price":10.3 if side=="BUY" else 9.7,"min_order":100,"lot_step":100,"adv20":1e8,"industry":"金融"}

def plan(orders):return {"orders":orders,"trade_date":"2026-09-10","nav":100000}

def test_lot_sizes_and_residual_liquidation():
    assert round_order(199,200,1)==0
    assert round_order(237,200,1)==237
    assert round_order(199,100,100)==100
    assert round_order(63,100,100,"SELL",63)==63
    assert round_order(63,100,100,"SELL",163)==0
    assert round_order(160,200,1,"SELL",160)==160

def test_fees_are_historical_and_sell_only(cfg,fees):
    assert trade_fee(100000,"SELL","2023-08-25",cfg,fees)-trade_fee(100000,"BUY","2023-08-25",cfg,fees)==100
    assert trade_fee(100000,"SELL","2023-08-28",cfg,fees)-trade_fee(100000,"BUY","2023-08-28",cfg,fees)==50
    assert trade_fee(100,"BUY","2026-09-10",cfg,fees)>5

def test_limit_up_blocks_buys_but_allows_sells(cfg,fees,market):
    market.loc[0,"open"]=11.
    p=plan([order("BUY")]);p["orders"][0]["limit_price"]=12
    h,c,logs=execute_plan(p,market,{},100000,cfg,fees)
    assert not h and c==100000 and logs[0]["fill_status"]=="REJECTED"
    h,c,logs=execute_plan(plan([order("SELL")]),market,{"600000.SH":500},50000,cfg,fees)
    assert not h and logs[0]["filled_shares"]==500

def test_limit_down_blocks_sells(cfg,fees,market):
    market.loc[0,"open"]=9.
    h,c,logs=execute_plan(plan([order("SELL")]),market,{"600000.SH":500},50000,cfg,fees)
    assert h["600000.SH"]==500 and logs[0]["filled_shares"]==0

def test_suspension_and_gap_cancel(cfg,fees,market):
    market.loc[0,"suspended"]=1
    assert execute_plan(plan([order()]),market,{},100000,cfg,fees)[2][0]["filled_shares"]==0
    market.loc[0,"suspended"]=0;market.loc[0,"open"]=10.5
    assert execute_plan(plan([order()]),market,{},100000,cfg,fees)[2][0]["filled_shares"]==0

def test_no_selling_same_day_purchases(cfg,fees,market):
    buy=order();sell=order("SELL")
    h,c,logs=execute_plan(plan([buy,sell]),market,{},100000,cfg,fees)
    assert h["600000.SH"]==500
    assert logs[0]["filled_shares"]==500 and logs[1]["filled_shares"]==0

def test_cash_never_goes_negative_and_partial_fill(cfg,fees,market):
    h,c,logs=execute_plan(plan([order(shares=10000)]),market,{},1500,cfg,fees)
    assert c>=0 and h=={} # 20% cap cannot fit 100-share minimum

def test_execution_rejects_budget_truncated_tiny_trade(cfg,fees,market):
    tiny=order(shares=100);tiny["full_liquidation"]=False
    h,c,logs=execute_plan(plan([tiny]),market,{},100000,cfg,fees)
    assert h=={} and c==100000
    assert logs[0]["fill_reason"]=="实际可成交金额低于最小交易占比"

def test_failed_sale_does_not_fund_buy(cfg,fees,market):
    other=market.copy();other["symbol"]="000001.SZ";other["suspended"]=1
    market=pd.concat([market,other],ignore_index=True)
    sell=order("SELL");sell["symbol"]="000001.SZ"
    h,c,logs=execute_plan(plan([sell,order()]),market,{"000001.SZ":500},0,cfg,fees)
    assert h=={"000001.SZ":500} and c==0
    assert all(l["filled_shares"]==0 for l in logs)

def test_live_sector_cap_is_rechecked(cfg,fees,market):
    second=market.copy();second["symbol"]="000001.SZ"
    market=pd.concat([market,second],ignore_index=True)
    h,c,logs=execute_plan(plan([order()]),market,{"000001.SZ":4500},55000,cfg,fees)
    assert "600000.SH" not in h and logs[0]["filled_shares"]==0

def test_corporate_action_cancels_stale_order(cfg,fees,market):
    result=execute_plan(plan([order()]),market,{},100000,cfg,fees,action_symbols={"600000.SH"})
    assert result[2][0]["filled_shares"]==0

def test_execution_turnover_hard_cap(cfg,fees,market):
    cfg=replace(cfg,max_daily_turnover=.01,max_weight=1,max_sector_weight=1,slippage_bps=0)
    p=plan([order("SELL",1000)]);p["orders"][0]["limit_price"]=9.
    market.loc[0,"open"]=10.5
    _,_,logs=execute_plan(p,market,{"600000.SH":1000},90000,cfg,fees)
    assert sum(l["filled_shares"]*(l["fill_price"] or 0) for l in logs)<=1000

def test_weekly_rebalance_uses_last_exchange_session(cfg):
    weekly=replace(cfg,rebalance_frequency="weekly")
    assert should_rebalance("2026-09-04","2026-09-07",weekly)
    assert not should_rebalance("2026-09-03","2026-09-04",weekly)
    assert should_rebalance("2026-09-03","2026-09-04",replace(cfg,rebalance_frequency="daily"))

def test_worldquant_inventory_is_explicit():
    assert len(USABLE)==83 and len(UNAVAILABLE)==18
    assert set(USABLE)|set(UNAVAILABLE)==set(range(1,102))

def test_no_event_means_no_text_weight(cfg):
    rows=[]
    for symbol,prior,wq in [("A",.4,.9),("B",-.4,.1)]:
        row={"date":pd.Timestamp("2026-09-01"),"symbol":symbol,"worldquant_score":wq,
             "event_sentiment":0.,"event_risk":0.,"event_disagreement":0.,"event_coverage":0.}
        from ashare.models import PRIOR_POSITIVE,PRIOR_NEGATIVE
        row|={f"x_{c}":prior for c in PRIOR_POSITIVE}
        row|={f"x_{c}":-prior for c in PRIOR_NEGATIVE}
        rows.append(row)
    scored=production_scores(pd.DataFrame(rows),replace(cfg,enable_text_alpha=True))
    pd.testing.assert_series_equal(scored.structured_score,scored.combined_score,check_names=False)

def test_robust_prior_rewards_value_quality_and_low_risk():
    from ashare.models import PRIOR_POSITIVE,PRIOR_NEGATIVE
    good={f"x_{c}":.4 for c in PRIOR_POSITIVE}|{f"x_{c}":-.4 for c in PRIOR_NEGATIVE}
    bad={f"x_{c}":-.4 for c in PRIOR_POSITIVE}|{f"x_{c}":.4 for c in PRIOR_NEGATIVE}
    score=robust_prior(pd.DataFrame([good,bad]))
    assert score.iloc[0]>score.iloc[1]

def test_weights_obey_sector_and_name_caps(cfg):
    selected=pd.DataFrame({"symbol":[str(i) for i in range(10)],"industry":["A"]*6+["B"]*4,"volatility_20":[.2]*10})
    w=allocate_weights(selected,cfg)
    assert max(w.values())<=cfg.max_weight+1e-9
    assert sum(w[str(i)] for i in range(6))<=cfg.max_sector_weight+1e-9
    assert sum(w.values())<=1-cfg.cash_reserve+1e-9

def test_financial_disclosure_time_and_old_restatement():
    p=pd.DataFrame({"symbol":["600000.SH"]*3,"cutoff":[cutoff("2026-04-01"),cutoff("2026-05-01"),cutoff("2026-06-01")]})
    f=pd.DataFrame({"symbol":["600000.SH"]*3,"report_period":pd.to_datetime(["2025-12-31","2026-03-31","2025-12-31"]),"available_at":[cutoff("2026-03-30"),cutoff("2026-04-29"),cutoff("2026-05-20")],"net_profit_ttm":[10,20,999]})
    out=attach_fundamentals(p,f)
    assert out.net_profit_ttm.tolist()==[10,20,20]

def test_after_cutoff_financial_not_used():
    p=pd.DataFrame({"symbol":["600000.SH"],"cutoff":[cutoff("2026-04-01")]})
    f=pd.DataFrame({"symbol":["600000.SH"],"report_period":pd.to_datetime(["2025-12-31"]),"available_at":[cutoff("2026-04-01","21:00:00")],"net_profit_ttm":[99]})
    assert pd.isna(attach_fundamentals(p,f).net_profit_ttm.iloc[0])

def test_text_availability_uses_processing_time_and_dedup(cfg):
    p=pd.DataFrame({"symbol":["600000.SH"]*2,"cutoff":[cutoff("2026-09-09"),cutoff("2026-09-10")]})
    e=pd.DataFrame({"symbol":["600000.SH"]*2,"published_at":[cutoff("2020-01-01")]*2,"first_seen_at":[cutoff("2020-01-01")]*2,"processed_at":[cutoff("2026-09-10","18:00:00")]*2,"source_type":["filing"]*2,"confidence":[1.]*2,"materiality":[1.]*2,"novelty":[1.]*2,"sentiment":[1.]*2,"text_sha256":["same"]*2})
    out=attach_events(p,e,cfg)
    assert out.event_coverage.tolist()==[0.,1.]
    assert 0<out.event_intensity.iloc[1]<=1 # duplicate not counted twice

def test_training_purges_future_labels(cfg):
    dates=pd.bdate_range("2025-01-01",periods=100)
    p=pd.DataFrame({"date":dates,"symbol":"600000.SH","label_end":pd.Series(dates).shift(-6),"label_available_at":[cutoff(d) for d in dates],"eligible":True,"target":.1})
    cfg=replace(cfg,min_train_days=40,train_window_days=100,embargo_days=5)
    train=training_slice(p,dates[-1],cfg)
    boundary=dates[-7]
    assert (train.label_end<boundary).all()
    assert (train.date<boundary).all()

def test_plan_uses_signal_close_and_affordability(cfg,fees,market):
    market["date"]=pd.Timestamp("2026-09-09")
    scored=market.assign(rank=1,eligible=True,volatility_20=.2,score=100)
    p=make_plan(scored,{},100000,market,"2026-09-10",cfg,fees)
    assert p["orders"][0]["limit_price"]==10.3
    tiny=make_plan(scored,{},1000,market,"2026-09-10",cfg,fees)
    assert not tiny["orders"] and tiny["minimum_capital_for_topk"]>1000

def test_plan_freezes_quarantined_holding_without_blocking_other_names(cfg,fees,market):
    locked=market.copy();locked["suspended"]=1;locked["adv20"]=0.
    other=market.copy();other["symbol"]="000001.SZ";other["industry"]="消费"
    combined=pd.concat([locked,other],ignore_index=True)
    scored=combined.assign(rank=[2,1],eligible=[False,True],volatility_20=.2,score=[50,100])
    result=make_plan(scored,{"600000.SH":1000},90000,combined,"2026-09-10",cfg,fees)
    assert result["target_shares"]["600000.SH"]==1000
    assert all(order["symbol"]!="600000.SH" for order in result["orders"])
    assert any(row["symbol"]=="600000.SH" and "仓位冻结" in row["reason"] for row in result["excluded"])
    assert any(order["symbol"]=="000001.SZ" for order in result["orders"])

def test_plan_caps_daily_orders_without_buying_below_top_k(cfg,fees):
    symbols=[f"{i:06d}.SZ" for i in range(1,21)]
    market=pd.DataFrame({
        "date":pd.Timestamp("2026-09-09"),"symbol":symbols,"close":10.,
        "industry":[f"行业{i%4}" for i in range(20)],"min_order":100,"lot_step":100,
    })
    scored=market.assign(rank=range(1,21),eligible=True,volatility_20=.2,
                         score=np.linspace(100,1,20),adv20=1e9)
    holdings={symbol:10_000 for symbol in symbols[10:]}
    capped=replace(cfg,initial_cash=1_000_000,top_k=10,max_weight=.1,
                   max_sector_weight=1.,max_daily_turnover=2.,cash_reserve=0.,
                   min_trade_weight=.025,max_daily_orders=15,hold_buffer=0)
    result=make_plan(scored,holdings,0.,market,"2026-09-10",capped,fees)
    assert len(result["orders"])==15
    assert result["deferred_by_order_cap"]==5
    assert all(order["symbol"] in set(symbols[:10]) for order in result["orders"] if order["side"]=="BUY")
    assert all(order["side"]=="SELL" for order in result["orders"][:10])

def test_precheck_rejects_stale_auction():
    a=pd.DataFrame({"date":[pd.Timestamp("2026-09-10")],"symbol":["600000.SH"],"indicative_open":[10.],"upper_limit":[11.],"lower_limit":[9.],"suspended":[0],"available_at":[cutoff("2026-09-10","09:00:00")]})
    assert auction_precheck(plan([order()]),a)[0]["precheck"]=="BLOCKED"

@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    return load_dataset(generate_demo(tmp_path_factory.mktemp("sample"),days=170,n=12))

def test_data_duplicate_and_missing_component_rejected(sample):
    tables={k:v.copy() for k,v in sample.tables.items()}
    tables["prices"]=pd.concat([tables["prices"],tables["prices"].iloc[:1]],ignore_index=True)
    assert any("重复主键" in s for s in validate(tables)["errors"])
    tables={k:v.copy() for k,v in sample.tables.items()}
    tables["prices"]=tables["prices"].iloc[1:]
    assert any("缺少当日日线" in s for s in validate(tables)["errors"])

def test_features_invariant_to_future_fundamentals(sample,cfg):
    cfg=replace(cfg,min_train_days=40)
    p1=build_panel(sample,cfg)
    original=sample.tables["fundamentals"].copy()
    last=sample["prices"].date.max()
    new=original.iloc[:1].copy();new["available_at"]=cutoff(last+pd.Timedelta(days=1));new["net_profit_ttm"]=1e20
    sample.tables["fundamentals"]=pd.concat([original,new],ignore_index=True)
    p2=build_panel(sample,cfg)
    sample.tables["fundamentals"]=original
    pd.testing.assert_frame_equal(p1[["x_"+c for c in PRICE_FEATURES+FUND_FEATURES]],p2[["x_"+c for c in PRICE_FEATURES+FUND_FEATURES]])

def test_label_is_next_open_to_future_open(sample,cfg):
    cfg=replace(cfg,min_train_days=40)
    p=build_panel(sample,cfg)
    r=p.iloc[70]
    prices=sample["prices"][sample["prices"].symbol==r.symbol].sort_values("date").reset_index(drop=True)
    i=prices.index[prices.date==r.date][0]
    expected=prices.open.iloc[i+6]/prices.open.iloc[i+1]-1
    assert r.forward_return==pytest.approx(expected)

def test_end_to_end_real_algorithms(sample,cfg,tmp_path):
    from ashare.pipeline import run_engine
    cfg=replace(cfg,min_train_days=40,refit_every=25,train_window_days=120,top_k=5)
    result=run_engine(sample.folder,cfg,output_root=tmp_path)
    assert result["mode"]=="synthetic"
    assert result["metrics"]["sessions"]>20
    assert len(result["rankings"])==12
    assert result["model_card"]["live_ready"] is False
    for fold in result["folds"]:
        assert fold["latest_label_end"]<fold["signal_start"]
    assert (tmp_path/result["run_id"]/"executions.csv").exists()
    assert json.loads((tmp_path/"latest.json").read_text())["run_id"]==result["run_id"]

def test_config_rejects_impossible_inputs():
    with pytest.raises(ValueError):Config(initial_cash=-1)
    with pytest.raises(ValueError):Config(max_sector_weight=.01,max_weight=.1)

def test_daily_plan_without_replaying_backtest(sample,cfg,tmp_path):
    from ashare.pipeline import run_engine
    cfg=replace(cfg,min_train_days=40,train_window_days=120)
    report=run_engine(sample.folder,cfg,kind="daily",output_root=tmp_path)
    assert report["kind"]=="daily" and report["curve"]==[]
    assert len(report["rankings"])==12
    assert report["plan"]["trade_date"]>report["as_of"]
    assert (tmp_path/report["run_id"]/"plan.json").exists()

def test_zero_benchmark_and_unknown_fees_are_rejected(sample):
    tables={k:v.copy() for k,v in sample.tables.items()}
    tables["benchmark"].loc[0,"total_return_close"]=0
    tables["fees"].loc[0,"sell_stamp_rate"]=np.nan
    errors=validate(tables)["errors"]
    assert any("benchmark" in e for e in errors)
    assert any("fees" in e for e in errors)
