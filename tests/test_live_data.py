import pandas as pd

from ashare.config import Config
from ashare.live_data import AkShareProvider, _build_prices, _limit_price, _live_gate, _reference_actions, merge_table, normalize_symbol


def test_live_symbol_normalization_and_board_limits():
    assert normalize_symbol("sh.600000") == "600000.SH"
    assert normalize_symbol("000001", "深圳") == "000001.SZ"
    assert _limit_price(10, 1, 0, "600000.SH") == 11.0
    assert _limit_price(10, 1, 1, "600000.SH") == 10.5
    assert _limit_price(10, 1, 0, "300001.SZ") == 12.0


def test_live_overlay_replaces_same_primary_key():
    base = pd.DataFrame({"date": pd.to_datetime(["2026-09-18"]), "symbol": ["600000.SH"], "close": [10.0]})
    overlay = base.assign(close=10.2)
    merged = merge_table(base, overlay, "prices")
    assert len(merged) == 1
    assert merged.iloc[0].close == 10.2


def test_live_gate_requires_every_constituent_close():
    latest = pd.Timestamp("2026-09-18")
    symbols = [f"{index:06d}.SZ" for index in range(300)]
    universe = pd.DataFrame({"date": latest, "symbol": symbols})
    prices = universe.iloc[:-1].copy()
    calendar = pd.DataFrame({"date": pd.bdate_range(latest + pd.Timedelta(days=1), periods=10)})
    tables = {"universe": universe, "prices": prices, "calendar": calendar}
    blocked = _live_gate(tables, {"errors": [], "warnings": []}, latest, Config())
    assert not blocked["passed"]
    assert blocked["price_coverage"] < 1
    tables["prices"] = universe.copy()
    passed = _live_gate(tables, {"errors": [], "warnings": []}, latest, Config())
    assert passed["passed"]
    assert passed["price_coverage"] == 1


def test_official_universe_retries_with_network_timeout(monkeypatch):
    import requests

    calls=[]
    class Response:
        def __init__(self,body):self.content=body
        def raise_for_status(self):pass

    def fetch(url,timeout):
        calls.append((url,timeout))
        if len(calls)==1:raise requests.ConnectionError('temporary disconnect')
        return Response(b'weights' if 'closeweight' in url else b'constituents')

    def read_excel(stream):
        names=['日期','指数代码','指数名称','指数英文名称','成分券代码','成分券名称','成分券英文名称','交易所','交易所英文名称']
        if stream.read()==b'weights':names.append('权重')
        values=['20260922','000300','沪深300','CSI300','600000','浦发银行','SPD Bank','上海证券交易所','SSE']
        if len(names)==10:values.append('1.2')
        return pd.DataFrame([values],columns=names)

    sessions=[]
    class DirectSession:
        def __init__(self):
            self.trust_env=True
            sessions.append(self)

        def get(self,url,timeout):
            return fetch(url,timeout)

    monkeypatch.setattr(requests,'Session',DirectSession)
    monkeypatch.setattr(pd,'read_excel',read_excel)
    monkeypatch.setattr('ashare.live_data.time.sleep',lambda _:None)
    constituents,weights=AkShareProvider.__new__(AkShareProvider).universe()
    assert sessions[0].trust_env is False
    assert len(calls)==3 and all(timeout==(5,20) for _,timeout in calls)
    assert constituents.iloc[0]['成分券代码']=='600000'
    assert weights.iloc[0]['权重']==1.2


def test_source_cited_action_repairs_vendor_omission():
    result=_reference_actions(pd.DatetimeIndex(["2026-09-24"]),{"000538.SZ"})
    assert len(result)==1
    assert result.iloc[0].cash_per_share==1.038
    assert str(result.iloc[0].known_at)=="2026-09-17 20:00:00+08:00"


def test_missing_action_detail_quarantines_only_affected_symbol(tmp_path):
    day=pd.Timestamp("2026-10-19")
    class Provider:
        def history(self,*_):
            return pd.DataFrame([{"date":day,"open":8.1,"high":8.4,"low":8.,"close":8.2,
                                  "preclose":8.,"volume":1000,"amount":8200,"turn":1.,
                                  "tradestatus":"1","isST":"0"}])
        def adjustments(self,*_):return pd.DataFrame()
        def dividends(self,*_):return pd.DataFrame()
    universe=pd.DataFrame({"date":[day],"symbol":["600000.SH"]})
    metadata={"600000.SH":{"date":pd.Timestamp("2026-10-16"),"close":10.,"adj_factor":1.,
                              "market_cap":100_000.,"float_market_cap":80_000.,"industry":"金融",
                              "min_order":100,"lot_step":100,"listed_days":1000,"is_st":0}}
    prices,actions,errors,warnings,_,quarantines=_build_prices(
        Provider(),object(),pd.DatetimeIndex([day]),universe,metadata,tmp_path,lambda *_:None
    )
    assert errors==[] and actions.empty
    assert quarantines==[{"symbol":"600000.SH","date":"2026-10-19","reason":"600000.SH 2026-10-19 检测到除权参考价变化，但公司行动明细未取得"}]
    assert prices.iloc[0].suspended==1 and prices.iloc[0].close==10. and prices.iloc[0].amount==0
    assert any("其他股票继续更新" in warning for warning in warnings)
