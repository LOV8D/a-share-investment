from dataclasses import replace
import numpy as np
import pandas as pd
from ashare.alpha158 import compute_alpha158,catalog
from ashare.config import Config
from ashare.factor_screen import screen_factors
from ashare.models import training_slice


def prices_fixture():
    dates=pd.bdate_range('2025-01-01',periods=90)
    parts=[]
    for i in range(12):
        rng=np.random.default_rng(i);close=30*np.exp(np.cumsum(rng.normal(.001,.02,len(dates))))
        parts.append(pd.DataFrame({'date':dates,'symbol':f'{i:06d}.SZ','close':close,'open':close*.995,'high':close*1.02,'low':close*.98,'volume':rng.uniform(1e5,2e5,len(dates)),'amount':close*1e5,'adj_factor':1.}))
    return pd.concat(parts,ignore_index=True)


def test_alpha158_catalog_formulas_and_no_future_feature_leakage():
    source=prices_fixture();result=compute_alpha158(source)
    assert len(catalog())==158 and len({x['factor'] for x in catalog()})==158
    assert set(result.filter(like='x_q158_'))=={'x_'+x['factor'] for x in catalog()}
    split=source.date.unique()[75];future=source.copy()
    future.loc[future.date>split,['close','open','high','low','volume','amount']]*=3
    changed=compute_alpha158(future)
    features=['date','symbol']+list(result.filter(like='x_q158_'))
    pd.testing.assert_frame_equal(result.loc[result.date<=split,features].reset_index(drop=True),changed.loc[changed.date<=split,features].reset_index(drop=True))
    # Independently verify a rolling formula and an OHLC shape, including CS ranks.
    raw=source.sort_values(['symbol','date']).copy()
    raw['roc5']=raw.groupby('symbol').close.shift(5)/raw.close
    expected=raw.groupby('date').roc5.rank(pct=True)-.5
    actual=result.set_index(['date','symbol']).x_q158_roc5.reindex(pd.MultiIndex.from_frame(raw[['date','symbol']]))
    np.testing.assert_allclose(actual,expected,equal_nan=True,atol=1e-6)


def screening_fixture():
    from ashare.features import PRICE_FEATURES,FUND_FEATURES
    from ashare.worldquant101 import USABLE
    rng=np.random.default_rng(2026);dates=pd.bdate_range('2024-01-01',periods=170);n=30
    f=pd.MultiIndex.from_product([dates,[f'{i:06d}.SZ' for i in range(n)]],names=['date','symbol']).to_frame(index=False)
    signal=rng.normal(size=len(f));f['target']=signal*.01+rng.normal(0,.003,len(f));f['forward_return']=f.target
    f['factor_return_1d']=signal*.005+rng.normal(0,.002,len(f));f['eligible']=True
    f['label_end']=f.date+pd.Timedelta(days=9);f['label_available_at']=pd.to_datetime(f.label_end).dt.tz_localize('Asia/Shanghai')+pd.Timedelta(hours=18)
    columns={**{f'x_{c}':rng.normal(size=len(f)) for c in PRICE_FEATURES+FUND_FEATURES},**{f'x_wq{k:03d}':rng.normal(size=len(f)) for k in USABLE}}
    columns['x_q158_good']=signal;columns['x_q158_duplicate']=-signal;columns['x_q158_missing']=np.nan
    f=pd.concat([f,pd.DataFrame(columns)],axis=1)
    return f


def test_screen_excludes_missing_and_correlated_and_never_reads_future():
    f=screening_fixture();cfg=replace(Config(),min_train_days=40,train_window_days=150,factor_screen_cost_bps=0.)
    signal=f.date.unique()[150];train=training_slice(f,signal,cfg)
    selected=screen_factors(train,cfg)
    pair={'x_q158_good','x_q158_duplicate'}
    assert len(pair.intersection(selected['selected_features']))==1
    missing=next(x for x in selected['rows'] if x['feature']=='x_q158_missing')
    assert not missing['selected'] and '有效覆盖率不足' in missing['reasons']
    future=f.copy();future.loc[future.date>=signal,['target','forward_return','factor_return_1d']]=1e12
    assert selected==screen_factors(training_slice(future,signal,cfg),cfg)
    assert pd.Timestamp(selected['latest_label_end'])<pd.Timestamp(signal)
    assert selected['new_candidates']==3


def test_alpha158_cross_section_uses_historical_members_only():
    from ashare.features import normalize_zoo_cross_section
    source=prices_fixture();members=sorted(source.symbol.unique())[:6]
    wide=compute_alpha158(source);within=compute_alpha158(source[source.symbol.isin(members)])
    wide=normalize_zoo_cross_section(wide[wide.symbol.isin(members)].copy())
    within=normalize_zoo_cross_section(within.copy())
    columns=['date','symbol']+list(wide.filter(like='x_q158_'))
    pd.testing.assert_frame_equal(wide[columns].reset_index(drop=True),within[columns].reset_index(drop=True))


def test_enabled_zoo_rejects_incomplete_candidates():
    import pytest
    from ashare.models import fit_xgboost
    frame=screening_fixture()
    with pytest.raises(ValueError,match='158个Alpha158候选不完整'):
        fit_xgboost(frame,Config())
