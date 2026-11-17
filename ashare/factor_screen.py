"""Training-only factor diagnostics and conservative coarse selection.

Quintile long-short returns are a theoretical diagnostic, not an executable
A-share short book. Daily Sharpe uses one-session next-open returns, not
annualized overlapping five-session labels. No eval-period row is accepted.
"""
import numpy as np
import pandas as pd
from .alpha158 import catalog
from .factor_definitions import definition_for


def candidate_columns(frame):
    from .features import PRICE_FEATURES,FUND_FEATURES
    return [f'x_{c}' for c in PRICE_FEATURES+FUND_FEATURES]+sorted(c for c in frame if c.startswith(('x_wq','x_q158_')))


def factor_catalog():
    from .features import PRICE_FEATURES,FUND_FEATURES
    from .worldquant101 import USABLE
    rows=([{'factor':c,'name':c,'family':'基本面' if c in FUND_FEATURES else '核心量价','source':'本项目核心经济因子','formula':c} for c in PRICE_FEATURES+FUND_FEATURES]
            +[{'factor':f'wq{n:03d}','name':f'WorldQuant Alpha #{n}','family':'WorldQuant 101','source':'https://arxiv.org/abs/1601.00991','formula':f'见 worldquant101.py::alpha{n:03d}'} for n in USABLE]+catalog())
    return [row|definition_for(row['factor']) for row in rows]


def _hac_t(values,lags):
    values=np.asarray(values);values=values[np.isfinite(values)];n=len(values)
    if n<20:return None
    centered=values-values.mean();variance=float(centered@centered)/n
    for lag in range(1,min(lags,n-1)+1):variance+=2*(1-lag/(lags+1))*float(centered[lag:]@centered[:-lag])/n
    return float(values.mean()/np.sqrt(variance/n)) if variance>1e-15 else None


def screen_factors(train,cfg):
    columns=candidate_columns(train)
    if 'factor_return_1d' not in train:raise ValueError('扩展因子筛选缺少单交易日收益，不能用重叠标签伪造Sharpe')
    if train.duplicated(['date','symbol']).any():raise ValueError('因子训练主键重复')
    dates=pd.DatetimeIndex(train.date.unique()).sort_values();split=int(len(dates)*.7)
    if split<40 or len(dates)-split<20:raise ValueError('因子筛选的训练内验证日期不足')
    validation_start=dates[split]
    sign_dates=set(dates[:max(0,split-cfg.embargo_days-cfg.label_horizon-1)])
    daily_ic=[];daily_returns=[];daily_turnover=[];coverage=[];previous={};day_labels=[]
    for day,frame in train.sort_values(['date','symbol']).groupby('date',sort=True):
        x=frame[columns].replace([np.inf,-np.inf],np.nan)
        for col in columns:
            raw=col.removeprefix('x_')
            if raw in frame:x[col]=x[col].where(frame[raw].notna())
        y=frame.forward_return if 'forward_return' in frame else frame.target
        # Pairwise Spearman: re-rank labels separately where each factor is observed.
        ranks=x.rank(pct=True)
        paired=pd.DataFrame(np.broadcast_to(y.to_numpy()[:,None],x.shape),index=x.index,columns=x.columns).where(x.notna()).rank(pct=True)
        ic=ranks.corrwith(paired)
        q=ranks;long=q.ge(.8)&x.notna();short=q.le(.2)&x.notna()
        weights=long.div(long.sum().replace(0,np.nan)).fillna(0)-short.div(short.sum().replace(0,np.nan)).fillna(0)
        weights.index=frame.symbol.values
        returns=frame.factor_return_1d.copy();returns.index=frame.symbol.values
        weighted=weights.mul(returns,axis=0).sum(min_count=1)
        valid_returns=returns.notna().mean()
        if valid_returns<.95:weighted[:]=np.nan
        prior=previous.get('weights')
        turnover=weights.abs().sum() if prior is None else weights.subtract(prior,fill_value=0).abs().sum()
        previous['weights']=weights
        day_labels.append(day);daily_ic.append(ic);daily_returns.append(weighted);daily_turnover.append(turnover);coverage.append(x.notna().mean())
    ic=pd.DataFrame(daily_ic,index=day_labels);rets=pd.DataFrame(daily_returns,index=day_labels);turn=pd.DataFrame(daily_turnover,index=day_labels);cover=pd.DataFrame(coverage,index=day_labels)
    fit_mask=ic.index.isin(sign_dates);val_mask=ic.index>=validation_start
    metadata={r['factor']:r for r in factor_catalog()};rows=[]
    for col in columns:
        v=ic[col].dropna();fit=ic.loc[fit_mask,col].dropna();validation=ic.loc[val_mask,col].dropna()
        direction=1 if fit.mean()>=0 else -1
        net=rets[col]*direction-turn[col]*cfg.factor_screen_cost_bps/10000
        val_net=net.loc[val_mask].dropna();std=val_net.std()
        sharpe=float(val_net.mean()/std*np.sqrt(252)) if len(val_net)>=20 and std>1e-12 else None
        deviations=v.std();rank_ic=float(v.mean()) if len(v) else None
        fit_ic=float(fit.mean()*direction) if len(fit) else None;val_ic=float(validation.mean()*direction) if len(validation) else None
        reasons=[];flags=[]
        if cover[col].mean()<cfg.factor_screen_min_coverage:reasons.append('有效覆盖率不足')
        if len(v)<cfg.min_train_days//2:reasons.append('有效交易日不足')
        if fit_ic is None or val_ic is None:reasons.append('缺少有效训练/验证IC')
        if sharpe is None:reasons.append('缺少有效验证日收益Sharpe')
        if fit_ic is not None and fit_ic<cfg.factor_screen_min_ic:flags.append('前段IC较弱')
        if val_ic is not None and val_ic<=0:flags.append('验证IC方向不稳定')
        if sharpe is not None and sharpe<=cfg.factor_screen_min_sharpe:flags.append('单因子成本后Sharpe未转正')
        key=col.removeprefix('x_');info=metadata.get(key,{})
        rows.append({**info,'factor':key,'feature':col,'direction':direction,'rank_ic':rank_ic,'fit_ic':fit_ic,'validation_ic':val_ic,'icir':float(abs(v.mean())/deviations) if deviations>1e-12 else None,'ic_hac_t':_hac_t(v*direction,max(cfg.label_horizon,5)),'positive_rate':float((v*direction>0).mean()) if len(v) else None,'coverage':float(cover[col].mean()),'days':len(v),'net_sharpe':sharpe,'gross_sharpe':float((rets.loc[val_mask,col]*direction).mean()/(rets.loc[val_mask,col].std())*np.sqrt(252)) if rets.loc[val_mask,col].std()>1e-12 else None,'average_turnover':float(turn[col].mean()),'validation_net_mean':float(val_net.mean()) if len(val_net) else None,'status':'eligible' if not reasons else 'rejected','reasons':reasons,'diagnostic_flags':flags,'selected':False})
    table=pd.DataFrame(rows)
    # Fixed predeclared ranking; the evaluation/replay period is never consulted.
    components=[('fit_ic',.25),('validation_ic',.30),('icir',.15),('net_sharpe',.30)]
    score=sum(table[k].rank(pct=True).fillna(0)*w for k,w in components)
    table['screen_score']=score
    rows=table.sort_values(['screen_score','factor'],ascending=[False,True]).to_dict('records')
    # Correlation pruning uses only training data; deterministic bounded sample.
    sample=train.iloc[::max(1,len(train)//20000)][columns]
    corr=sample.corr().abs();selected=[]
    for rank,row in enumerate(rows,1):
        row['screen_rank']=rank
        if row['status']!='eligible':continue
        similar=next((c for c in selected if corr.loc[row['feature'],c]>cfg.factor_screen_max_correlation),None)
        if similar:
            row['status']='correlated';row['reasons']=['与已入选因子高度相关：'+similar.removeprefix('x_')]
        elif len(selected)>=cfg.factor_screen_target:
            row['status']='limit';row['reasons']=['已达到本窗口粗筛目标数量']
        else:
            selected.append(row['feature']);row['selected']=True;row['status']='selected'
    if not selected:raise ValueError('没有因子通过训练期粗筛；拒绝静默回退为全部因子')
    result={'version':'alpha158-screen-v2-ranked-budget','range_met':cfg.factor_screen_min<=len(selected)<=cfg.factor_screen_max,'warnings':[] if len(selected)>=cfg.factor_screen_min else ['有效且去相关的因子不足最低数量，未用无效或重复因子凑数'],'candidates':len(columns),'new_candidates':sum(c.startswith('x_q158_') for c in columns),'selected_count':len(selected),'selected_features':selected,'rows':rows,'train_start':str(dates.min().date()),'train_end':str(dates.max().date()),'validation_start':str(validation_start.date()),'latest_label_end':str(pd.to_datetime(train.label_end).max().date()),'rules':{'min_selected':cfg.factor_screen_min,'target_selected':cfg.factor_screen_target,'selection_mode':'ranked_budget','min_coverage':cfg.factor_screen_min_coverage,'min_fit_ic':cfg.factor_screen_min_ic,'min_validation_net_sharpe':cfg.factor_screen_min_sharpe,'max_correlation':cfg.factor_screen_max_correlation,'max_selected':cfg.factor_screen_max,'cost_bps_per_gross_turnover':cfg.factor_screen_cost_bps},'method':f'仅用成熟的历史训练数据：前70%确定方向，隔离后用后30%计算验证IC和成本后Sharpe。覆盖率、有效样本和指标可计算性是硬门槛；IC/ICIR/净Sharpe按25%/30%/15%/30%的百分位粗排，再去相关，目标{cfg.factor_screen_target}个、允许{cfg.factor_screen_min}–{cfg.factor_screen_max}个。单因子IC或净Sharpe偏弱会降低排名并标注，不再一票否决；入模不代表其可独立交易。HAC t值未做多重检验。','sharpe_note':f'理论等权顶部20%减底部20%的次日开盘至再下一开盘日收益；扣每单位双边换手{cfg.factor_screen_cost_bps:g}bp代理成本，不含借券约束和冲击。仅用于训练内筛选，不是A股可执行策略收益。','sources':[{'name':'Microsoft Qlib Alpha158','url':'https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py'}]}

    def safe(value):
        if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
        if isinstance(value,list):return [safe(v) for v in value]
        if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
        return value
    return safe(result)
