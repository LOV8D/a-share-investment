"""Alpha158 candidates, independently evaluated from Microsoft Qlib formulas.

Source: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py
Qlib is MIT licensed, Copyright Microsoft Corporation. We implement the
expressions in pandas, without importing Qlib or executing expression strings.
Prices use a common adjustment factor; volume is split-adjusted inversely.
"""
import re
import numpy as np
import pandas as pd

SOURCE = 'https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py'
WINDOWS = (5, 10, 20, 30, 60)
ROLLING = {
 'ROC': 'Ref(C,n)/C', 'MA': 'Mean(C,n)/C', 'STD': 'Std(C,n)/C',
 'BETA': 'Slope(C,n)/C', 'RSQR': 'Rsquare(C,n)', 'RESI': 'Resi(C,n)/C',
 'MAX': 'Max(H,n)/C', 'MIN': 'Min(L,n)/C', 'QTLU': 'Quantile(C,n,.8)/C',
 'QTLD': 'Quantile(C,n,.2)/C', 'RANK': 'TsRank(C,n)',
 'RSV': '(C-Min(L,n))/(Max(H,n)-Min(L,n)+eps)',
 'IMAX': 'IdxMax(H,n)/n', 'IMIN': 'IdxMin(L,n)/n', 'IMXD': '(IdxMax(H,n)-IdxMin(L,n))/n',
 'CORR': 'Corr(C,log(V+1),n)', 'CORD': 'Corr(C/Ref(C,1),log(V/Ref(V,1)+1),n)',
 'CNTP': 'Mean(C>Ref(C,1),n)', 'CNTN': 'Mean(C<Ref(C,1),n)', 'CNTD': 'CNTP-CNTN',
 'SUMP': 'Sum(max(delta(C),0),n)/(Sum(abs(delta(C)),n)+eps)',
 'SUMN': 'Sum(max(-delta(C),0),n)/(Sum(abs(delta(C)),n)+eps)', 'SUMD': 'SUMP-SUMN',
 'VMA': 'Mean(V,n)/(V+eps)', 'VSTD': 'Std(V,n)/(V+eps)',
 'WVMA': 'Std(abs(C/Ref(C,1)-1)*V,n)/(Mean(abs(C/Ref(C,1)-1)*V,n)+eps)',
 'VSUMP': 'Sum(max(delta(V),0),n)/(Sum(abs(delta(V)),n)+eps)',
 'VSUMN': 'Sum(max(-delta(V),0),n)/(Sum(abs(delta(V)),n)+eps)', 'VSUMD': 'VSUMP-VSUMN',
}
KBAR = {
 'KMID':'(C-O)/O','KLEN':'(H-L)/O','KMID2':'(C-O)/(H-L+eps)',
 'KUP':'(H-max(O,C))/O','KUP2':'(H-max(O,C))/(H-L+eps)',
 'KLOW':'(min(O,C)-L)/O','KLOW2':'(min(O,C)-L)/(H-L+eps)',
 'KSFT':'(2*C-H-L)/O','KSFT2':'(2*C-H-L)/(H-L+eps)',
}

def catalog():
    rows=[{'factor':'q158_'+name.lower(),'name':name,'formula':formula} for name,formula in KBAR.items()]
    rows += [{'factor':'q158_'+name.lower()+'0','name':name+'0','formula':name+'/C'} for name in ('OPEN','HIGH','LOW','VWAP')]
    rows += [{'factor':f'q158_{name.lower()}{n}','name':f'{name}{n}','formula':formula.replace('n',str(n)),'window':n} for name,formula in ROLLING.items() for n in WINDOWS]
    return [r|{'family':'Qlib Alpha158','source':SOURCE,'status':'candidate','max_lookback':61} for r in rows]


def compute_alpha158(prices):
    """Return date-local ranked features; missing values remain NaN for screening."""
    source=prices.sort_values(['date','symbol']).copy()
    def wide(name):return source.pivot(index='date',columns='symbol',values=name).sort_index()
    adj=wide('adj_factor');o=wide('open')*adj;h=wide('high')*adj;l=wide('low')*adj;c=wide('close')*adj
    v=wide('volume')/adj;vwap=wide('amount')/wide('volume').replace(0,np.nan)*adj
    eps=1e-12;span=h-l;mx=np.maximum(o,c);mn=np.minimum(o,c)
    values={'KMID':(c-o)/o,'KLEN':span/o,'KMID2':(c-o)/(span+eps),'KUP':(h-mx)/o,'KUP2':(h-mx)/(span+eps),'KLOW':(mn-l)/o,'KLOW2':(mn-l)/(span+eps),'KSFT':(2*c-h-l)/o,'KSFT2':(2*c-h-l)/(span+eps),'OPEN0':o/c,'HIGH0':h/c,'LOW0':l/c,'VWAP0':vwap/c}
    dc=c.diff();dv=v.diff();pc=c/c.shift(1);vc=np.log(v/v.shift(1)+1);wv=pc.sub(1).abs()*v
    for n in WINDOWS:
        rc=c.rolling(n);rv=v.rolling(n);rh=h.rolling(n);rl=l.rolling(n)
        mean=rc.mean();std=rc.std();t=np.arange(n,dtype=float);center=t-t.mean();ss=float(center@center)
        slope=rc.apply(lambda x:float(x@center)/ss,raw=True)
        rsq=slope.pow(2)*ss/(rc.var(ddof=0)*n).replace(0,np.nan)
        ih=rh.apply(lambda x:float(np.argmax(x)+1),raw=True);il=rl.apply(lambda x:float(np.argmin(x)+1),raw=True)
        positive=dc.gt(0).where(dc.notna()).rolling(n).mean();negative=dc.lt(0).where(dc.notna()).rolling(n).mean()
        sump=dc.clip(lower=0).rolling(n).sum()/(dc.abs().rolling(n).sum()+eps);sumn=(-dc).clip(lower=0).rolling(n).sum()/(dc.abs().rolling(n).sum()+eps)
        vp=dv.clip(lower=0).rolling(n).sum()/(dv.abs().rolling(n).sum()+eps);vn=(-dv).clip(lower=0).rolling(n).sum()/(dv.abs().rolling(n).sum()+eps)
        batch={'ROC':c.shift(n)/c,'MA':mean/c,'STD':std/c,'BETA':slope/c,'RSQR':rsq,'RESI':(c-mean-slope*(n-1-t.mean()))/c,'MAX':rh.max()/c,'MIN':rl.min()/c,'QTLU':rc.quantile(.8)/c,'QTLD':rc.quantile(.2)/c,'RANK':rc.rank(pct=True),'RSV':(c-rl.min())/(rh.max()-rl.min()+eps),'IMAX':ih/n,'IMIN':il/n,'IMXD':(ih-il)/n,'CORR':rc.corr(np.log(v+1)),'CORD':pc.rolling(n).corr(vc),'CNTP':positive,'CNTN':negative,'CNTD':positive-negative,'SUMP':sump,'SUMN':sumn,'SUMD':sump-sumn,'VMA':rv.mean()/(v+eps),'VSTD':rv.std()/(v+eps),'WVMA':wv.rolling(n).std()/(wv.rolling(n).mean()+eps),'VSUMP':vp,'VSUMN':vn,'VSUMD':vp-vn}
        values.update({f'{name}{n}':value for name,value in batch.items()})
    series=[]
    for name,value in values.items():
        ranked=value.replace([np.inf,-np.inf],np.nan).rank(axis=1,pct=True).sub(.5).astype('float32')
        series.append(ranked.stack(future_stack=True).rename('x_q158_'+name.lower()))
    result=pd.concat(series,axis=1).reset_index()
    # Daily Sharpe diagnostics use non-overlapping one-session next-open returns.
    result=result.merge((o.shift(-2)/o.shift(-1)-1).stack(future_stack=True).rename('factor_return_1d').reset_index(),on=['date','symbol'],validate='one_to_one')
    return result


def catalog():
    rows=[]
    for name,formula in KBAR.items():
        rows.append({'factor':'q158_'+name.lower(),'name':'Alpha158 '+name,'family':'Qlib Alpha158','formula':formula,'source':SOURCE,'lookback':1})
    for name in ['OPEN','HIGH','LOW','VWAP']:
        rows.append({'factor':'q158_'+name.lower()+'0','name':'Alpha158 '+name+'0','family':'Qlib Alpha158','formula':name+'/C','source':SOURCE,'lookback':1})
    for op,formula in ROLLING.items():
        for n in WINDOWS:
            rows.append({'factor':f'q158_{op.lower()}{n}','name':f'Alpha158 {op}{n}','family':'Qlib Alpha158','formula':re.sub(r'\bn\b',str(n),formula),'source':SOURCE,'lookback':n+1})
    return rows
