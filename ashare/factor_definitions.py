"""Human-readable definitions of the exact model inputs, shared by UI exports."""
import ast
from functools import lru_cache
from pathlib import Path
import re
from .alpha158 import catalog

CORE = {
 'momentum_5':('5日动量','过去5个交易日的复权收盘涨跌幅','C_t / C_(t-5) - 1'),
 'momentum_20':('20日动量','过去20个交易日的复权收盘涨跌幅','C_t / C_(t-20) - 1'),
 'momentum_60':('60日动量','过去60个交易日的复权收盘涨跌幅','C_t / C_(t-60) - 1'),
 'reversal_1':('短期反转','上一交易日复权收盘收益的相反数','-(C_t / C_(t-1) - 1)'),
 'volatility_20':('20日波动率','20日收盘收益的年化样本标准差','Std(r,20) × sqrt(252)'),
 'downside_20':('下行波动','20日负收益平方均值的年化平方根','sqrt(Mean(min(r,0)^2,20)) × sqrt(252)'),
 'volume_ratio':('量能变化','短期成交股数均值相对中期成交股数均值','Mean(volume,5) / Mean(volume,20)'),
 'turnover_20':('换手率','20日平均换手率','Mean(turnover_rate,20)'),
 'amihud_20':('非流动性','单位成交额对应的绝对价格变化，乘1亿调整量纲','Mean(abs(r) / amount × 100000000,20)'),
 'intraday_return':('日内收益','当日收盘相对开盘的涨跌幅','close / open - 1'),
 'range_20':('20日振幅','20日平均高低价差相对前收盘价','Mean((high-low) / prev_close,20)'),
 'ma_distance':('均线偏离','复权收盘价相对20日均线的偏离','C_t / Mean(C,20) - 1'),
 'log_size':('市值','股票总市值的自然对数','ln(market_cap)'),
 'earnings_yield':('盈利收益率','近12个月净利润相对股票总市值','net_profit_TTM / market_cap'),
 'book_to_price':('账面市值比','账面净资产相对股票总市值','book_equity / market_cap'),
 'cashflow_yield':('现金流收益率','近12个月经营活动现金流相对股票总市值','operating_cashflow_TTM / market_cap'),
 'roe':('ROE','数据库提供的净资产收益率；只接入当时已可用的报表版本','roe（报表字段，按available_at接入）'),
 'gross_margin':('毛利率','数据库提供的报表毛利率；只接入当时已可用版本','gross_margin（报表字段，按available_at接入）'),
 'revenue_growth':('收入增长','数据库提供的营业收入增长率','revenue_growth（报表字段，按available_at接入）'),
 'profit_growth':('利润增长','数据库提供的净利润增长率','profit_growth（报表字段，按available_at接入）'),
 'leverage':('资产负债率','总负债相对总资产','total_liabilities / total_assets'),
 'accruals':('应计利润','净利润与经营现金流的差额，相对总资产','(net_profit_TTM - operating_cashflow_TTM) / total_assets'),
 'fund_age':('财报时效','当前决策时点距离最新可用财报的天数','(decision_cutoff - fund_available_at) / 86400 seconds'),
}
ROLLING_MEANING={'roc':'历史收盘价与当日收盘价之比','ma':'平均收盘价相对当日收盘价','std':'收盘价格标准差相对当日收盘价','beta':'价格对时间的线性回归斜率，相对当日收盘价','rsqr':'价格对时间线性回归的R²','resi':'时间回归的末日残差，相对当日收盘价','max':'最高价相对当日收盘价','min':'最低价相对当日收盘价','qtlu':'收盘价80%分位相对当日收盘价','qtld':'收盘价20%分位相对当日收盘价','rank':'当日收盘在历史窗口中的百分位','rsv':'当日收盘在窗口高低价区间中的位置','imax':'窗口内最高价的位置除以窗口长度','imin':'窗口内最低价的位置除以窗口长度','imxd':'窗口最高、最低价位置的差除以窗口长度','corr':'收盘价与对数成交量的相关性','cord':'价格变化比率与对数成交量变化比率的相关性','cntp':'上涨交易日比例','cntn':'下跌交易日比例','cntd':'上涨日比例减下跌日比例','sump':'累计正价格变动相对累计绝对变动','sumn':'累计负价格变动相对累计绝对变动','sumd':'累计正负价格变动之差相对累计绝对变动','vma':'平均成交量相对当日成交量','vstd':'成交量标准差相对当日成交量','wvma':'成交量加权绝对收益的波动系数','vsump':'累计正成交量变化相对累计绝对成交量变化','vsumn':'累计负成交量变化相对累计绝对成交量变化','vsumd':'累计正负成交量变化之差相对累计绝对变化'}
KBAR_MEANING={'kmid':'实体长度相对开盘价','klen':'高低价差相对开盘价','kmid2':'实体长度相对高低价差','kup':'上影线相对开盘价','kup2':'上影线相对高低价差','klow':'下影线相对开盘价','klow2':'下影线相对高低价差','ksft':'收盘偏离高低价中点的两倍，相对开盘价','ksft2':'收盘偏离高低价中点的两倍，相对高低价差'}

@lru_cache(maxsize=1)
def _worldquant_formulas():
    tree=ast.parse((Path(__file__).parent/'worldquant101.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Alphas')
    result={}
    for function in cls.body:
        if not isinstance(function,ast.FunctionDef) or not function.name.startswith('alpha'):continue
        statements=[node for node in function.body if not (isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str))]
        formula='\n'.join(ast.unparse(node).removeprefix('return ') for node in statements).replace('self.','')
        result['wq'+function.name[5:]]=formula
    result['wq027']='where(rank(Mean(Corr(rank(V),rank(VWAP),6),2)) > 0.5, -1, +1)'
    return result

@lru_cache(maxsize=512)
def definition_for(factor):
    factor=factor.removeprefix('x_')
    if factor in CORE:
        name,meaning,formula=CORE[factor]
        return {'name':name,'definition':meaning,'formula':formula,'formula_source':'ashare/features.py'}
    if factor.startswith('q158_'):
        item=next((row for row in catalog() if row['factor']==factor),None)
        if item:
            key=factor[5:];match=re.fullmatch(r'([a-z]+)(\d+)',key)
            meaning=KBAR_MEANING.get(key)
            if match:
                op,n=match.groups();meaning=f'{n}日'+ROLLING_MEANING[op] if op in ROLLING_MEANING else {'open':'开盘价相对当日收盘价','high':'最高价相对当日收盘价','low':'最低价相对当日收盘价','vwap':'成交均价相对当日收盘价'}.get(op)
            return {'name':item['name'],'definition':meaning or item['name'],'formula':item['formula'],'formula_source':item['source']}
    if factor in _worldquant_formulas():
        n=int(factor[2:]);meaning='成交量与成交均价的截面排名相关性，经平滑和排名后输出二值信号' if n==27 else f'WorldQuant Alpha #{n}；下列为本项目实际计算表达式'
        return {'name':f'WorldQuant Alpha #{n}','definition':meaning,'formula':_worldquant_formulas()[factor],'formula_source':f'ashare/worldquant101.py::alpha{n:03}'}
    return {'name':factor,'definition':'未提供定义','formula':factor,'formula_source':None}
