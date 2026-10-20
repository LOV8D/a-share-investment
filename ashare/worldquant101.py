"""Point-in-time implementation of the data-available WorldQuant 101 alphas.

Formulas follow Kakushadze (2016), "101 Formulaic Alphas":
https://arxiv.org/abs/1601.00991.  Eighteen formulas need the paper's
sector/industry/subindustry hierarchy, which the current data does not contain;
they are deliberately excluded instead of silently substituting one label.
"""

import numpy as np
import pandas as pd
from scipy import stats


UNAVAILABLE = {
    48: "subindustry", 58: "sector", 59: "industry", 63: "industry",
    67: "sector+subindustry", 69: "industry", 70: "industry", 76: "sector",
    79: "sector", 80: "industry", 82: "sector", 87: "industry",
    89: "industry", 90: "subindustry", 91: "industry", 93: "industry",
    97: "industry", 100: "subindustry",
}
USABLE = tuple(i for i in range(1, 102) if i not in UNAVAILABLE)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank, normalized to [0,1]."""
    return df.rank(axis=1, pct=True)


def ts_rank(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Time-series rolling rank over `window` days."""
    return df.rolling(window).apply(lambda x: stats.rankdata(x)[-1] / window, raw=True)


def ts_sum(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window).sum()


def ts_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window).mean()


def ts_min(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window).min()


def ts_max(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window).max()


def ts_argmin(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window).apply(np.argmin, raw=True) + 1


def ts_argmax(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window).apply(np.argmax, raw=True) + 1


def delay(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.shift(d)


def delta(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.diff(d)


def correlation(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window).corr(y)


def covariance(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window).cov(y)


def stddev(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window).std()


def scale(df: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    """Scale each row so absolute values sum to `a`."""
    return df.div(df.abs().sum(axis=1), axis=0) * a


def signedpower(df: pd.DataFrame, e: float) -> pd.DataFrame:
    return np.sign(df) * np.power(np.abs(df), e)


def log(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df)


def sign(df: pd.DataFrame) -> pd.DataFrame:
    return np.sign(df)


def decay_linear(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Weighted moving average with linearly increasing weights."""
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    return df.rolling(window).apply(lambda x: (x * weights).sum(), raw=True)


def product(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window).apply(np.prod, raw=True)


def indneutralize(df: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Subtract industry/group mean from each stock (cross-sectional)."""
    result = df.copy()
    for date in df.index:
        row = df.loc[date]
        group_means = row.groupby(groups).transform("mean")
        result.loc[date] = row - group_means
    return result


# ---------------------------------------------------------------------------
# Main Alphas class
# ---------------------------------------------------------------------------

class Alphas:
    """
    Implements all 101 WorldQuant formulaic alphas.

    Parameters
    ----------
    open_, high, low, close, volume : pd.DataFrame
        OHLCV data — rows = dates, columns = tickers.
    vwap : pd.DataFrame
        Volume-weighted average price.
    returns : pd.DataFrame
        Daily returns (close.pct_change()).
    cap : pd.DataFrame, optional
        Market capitalization.
    adv : dict, optional
        Pre-computed average daily volumes, e.g. {20: adv20_df, 5: adv5_df}.
    industry : pd.Series, optional
        Industry group label per ticker (used in indneutralize calls).
    subindustry : pd.Series, optional
        Sub-industry label per ticker.
    """

    def __init__(self, open_, high, low, close, volume, vwap, returns,
                 cap=None, adv=None, industry=None, subindustry=None):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.vwap = vwap
        self.returns = returns
        self.cap = cap if cap is not None else pd.DataFrame(np.ones_like(close), index=close.index, columns=close.columns)
        self._adv_cache = adv or {}
        self.industry = industry
        self.subindustry = subindustry

    def adv(self, n: int) -> pd.DataFrame:
        if n not in self._adv_cache:
            self._adv_cache[n] = ts_mean(self.volume, n)
        return self._adv_cache[n]

    # ------------------------------------------------------------------
    # Alpha definitions
    # ------------------------------------------------------------------

    def alpha001(self):
        """rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns,20) : close), 2.), 5)) - 0.5"""
        cond = self.returns < 0
        inner = np.where(cond, stddev(self.returns, 20), self.close)
        inner = pd.DataFrame(inner, index=self.close.index, columns=self.close.columns)
        return rank(ts_argmax(signedpower(inner, 2), 5)) - 0.5

    def alpha002(self):
        """-1 * correlation(rank(delta(log(volume),2)), rank((close-open)/open), 6)"""
        return -1 * correlation(rank(delta(log(self.volume), 2)),
                                rank((self.close - self.open) / self.open), 6)

    def alpha003(self):
        """-1 * correlation(rank(open), rank(volume), 10)"""
        return -1 * correlation(rank(self.open), rank(self.volume), 10)

    def alpha004(self):
        """-1 * Ts_Rank(rank(low), 9)"""
        return -1 * ts_rank(rank(self.low), 9)

    def alpha005(self):
        """(rank((open-(ts_sum(vwap,10)/10)))*(-1*abs(rank((close-vwap)))))"""
        return rank(self.open - ts_mean(self.vwap, 10)) * (-1 * abs(rank(self.close - self.vwap)))

    def alpha006(self):
        """-1 * correlation(open, volume, 10)"""
        return -1 * correlation(self.open, self.volume, 10)

    def alpha007(self):
        """((adv20 < volume) ? ((-1*ts_rank(abs(delta(close,7)),60))*sign(delta(close,7))) : -1)"""
        cond = self.adv(20) < self.volume
        val = -1 * ts_rank(abs(delta(self.close, 7)), 60) * sign(delta(self.close, 7))
        return pd.DataFrame(np.where(cond, val, -1), index=self.close.index, columns=self.close.columns)

    def alpha008(self):
        """-1 * rank(((ts_sum(open,5)*ts_sum(returns,5))-delay((ts_sum(open,5)*ts_sum(returns,5)),10)))"""
        x = ts_sum(self.open, 5) * ts_sum(self.returns, 5)
        return -1 * rank(x - delay(x, 10))

    def alpha009(self):
        """(0<ts_min(delta(close,1),5)) ? delta(close,1) : ((ts_max(delta(close,1),5)<0) ? delta(close,1) : (-1*delta(close,1)))"""
        d = delta(self.close, 1)
        cond1 = ts_min(d, 5) > 0
        cond2 = ts_max(d, 5) < 0
        return pd.DataFrame(
            np.where(cond1, d, np.where(cond2, d, -1 * d)),
            index=self.close.index, columns=self.close.columns
        )

    def alpha010(self):
        """rank(((0<ts_min(delta(close,1),4))?delta(close,1):((ts_max(delta(close,1),4)<0)?delta(close,1):(-1*delta(close,1)))))"""
        d = delta(self.close, 1)
        cond1 = ts_min(d, 4) > 0
        cond2 = ts_max(d, 4) < 0
        inner = pd.DataFrame(
            np.where(cond1, d, np.where(cond2, d, -1 * d)),
            index=self.close.index, columns=self.close.columns
        )
        return rank(inner)

    def alpha011(self):
        """((rank(ts_max((vwap-close),3))+rank(ts_min((vwap-close),3)))*rank(delta(volume,3)))"""
        x = self.vwap - self.close
        return (rank(ts_max(x, 3)) + rank(ts_min(x, 3))) * rank(delta(self.volume, 3))

    def alpha012(self):
        """(sign(delta(volume,1))*(-1*delta(close,1)))"""
        return sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    def alpha013(self):
        """-1 * rank(covariance(rank(close), rank(volume), 5))"""
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    def alpha014(self):
        """-1 * rank(delta(returns,3)) * correlation(open, volume, 10)"""
        return -1 * rank(delta(self.returns, 3)) * correlation(self.open, self.volume, 10)

    def alpha015(self):
        """-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)"""
        return -1 * ts_sum(rank(correlation(rank(self.high), rank(self.volume), 3)), 3)

    def alpha016(self):
        """-1 * rank(covariance(rank(high), rank(volume), 5))"""
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))

    def alpha017(self):
        """-1*rank(ts_rank(close,10))*rank(delta(delta(close,1),1))*rank(ts_rank((volume/adv20),5))"""
        return (-1 * rank(ts_rank(self.close, 10))
                * rank(delta(delta(self.close, 1), 1))
                * rank(ts_rank(self.volume / self.adv(20), 5)))

    def alpha018(self):
        """-1*rank((stddev(abs(close-open),5)+(close-open))+correlation(close,open,10))"""
        return -1 * rank(stddev(abs(self.close - self.open), 5)
                         + (self.close - self.open)
                         + correlation(self.close, self.open, 10))

    def alpha019(self):
        """(-1*sign((close-delay(close,7))+delta(close,7)))*(1+rank(1+ts_sum(returns,250)))"""
        return ((-1 * sign(delta(self.close, 7) + delta(self.close, 7)))
                * (1 + rank(1 + ts_sum(self.returns, 250))))

    def alpha020(self):
        """-1*rank(open-delay(high,1))*rank(open-delay(close,1))*rank(open-delay(low,1))"""
        return (-1 * rank(self.open - delay(self.high, 1))
                * rank(self.open - delay(self.close, 1))
                * rank(self.open - delay(self.low, 1)))

    def alpha021(self):
        """((ts_mean(close,8)+stddev(close,8))<ts_mean(close,2))?-1:((ts_mean(close,2)<(ts_mean(close,8)-stddev(close,8)))?1:((volume/adv20<=1)?-1:1))"""
        sma8 = ts_mean(self.close, 8)
        std8 = stddev(self.close, 8)
        sma2 = ts_mean(self.close, 2)
        cond1 = (sma8 + std8) < sma2
        cond2 = sma2 < (sma8 - std8)
        cond3 = (self.volume / self.adv(20)) >= 1
        result = np.where(cond1, -1, np.where(cond2, 1, np.where(cond3, 1, -1)))
        return pd.DataFrame(result, index=self.close.index, columns=self.close.columns)

    def alpha022(self):
        """-1*delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))"""
        return -1 * delta(correlation(self.high, self.volume, 5), 5) * rank(stddev(self.close, 20))

    def alpha023(self):
        """(ts_mean(high,20)<high)?(-1*delta(high,2)):0"""
        cond = ts_mean(self.high, 20) < self.high
        return pd.DataFrame(np.where(cond, -1 * delta(self.high, 2), 0),
                            index=self.close.index, columns=self.close.columns)

    def alpha024(self):
        """((delta(ts_mean(close,100),100)/delay(close,100))<=0.05)?(-1*(close-ts_min(close,100))):(-1*delta(close,3))"""
        cond = (delta(ts_mean(self.close, 100), 100) / delay(self.close, 100)) <= 0.05
        return pd.DataFrame(
            np.where(cond, -1 * (self.close - ts_min(self.close, 100)), -1 * delta(self.close, 3)),
            index=self.close.index, columns=self.close.columns
        )

    def alpha025(self):
        """rank(((-1*returns)*adv20*vwap*(high-close)))"""
        return rank(-1 * self.returns * self.adv(20) * self.vwap * (self.high - self.close))

    def alpha026(self):
        """-1*ts_max(correlation(ts_rank(volume,5),ts_rank(high,5),5),3)"""
        return -1 * ts_max(correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3)

    def alpha027(self):
        """Paper Alpha 27."""
        cond = rank(ts_mean(correlation(rank(self.volume), rank(self.vwap), 6), 2)) > 0.5
        return pd.DataFrame(np.where(cond, -1, 1), index=self.close.index, columns=self.close.columns)

    def alpha028(self):
        """scale(((correlation(adv20,low,5)+((high+low)/2))-close))"""
        return scale(correlation(self.adv(20), self.low, 5) + ((self.high + self.low) / 2) - self.close)

    def alpha029(self):
        """(min(product(rank(rank(scale(log(ts_sum(rank(rank(-1*rank(delta(close-1,5)))),2))))),1),5)+ts_rank(delay(-1*returns,6),5))"""
        base = rank(rank(-1 * rank(delta(self.close - 1, 5))))
        inner = rank(rank(scale(log(ts_sum(ts_min(base, 2), 1)))))
        return ts_min(product(inner, 1), 5) + ts_rank(delay(-1 * self.returns, 6), 5)

    def alpha030(self):
        """(1.0-rank(((sign((close-delay(close,1)))+sign((delay(close,1)-delay(close,2))))+sign((delay(close,2)-delay(close,3))))))*ts_sum(volume,5)/ts_sum(volume,20)"""
        s = (sign(delta(self.close, 1))
             + sign(delay(self.close, 1) - delay(self.close, 2))
             + sign(delay(self.close, 2) - delay(self.close, 3)))
        return (1.0 - rank(s)) * ts_sum(self.volume, 5) / ts_sum(self.volume, 20)

    def alpha031(self):
        """((rank(rank(rank(decay_linear((-1*rank(rank(delta(close,10)))),10))))+rank((-1*delta(close,3))))+sign(scale(correlation(adv20,low,12))))"""
        return (rank(rank(rank(decay_linear(-1 * rank(rank(delta(self.close, 10))), 10))))
                + rank(-1 * delta(self.close, 3))
                + sign(scale(correlation(self.adv(20), self.low, 12))))

    def alpha032(self):
        """Paper Alpha 32."""
        return (scale(ts_mean(self.close, 7) - self.close)
                + 20 * scale(correlation(self.vwap, delay(self.close, 5), 230)))

    def alpha033(self):
        """rank(-1+open/close)"""
        return rank(-1 + self.open / self.close)

    def alpha034(self):
        """rank(((1-rank(stddev(returns,2)/stddev(returns,5)))+(1-rank(delta(close,1)))))"""
        return rank((1 - rank(stddev(self.returns, 2) / stddev(self.returns, 5)))
                    + (1 - rank(delta(self.close, 1))))

    def alpha035(self):
        """((Ts_Rank(volume,32)*(1-Ts_Rank(((close+high)-low),16)))*(1-Ts_Rank(returns,32)))"""
        return (ts_rank(self.volume, 32)
                * (1 - ts_rank(self.close + self.high - self.low, 16))
                * (1 - ts_rank(self.returns, 32)))

    def alpha036(self):
        """(2.21*rank(correlation(close-open,delay(volume,1),15))+0.7*rank(open-close)+0.73*rank(Ts_Rank(delay(-1*returns,6),5))+rank(abs(correlation(vwap,adv20,6)))+0.6*rank((ts_mean(close,200)-open)*(close-open)))"""
        return (2.21 * rank(correlation(self.close - self.open, delay(self.volume, 1), 15))
                + 0.7 * rank(self.open - self.close)
                + 0.73 * rank(ts_rank(delay(-1 * self.returns, 6), 5))
                + rank(abs(correlation(self.vwap, self.adv(20), 6)))
                + 0.6 * rank((ts_mean(self.close, 200) - self.open) * (self.close - self.open)))

    def alpha037(self):
        """(rank(correlation(delay(open-close,1),close,200))+rank(open-close))"""
        return rank(correlation(delay(self.open - self.close, 1), self.close, 200)) + rank(self.open - self.close)

    def alpha038(self):
        """-1*rank(Ts_Rank(close,10))*rank(close/open)"""
        return -1 * rank(ts_rank(self.close, 10)) * rank(self.close / self.open)

    def alpha039(self):
        """(-1*rank(delta(close,7)*(1-rank(decay_linear(volume/adv20,9)))))*(1+rank(ts_sum(returns,250)))"""
        return ((-1 * rank(delta(self.close, 7) * (1 - rank(decay_linear(self.volume / self.adv(20), 9)))))
                * (1 + rank(ts_sum(self.returns, 250))))

    def alpha040(self):
        """-1*rank(stddev(high,10))*correlation(high,volume,10)"""
        return -1 * rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)

    def alpha041(self):
        """(((high*low)^0.5)-vwap)"""
        return ((self.high * self.low) ** 0.5) - self.vwap

    def alpha042(self):
        """rank(vwap-close)/rank(vwap+close)"""
        return rank(self.vwap - self.close) / rank(self.vwap + self.close)

    def alpha043(self):
        """(ts_rank(volume/adv20,20)*ts_rank(-1*delta(close,7),8))"""
        return ts_rank(self.volume / self.adv(20), 20) * ts_rank(-1 * delta(self.close, 7), 8)

    def alpha044(self):
        """-1*correlation(high,rank(volume),5)"""
        return -1 * correlation(self.high, rank(self.volume), 5)

    def alpha045(self):
        """-1*((rank(ts_mean(delay(close,5),20))*correlation(close,volume,2))*rank(correlation(ts_sum(close,5),ts_sum(close,20),2)))"""
        return (-1 * rank(ts_mean(delay(self.close, 5), 20))
                * correlation(self.close, self.volume, 2)
                * rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)))

    def alpha046(self):
        """(0.25<(delay(close,20)-delay(close,10))/10-((delay(close,10)-close)/10))?-1:((((delay(close,20)-delay(close,10))/10-((delay(close,10)-close)/10))<0)?1:((-1*(close-delay(close,1)))))"""
        x = (delay(self.close, 20) - delay(self.close, 10)) / 10 - (delay(self.close, 10) - self.close) / 10
        return pd.DataFrame(
            np.where(x > 0.25, -1, np.where(x < 0, 1, -1 * delta(self.close, 1))),
            index=self.close.index, columns=self.close.columns
        )

    def alpha047(self):
        """((rank(1/close)*volume/adv20)*((high*rank(high-close))/(ts_sum(high,5)/5)))-rank(vwap-delay(vwap,5))"""
        return ((rank(1 / self.close) * self.volume / self.adv(20))
                * (self.high * rank(self.high - self.close) / (ts_mean(self.high, 5)))
                - rank(self.vwap - delay(self.vwap, 5)))

    def alpha049(self):
        """((delay(close,20)-delay(close,10))/10-(delay(close,10)-close)/10)<-0.1?1:(-1*(close-delay(close,1)))"""
        x = (delay(self.close, 20) - delay(self.close, 10)) / 10 - (delay(self.close, 10) - self.close) / 10
        return pd.DataFrame(
            np.where(x < -0.1, 1, -1 * delta(self.close, 1)),
            index=self.close.index, columns=self.close.columns
        )

    def alpha050(self):
        """-1*ts_max(rank(correlation(rank(volume),rank(vwap),5)),5)"""
        return -1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5)

    def alpha051(self):
        """((delay(close,20)-delay(close,10))/10-(delay(close,10)-close)/10)<-0.05?1:(-1*(close-delay(close,1)))"""
        x = (delay(self.close, 20) - delay(self.close, 10)) / 10 - (delay(self.close, 10) - self.close) / 10
        return pd.DataFrame(
            np.where(x < -0.05, 1, -1 * delta(self.close, 1)),
            index=self.close.index, columns=self.close.columns
        )

    def alpha052(self):
        """((-1*ts_min(low,5))+delay(ts_min(low,5),5))*rank((ts_sum(returns,240)-ts_sum(returns,20))/220)*ts_rank(volume,5)"""
        return ((-1 * ts_min(self.low, 5) + delay(ts_min(self.low, 5), 5))
                * rank((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220)
                * ts_rank(self.volume, 5))

    def alpha053(self):
        """-1*delta(1-((high-close)/(close-low+1e-8)),9)"""
        return -1 * delta(1 - (self.high - self.close) / (self.close - self.low + 1e-8), 9)

    def alpha054(self):
        """Paper Alpha 54."""
        return -1 * (self.low - self.close) * (self.open ** 5) / ((self.low - self.high + 1e-8) * (self.close ** 5))

    def alpha055(self):
        """-1*correlation(rank((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12))),rank(volume),6)"""
        x = (self.close - ts_min(self.low, 12)) / (ts_max(self.high, 12) - ts_min(self.low, 12) + 1e-8)
        return -1 * correlation(rank(x), rank(self.volume), 6)

    def alpha056(self):
        """Paper Alpha 56; this formula needs cap but no industry hierarchy."""
        denominator = ts_sum(ts_sum(self.returns, 2), 3).replace(0, np.nan)
        return -1 * rank(ts_sum(self.returns, 10) / denominator) * rank(self.returns * self.cap)

    def alpha057(self):
        """Paper Alpha 57."""
        return -1 * (self.close - self.vwap) / decay_linear(rank(ts_argmax(self.close, 30)), 2)

    def alpha060(self):
        """-1*(2*scale(rank((((close-low)-(high-close))/(high-low+1e-8))*volume))-scale(rank(ts_argmax(close,10))))"""
        x = ((self.close - self.low - (self.high - self.close)) / (self.high - self.low + 1e-8)) * self.volume
        return -1 * (2 * scale(rank(x)) - scale(rank(ts_argmax(self.close, 10))))

    def alpha061(self):
        """(rank(vwap-ts_min(vwap,16))<rank(correlation(vwap,adv180,18)))"""
        return (rank(self.vwap - ts_min(self.vwap, 16)) < rank(correlation(self.vwap, self.adv(180), 17))).astype(float)

    def alpha062(self):
        """((rank(correlation(vwap,ts_sum(adv20,22),9))<rank(((rank(open)+rank(open))<(rank((high+low)/2)+rank(high)))))*-1)"""
        cond1 = rank(correlation(self.vwap, ts_sum(self.adv(20), 22), 9))
        cond2 = rank((rank(self.open) + rank(self.open)) < (rank((self.high + self.low) / 2) + rank(self.high)))
        return -1 * (cond1 < cond2).astype(float)

    def alpha064(self):
        """((rank(correlation(ts_sum((open*0.178404+low*(1-0.178404)),12),ts_sum(adv120,12),16))<rank(delta(((high+low)/2*0.178404+vwap*(1-0.178404)),3.69455)))*-1)"""
        x = ts_sum(self.open * 0.178404 + self.low * (1 - 0.178404), 12)
        y = ts_sum(self.adv(120), 12)
        z = (self.high + self.low) / 2 * 0.178404 + self.vwap * (1 - 0.178404)
        return -1 * (rank(correlation(x, y, 16)) < rank(delta(z, 3))).astype(float)

    def alpha065(self):
        """((rank(correlation(open*0.00817205+vwap*(1-0.00817205),ts_sum(adv60,9),6))<rank(open-ts_min(open,13)))*-1)"""
        x = self.open * 0.00817205 + self.vwap * (1 - 0.00817205)
        return -1 * (rank(correlation(x, ts_sum(self.adv(60), 8), 6)) < rank(self.open - ts_min(self.open, 13))).astype(float)

    def alpha066(self):
        """Paper Alpha 66, with all noninteger windows floored."""
        denominator = self.open - (self.high + self.low) / 2
        x = (self.low - self.vwap) / denominator.replace(0, np.nan)
        return -1 * (rank(decay_linear(delta(self.vwap, 3), 7)) + ts_rank(decay_linear(x, 11), 6))

    def alpha068(self):
        """((Ts_Rank(correlation(rank(high),rank(adv15),9),14)<rank(delta(close*0.518371+low*0.481629,1.06383)))*-1)"""
        x = self.close * 0.518371 + self.low * 0.481629
        return -1 * (ts_rank(correlation(rank(self.high), rank(self.adv(15)), 8), 13) < rank(delta(x, 1))).astype(float)

    def alpha071(self):
        """max(Ts_Rank(decay_linear(correlation(Ts_Rank(close,3),Ts_Rank(adv180,12),18),4),16),Ts_Rank(decay_linear(rank(low+open-2*vwap)^2,16),4))"""
        x = decay_linear(correlation(ts_rank(self.close, 3), ts_rank(self.adv(180), 12), 18), 4)
        y = decay_linear(signedpower(rank(self.low + self.open - 2 * self.vwap), 2), 16)
        return pd.DataFrame(np.maximum(ts_rank(x, 15), ts_rank(y, 4)),
                            index=self.close.index, columns=self.close.columns)

    def alpha072(self):
        """rank(decay_linear(correlation(((high+low)/2),adv40,8.93345),10.1519))/rank(decay_linear(correlation(Ts_Rank(vwap,3.72469),Ts_Rank(volume,18.5188),6.86671),2.95011))"""
        num = rank(decay_linear(correlation((self.high + self.low) / 2, self.adv(40), 8), 10))
        den = rank(decay_linear(correlation(ts_rank(self.vwap, 3), ts_rank(self.volume, 18), 6), 2))
        return num / (den + 1e-8)

    def alpha073(self):
        """(max(rank(decay_linear(delta(vwap,4.72775),2.91864)),Ts_Rank(decay_linear(((delta(open*0.147155+low*0.852845,2.07536))/(open*0.147155+low*0.852845))*-1,3.33012),16.1223))*-1)"""
        x = self.open * 0.147155 + self.low * 0.852845
        a = rank(decay_linear(delta(self.vwap, 4), 2))
        b = ts_rank(decay_linear(-1 * delta(x, 2) / x, 3), 16)
        return -1 * pd.DataFrame(np.maximum(a, b), index=self.close.index, columns=self.close.columns)

    def alpha074(self):
        """((rank(correlation(close,ts_sum(adv30,37.4843),15.1365))<rank(correlation(rank(high*0.0261661+vwap*(1-0.0261661)),rank(volume),11.4791)))*-1)"""
        x = self.high * 0.0261661 + self.vwap * (1 - 0.0261661)
        return -1 * (rank(correlation(self.close, ts_sum(self.adv(30), 37), 15))
                     < rank(correlation(rank(x), rank(self.volume), 11))).astype(float)

    def alpha075(self):
        """(rank(correlation(vwap,volume,4.24304))<rank(correlation(rank(low),rank(adv50),12.4413)))"""
        return (rank(correlation(self.vwap, self.volume, 4))
                < rank(correlation(rank(self.low), rank(self.adv(50)), 12))).astype(float)

    def alpha077(self):
        """min(rank(decay_linear(((high+low)/2+(high-low))-vwap,20)),rank(decay_linear(correlation(((high+low)/2),adv40,3),6)))"""
        x = (self.high + self.low) / 2 - self.vwap
        a = rank(decay_linear(x, 20))
        b = rank(decay_linear(correlation((self.high + self.low) / 2, self.adv(40), 3), 5))
        return pd.DataFrame(np.minimum(a, b), index=self.close.index, columns=self.close.columns)

    def alpha078(self):
        """(rank(correlation(ts_sum(((low*0.352233+vwap*(1-0.352233))),19.7428),ts_sum(adv40,19.7428),6.83313))^rank(correlation(rank(vwap),rank(volume),5.77492)))"""
        x = ts_sum(self.low * 0.352233 + self.vwap * (1 - 0.352233), 19)
        return (rank(correlation(x, ts_sum(self.adv(40), 19), 6))
                ** rank(correlation(rank(self.vwap), rank(self.volume), 5)))

    def alpha081(self):
        """((rank(log(product(rank(rank(correlation(vwap,ts_sum(adv10,49.6054),8.47743))^4),14.9655)))<rank(correlation(rank(vwap),rank(volume),5.07914)))*-1)"""
        x = product(rank(rank(correlation(self.vwap, ts_sum(self.adv(10), 49), 8)) ** 4), 14)
        return -1 * (rank(log(x)) < rank(correlation(rank(self.vwap), rank(self.volume), 5))).astype(float)

    def alpha083(self):
        """(rank(delay(((high-low)/(ts_sum(close,5)/5)),2))*rank(rank(volume)))/((high-low)/(ts_sum(close,5)/5)/(vwap-close+1e-8))"""
        hl_ratio = (self.high - self.low) / (ts_mean(self.close, 5))
        return (rank(delay(hl_ratio, 2)) * rank(rank(self.volume))) / (hl_ratio / (self.vwap - self.close + 1e-8))

    def alpha084(self):
        """SignedPower(Ts_Rank((vwap-ts_max(vwap,15.3217)),20.7127),delta(close,4.96796))"""
        return signedpower(ts_rank(self.vwap - ts_max(self.vwap, 15), 20), delta(self.close, 4))

    def alpha085(self):
        """Paper Alpha 85."""
        x = self.high * 0.876703 + self.close * (1 - 0.876703)
        return (rank(correlation(x, self.adv(30), 9))
                ** rank(correlation(ts_rank((self.high + self.low) / 2, 3), ts_rank(self.volume, 10), 7)))

    def alpha086(self):
        """((Ts_Rank(correlation(close,ts_sum(adv20,14.7444),6.00049),20.4195)<rank(open+close-vwap))*-1)"""
        return -1 * (ts_rank(correlation(self.close, ts_sum(self.adv(20), 14), 6), 20)
                     < rank(self.close - self.vwap)).astype(float)

    def alpha088(self):
        """min(rank(decay_linear(((rank(open)+rank(low))-(rank(high)+rank(close))),8.06882)),Ts_Rank(decay_linear(correlation(Ts_Rank(close,8.44728),Ts_Rank(adv60,13.4132),8.01266),6.65053),2.69758))"""
        x = rank(self.open) + rank(self.low) - rank(self.high) - rank(self.close)
        a = rank(decay_linear(x, 8))
        b = ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(self.adv(60), 20), 8), 6), 2)
        return pd.DataFrame(np.minimum(a, b), index=self.close.index, columns=self.close.columns)

    def alpha092(self):
        """Paper Alpha 92."""
        x = (((self.high + self.low) / 2 + self.close) < (self.low + self.open)).astype(float)
        a = ts_rank(decay_linear(x, 14), 18)
        b = ts_rank(decay_linear(correlation(rank(self.low), rank(self.adv(30)), 7), 6), 6)
        return pd.DataFrame(np.minimum(a, b), index=self.close.index, columns=self.close.columns)

    def alpha094(self):
        """((rank(vwap-ts_min(vwap,11.5783))^Ts_Rank(correlation(Ts_Rank(vwap,19.6462),Ts_Rank(adv60,4.02401),18.0926),2.70756))*-1)"""
        return -1 * (rank(self.vwap - ts_min(self.vwap, 11))
                     ** ts_rank(correlation(ts_rank(self.vwap, 19), ts_rank(self.adv(60), 4), 18), 2))

    def alpha095(self):
        """(rank(open-ts_min(open,12.4105))<Ts_Rank(rank(correlation(ts_sum(((high+low)/2),19.8745),ts_sum(adv40,19.8745),12.9442)),11.7584))"""
        x = ts_sum((self.high + self.low) / 2, 19)
        return (rank(self.open - ts_min(self.open, 12))
                < ts_rank(rank(correlation(x, ts_sum(self.adv(40), 19), 12)) ** 5, 11)).astype(float)

    def alpha096(self):
        """(max(Ts_Rank(decay_linear(correlation(rank(vwap),rank(volume),3.83878),4.16783),8.38151),Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close,7.45404),Ts_Rank(adv60,4.13242),3.65459),12.6556),14.0365),13.4143))*-1)"""
        a = ts_rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 3), 4), 8)
        b = ts_rank(decay_linear(ts_argmax(correlation(ts_rank(self.close, 7), ts_rank(self.adv(60), 4), 3), 12), 14), 13)
        return -1 * pd.DataFrame(np.maximum(a, b), index=self.close.index, columns=self.close.columns)

    def alpha098(self):
        """(rank(decay_linear(correlation(vwap,ts_sum(adv5,26.4719),4.58418),7.18088))-rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open),rank(adv15),20.8187),8.62571),6.95668),8.07206)))"""
        a = rank(decay_linear(correlation(self.vwap, ts_sum(self.adv(5), 26), 4), 7))
        b = rank(decay_linear(ts_rank(ts_argmin(correlation(rank(self.open), rank(self.adv(15)), 20), 8), 6), 8))
        return a - b

    def alpha099(self):
        """((rank(correlation(ts_sum(((high+low)/2),19.8745),ts_sum(adv60,19.8745),8.43743))<rank(correlation(low,volume,6.22617)))*-1)"""
        x = ts_sum((self.high + self.low) / 2, 19)
        return -1 * (rank(correlation(x, ts_sum(self.adv(60), 19), 8))
                     < rank(correlation(self.low, self.volume, 6))).astype(float)

    def alpha101(self):
        """(close-open)/((high-low)+0.001)"""
        return (self.close - self.open) / (self.high - self.low + 0.001)

    # ------------------------------------------------------------------
    # Convenience: run all available alphas
    # ------------------------------------------------------------------

    def compute_all(self) -> dict:
        """
        Compute all implemented alphas.
        Returns a dict: {'alpha001': DataFrame, 'alpha002': DataFrame, ...}
        Silently skips any alpha that raises an exception (e.g., insufficient data).
        """
        results = {}
        for name in sorted(dir(self)):
            if name.startswith("alpha") and callable(getattr(self, name)):
                try:
                    results[name] = getattr(self, name)()
                except Exception as e:
                    results[name] = f"ERROR: {e}"
        return results


def compute_worldquant_composite(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Return an equal-weight rank ensemble of every currently usable formula.

    Prices must already be point-in-time safe. OHLC and VWAP are adjusted with
    the same factor, while share volume remains unadjusted.  ``advN`` follows
    the canonical Alpha101 implementations as N-day average share volume.
    Each published alpha's direction is retained; no signs or weights are fit
    on the replay period.
    """
    required = {"date", "symbol", "open", "high", "low", "close", "volume",
                "amount", "adj_factor", "market_cap"}
    missing = required - set(prices)
    if missing:
        raise ValueError(f"WorldQuant 101 缺少字段：{sorted(missing)}")
    source = prices.sort_values(["date", "symbol"]).copy()
    for column in ["open", "high", "low", "close"]:
        source[column] = source[column] * source.adj_factor
    source["vwap"] = source.amount / source.volume.replace(0, np.nan) * source.adj_factor

    def wide(column):
        return source.pivot(index="date", columns="symbol", values=column).sort_index()

    close = wide("close")
    engine = Alphas(
        wide("open"), wide("high"), wide("low"), close, wide("volume"),
        wide("vwap"), close.pct_change(fill_method=None), wide("market_cap"),
    )
    total = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    count = pd.DataFrame(0, index=close.index, columns=close.columns, dtype="int16")
    audit = []
    implemented = {int(name[5:]) for name in dir(engine) if name.startswith("alpha") and name[5:].isdigit()}
    if implemented != set(USABLE):
        raise RuntimeError(f"Alpha101 实现清单不完整：{sorted(set(USABLE) - implemented)}")
    for number in USABLE:
        name = f"alpha{number:03d}"
        value = getattr(engine, name)().replace([np.inf, -np.inf], np.nan)
        ranked = value.rank(axis=1, pct=True, method="average")
        valid = ranked.notna()
        total = total.add(ranked.fillna(0.0), fill_value=0.0)
        count = count.add(valid.astype("int16"), fill_value=0).astype("int16")
        finite = int(valid.to_numpy().sum())
        audit.append({
            "alpha": f"Alpha#{number}", "status": "usable",
            "finite_observations": finite,
            "coverage": float(finite / valid.size) if valid.size else 0.0,
        })
    composite = total.div(count.where(count > 0))
    result = pd.concat(
        [composite.stack(future_stack=True).rename("worldquant_score"),
         count.div(len(USABLE)).stack(future_stack=True).rename("worldquant_coverage")],
        axis=1,
    ).reset_index()
    result["worldquant_factor_count"] = len(USABLE)
    return result, audit


# ---------------------------------------------------------------------------
# Quick usage demo (runs only when executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    n_dates, n_stocks = 300, 50
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    tickers = [f"STK{i:03d}" for i in range(n_stocks)]

    def rand_df(base=100):
        prices = base + np.random.randn(n_dates, n_stocks).cumsum(axis=0)
        return pd.DataFrame(prices, index=dates, columns=tickers)

    close  = rand_df(100)
    open_  = close * (1 + np.random.randn(n_dates, n_stocks) * 0.005)
    high   = close * (1 + abs(np.random.randn(n_dates, n_stocks) * 0.01))
    low    = close * (1 - abs(np.random.randn(n_dates, n_stocks) * 0.01))
    volume = pd.DataFrame(np.abs(np.random.randn(n_dates, n_stocks)) * 1e6 + 1e6,
                          index=dates, columns=tickers)
    vwap   = (high + low + close) / 3
    returns = close.pct_change()

    alphas = Alphas(open_, high, low, close, volume, vwap, returns)

    print("=== WorldQuant 101 Alphas Demo ===\n")
    for name in ["alpha001", "alpha002", "alpha003", "alpha041", "alpha101"]:
        result = getattr(alphas, name)()
        print(f"{name}  |  last row mean: {result.iloc[-1].mean():.6f}")

    print("\nAll alphas computed successfully.")
