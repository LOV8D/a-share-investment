import numpy as np
import pandas as pd
from .data import cutoff
from .worldquant101 import compute_worldquant_composite

PRICE_FEATURES = ["momentum_5", "momentum_20", "momentum_60", "reversal_1", "volatility_20", "downside_20", "volume_ratio", "turnover_20", "amihud_20", "intraday_return", "range_20", "ma_distance", "log_size"]
FUND_FEATURES = ["earnings_yield", "book_to_price", "cashflow_yield", "roe", "gross_margin", "revenue_growth", "profit_growth", "leverage", "accruals", "fund_age"]
TEXT_FEATURES = ["event_sentiment", "event_intensity", "event_risk", "event_disagreement", "event_coverage"]
WORLDQUANT_FEATURES = ["worldquant_score"]

def attach_fundamentals(panel, fundamentals):
    parts = []
    for symbol, p in panel.groupby("symbol", sort=False):
        f = fundamentals.loc[fundamentals.symbol == symbol].sort_values(["available_at", "report_period"]).copy()
        if f.empty:
            for col in fundamentals.columns.difference(["symbol", "available_at"]):
                p[col] = pd.NaT if col == "report_period" else np.nan
            p["fund_available_at"] = pd.NaT
            parts.append(p)
            continue
        # A late restatement of an older period must not replace a newer report.
        f = f[f.report_period >= f.report_period.cummax()]
        f = f.rename(columns={"available_at": "fund_available_at"}).drop(columns="symbol")
        parts.append(pd.merge_asof(p.sort_values("cutoff"), f.sort_values("fund_available_at"), left_on="cutoff", right_on="fund_available_at", direction="backward"))
    return pd.concat(parts, ignore_index=True)

def attach_events(panel, events, cfg):
    for col in TEXT_FEATURES:
        panel[col] = 0.
    if events.empty:
        return panel
    e = events.copy()
    e["usable_at"] = e[["published_at", "first_seen_at", "processed_at"]].max(axis=1)
    e = e.sort_values("usable_at").drop_duplicates(["symbol", "text_sha256"])
    source_weight = {"filing": 1., "policy": .9, "news": .7, "forum": .2}
    for symbol, rows in panel.groupby("symbol").groups.items():
        se = e[e.symbol == symbol]
        if se.empty:
            continue
        # Chunk dates and retain only a 30-day event window to bound real-data memory.
        for start in range(0,len(rows),64):
            indices=rows[start:start+64]
            cuts = panel.loc[indices, "cutoff"].astype("int64").to_numpy()/86400e9
            all_times=se.usable_at.astype("int64").to_numpy()/86400e9
            active=se[(all_times>=cuts.min()-30)&(all_times<=cuts.max())]
            if active.empty:continue
            times = active.usable_at.astype("int64").to_numpy()/86400e9
            age = cuts[:, None]-times[None, :]
            mask = (age >= 0) & (age <= 30)
            base = (active.confidence*active.materiality*active.novelty*active.source_type.map(source_weight).fillna(.3)).to_numpy()
            w = np.exp(-np.log(2)*np.clip(age,0,30)/cfg.event_half_life_days)*base[None,:]*mask
            sentiment=active.sentiment.to_numpy()
            count=mask.sum(axis=1)
            mean=(mask@sentiment)/np.maximum(count,1)
            variance=(mask@(sentiment**2))/np.maximum(count,1)-mean**2
            panel.loc[indices, TEXT_FEATURES] = np.column_stack([w@sentiment,w.sum(axis=1),w@np.maximum(-sentiment,0),np.sqrt(np.maximum(variance,0)),(count>0).astype(float)])
    return panel

def build_panel(ds, cfg):
    prices = ds["prices"].sort_values(["symbol", "date"]).copy()
    # Keep full lookback history for every name that appears in the requested
    # universe, while avoiding thousands of irrelevant A-share histories.
    prices = prices[prices.symbol.isin(ds["universe"].symbol.unique())].copy()
    # Reindex each symbol to the actual exchange calendar, never shift across missing days.
    sessions = pd.DatetimeIndex(ds["calendar"].date.sort_values())
    sessions = sessions[(sessions >= prices.date.min()) & (sessions <= prices.date.max())]
    frames = []
    for symbol, frame in prices.groupby("symbol", sort=False):
        f = frame.set_index("date").reindex(sessions).rename_axis("date").reset_index()
        f["symbol"] = symbol
        f["cutoff"] = f.date.map(lambda d: cutoff(d, cfg.decision_time))
        # A Friday signal may legitimately be generated on a weekend before
        # Monday's open.  In forward-paper mode the manifest records the actual
        # decision timestamp, so filings/news first seen after the signal-day cutoff can be
        # used without pretending they were known on Friday.  Historical rows
        # keep their original point-in-time cutoff.
        if ds.manifest.get("mode") == "forward_paper" and ds.manifest.get("decision_as_of"):
            decision_as_of = pd.Timestamp(ds.manifest["decision_as_of"])
            if decision_as_of.tzinfo is None:
                decision_as_of = decision_as_of.tz_localize("Asia/Shanghai")
            else:
                decision_as_of = decision_as_of.tz_convert("Asia/Shanghai")
            f.loc[f.date == prices.date.max(), "cutoff"] = decision_as_of
        adjusted = f.close * f.adj_factor
        r = adjusted.pct_change(fill_method=None)
        for n in [5,20,60]:
            f[f"momentum_{n}"] = adjusted / adjusted.shift(n) - 1
        f["reversal_1"] = -r
        f["volatility_20"] = r.rolling(20, min_periods=15).std() * np.sqrt(252)
        f["downside_20"] = r.clip(upper=0).pow(2).rolling(20, min_periods=15).mean().pow(.5) * np.sqrt(252)
        f["volume_ratio"] = f.volume.rolling(5).mean() / f.volume.rolling(20).mean().replace(0, np.nan)
        f["turnover_20"] = f.turnover_rate.rolling(20).mean()
        f["amihud_20"] = (r.abs() / f.amount.replace(0, np.nan) * 1e8).rolling(20).mean()
        f["intraday_return"] = f.close / f.open - 1
        f["range_20"] = ((f.high - f.low) / f.prev_close).rolling(20).mean()
        f["ma_distance"] = adjusted / adjusted.rolling(20).mean() - 1
        f["log_size"] = np.log(f.market_cap)
        f["adv20"] = f.amount.rolling(20, min_periods=15).mean()
        # Signal t -> buy open(t+1), evaluate open(t+1+h); future unavailable labels stay NaN.
        adj_open = f.open * f.adj_factor
        entry, leave = adj_open.shift(-1), adj_open.shift(-(cfg.label_horizon + 1))
        f["forward_return"] = leave / entry - 1
        f["label_end"] = f.date.shift(-(cfg.label_horizon + 1))
        f["label_available_at"] = f.available_at.shift(-(cfg.label_horizon + 1))
        valid_label = (f.suspended.shift(-1) == 0) & (f.suspended.shift(-(cfg.label_horizon + 1)) == 0)
        f.loc[~valid_label, "forward_return"] = np.nan
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    wq, wq_audit = compute_worldquant_composite(panel)
    panel = panel.merge(wq, on=["date", "symbol"], how="left")
    panel = panel.merge(ds["universe"], on=["date", "symbol"], how="inner")
    late = (panel.known_at > panel.cutoff) | (panel.available_at > panel.cutoff)
    if late.any():
        raise ValueError("部分成分/日线数据在设定截止时刻尚不可用；请补齐当时版本或调整统一决策时间，不能静默剔除晚到股票")
    panel = panel[(panel.known_at <= panel.cutoff) & (panel.available_at <= panel.cutoff)].copy()
    panel = attach_fundamentals(panel, ds["fundamentals"])
    panel["earnings_yield"] = panel.net_profit_ttm / panel.market_cap
    panel["book_to_price"] = panel.book_equity / panel.market_cap
    panel["cashflow_yield"] = panel.operating_cashflow_ttm / panel.market_cap
    panel["leverage"] = panel.total_liabilities / panel.total_assets.replace(0, np.nan)
    panel["accruals"] = (panel.net_profit_ttm - panel.operating_cashflow_ttm) / panel.total_assets.replace(0, np.nan)
    panel["fund_age"] = (panel.cutoff - pd.to_datetime(panel.fund_available_at, utc=True)).dt.total_seconds() / 86400
    panel = attach_events(panel, ds["events"], cfg)
    raw = PRICE_FEATURES + FUND_FEATURES
    panel["missing_fraction"] = panel[raw].replace([np.inf,-np.inf], np.nan).isna().mean(axis=1)
    panel["eligible"] = (panel.suspended.eq(0) & panel.is_st.eq(0) & panel.listed_days.ge(cfg.min_listed_days) & panel.adv20.ge(cfg.min_liquidity) & panel.missing_fraction.le(.3) & panel.fund_age.le(550) & panel.worldquant_coverage.ge(.70))
    # Date-local transformations use no fitted full-history statistics.
    for col in raw + WORLDQUANT_FEATURES + TEXT_FEATURES:
        s = panel[col].replace([np.inf, -np.inf], np.nan)
        panel[f"x_{col}"] = s.groupby(panel.date).rank(pct=True).sub(.5).fillna(0.)
    panel["target"] = panel.groupby("date").forward_return.transform(lambda x: x - x.mean())
    panel["relevance"] = panel.groupby("date").forward_return.rank(pct=True).mul(4.999).fillna(0).astype(int)
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    panel.attrs["worldquant_audit"] = wq_audit
    return panel
