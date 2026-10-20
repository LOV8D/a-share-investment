from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import sys
if sys.platform == "darwin":
    # Reuse the OpenMP library bundled with scikit-learn; no system Homebrew mutation.
    import ctypes
    import sklearn
    from pathlib import Path
    omp = Path(sklearn.__file__).parent / ".dylibs" / "libomp.dylib"
    if omp.exists():
        ctypes.CDLL(str(omp), mode=ctypes.RTLD_GLOBAL)
from lightgbm import LGBMRanker
from .features import PRICE_FEATURES, FUND_FEATURES, TEXT_FEATURES, WORLDQUANT_FEATURES
from .data import cutoff

# The production score is deliberately anchored to a small, economically
# interpretable factor prior.  ML models remain challengers until forward
# evidence shows that their extra turnover earns its cost.
PRIOR_POSITIVE = [
    "earnings_yield", "book_to_price", "cashflow_yield",
    "roe", "gross_margin", "reversal_1",
]
PRIOR_NEGATIVE = [
    "accruals", "leverage", "volatility_20",
    "downside_20", "turnover_20", "range_20",
]

def robust_prior(frame):
    positive = frame[[f"x_{c}" for c in PRIOR_POSITIVE]].mean(axis=1)
    negative = frame[[f"x_{c}" for c in PRIOR_NEGATIVE]].mean(axis=1)
    return positive - negative

def production_scores(frame, cfg):
    """Transparent structured ensemble plus bounded, point-in-time text signal."""
    d = frame.copy()
    d["prior_raw"] = robust_prior(d)
    d["prior_score"] = d.groupby("date").prior_raw.rank(pct=True)
    d["worldquant_score_rank"] = d.groupby("date").worldquant_score.rank(pct=True)
    d["structured_score"] = ((1-cfg.worldquant_weight) * d.prior_score
                             + cfg.worldquant_weight * d.worldquant_score_rank)
    # GLM extracts events; deterministic aggregation keeps the portfolio rule
    # auditable. No event means zero active text weight rather than invented news.
    d["text_raw"] = d.event_sentiment - d.event_risk - .25*d.event_disagreement
    d["text_score"] = d.groupby("date").text_raw.rank(pct=True)
    active = cfg.enable_text_alpha & d.event_coverage.gt(0)
    d["text_model_active"] = active
    d["combined_score"] = d.structured_score
    d.loc[active, "combined_score"] = ((1-cfg.text_alpha_weight)*d.loc[active, "structured_score"]
                                       + cfg.text_alpha_weight*d.loc[active, "text_score"])
    d["score"] = d.groupby("date").combined_score.rank(pct=True).mul(100)
    d["rank"] = d.groupby("date").score.rank(ascending=False, method="first").astype(int)
    return d

@dataclass
class Ensemble:
    ranker: object
    ridge: object
    text_model: object
    columns: list
    text_columns: list
    trained_through: str
    trained_at: str
    rows: int

def training_slice(panel, signal_date, cfg):
    dates = pd.DatetimeIndex(panel.date.unique()).sort_values()
    prior = dates[dates < pd.Timestamp(signal_date)]
    if len(prior) <= cfg.embargo_days:
        return panel.iloc[:0]
    last_allowed = prior[-(cfg.embargo_days + 1)]
    first_allowed = prior[max(0, len(prior) - cfg.train_window_days)]
    frame = panel[(panel.date >= first_allowed) & (panel.date <= last_allowed) & (panel.label_end < last_allowed) & (panel.label_available_at < cutoff(last_allowed, cfg.decision_time)) & panel.eligible & panel.target.notna()].copy()
    return frame.sort_values(["date", "symbol"])

def fit_model(train, signal_date, cfg):
    if train.date.nunique() < cfg.min_train_days:
        raise ValueError(f"可用训练日不足 {cfg.min_train_days}，当前 {train.date.nunique()}")
    cols = [f"x_{c}" for c in PRICE_FEATURES + FUND_FEATURES + WORLDQUANT_FEATURES]
    tcols = [f"x_{c}" for c in TEXT_FEATURES]
    params = dict(objective="lambdarank", n_estimators=80, learning_rate=.04, num_leaves=15, max_depth=4, min_child_samples=100, colsample_bytree=.85, reg_lambda=10., random_state=cfg.seed, n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True)
    ranker = LGBMRanker(**params)
    ranker.fit(train[cols], train.relevance, group=train.groupby("date", sort=False).size().to_numpy())
    ridge = Ridge(alpha=100.)
    ridge.fit(train[cols], train.target.clip(-.2, .2))
    text_model = None
    if cfg.enable_text_alpha and train.event_coverage.mean() >= cfg.min_text_coverage:
        text_model = LGBMRanker(**params)
        text_model.fit(train[cols+tcols], train.relevance, group=train.groupby("date", sort=False).size().to_numpy())
    return Ensemble(ranker, ridge, text_model, cols, tcols, str(train.date.max().date()), str(signal_date)[:10], len(train))

def predict(model, today, cfg):
    d = production_scores(today, cfg)
    d["tree_raw"] = model.ranker.predict(d[model.columns])
    d["ridge_raw"] = model.ridge.predict(d[model.columns])
    d["ml_challenger_score"] = .7 * d.tree_raw.rank(pct=True) + .3 * d.ridge_raw.rank(pct=True)
    d["model_train_end"] = model.trained_through
    return d.sort_values("rank")

def walk_forward(panel, cfg, progress=None):
    scores, folds, model = [], [], None
    dates = sorted(panel.date.unique())
    last_fit = -cfg.refit_every
    for i, date in enumerate(dates):
        if i < cfg.min_train_days + 65 + cfg.label_horizon + cfg.embargo_days:
            continue
        today = panel[panel.date == date]
        if today.empty:
            continue
        if model is None or i-last_fit >= cfg.refit_every:
            train = training_slice(panel, date, cfg)
            if train.date.nunique() < cfg.min_train_days:
                continue
            model = fit_model(train, date, cfg)
            last_fit = i
            folds.append({"signal_start": str(date)[:10], "train_start": str(train.date.min())[:10], "train_end": model.trained_through, "latest_label_end": str(train.label_end.max())[:10], "train_days": train.date.nunique(), "rows": len(train), "text_active": model.text_model is not None})
            if progress:
                progress(f"滚动训练 {str(date)[:10]} · {len(folds)} 个窗口", i/len(dates))
        scores.append(predict(model, today, cfg))
    if not scores:
        raise ValueError("没有可用的样本外预测。请增加历史数据（默认至少约 340 个交易日）。")
    # Refit latest point using only fully matured historical labels for tomorrow's plan.
    latest = pd.Timestamp(dates[-1])
    final = fit_model(training_slice(panel, latest, cfg), latest, cfg)
    latest_score = predict(final, panel[panel.date == latest], cfg)
    return pd.concat(scores, ignore_index=True), folds, final, latest_score

def explanations(model, latest):
    # Exact LightGBM TreeSHAP contributions, for the tree component only.
    values = model.ranker.booster_.predict(latest[model.columns], pred_contrib=True)
    result = {}
    for symbol, row in zip(latest.symbol, values):
        order = np.argsort(np.abs(row[:-1]))[-4:][::-1]
        result[symbol] = [{"factor": model.columns[i][2:], "contribution": round(float(row[i]), 5)} for i in order]
    return result
