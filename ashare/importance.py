"""Export auditable XGBoost importance without retraining on evaluation data."""
import hashlib
from .factor_definitions import definition_for


def xgboost_importance(model, **metadata):
    if "rows" in metadata:
        metadata["train_rows"] = metadata.pop("rows")
    booster = model.get_booster()
    gains = booster.get_score(importance_type="gain")
    totals = booster.get_score(importance_type="total_gain")
    counts = booster.get_score(importance_type="weight")
    columns = booster.feature_names
    if not columns:
        raise ValueError("模型没有特征名称，无法可靠关联因子")
    denominator = sum(gains.values())
    rows = []
    for column in columns:
        gain = float(gains.get(column, 0.))
        rows.append({
            "factor": column.removeprefix("x_"), "feature": column,
            **definition_for(column),
            "family": "Qlib Alpha158" if column.startswith("x_q158_") else "WorldQuant 101" if column.startswith("x_wq") else "核心因子",
            "gain": gain, "share": gain / denominator if denominator else 0.,
            "total_gain": float(totals.get(column, 0.)),
            "splits": int(counts.get(column, 0)),
        })
    rows.sort(key=lambda row: (-row["gain"], row["factor"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return {
        "method": "normalized_average_gain", "algorithm": "XGBoost",
        "definition": "各特征在树分裂中的平均损失改善（gain），除以所有特征的gain之和；未使用特征记为0。",
        "scope": "latest_training_window", "features": len(rows), "rows": rows,
        "model_sha256": hashlib.sha256(booster.save_raw(raw_format="json")).hexdigest(),
        **metadata,
    }
