"""Attach the saved model's importance to an existing report, without rerunning it."""
import argparse
import json
from pathlib import Path

from ashare.config import ROOT
from ashare.importance import xgboost_importance
from ashare.pipeline import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        parser.error("非法运行编号")
    folder = ROOT / "runs" / args.run_id
    report = json.loads((folder / "report.json").read_text())
    fold = report["folds"][-1]
    model_root=ROOT / "runs/xgb-models"
    if report.get("config",{}).get("factor_zoo_enabled"):model_root=model_root/"zoo-v1"
    elif (model_root/"legacy").exists():model_root=model_root/"legacy"
    path = model_root / f"xgb-{fold['signal_start']}.json"
    # models imports the shared macOS OpenMP library before XGBoost is loaded.
    import ashare.models  # noqa: F401
    from xgboost import XGBRanker
    model = XGBRanker()
    model.load_model(path)
    artifact = xgboost_importance(model, **fold)
    if artifact["features"] != report["model_card"]["features"]:
        raise ValueError("保存的模型与报告特征数不一致")
    report["feature_importance"] = artifact
    write_json(folder / "report.json", report)
    write_json(folder / "factor_importance.json", artifact)
    print(json.dumps({"run_id": args.run_id, "features": artifact["features"],
                      "top5": artifact["rows"][:5]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
