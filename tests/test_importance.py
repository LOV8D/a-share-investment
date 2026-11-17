import json
import numpy as np
import pandas as pd

# Load the common OpenMP runtime before importing XGBoost on macOS.
import ashare.models  # noqa: F401
from xgboost import XGBRanker
from ashare.importance import xgboost_importance


def test_saved_importance_includes_unused_features_and_reproduces_after_reload(tmp_path):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(120, 105)), columns=[f"x_wq{i:03}" for i in range(105)])
    frame["x_constant"] = 0.
    target = (frame.x_wq000 > 0).astype(int)
    model = XGBRanker(n_estimators=8, max_depth=2, min_child_weight=1, n_jobs=1)
    model.fit(frame, target, group=np.repeat(10, 12))
    artifact = xgboost_importance(model, rows=120, train_end="2026-09-24")
    assert all("definition" in row and "formula" in row for row in artifact["rows"])
    assert artifact["features"] == 106 and artifact["train_rows"] == 120
    assert abs(sum(row["share"] for row in artifact["rows"]) - 1) < 1e-12
    assert artifact["rows"][0]["factor"] == "wq000"
    constant = next(row for row in artifact["rows"] if row["factor"] == "constant")
    assert constant["share"] == constant["splits"] == 0
    path = tmp_path / "model.json"
    model.save_model(path)
    restored = XGBRanker()
    restored.load_model(path)
    assert artifact == xgboost_importance(restored, rows=120, train_end="2026-09-24")
    json.dumps(artifact, allow_nan=False)


def test_importance_download_is_scoped_to_run(tmp_path, monkeypatch):
    import ashare.web as web
    from fastapi.testclient import TestClient
    monkeypatch.setattr(web, "ROOT", tmp_path)
    folder = tmp_path / "runs/study-1"
    folder.mkdir(parents=True)
    (folder / "factor_importance.json").write_text('{"features":106}')
    client = TestClient(web.app)
    response = client.get("/api/download/study-1/factor_importance.json")
    assert response.status_code == 200
    assert response.json()["features"] == 106
    assert client.get("/api/download/study-1/.env").status_code == 400
