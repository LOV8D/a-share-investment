from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .schema import SCHEMAS, REQUIRED, DESCRIPTIONS

def timestamp(values):
    # Naive vendor times are explicitly interpreted as China local time.
    def one(v):
        if pd.isna(v) or str(v).strip() == "":
            return pd.NaT
        t = pd.Timestamp(v)
        return t.tz_localize("Asia/Shanghai") if t.tzinfo is None else t.tz_convert("Asia/Shanghai")
    return pd.to_datetime(values.map(one), utc=True).dt.tz_convert("Asia/Shanghai")

def cutoff(date, time="20:00:00"):
    return pd.Timestamp(f"{str(date)[:10]} {time}", tz="Asia/Shanghai")

def read_table(folder, name):
    folder = Path(folder)
    for ext in ("parquet", "csv"):
        path = folder / f"{name}.{ext}"
        if path.exists():
            frame = pd.read_parquet(path) if ext == "parquet" else pd.read_csv(path, dtype={"symbol": str}, keep_default_na=True)
            for col in ("date", "report_period", "start_date", "end_date"):
                if col in frame:
                    frame[col] = pd.to_datetime(frame[col], errors="raise").dt.normalize()
            for col in ("available_at", "known_at", "published_at", "first_seen_at", "processed_at"):
                if col in frame:
                    frame[col] = timestamp(frame[col])
            return frame
    return pd.DataFrame(columns=SCHEMAS[name])

@dataclass
class Dataset:
    folder: Path
    tables: dict
    manifest: dict
    quality: dict
    def __getitem__(self, key):
        return self.tables[key]

def load_dataset(folder, strict=True):
    folder = Path(folder).resolve()
    tables = {key: read_table(folder, key) for key in SCHEMAS}
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"mode": "real", "source": "user export"}
    quality = validate(tables, folder)
    if strict and quality["errors"]:
        raise ValueError("数据校验失败：" + "；".join(quality["errors"][:10]))
    return Dataset(folder, tables, manifest, quality)

def load_daily_dataset(folder, lookback_sessions=860, strict=True):
    """Read only the rolling window needed for a daily refit.

    The full immutable history remains on disk and is still hashed in the run
    manifest.  Filtering at Parquet read time keeps the cloud worker's memory
    bounded without changing the model's configured training window.
    """
    folder = Path(folder).resolve()
    calendar = read_table(folder, "calendar")
    price_path = next((folder/f"prices.{ext}" for ext in ("parquet","csv") if (folder/f"prices.{ext}").exists()), None)
    if price_path is None:
        return load_dataset(folder, strict)
    dates = pd.read_parquet(price_path, columns=["date"])["date"] if price_path.suffix==".parquet" else pd.read_csv(price_path,usecols=["date"])["date"]
    latest = pd.to_datetime(dates, errors="raise").max().normalize()
    sessions = pd.DatetimeIndex(calendar.loc[calendar.date<=latest,"date"].sort_values().unique())
    start = sessions[max(0, len(sessions)-int(lookback_sessions))]

    def windowed(name):
        path = folder/f"{name}.parquet"
        if path.exists() and name in {"prices","universe","benchmark","actions","auction"}:
            try:
                frame = pd.read_parquet(path, filters=[("date", ">=", start)])
                for col in ("date","report_period","start_date","end_date"):
                    if col in frame:frame[col]=pd.to_datetime(frame[col],errors="raise").dt.normalize()
                for col in ("available_at","known_at","published_at","first_seen_at","processed_at"):
                    if col in frame:frame[col]=timestamp(frame[col])
                return frame
            except Exception:
                pass
        frame=read_table(folder,name)
        if name in {"prices","universe","benchmark","actions","auction"} and len(frame):
            frame=frame[frame.date>=start].copy()
        return frame

    tables={name:(calendar[calendar.date>=start].copy() if name=="calendar" else windowed(name)) for name in SCHEMAS}
    manifest_path=folder/"manifest.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {"mode":"real","source":"user export"}
    quality=validate(tables,folder)
    if strict and quality["errors"]:
        raise ValueError("数据校验失败："+"；".join(quality["errors"][:10]))
    manifest=manifest|{"daily_window_start":str(pd.Timestamp(start).date()),"daily_window_sessions":len(sessions[sessions>=start])}
    return Dataset(folder,tables,manifest,quality)

def validate(tables, folder=None):
    errors, warnings, inventory = [], [], []
    for key, cols in SCHEMAS.items():
        d = tables[key]
        exists = folder is None or any((Path(folder) / f"{key}.{ext}").exists() for ext in ("csv", "parquet"))
        missing = sorted(set(cols) - set(d.columns))
        if key in REQUIRED and (not exists or (d.empty and key != "actions")):
            errors.append(f"{key}: 缺少必需数据")
        if not d.empty and missing:
            errors.append(f"{key}: 缺少字段 {missing}")
        inventory.append({"name": key, "description": DESCRIPTIONS[key], "rows": len(d), "required": key in REQUIRED, "status": "missing" if not exists else ("error" if missing else "ready"), "columns": cols})
        if d.empty or missing:
            continue
        keys = {"calendar": ["date"], "prices": ["date", "symbol"], "universe": ["date", "symbol"], "benchmark": ["date"], "fundamentals": ["symbol", "report_period", "available_at"], "events": ["event_id"], "documents": ["document_id"], "actions": ["date", "symbol"], "holdings": ["symbol"], "auction": ["date", "symbol"]}.get(key)
        if keys and d.duplicated(keys).any():
            errors.append(f"{key}: 重复主键 {keys}")
        if "symbol" in d and not d.symbol.astype(str).str.fullmatch(r"\d{6}\.(SH|SZ)").all():
            errors.append(f"{key}: 证券代码必须为 000001.SZ / 600000.SH 格式")
        for col in ("available_at", "known_at", "published_at", "first_seen_at", "processed_at"):
            if col in d and d[col].isna().any():
                errors.append(f"{key}: {col} 不得缺失")
    p, u = tables["prices"], tables["universe"]
    if len(p) and not (set(SCHEMAS["prices"]) - set(p)):
        for col in ["open", "high", "low", "close", "prev_close", "volume", "amount", "adj_factor", "market_cap", "float_market_cap", "turnover_rate", "min_order", "lot_step", "listed_days"]:
            if not pd.api.types.is_numeric_dtype(p[col]) or not np.isfinite(pd.to_numeric(p[col],errors="coerce")).all():
                errors.append(f"prices: {col} 必须为完整数值")
        if not errors:
            if (p[["close", "prev_close", "adj_factor", "market_cap", "min_order", "lot_step"]] <= 0).any().any():
                errors.append("prices: 价格、复权因子、市值或最小申报数量不合法")
            if (p[["volume", "amount", "turnover_rate"]] < 0).any().any():
                errors.append("prices: 成交量/额/换手率不能为负")
            active = p.loc[p.suspended == 0]
            if ((active.high < active[["open", "close"]].max(axis=1)) | (active.low > active[["open", "close"]].min(axis=1)) | (active.low <= 0)).any():
                errors.append("prices: OHLC 不满足价格关系")
            if not p.suspended.isin([0, 1]).all() or not p.is_st.isin([0, 1]).all():
                errors.append("prices: suspended / is_st 必须为 0/1")
            if ((p.min_order%1!=0)|(p.lot_step%1!=0)|(p.listed_days%1!=0)).any():
                errors.append("prices: 申报数量、递增单位与上市天数必须为整数")
            if p.industry.isna().any():
                errors.append("prices: 行业不能为空，否则无法控制行业权重")
        if not set(SCHEMAS["universe"]) - set(u) and len(u):
            weights=pd.to_numeric(u.weight,errors="coerce")
            if not np.isfinite(weights).all() or (weights<=0).any():
                errors.append("universe: 成分权重必须为正的有限数值")
            missing_rows = u.merge(p[["date", "symbol"]], on=["date", "symbol"], how="left", indicator=True)
            if (missing_rows._merge == "left_only").any():
                errors.append("universe: 部分历史成分缺少当日日线，禁止静默剔除")
            counts = u.groupby("date").size()
            if counts.min() != 300 or counts.max() != 300:
                verified_transition = False
                if folder is not None:
                    manifest_path = Path(folder) / "manifest.json"
                    if manifest_path.exists():
                        try:
                            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
                            expected = pd.DatetimeIndex(
                                pd.to_datetime(
                                    metadata.get("universe", {}).get("temporary_298_dates", []),
                                    errors="coerce",
                                )
                            ).dropna()
                            affected = counts[counts != 300]
                            verified_transition = (
                                len(expected) > 0
                                and counts.min() == 298
                                and counts.max() == 300
                                and (affected == 298).all()
                                and set(affected.index) == set(expected)
                            )
                        except (OSError, ValueError, TypeError, json.JSONDecodeError):
                            verified_transition = False
                if verified_transition:
                    warnings.append(
                        f"成分数范围 {counts.min()}–{counts.max()}；"
                        f"{len(expected)} 个 298 只交易日已在 manifest 标记为官方调样过渡期"
                    )
                else:
                    warnings.append(f"成分数范围 {counts.min()}–{counts.max()}；请核对缺失或临时变更")
            sums = weights.groupby(u.date).sum()
            if ((sums - 1).abs() > .005).any():
                errors.append("universe: 逐日权重合计必须约等于 1（不是百分数 100）")
        dates = pd.DatetimeIndex(tables["calendar"].date) if len(tables["calendar"]) else []
        if len(dates) and not p.date.isin(dates).all():
            errors.append("prices: 存在交易日历以外的日期")
        if len(tables["benchmark"]) and not p.date.isin(tables["benchmark"].date).all():
            errors.append("benchmark: 缺少价格历史覆盖内的基准")
    f = tables["fundamentals"]
    if len(f) and not set(SCHEMAS["fundamentals"])-set(f):
        if (f.report_period > f.available_at.dt.tz_localize(None).dt.normalize()).any():
            errors.append("fundamentals: 财报可用时间早于报告期末")
    b=tables["benchmark"]
    if len(b) and not set(SCHEMAS["benchmark"])-set(b):
        values=b[["close","total_return_close"]].apply(pd.to_numeric,errors="coerce")
        if not np.isfinite(values).all().all() or (values<=0).any().any():
            errors.append("benchmark: 价格和全收益指数必须为正的有限数值")
    e = tables["events"]
    if len(e) and not set(SCHEMAS["events"]) - set(e):
        for col, lo, hi in [("sentiment", -1, 1), ("materiality", 0, 1), ("confidence", 0, 1), ("novelty", 0, 1)]:
            if not pd.to_numeric(e[col], errors="coerce").between(lo, hi).all():
                errors.append(f"events: {col} 超出 [{lo}, {hi}]")
        if e.evidence.fillna("").str.len().lt(4).any():
            errors.append("events: 缺少可追溯的原文证据")
    if tables["documents"].empty and e.empty:
        warnings.append("按当前方案不做历史文本回测；模拟盘启动后再记录实时事件")
    if len(tables["actions"]) and not set(SCHEMAS["actions"])-set(tables["actions"]):
        a = tables["actions"]
        values=a[["share_multiplier","cash_per_share"]].apply(pd.to_numeric,errors="coerce")
        if not np.isfinite(values).all().all() or (values.share_multiplier <= 0).any() or (values.cash_per_share < 0).any():
            errors.append("actions: 股数乘数或现金分红不合法")
    if len(tables["holdings"]) and not set(SCHEMAS["holdings"])-set(tables["holdings"]):
        h = tables["holdings"][["shares","sellable_shares","cost_price"]].apply(pd.to_numeric,errors="coerce")
        if not np.isfinite(h).all().all() or ((h.shares < 0) | (h.sellable_shares < 0) | (h.sellable_shares > h.shares) | (h.shares % 1 != 0) | (h.sellable_shares % 1 != 0)).any():
            errors.append("holdings: 持仓/可卖股数不合法")
    fees = tables["fees"]
    if len(fees) and not set(SCHEMAS["fees"]) - set(fees):
        rates=fees[["sell_stamp_rate","transfer_rate"]].apply(pd.to_numeric,errors="coerce")
        if not np.isfinite(rates).all().all():
            errors.append("fees: 税费不得缺失或为无穷值")
        if ((fees.start_date > fees.end_date) | (rates.sell_stamp_rate < 0) | (rates.transfer_rate < 0)).any():
            errors.append("fees: 区间或税费不合法")
        if len(p):
            for date in p.date.unique():
                if len(fees[(fees.start_date <= date) & (fees.end_date >= date)]) != 1:
                    errors.append("fees: 费率区间必须完整覆盖交易日期且不得重叠")
                    break
    return {"errors": errors, "warnings": warnings, "inventory": inventory, "passed": not errors}

def dataset_hash(folder):
    result = {}
    for p in sorted(Path(folder).iterdir()):
        if p.suffix in (".csv", ".parquet", ".json"):
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            result[p.name] = h.hexdigest()
    return result

def write_templates(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, cols in SCHEMAS.items():
        pd.DataFrame(columns=cols).to_csv(folder / f"{name}.csv", index=False)
    (folder / "schema.json").write_text(json.dumps({n: {"columns": c, "description": DESCRIPTIONS[n], "required": n in REQUIRED} for n,c in SCHEMAS.items()}, ensure_ascii=False, indent=2))
