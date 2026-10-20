"""Explicit vendor column/unit conversion, never guess ambiguous vendor field names."""
import json
from pathlib import Path
import pandas as pd
from .schema import SCHEMAS

def convert(source,table,mapping_path,dest):
    if table not in SCHEMAS:raise ValueError("未知数据表")
    mapping=json.loads(Path(mapping_path).read_text())
    source=Path(source)
    df=pd.read_parquet(source) if source.suffix==".parquet" else pd.read_csv(source,dtype=str,encoding=mapping.get("encoding","utf-8-sig"))
    columns=mapping["columns"] # canonical -> source
    absent=set(columns.values())-set(df)
    if absent:raise ValueError(f"源文件缺少映射字段：{sorted(absent)}")
    out=df[[columns[c] for c in columns]].copy()
    out.columns=list(columns)
    for key,value in mapping.get("constants",{}).items():out[key]=value
    for key,multiplier in mapping.get("multipliers",{}).items():out[key]=pd.to_numeric(out[key],errors="raise")*float(multiplier)
    if mapping.get("symbol_exchange_column"):
        raw=out.symbol.astype(str).str.zfill(6)
        exchange=df[mapping["symbol_exchange_column"]].map(mapping.get("exchange_mapping",{"SH":"SH","SZ":"SZ"}))
        if exchange.isna().any():raise ValueError("未知交易所，必须显式映射")
        out["symbol"]=raw+"."+exchange
    missing=set(SCHEMAS[table])-set(out)
    if missing:raise ValueError(f"转换缺少必需字段：{sorted(missing)}。不能推测历史时间/状态。")
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    out[SCHEMAS[table]].to_csv(dest/f"{table}.csv",index=False)
    return len(out)
