"""Autonomous, auditable forward-paper data collection.

The paid historical research set in ``data/inbox`` is immutable.  This module
stores every public-source response under ``data/live/raw`` and only publishes
``data/live/current`` after coverage, schema, timing and freshness gates pass.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import ROOT, Config, load_config
from .data import load_dataset, read_table, validate
from .schema import SCHEMAS

TZ = ZoneInfo("Asia/Shanghai")
LIVE_ROOT = ROOT / "data/live"
BASE_ROOT = ROOT / "data/inbox"
PRIMARY_KEYS = {
    "calendar": ["date"],
    "prices": ["date", "symbol"],
    "universe": ["date", "symbol"],
    "fundamentals": ["symbol", "report_period", "available_at"],
    "benchmark": ["date"],
    "actions": ["date", "symbol"],
    "documents": ["document_id"],
    "events": ["event_id"],
    "holdings": ["symbol"],
    "auction": ["date", "symbol"],
    "fees": ["start_date", "end_date"],
}
POLICY_SOURCES = {
    "state_council": "https://sousuo.www.gov.cn/zcwjk/",
    "csrc": "https://www.csrc.gov.cn/csrc/c100028/common_list.shtml",
    "pboc": "https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/index.html",
    "ndrc": "https://www.ndrc.gov.cn/xxgk/zcfb/",
}


class LiveCollectionError(RuntimeError):
    pass


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Shanghai")


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NaT or value is pd.NA:
        return None
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def _append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_value(value), ensure_ascii=False, allow_nan=False) + "\n")


def _save_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _read_optional(folder: Path, name: str) -> pd.DataFrame:
    if not folder.exists():
        return pd.DataFrame(columns=SCHEMAS[name])
    return read_table(folder, name)


def normalize_symbol(code, exchange=None) -> str:
    raw = str(code).strip().upper().replace(" ", "")
    if raw.endswith((".SH", ".SZ")) and len(raw) == 9:
        return raw
    if raw.startswith(("SH.", "SZ.")) and len(raw) >= 9:
        return raw[3:9] + "." + raw[:2]
    if raw.startswith(("SH", "SZ")) and len(raw) >= 8:
        return raw[2:8] + "." + raw[:2]
    digits = "".join(ch for ch in raw if ch.isdigit())[-6:].zfill(6)
    market = str(exchange or "")
    suffix = "SH" if ("上海" in market or "SHANGHAI" in market.upper() or digits.startswith(("5", "6", "9"))) else "SZ"
    return f"{digits}.{suffix}"


def bao_code(symbol: str) -> str:
    code, exchange = normalize_symbol(symbol).split(".")
    return f"{exchange.lower()}.{code}"


def sina_code(symbol: str) -> str:
    code, exchange = normalize_symbol(symbol).split(".")
    return f"{exchange.lower()}{code}"


def _numeric(value, default=np.nan) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _pick(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def _local_time(date, clock="18:30:00") -> pd.Timestamp:
    return pd.Timestamp(f"{pd.Timestamp(date).date()} {clock}", tz="Asia/Shanghai")


def _limit_price(reference: float, direction: int, is_st: int, symbol: str) -> float:
    special_board = normalize_symbol(symbol).startswith(("300", "301", "688", "689"))
    ratio = Decimal("0.20") if special_board else (Decimal("0.05") if is_st else Decimal("0.10"))
    value = Decimal(str(reference)) * (Decimal("1") + Decimal(direction) * ratio)
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def merge_table(base: pd.DataFrame, overlay: pd.DataFrame, name: str) -> pd.DataFrame:
    frames = [frame for frame in (base, overlay) if frame is not None and len(frame)]
    if not frames:
        return pd.DataFrame(columns=SCHEMAS[name])
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(PRIMARY_KEYS[name], keep="last")
    sort = [column for column in PRIMARY_KEYS[name] if column in result]
    return result.sort_values(sort).reset_index(drop=True) if sort else result.reset_index(drop=True)


@contextmanager
def collection_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".collect.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, f"{os.getpid()} {_now().isoformat()}".encode())
        os.close(descriptor)
    except FileExistsError as exc:
        age = time.time() - lock.stat().st_mtime
        if age < 3600:
            raise LiveCollectionError("另一个实时采集任务正在运行") from exc
        lock.unlink(missing_ok=True)
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(descriptor)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


class BaoStockProvider:
    """BaoStock is the primary EOD source; one logged-in session is reused."""

    def __init__(self):
        self.bs = None

    def __enter__(self):
        import baostock as bs
        result = bs.login()
        if result.error_code != "0":
            raise LiveCollectionError(f"BaoStock 登录失败：{result.error_msg}")
        self.bs = bs
        return self

    def __exit__(self, *_):
        if self.bs is not None:
            try:
                self.bs.logout()
            except Exception:
                pass

    def set_socket_timeout(self, seconds: float) -> None:
        """Bound the vendor's otherwise unbounded recv loop."""
        from baostock.common import context
        connection = getattr(context, "default_socket", None)
        if connection is not None:
            connection.settimeout(seconds)

    def reconnect(self) -> None:
        """Replace a socket that may have timed out during optional enrichment."""
        from baostock.common import context
        connection = getattr(context, "default_socket", None)
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        setattr(context, "default_socket", None)
        result = self.bs.login()
        if result.error_code != "0":
            raise LiveCollectionError(f"BaoStock 重新登录失败：{result.error_msg}")

    @staticmethod
    def _frame(result) -> pd.DataFrame:
        rows = []
        while result.error_code == "0" and result.next():
            rows.append(result.get_row_data())
        if result.error_code != "0":
            raise LiveCollectionError(f"BaoStock 查询失败：{result.error_msg}")
        return pd.DataFrame(rows, columns=result.fields)

    def _retry(self, callback, attempts=3) -> pd.DataFrame:
        last = None
        for attempt in range(attempts):
            try:
                return self._frame(callback())
            except Exception as exc:
                last = exc
                if attempt + 1 < attempts:
                    time.sleep(.25 * (attempt + 1))
        raise last

    def calendar(self, start, end) -> pd.DataFrame:
        return self._retry(lambda: self.bs.query_trade_dates(str(start)[:10], str(end)[:10]))

    def history(self, symbol, start, end) -> pd.DataFrame:
        fields = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST"
        return self._retry(lambda: self.bs.query_history_k_data_plus(
            bao_code(symbol), fields, str(start)[:10], str(end)[:10], frequency="d", adjustflag="3"
        ))

    def benchmark(self, start, end) -> pd.DataFrame:
        fields = "date,code,open,high,low,close,preclose,volume,amount"
        return self._retry(lambda: self.bs.query_history_k_data_plus(
            "sh.000300", fields, str(start)[:10], str(end)[:10], frequency="d", adjustflag="3"
        ))

    def adjustments(self, symbol, end) -> pd.DataFrame:
        return self._retry(lambda: self.bs.query_adjust_factor(bao_code(symbol), "1990-01-01", str(end)[:10]), attempts=1)

    def dividends(self, symbol, year) -> pd.DataFrame:
        return self._retry(lambda: self.bs.query_dividend_data(bao_code(symbol), year=int(year), yearType="operate"), attempts=1)

    def stock_basic(self, symbol) -> pd.DataFrame:
        return self._retry(lambda: self.bs.query_stock_basic(code=bao_code(symbol)))

    def industry(self, symbol) -> pd.DataFrame:
        return self._retry(lambda: self.bs.query_stock_industry(code=bao_code(symbol)))

    def financial(self, name, symbol, year, quarter) -> pd.DataFrame:
        method = getattr(self.bs, f"query_{name}_data")
        return self._retry(lambda: method(bao_code(symbol), year=int(year), quarter=int(quarter)))


class AkShareProvider:
    """Official CSI adapter plus public fallbacks and text observation feeds."""

    def __init__(self):
        import akshare as ak
        self.ak = ak

    def universe(self):
        # AKShare's wrappers use requests.get without a timeout.  A stalled
        # official endpoint can otherwise keep the nightly task hung for many
        # minutes and make the web action appear to do nothing.
        from io import BytesIO
        import requests

        base = "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile"
        specs = [
            ("cons/000300cons.xls", ["日期", "指数代码", "指数名称", "指数英文名称", "成分券代码", "成分券名称", "成分券英文名称", "交易所", "交易所英文名称"]),
            ("closeweight/000300closeweight.xls", ["日期", "指数代码", "指数名称", "指数英文名称", "成分券代码", "成分券名称", "成分券英文名称", "交易所", "交易所英文名称", "权重"]),
        ]
        result = []
        # This official OSS host is publicly reachable directly.  Some desktop
        # environments inject a stale HTTP(S) proxy into long-running GUI
        # processes; requests would then fail with ProxyError even though the
        # endpoint itself is healthy.  Bypass inherited proxy settings only for
        # these two public CSI files.
        direct = requests.Session()
        direct.trust_env = False
        for suffix, columns in specs:
            error = None
            for attempt in range(3):
                try:
                    response = direct.get(f"{base}/{suffix}", timeout=(5, 20))
                    response.raise_for_status()
                    frame = pd.read_excel(BytesIO(response.content))
                    if len(frame.columns) != len(columns) or frame.empty:
                        raise ValueError(f"中证指数 {suffix} 文件结构不完整")
                    frame.columns = columns
                    frame["日期"] = pd.to_datetime(frame["日期"], format="%Y%m%d", errors="coerce").dt.date
                    frame["指数代码"] = frame["指数代码"].astype(str).str.zfill(6)
                    frame["成分券代码"] = frame["成分券代码"].astype(str).str.zfill(6)
                    if "权重" in frame:
                        frame["权重"] = pd.to_numeric(frame["权重"], errors="coerce")
                    result.append(frame)
                    break
                except (requests.RequestException, ValueError, OSError) as exc:
                    error = exc
                    if attempt < 2:
                        time.sleep(attempt + 1)
            else:
                raise LiveCollectionError(f"中证指数官方文件暂不可用（{suffix}）：{error}；保留上一份成功报告，可稍后手动重跑") from error
        return result[0], result[1]

    def calendar(self) -> pd.DataFrame:
        return self.ak.tool_trade_date_hist_sina()

    def fallback_history(self, symbol, start, end) -> pd.DataFrame:
        return self.ak.stock_zh_a_daily(
            symbol=sina_code(symbol),
            start_date=pd.Timestamp(start).strftime("%Y%m%d"),
            end_date=pd.Timestamp(end).strftime("%Y%m%d"),
            adjust="",
        )

    def notices(self, date) -> pd.DataFrame:
        return self.ak.stock_notice_report(symbol="全部", date=pd.Timestamp(date).strftime("%Y%m%d"))

    def hot_rank(self) -> pd.DataFrame:
        return self.ak.stock_hot_rank_em()

    def news(self, symbol) -> pd.DataFrame:
        return self.ak.stock_news_em(symbol=normalize_symbol(symbol)[:6])


def _canonical_universe(cons: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    code_col = _pick(cons, ["成分券代码", "证券代码", "code"])
    exchange_col = _pick(cons, ["交易所", "exchange"])
    date_col = _pick(cons, ["日期", "date"])
    if not code_col or not date_col:
        raise LiveCollectionError("中证指数成分接口字段发生变化")
    snapshot_date = pd.to_datetime(cons[date_col]).max().normalize()
    symbols = cons.apply(lambda row: normalize_symbol(row[code_col], row[exchange_col] if exchange_col else None), axis=1)
    result = pd.DataFrame({"symbol": symbols})
    w_code = _pick(weights, ["成分券代码", "证券代码", "code"])
    w_exchange = _pick(weights, ["交易所", "exchange"])
    w_value = _pick(weights, ["权重", "weight"])
    if w_code and w_value:
        mapping = {
            normalize_symbol(row[w_code], row[w_exchange] if w_exchange else None): _numeric(row[w_value], 0.)
            for _, row in weights.iterrows()
        }
        result["weight"] = result.symbol.map(mapping)
    else:
        result["weight"] = np.nan
    if len(result) != 300 or result.symbol.nunique() != 300:
        raise LiveCollectionError(f"中证官方当前成分应为300只，实际取得 {result.symbol.nunique()} 只")
    if result.weight.isna().any() or (result.weight <= 0).any():
        result["weight"] = 1 / len(result)
    else:
        result["weight"] = result.weight / result.weight.sum()
    result["date"] = snapshot_date
    result["known_at"] = _local_time(snapshot_date, "00:00:00")
    return result[SCHEMAS["universe"]], snapshot_date


def _expand_universe(base: pd.DataFrame, snapshots: list[pd.DataFrame], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    base_date = pd.Timestamp(base.date.max())
    base_snapshot = base[base.date == base_date][["symbol", "weight", "known_at"]].copy()
    candidates = [(base_date, base_snapshot)]
    for snapshot in snapshots:
        if len(snapshot):
            date = pd.Timestamp(snapshot.date.max())
            candidates.append((date, snapshot[["symbol", "weight", "known_at"]].copy()))
    candidates.sort(key=lambda item: item[0])
    rows = []
    for day in sessions:
        eligible = [item for item in candidates if item[0] <= day]
        _, selected = eligible[-1] if eligible else candidates[0]
        frame = selected.copy()
        frame["date"] = day
        rows.append(frame[SCHEMAS["universe"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=SCHEMAS["universe"])


def _fallback_rows(ak: AkShareProvider, symbol: str, start, end) -> pd.DataFrame:
    frame = ak.fallback_history(symbol, start, end).copy()
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["code"] = bao_code(symbol)
    frame["preclose"] = frame.close.shift(1)
    frame["turn"] = pd.to_numeric(frame.get("turnover"), errors="coerce") * 100
    frame["tradestatus"] = "1"
    frame["isST"] = np.nan
    return frame


def _last_metadata(base_prices: pd.DataFrame, overlay_prices: pd.DataFrame, before=None) -> dict[str, dict]:
    columns = SCHEMAS["prices"]
    frames = [frame[columns] for frame in (base_prices, overlay_prices) if len(frame)]
    data = pd.concat(frames, ignore_index=True)
    if before is not None:
        data = data[data.date < pd.Timestamp(before)]
    data = data.sort_values(["symbol", "date"])
    return {row.symbol: row._asdict() for row in data.groupby("symbol", sort=False).tail(1).itertuples(index=False)}


def _new_symbol_metadata(provider: BaoStockProvider, symbol: str, day) -> dict:
    basic = provider.stock_basic(symbol)
    industry = provider.industry(symbol)
    ipo = pd.NaT
    if len(basic) and "ipoDate" in basic:
        ipo = pd.to_datetime(basic.iloc[-1].ipoDate, errors="coerce")
    industry_name = "未知行业"
    if len(industry):
        column = _pick(industry, ["industry", "industryClassification"])
        if column and pd.notna(industry.iloc[-1][column]):
            industry_name = str(industry.iloc[-1][column])
    star = normalize_symbol(symbol).startswith(("688", "689"))
    return {
        "date": pd.Timestamp(day) - pd.Timedelta(days=1), "close": np.nan, "adj_factor": 1.,
        "market_cap": np.nan, "float_market_cap": np.nan, "industry": industry_name,
        "min_order": 200 if star else 100, "lot_step": 1 if star else 100,
        "listed_days": max(0, (pd.Timestamp(day) - ipo).days) if pd.notna(ipo) else 0,
        "is_st": 0,
    }


def _parse_actions(provider: BaoStockProvider, symbols: set[str], years: set[int], raw_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    action_rows, factors, errors = [], {}, []
    raw_dividends = []
    for symbol in sorted(symbols):
        try:
            factor = provider.adjustments(symbol, pd.Timestamp(max(years), 12, 31))
            factor["symbol"] = symbol
            factors[symbol] = factor
        except Exception as exc:
            errors.append(f"{symbol} 复权因子失败：{type(exc).__name__}")
            continue
        for year in sorted(years):
            try:
                dividend = provider.dividends(symbol, year)
            except Exception as exc:
                errors.append(f"{symbol} 公司行动失败：{type(exc).__name__}")
                continue
            if dividend.empty:
                continue
            dividend["symbol"] = symbol
            raw_dividends.append(dividend)
            for row in dividend.to_dict("records"):
                event_date = pd.to_datetime(row.get("dividOperateDate"), errors="coerce")
                if pd.isna(event_date):
                    continue
                announcement = pd.to_datetime(row.get("dividPlanAnnounceDate"), errors="coerce")
                cash = _numeric(row.get("dividCashPsBeforeTax"), 0.)
                stock = _numeric(row.get("dividStocksPs"), 0.)
                reserve = _numeric(row.get("dividReserveToStockPs"), 0.)
                action_rows.append({
                    "date": event_date.normalize(), "symbol": symbol,
                    "share_multiplier": 1 + max(0., stock) + max(0., reserve),
                    "cash_per_share": max(0., cash),
                    "known_at": _local_time(announcement if pd.notna(announcement) else event_date, "20:00:00"),
                })
    if factors:
        _save_frame(raw_dir / "baostock_adjust_factors.parquet", pd.concat(factors.values(), ignore_index=True))
    if raw_dividends:
        _save_frame(raw_dir / "baostock_dividends.parquet", pd.concat(raw_dividends, ignore_index=True))
    actions = pd.DataFrame(action_rows, columns=SCHEMAS["actions"])
    if len(actions):
        actions = actions.sort_values(["date", "symbol", "known_at"]).drop_duplicates(["date", "symbol"], keep="last")
    return actions, factors, errors


def _reference_actions(sessions: pd.DatetimeIndex, symbols: set[str]) -> pd.DataFrame:
    """Load small, source-cited corrections for vendor omissions.

    The collector still fails closed when an unexplained adjustment is found.
    A correction is accepted only when it is explicitly recorded in the
    repository with its original announcement time and source URL.
    """
    path = Path(__file__).with_name("reference_actions.csv")
    if not path.exists():
        return pd.DataFrame(columns=SCHEMAS["actions"])
    frame = pd.read_csv(path, dtype={"symbol": str})
    frame["date"] = pd.to_datetime(frame.date, errors="raise").dt.normalize()
    frame["known_at"] = pd.to_datetime(frame.known_at, errors="raise", utc=True).dt.tz_convert("Asia/Shanghai")
    frame = frame[frame.date.isin(pd.DatetimeIndex(sessions)) & frame.symbol.isin(symbols)].copy()
    return frame[SCHEMAS["actions"]]


def _factor_ratio(factor: pd.DataFrame, prior_date, day) -> float:
    if factor is None or factor.empty or "dividOperateDate" not in factor:
        return 1.
    f = factor.copy()
    f["date"] = pd.to_datetime(f.dividOperateDate, errors="coerce")
    f["value"] = pd.to_numeric(f.get("backAdjustFactor", f.get("adjustFactor")), errors="coerce")
    f = f.dropna(subset=["date", "value"]).sort_values("date")
    before = f[f.date <= pd.Timestamp(prior_date)]
    current = f[f.date <= pd.Timestamp(day)]
    baseline = float(before.iloc[-1].value) if len(before) else 1.
    value = float(current.iloc[-1].value) if len(current) else baseline
    return value / baseline if baseline > 0 else 1.


def _build_prices(
    provider: BaoStockProvider,
    ak: AkShareProvider,
    sessions: pd.DatetimeIndex,
    universe: pd.DataFrame,
    metadata: dict[str, dict],
    raw_dir: Path,
    progress,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[dict], list[dict]]:
    expected = universe[universe.date.isin(sessions)].groupby("symbol").date.apply(lambda x: set(pd.to_datetime(x))).to_dict()
    histories, sources, errors, warnings, action_symbols = {}, [], [], [], set()
    quarantines = []
    raw_primary, raw_fallback = [], []
    start, end = sessions.min(), sessions.max()
    for index, symbol in enumerate(sorted(expected)):
        source = "baostock"
        try:
            history = provider.history(symbol, start, end)
        except Exception:
            history = pd.DataFrame()
        if len(history):
            history["date"] = pd.to_datetime(history.date, errors="coerce").dt.normalize()
            history["symbol"] = symbol
            raw_primary.append(history)
        missing = expected[symbol] - set(history.date if len(history) else [])
        if missing:
            try:
                fallback = _fallback_rows(ak, symbol, min(missing), max(missing))
            except Exception:
                fallback = pd.DataFrame()
            if len(fallback):
                fallback["symbol"] = symbol
                raw_fallback.append(fallback)
                history = pd.concat([history, fallback], ignore_index=True).sort_values("date").drop_duplicates("date", keep="first")
                source = "baostock+sina" if len(raw_primary) else "sina"
        still_missing = expected[symbol] - set(history.date if len(history) else [])
        if still_missing:
            errors.append(f"{symbol} 缺少 {len(still_missing)} 个应有交易日行情")
        histories[symbol] = history
        sources.append({"symbol": symbol, "source": source, "rows": len(history), "missing_sessions": len(still_missing)})
        previous_close = _numeric(metadata.get(symbol, {}).get("close"))
        for row in history.sort_values("date").to_dict("records"):
            preclose = _numeric(row.get("preclose"))
            if np.isfinite(previous_close) and np.isfinite(preclose) and previous_close > 0 and abs(preclose / previous_close - 1) > .0005:
                action_symbols.add(symbol)
            close = _numeric(row.get("close"))
            if close > 0:
                previous_close = close
        if index % 20 == 0 or index + 1 == len(expected):
            progress(f"采集沪深300日线 {index + 1}/{len(expected)}", .10 + .45 * (index + 1) / max(len(expected), 1))
    if raw_primary:
        _save_frame(raw_dir / "baostock_prices.parquet", pd.concat(raw_primary, ignore_index=True))
    if raw_fallback:
        _save_frame(raw_dir / "sina_fallback_prices.parquet", pd.concat(raw_fallback, ignore_index=True))
    # Corporate-action enrichment is useful but optional.  BaoStock implements
    # it over a raw socket with no timeout, so cap the whole connection's recv
    # wait.  Any timeout becomes a per-symbol quarantine; reconnect before the
    # benchmark and fundamental queries continue.
    manages_socket = hasattr(provider, "set_socket_timeout") and hasattr(provider, "reconnect")
    if manages_socket:
        provider.set_socket_timeout(float(os.getenv("ACTION_LOOKUP_TIMEOUT_SECONDS", "8")))
    try:
        actions, factors, action_errors = _parse_actions(
            provider, action_symbols, set(pd.DatetimeIndex(sessions).year), raw_dir
        )
    finally:
        if manages_socket:
            provider.reconnect()
    if len(actions):
        actions = actions[actions.date.isin(sessions)].copy()
    references = _reference_actions(sessions, set(expected))
    if len(references):
        _save_frame(raw_dir / "source_cited_reference_actions.parquet", references)
        actions = merge_table(actions, references, "actions")
    # Missing vendor action details are handled at the symbol level below.
    # They must not prevent the other CSI 300 names from updating.
    warnings.extend(action_errors)
    action_dates = {(row.symbol, pd.Timestamp(row.date)) for row in actions.itertuples()}
    rows = []
    for symbol, wanted in expected.items():
        prior = dict(metadata.get(symbol) or _new_symbol_metadata(provider, symbol, start))
        history = histories[symbol].set_index("date") if len(histories[symbol]) else pd.DataFrame()
        prior_date = pd.Timestamp(prior["date"])
        prior_close = _numeric(prior.get("close"))
        total_shares = _numeric(prior.get("market_cap")) / prior_close if prior_close > 0 else np.nan
        float_shares = _numeric(prior.get("float_market_cap")) / prior_close if prior_close > 0 else np.nan
        prior_factor = _numeric(prior.get("adj_factor"), 1.)
        for day in sorted(wanted):
            if history.empty or day not in history.index:
                continue
            raw = history.loc[day]
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[-1]
            active = str(raw.get("tradestatus", "1")) == "1"
            trusted_total_shares, trusted_float_shares = total_shares, float_shares
            close = _numeric(raw.get("close"))
            suspended = int(not active or not np.isfinite(close) or close <= 0)
            if suspended:
                if not np.isfinite(prior_close) or prior_close <= 0:
                    errors.append(f"{symbol} {day.date()} 停牌且没有可用前收盘价")
                    continue
                close = prior_close
            prev_close = _numeric(raw.get("preclose"), prior_close)
            if not np.isfinite(prev_close) or prev_close <= 0:
                prev_close = prior_close
            open_ = _numeric(raw.get("open"), close) if not suspended else close
            high = _numeric(raw.get("high"), close) if not suspended else close
            low = _numeric(raw.get("low"), close) if not suspended else close
            volume = max(0., _numeric(raw.get("volume"), 0.))
            amount = max(0., _numeric(raw.get("amount"), 0.))
            turnover = max(0., _numeric(raw.get("turn"), 0.) / 100)
            if turnover > 0 and volume > 0:
                float_shares = volume / turnover
            is_st_value = _numeric(raw.get("isST"), _numeric(prior.get("is_st"), 0.))
            is_st = int(is_st_value > .5)
            action = actions[(actions.symbol == symbol) & (actions.date == day)] if len(actions) else actions
            if len(action):
                total_shares *= float(action.iloc[-1].share_multiplier)
            factor = prior_factor * _factor_ratio(factors.get(symbol), prior_date, day)
            unknown_action = (
                symbol in action_symbols and np.isfinite(prior_close) and prior_close > 0
                and abs(prev_close / prior_close - 1) > .0005 and (symbol, day) not in action_dates
            )
            if unknown_action:
                reason = f"{symbol} {day.date()} 检测到除权参考价变化，但公司行动明细未取得"
                warnings.append(reason + "；已冻结该证券，其他股票继续更新")
                quarantines.append({"symbol": symbol, "date": str(day.date()), "reason": reason})
                # Keep the last trusted mark and share counts.  This prevents an
                # unexplained ex-rights gap from entering factors, NAV or orders.
                close = prior_close
                open_ = high = low = prior_close
                prev_close = prior_close
                volume = amount = turnover = 0.
                suspended = 1
                total_shares, float_shares = trusted_total_shares, trusted_float_shares
                factor = prior_factor
            if not np.isfinite(total_shares) or total_shares <= 0:
                total_shares = float_shares
            if not np.isfinite(float_shares) or float_shares <= 0:
                float_shares = total_shares
            listed = int(_numeric(prior.get("listed_days"), 0) + max(0, (day - prior_date).days))
            min_order = int(_numeric(prior.get("min_order"), 200 if symbol.startswith("688") else 100))
            lot_step = int(_numeric(prior.get("lot_step"), 1 if symbol.startswith("688") else 100))
            rows.append({
                "date": day, "symbol": symbol, "open": open_, "high": max(high, open_, close),
                "low": min(low, open_, close), "close": close, "prev_close": prev_close,
                "volume": volume, "amount": amount, "adj_factor": factor,
                "upper_limit": _limit_price(prev_close, 1, is_st, symbol),
                "lower_limit": _limit_price(prev_close, -1, is_st, symbol),
                "suspended": suspended, "is_st": is_st,
                "market_cap": close * total_shares, "float_market_cap": close * float_shares,
                "turnover_rate": turnover, "industry": str(prior.get("industry") or "未知行业"),
                "min_order": min_order, "lot_step": lot_step, "listed_days": listed,
                "available_at": _local_time(day),
            })
            prior.update(date=day, close=close, market_cap=close * total_shares,
                         float_market_cap=close * float_shares, listed_days=listed, is_st=is_st,
                         adj_factor=factor)
            prior_date, prior_close, prior_factor = day, close, factor
    unique_quarantines = list({(row["symbol"], row["date"]): row for row in quarantines}.values())
    return pd.DataFrame(rows, columns=SCHEMAS["prices"]), actions, errors, warnings, sources, unique_quarantines


def _build_benchmark(provider: BaoStockProvider, sessions: pd.DatetimeIndex, base: pd.DataFrame, raw_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    errors = []
    raw = provider.benchmark(sessions.min(), sessions.max())
    _save_frame(raw_dir / "baostock_csi300.parquet", raw)
    raw["date"] = pd.to_datetime(raw.date, errors="coerce").dt.normalize()
    raw["close"] = pd.to_numeric(raw.close, errors="coerce")
    raw = raw[raw.date.isin(sessions)].dropna(subset=["date", "close"])
    missing = set(sessions) - set(raw.date)
    if missing:
        errors.append(f"沪深300基准缺少 {len(missing)} 个交易日")
    anchor = base.sort_values("date").iloc[-1]
    result = raw[["date", "close"]].copy()
    result["total_return_close"] = float(anchor.total_return_close) * result.close / float(anchor.close)
    return result[SCHEMAS["benchmark"]], errors


def _expected_report_period(day: pd.Timestamp) -> tuple[int, int, pd.Timestamp]:
    year, month = day.year, day.month
    if month <= 4:
        return year - 1, 4, pd.Timestamp(year - 1, 12, 31)
    if month <= 7:
        return year, 1, pd.Timestamp(year, 3, 31)
    if month <= 9:
        return year, 2, pd.Timestamp(year, 6, 30)
    return year, 3, pd.Timestamp(year, 9, 30)


def _one(frame: pd.DataFrame) -> dict:
    return frame.iloc[-1].to_dict() if len(frame) else {}


def _refresh_fundamentals(provider: BaoStockProvider, existing: pd.DataFrame, symbols: set[str], day: pd.Timestamp, raw_dir: Path, progress) -> tuple[pd.DataFrame, list[str]]:
    year, quarter, period = _expected_report_period(day)
    latest = existing.groupby("symbol").report_period.max().to_dict() if len(existing) else {}
    needed = [symbol for symbol in sorted(symbols) if pd.Timestamp(latest.get(symbol, "1900-01-01")) < period]
    if not needed:
        return pd.DataFrame(columns=SCHEMAS["fundamentals"]), []
    rows, warnings, raw = [], [], []
    for index, symbol in enumerate(needed):
        try:
            profit = provider.financial("profit", symbol, year, quarter)
            if profit.empty:
                continue  # report not published yet
            growth = provider.financial("growth", symbol, year, quarter)
            balance = provider.financial("balance", symbol, year, quarter)
            cash = provider.financial("cash_flow", symbol, year, quarter)
            prev_profit = provider.financial("profit", symbol, year - 1, quarter)
            annual_profit = provider.financial("profit", symbol, year - 1, 4)
            prev_cash = provider.financial("cash_flow", symbol, year - 1, quarter)
            annual_cash = provider.financial("cash_flow", symbol, year - 1, 4)
            for name, frame in [("profit", profit), ("growth", growth), ("balance", balance), ("cash", cash)]:
                if len(frame):
                    f = frame.copy(); f["dataset"] = name; f["symbol"] = symbol; raw.append(f)
            p, g, b, c = map(_one, (profit, growth, balance, cash))
            pp, ap, pc, ac = map(_one, (prev_profit, annual_profit, prev_cash, annual_cash))
            published = pd.to_datetime(p.get("pubDate"), errors="coerce")
            if pd.isna(published):
                continue
            shares = _numeric(p.get("totalShare"))
            eps_ttm = _numeric(p.get("epsTTM"))
            net_ttm = shares * eps_ttm
            current_revenue = _numeric(p.get("MBRevenue"))
            previous_revenue = _numeric(pp.get("MBRevenue"))
            annual_revenue = _numeric(ap.get("MBRevenue"))
            revenue_ttm = current_revenue if quarter == 4 else annual_revenue + current_revenue - previous_revenue
            def ocf(cash_row, profit_row):
                return _numeric(cash_row.get("CFOToNP")) * _numeric(profit_row.get("netProfit"))
            current_ocf, previous_ocf, annual_ocf = ocf(c, p), ocf(pc, pp), ocf(ac, ap)
            ocf_ttm = current_ocf if quarter == 4 else annual_ocf + current_ocf - previous_ocf
            prior_period = period - pd.DateOffset(years=1)
            prior = existing[(existing.symbol == symbol) & (existing.report_period == prior_period)]
            if prior.empty:
                prior = existing[existing.symbol == symbol].sort_values("report_period").tail(1)
            prior_row = _one(prior)
            assets = _numeric(prior_row.get("total_assets")) * (1 + _numeric(g.get("YOYAsset"), 0.))
            equity = _numeric(prior_row.get("book_equity")) * (1 + _numeric(g.get("YOYEquity"), 0.))
            liabilities = assets * _numeric(b.get("liabilityToAsset"))
            values = [revenue_ttm, net_ttm, equity, ocf_ttm, assets, liabilities]
            if not all(np.isfinite(values)) or min(assets, equity) <= 0:
                raise ValueError("公开财务字段不足以完成规范化")
            rows.append({
                "symbol": symbol, "report_period": period,
                "available_at": _local_time(published, "20:00:00"),
                "revenue_ttm": revenue_ttm, "net_profit_ttm": net_ttm,
                "book_equity": equity, "operating_cashflow_ttm": ocf_ttm,
                "total_assets": assets, "total_liabilities": liabilities,
                "roe": _numeric(p.get("roeAvg")), "gross_margin": _numeric(p.get("gpMargin")),
                "revenue_growth": current_revenue / previous_revenue - 1 if previous_revenue else np.nan,
                "profit_growth": _numeric(g.get("YOYNI")),
            })
        except Exception as exc:
            warnings.append(f"{symbol} 新财报自动规范化失败：{type(exc).__name__}")
        if index % 25 == 0:
            progress(f"检查新披露财报 {index + 1}/{len(needed)}", .68 + .08 * (index + 1) / max(len(needed), 1))
    if raw:
        _save_frame(raw_dir / "baostock_financial_updates.parquet", pd.concat(raw, ignore_index=True))
    return pd.DataFrame(rows, columns=SCHEMAS["fundamentals"]), warnings


def _timestamp_or_now(value, now: pd.Timestamp) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return now
    parsed = pd.Timestamp(parsed)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("Asia/Shanghai")
    return parsed.tz_convert("Asia/Shanghai")


def _document(symbol, source_type, url, published, first_seen, title, text) -> dict:
    identity = f"{source_type}|{symbol}|{url}|{title}|{published}"
    return {
        "document_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
        "symbol": normalize_symbol(symbol), "source_type": source_type,
        "source_url": str(url or ""), "published_at": published,
        "first_seen_at": first_seen, "title": str(title or "").strip(),
        "text": str(text or title or "").strip(),
    }


def _watch_symbols() -> list[str]:
    pointer = ROOT / "runs/latest.json"
    if not pointer.exists():
        return []
    try:
        run_id = json.loads(pointer.read_text())["run_id"]
        report = json.loads((ROOT / "runs" / run_id / "report.json").read_text())
        ranked = report.get("rankings", [])
        selected = [row["symbol"] for row in ranked if row.get("target_weight", 0) > 0]
        return (selected + [row["symbol"] for row in ranked[:25]])[:40]
    except Exception:
        return []


def _collect_text(ak: AkShareProvider, day: pd.Timestamp, universe: set[str], raw_dir: Path, now: pd.Timestamp) -> tuple[pd.DataFrame, list[str], list[dict]]:
    documents, warnings, sources = [], [], []
    try:
        notices = ak.notices(day)
        _save_frame(raw_dir / "announcements.parquet", notices)
        code_col = _pick(notices, ["代码", "股票代码", "证券代码"])
        title_col = _pick(notices, ["公告标题", "公告名称", "标题"])
        date_col = _pick(notices, ["公告日期", "日期", "发布时间"])
        url_col = _pick(notices, ["网址", "公告链接", "链接"])
        if code_col and title_col:
            for row in notices.to_dict("records"):
                symbol = normalize_symbol(row.get(code_col))
                if symbol not in universe:
                    continue
                published = _timestamp_or_now(row.get(date_col), now) if date_col else now
                documents.append(_document(symbol, "filing", row.get(url_col, "") if url_col else "", published, now, row.get(title_col), row.get(title_col)))
        sources.append({"name": "announcements", "status": "ok", "rows": len(notices)})
    except Exception as exc:
        warnings.append(f"公告观察源暂不可用：{type(exc).__name__}")
        sources.append({"name": "announcements", "status": "warning"})
    try:
        hot = ak.hot_rank()
        _save_frame(raw_dir / "forum_hot_rank.parquet", hot)
        code_col = _pick(hot, ["代码", "股票代码"])
        rank_col = _pick(hot, ["当前排名", "排名"])
        if code_col:
            for row in hot.to_dict("records"):
                symbol = normalize_symbol(row.get(code_col))
                if symbol not in universe:
                    continue
                rank = row.get(rank_col, "—") if rank_col else "—"
                title = f"东方财富股吧热度排名 {rank}"
                url = f"https://guba.eastmoney.com/list,{symbol[:6]}.html"
                documents.append(_document(symbol, "forum", url, now, now, title, title))
        sources.append({"name": "forum_hot_rank", "status": "ok", "rows": len(hot)})
    except Exception as exc:
        warnings.append(f"论坛热度观察源暂不可用：{type(exc).__name__}")
        sources.append({"name": "forum_hot_rank", "status": "warning"})
    news_frames = []
    for symbol in list(dict.fromkeys(_watch_symbols()))[:20]:
        if symbol not in universe:
            continue
        try:
            news = ak.news(symbol)
            if news.empty:
                continue
            news["symbol"] = symbol
            news_frames.append(news)
            title_col = _pick(news, ["新闻标题", "标题"])
            text_col = _pick(news, ["新闻内容", "内容"])
            time_col = _pick(news, ["发布时间", "时间"])
            url_col = _pick(news, ["新闻链接", "链接", "网址"])
            if title_col:
                for row in news.head(100).to_dict("records"):
                    published = _timestamp_or_now(row.get(time_col), now) if time_col else now
                    if published < now - pd.Timedelta(days=3):
                        continue
                    documents.append(_document(symbol, "news", row.get(url_col, "") if url_col else "", published, now, row.get(title_col), row.get(text_col, row.get(title_col)) if text_col else row.get(title_col)))
        except Exception:
            continue
    if news_frames:
        _save_frame(raw_dir / "watched_stock_news.parquet", pd.concat(news_frames, ignore_index=True))
    sources.append({"name": "watched_stock_news", "status": "ok" if news_frames else "warning", "symbols": len(news_frames)})
    # Policy pages are archived as evidence snapshots.  Entity/industry mapping
    # is intentionally deferred until an evidence-checked extractor is present.
    try:
        import httpx
        with httpx.Client(timeout=15, follow_redirects=True, headers={"User-Agent": "AShareResearch/0.2"}) as client:
            for name, url in POLICY_SOURCES.items():
                try:
                    response = client.get(url)
                    response.raise_for_status()
                    body = response.content[:5_000_000]
                    (raw_dir / f"policy_{name}.html").write_bytes(body)
                    sources.append({"name": f"policy_{name}", "status": "ok", "sha256": hashlib.sha256(body).hexdigest()})
                except Exception:
                    sources.append({"name": f"policy_{name}", "status": "warning"})
    except Exception as exc:
        warnings.append(f"政策网页快照暂不可用：{type(exc).__name__}")
    frame = pd.DataFrame(documents, columns=SCHEMAS["documents"])
    if len(frame):
        frame = frame.drop_duplicates("document_id")
    return frame, warnings, sources


def _publish_current(temp: Path, current: Path) -> None:
    previous = current.parent / ".previous"
    if previous.exists():
        shutil.rmtree(previous)
    if current.exists():
        current.replace(previous)
    temp.replace(current)
    if previous.exists():
        shutil.rmtree(previous)


def _live_gate(tables: dict[str, pd.DataFrame], quality: dict, latest: pd.Timestamp, cfg: Config) -> dict:
    errors, warnings = list(quality["errors"]), list(quality["warnings"])
    universe = tables["universe"]
    prices = tables["prices"]
    current_u = universe[universe.date == latest]
    current_p = prices[prices.date == latest]
    count = current_u.symbol.nunique()
    covered = current_u[["date", "symbol"]].merge(current_p[["date", "symbol"]], how="left", indicator=True)
    coverage = float((covered._merge == "both").mean()) if len(covered) else 0.
    if count != 300:
        errors.append(f"最新沪深300成分不是300只：{count}")
    if coverage < cfg.live_price_coverage_min:
        errors.append(f"最新收盘行情覆盖率 {coverage:.2%} 低于阈值 {cfg.live_price_coverage_min:.2%}")
    # Canonical dataset requires complete constituent coverage even if the
    # configurable warning threshold is slightly lower.
    if coverage < 1:
        errors.append("仍有成分股缺少最新日线，禁止生成订单")
    future = tables["calendar"][tables["calendar"].date > latest].date.nunique()
    if future < 10:
        errors.append(f"未来交易日历只有 {future} 天，至少需要10天")
    return {"passed": not errors, "errors": list(dict.fromkeys(errors)), "warnings": list(dict.fromkeys(warnings)),
            "signal_date": str(latest.date()), "constituents": count, "price_coverage": coverage,
            "future_sessions": int(future)}


def get_live_status(root: Path = LIVE_ROOT) -> dict:
    path = Path(root) / "collection_status.json"
    if not path.exists():
        return {"status": "not_started", "passed": False, "message": "尚未启动自动采集"}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"status": "invalid", "passed": False, "message": "采集状态文件损坏"}


def collect_live(
    as_of: str | None = None,
    *,
    base_root: Path = BASE_ROOT,
    live_root: Path = LIVE_ROOT,
    include_text: bool = True,
    cfg: Config | None = None,
    progress=None,
) -> dict:
    """Collect, normalize, gate and atomically publish the forward dataset."""
    cfg = cfg or load_config()
    progress = progress or (lambda *_: None)
    base_root, live_root = Path(base_root), Path(live_root)
    now = _now()
    collection_id = now.strftime("%Y%m%d-%H%M%S-%f")
    raw_dir = live_root / "raw" / collection_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    status_path = live_root / "collection_status.json"
    status = {"collection_id": collection_id, "status": "running", "passed": False,
              "started_at": now, "sources": [], "warnings": [], "errors": []}
    _write_json(status_path, status)
    with collection_lock(live_root):
        try:
            progress("读取冻结历史库", .02)
            base = load_dataset(base_root)
            cutoff = pd.Timestamp(base.manifest.get("historical_research_cutoff", base["prices"].date.max())).normalize()
            overlay_root = live_root / "overlay"
            overlay_prices = _read_optional(overlay_root, "prices")
            overlay_universe = _read_optional(overlay_root, "universe")
            ak = AkShareProvider()
            with BaoStockProvider() as provider:
                progress("读取交易日历与中证官方成分", .05)
                requested = pd.Timestamp(as_of).normalize() if as_of else now.tz_localize(None).normalize()
                calendar_raw = provider.calendar(cutoff - pd.Timedelta(days=7), requested + pd.Timedelta(days=75))
                _save_frame(raw_dir / "baostock_calendar.parquet", calendar_raw)
                date_col = _pick(calendar_raw, ["calendar_date", "date"])
                flag_col = _pick(calendar_raw, ["is_trading_day", "isOpen"])
                calendar_dates = pd.to_datetime(calendar_raw.loc[calendar_raw[flag_col].astype(str) == "1", date_col]).dt.normalize()
                if not as_of and requested in set(calendar_dates) and now.time() < pd.Timestamp("18:30").time():
                    completed = calendar_dates[calendar_dates < requested]
                else:
                    completed = calendar_dates[calendar_dates <= requested]
                if completed.empty:
                    raise LiveCollectionError("公开交易日历没有已完成交易日")
                latest = pd.Timestamp(completed.max())
                live_sessions = pd.DatetimeIndex(calendar_dates[(calendar_dates > cutoff) & (calendar_dates <= latest)]).sort_values()
                if live_sessions.empty:
                    raise LiveCollectionError(f"公开行情尚未晚于冻结日 {cutoff.date()}")
                # An explicitly historical rerun can follow a later failed
                # attempt. Never retain rows after its requested signal date.
                if len(overlay_prices):
                    overlay_prices = overlay_prices[overlay_prices.date <= latest].copy()
                if len(overlay_universe):
                    overlay_universe = overlay_universe[overlay_universe.date <= latest].copy()
                cons_raw, weight_raw = ak.universe()
                _save_frame(raw_dir / "csindex_constituents.parquet", cons_raw)
                _save_frame(raw_dir / "csindex_weights.parquet", weight_raw)
                snapshot, snapshot_date = _canonical_universe(cons_raw, weight_raw)
                snapshot_dir = live_root / "snapshots/universe"
                _save_frame(snapshot_dir / f"{snapshot_date.date()}.parquet", snapshot)
                snapshots = [pd.read_parquet(path) for path in sorted(snapshot_dir.glob("*.parquet"))]
                for frame in snapshots:
                    frame["date"] = pd.to_datetime(frame.date).dt.normalize()
                    frame["known_at"] = pd.to_datetime(frame.known_at, utc=True).dt.tz_convert("Asia/Shanghai")
                expanded = _expand_universe(base["universe"], snapshots, live_sessions)
                all_overlay_universe = merge_table(overlay_universe, expanded, "universe")
                _save_frame(overlay_root / "universe.parquet", all_overlay_universe)
                if len(overlay_prices):
                    refresh_floor = pd.Timestamp(overlay_prices.date.max()) - pd.Timedelta(days=7)
                    fetch_sessions = live_sessions[live_sessions >= refresh_floor]
                else:
                    fetch_sessions = live_sessions
                fetch_universe = all_overlay_universe[all_overlay_universe.date.isin(fetch_sessions)]
                # A removed constituent can still be held by the paper account;
                # keep collecting it until the position is actually liquidated.
                try:
                    from .paper import paper_symbols
                    held = paper_symbols(cfg, live_root)
                except Exception:
                    held = set()
                missing_held = held - set(fetch_universe.symbol)
                if missing_held:
                    extra = pd.MultiIndex.from_product(
                        [fetch_sessions, sorted(missing_held)], names=["date", "symbol"]
                    ).to_frame(index=False)
                    extra["weight"] = 0.
                    extra["known_at"] = now
                    fetch_universe = pd.concat([fetch_universe, extra[SCHEMAS["universe"]]], ignore_index=True)
                metadata = _last_metadata(base["prices"], overlay_prices, fetch_sessions.min())
                prices, new_actions, price_errors, price_warnings, price_sources, action_quarantines = _build_prices(
                    provider, ak, fetch_sessions, fetch_universe, metadata, raw_dir, progress
                )
                if price_errors:
                    raise LiveCollectionError("；".join(price_errors[:12]))
                all_overlay_prices = merge_table(overlay_prices, prices, "prices")
                _save_frame(overlay_root / "prices.parquet", all_overlay_prices)
                _save_frame(raw_dir / "price_source_coverage.parquet", pd.DataFrame(price_sources))
                overlay_actions = _read_optional(overlay_root, "actions")
                all_overlay_actions = merge_table(overlay_actions, new_actions, "actions")
                _save_frame(overlay_root / "actions.parquet", all_overlay_actions)
                benchmark, benchmark_errors = _build_benchmark(provider, live_sessions, base["benchmark"], raw_dir)
                if benchmark_errors:
                    raise LiveCollectionError("；".join(benchmark_errors))
                overlay_benchmark = merge_table(_read_optional(overlay_root, "benchmark"), benchmark, "benchmark")
                _save_frame(overlay_root / "benchmark.parquet", overlay_benchmark)
                live_calendar = pd.DataFrame({"date": pd.DatetimeIndex(calendar_dates[calendar_dates > cutoff]).sort_values()})
                overlay_calendar = merge_table(_read_optional(overlay_root, "calendar"), live_calendar, "calendar")
                _save_frame(overlay_root / "calendar.parquet", overlay_calendar)
                current_symbols = set(all_overlay_universe.loc[all_overlay_universe.date == latest, "symbol"])
                fundamentals, fundamental_warnings = _refresh_fundamentals(
                    provider, base["fundamentals"], current_symbols, latest, raw_dir, progress
                )
                overlay_fundamentals = merge_table(_read_optional(overlay_root, "fundamentals"), fundamentals, "fundamentals")
                _save_frame(overlay_root / "fundamentals.parquet", overlay_fundamentals)
            text_sources = []
            if include_text:
                progress("采集公告、新闻、论坛热度与政策快照", .80)
                documents, text_warnings, text_sources = _collect_text(ak, latest, current_symbols, raw_dir, now)
                overlay_documents = merge_table(_read_optional(overlay_root, "documents"), documents, "documents")
                _save_frame(overlay_root / "documents.parquet", overlay_documents)
            else:
                text_warnings = ["本次按参数跳过文本观察源"]
            progress("合并历史与实时层并执行质量闸门", .88)
            temp = live_root / f".current-{collection_id}"
            temp.mkdir(parents=True, exist_ok=False)
            tables = {}
            for name in SCHEMAS:
                overlay = _read_optional(overlay_root, name)
                tables[name] = merge_table(base[name], overlay, name)
                _save_frame(temp / f"{name}.parquet", tables[name])
            decision_as_of = _now()
            next_sessions = tables["calendar"][tables["calendar"].date > latest].date.sort_values()
            next_session = pd.Timestamp(next_sessions.iloc[0]) if len(next_sessions) else None
            manifest = {
                "mode": "forward_paper", "source": "frozen paid history + audited public incremental sources",
                "historical_research_cutoff": str(cutoff.date()), "research_locked": True,
                "collection_id": collection_id, "collected_at": decision_as_of,
                "decision_as_of": decision_as_of, "live_signal_date": str(latest.date()),
                "next_session": str(next_session.date()) if next_session is not None else None,
                "price_primary": "BaoStock", "price_fallback": "Sina via AKShare",
                "constituents": "CSI official via AKShare", "benchmark_total_return_live": "price-return proxy; historical metrics remain frozen",
                "text_backfill": False, "universe": base.manifest.get("universe", {}),
                "corporate_action_quarantine": action_quarantines,
            }
            _write_json(temp / "manifest.json", manifest)
            quality = validate(tables, temp)
            gate = _live_gate(tables, quality, latest, cfg)
            if not gate["passed"]:
                shutil.rmtree(temp)
                raise LiveCollectionError("；".join(gate["errors"][:12]))
            _publish_current(temp, live_root / "current")
            status = {
                "collection_id": collection_id, "status": "ready", "passed": True,
                "started_at": status["started_at"], "completed_at": _now(),
                "historical_research_cutoff": str(cutoff.date()), "signal_date": str(latest.date()),
                "next_session": manifest["next_session"], "decision_as_of": decision_as_of,
                "sources": [
                    {"name": "BaoStock EOD", "status": "ok", "symbols": len(price_sources)},
                    {"name": "CSI official constituents", "status": "ok", "as_of": str(snapshot_date.date()), "rows": 300},
                    {"name": "BaoStock CSI300 benchmark", "status": "ok", "rows": len(benchmark)},
                ] + text_sources,
                "gate": gate, "warnings": price_warnings + fundamental_warnings + text_warnings + gate["warnings"], "errors": [],
                "raw_snapshot": str(raw_dir), "current_dataset": str(live_root / "current"),
            }
            _write_json(status_path, status)
            _append_jsonl(live_root / "collection_ledger.jsonl", status)
            progress("实时数据已通过质量闸门", 1.)
            return status
        except Exception as exc:
            failed = status | {"status": "blocked", "passed": False, "completed_at": _now(),
                               "errors": [str(exc)[:1200]], "raw_snapshot": str(raw_dir)}
            _write_json(status_path, failed)
            _append_jsonl(live_root / "collection_ledger.jsonl", failed)
            if isinstance(exc, LiveCollectionError):
                raise
            raise LiveCollectionError(str(exc)) from exc


def crosscheck_symbols(symbols: list[str], date, *, tolerance_bps=20., raw_output: Path | None = None) -> dict:
    """Verify order-sensitive closes against the independent Sina endpoint."""
    current = read_table(LIVE_ROOT / "current", "prices")
    day = pd.Timestamp(date).normalize()
    primary = current[current.date == day].set_index("symbol")
    ak = AkShareProvider()
    rows, errors = [], []
    for symbol in list(dict.fromkeys(symbols)):
        try:
            secondary = ak.fallback_history(symbol, day, day)
            if secondary.empty or symbol not in primary.index:
                raise ValueError("missing")
            close = float(pd.to_numeric(secondary.close, errors="coerce").iloc[-1])
            expected = float(primary.loc[symbol, "close"])
            difference = abs(close / expected - 1) * 1e4
            rows.append({"date": day, "symbol": symbol, "primary_close": expected,
                         "secondary_close": close, "difference_bps": difference, "passed": difference <= tolerance_bps})
            if difference > tolerance_bps:
                errors.append(f"{symbol} 双源收盘价相差 {difference:.1f}bp")
        except Exception:
            errors.append(f"{symbol} 未取得独立收盘价复核")
    frame = pd.DataFrame(rows)
    if raw_output is not None:
        _save_frame(Path(raw_output), frame)
    return {"passed": not errors, "checked": len(rows), "required": len(set(symbols)),
            "tolerance_bps": tolerance_bps, "errors": errors,
            "max_difference_bps": float(frame.difference_bps.max()) if len(frame) else None}
