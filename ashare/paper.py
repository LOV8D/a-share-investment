"""Persistent forward-paper account and order-source verification."""
from __future__ import annotations

import json
import os
from pathlib import Path
from contextlib import contextmanager
import time

import numpy as np
import pandas as pd

from .config import ROOT, Config
from .data import read_table
from .live_data import LIVE_ROOT, crosscheck_symbols
from .portfolio import execute_plan
from .schema import SCHEMAS


def _write_json(path: Path, value) -> None:
    from .pipeline import write_json
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    write_json(temp, value)
    temp.replace(path)


def _state_path(live_root=LIVE_ROOT) -> Path:
    return Path(live_root) / "paper_state.json"


@contextmanager
def paper_job_lock(live_root=LIVE_ROOT):
    root = Path(live_root); root.mkdir(parents=True, exist_ok=True)
    path = root / ".paper-job.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, f"{os.getpid()} {pd.Timestamp.now(tz='Asia/Shanghai').isoformat()}".encode())
        os.close(descriptor)
    except FileExistsError as exc:
        if time.time() - path.stat().st_mtime < 7200:
            raise RuntimeError("另一个模拟盘任务正在运行") from exc
        path.unlink(missing_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(descriptor)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def load_paper_state(cfg: Config, live_root=LIVE_ROOT, cutoff=None) -> dict:
    path = _state_path(live_root)
    if path.exists():
        state = json.loads(path.read_text())
        state.setdefault("holdings", {})
        state.setdefault("cost_prices", {})
        state.setdefault("pending_plan", None)
        state.setdefault("pending_corporate_actions", [])
        return state
    return {
        "version": 1, "mode": "paper", "initial_cash": cfg.initial_cash,
        "cash": cfg.initial_cash, "holdings": {}, "cost_prices": {},
        "as_of": str(pd.Timestamp(cutoff).date()) if cutoff is not None else None,
        "pending_plan": None, "last_plan_run": None, "executions": 0,
        "pending_corporate_actions": [],
    }


def paper_symbols(cfg: Config | None = None, live_root=LIVE_ROOT) -> set[str]:
    path = _state_path(live_root)
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()).get("holdings", {}))
    except Exception:
        return set()


def _market_for_day(folder: Path, day) -> pd.DataFrame:
    path = folder / "prices.parquet"
    date = pd.Timestamp(day).normalize()
    try:
        return pd.read_parquet(path, filters=[("date", "=", date)])
    except Exception:
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame.date).dt.normalize()
        return frame[frame.date == date]


def _append_executions(live_root: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path = live_root / "paper_executions.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    frame = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    keys = [column for column in ["signal_date", "trade_date", "symbol", "side"] if column in frame]
    if keys:
        frame = frame.drop_duplicates(keys, keep="last")
    frame.to_parquet(path, index=False)


def _write_account_files(folder: Path, state: dict, latest, market: pd.DataFrame) -> dict:
    holdings = {symbol: int(shares) for symbol, shares in state.get("holdings", {}).items() if int(shares) > 0}
    rows = [{"symbol": symbol, "shares": shares, "sellable_shares": shares,
             "cost_price": float(state.get("cost_prices", {}).get(symbol, 0.))}
            for symbol, shares in sorted(holdings.items())]
    pd.DataFrame(rows, columns=SCHEMAS["holdings"]).to_parquet(folder / "holdings.parquet", index=False)
    marks = market.set_index("symbol") if len(market) else pd.DataFrame()
    missing = set(holdings) - set(marks.index)
    if missing:
        raise ValueError(f"模拟账户持仓缺少最新收盘价：{sorted(missing)}")
    market_value = sum(shares * float(marks.loc[symbol, "close"]) for symbol, shares in holdings.items())
    account = {"mode": "paper", "as_of": str(pd.Timestamp(latest).date()),
               "cash": float(state["cash"]), "market_value": market_value,
               "nav": float(state["cash"] + market_value), "positions": len(holdings),
               "quarantined_corporate_actions": state.get("pending_corporate_actions", [])}
    _write_json(folder / "account.json", account)
    return account


def advance_paper_account(dataset_folder: Path, cfg: Config, live_root=LIVE_ROOT) -> dict:
    """Apply prior conditional plan at the observed open and carry the account."""
    folder, live_root = Path(dataset_folder), Path(live_root)
    manifest = json.loads((folder / "manifest.json").read_text())
    latest = pd.Timestamp(manifest["live_signal_date"]).normalize()
    cutoff = pd.Timestamp(manifest["historical_research_cutoff"]).normalize()
    state = load_paper_state(cfg, live_root, cutoff)
    if state.get("as_of") is None:
        state["as_of"] = str(cutoff.date())
    last = pd.Timestamp(state["as_of"]).normalize()
    # Before the first paper trade, changing the configured starting capital is
    # safe and expected.  Once a fill exists the account is never silently reset.
    if state.get("executions", 0) == 0 and not state.get("holdings") and not state.get("pending_plan"):
        state["initial_cash"] = cfg.initial_cash
        state["cash"] = cfg.initial_cash
    actions = read_table(folder, "actions")
    fees = read_table(folder, "fees")
    calendar = read_table(folder, "calendar")
    sessions = calendar[(calendar.date > last) & (calendar.date <= latest)].date.sort_values()
    pending = state.get("pending_plan")
    execution_rows = []
    holdings = {symbol: int(value) for symbol, value in state.get("holdings", {}).items()}
    costs = {symbol: float(value) for symbol, value in state.get("cost_prices", {}).items()}
    cash = float(state["cash"])
    pending_actions = {
        (str(row["symbol"]), str(pd.Timestamp(row["date"]).date())): dict(row)
        for row in state.get("pending_corporate_actions", [])
        if row.get("symbol") and row.get("date")
    }
    current_quarantines = {
        (str(row["symbol"]), str(pd.Timestamp(row["date"]).date())): dict(row)
        for row in manifest.get("corporate_action_quarantine", [])
    }
    # A later refresh may successfully resolve a formerly quarantined row.  If
    # the corrected market row is tradable again, remove the stale quarantine
    # instead of carrying it forever in the paper account.
    prices = read_table(folder, "prices")
    resolved_market_keys = {
        (str(row.symbol), str(pd.Timestamp(row.date).date()))
        for row in prices.loc[prices.suspended.eq(0), ["symbol", "date"]].itertuples(index=False)
    }
    for key in list(pending_actions):
        if key not in current_quarantines and key in resolved_market_keys:
            pending_actions.pop(key, None)
    for row in current_quarantines.values():
        pending_actions[(str(row["symbol"]), str(pd.Timestamp(row["date"]).date()))] = dict(row)

    # If a vendor supplies the missing detail later, repair the frozen holding
    # exactly once even when the account has already advanced past the event day.
    for key,row in list(pending_actions.items()):
        event_date=pd.Timestamp(row["date"]).normalize()
        matching=actions[(actions.symbol==row["symbol"]) & (actions.date==event_date)] if len(actions) else actions
        if event_date<=last and len(matching):
            action=matching.sort_values("known_at").iloc[-1]
            symbol=row["symbol"]
            if symbol in holdings:
                shares=holdings[symbol]
                cash += shares*float(action.cash_per_share)
                multiplier=float(action.share_multiplier)
                holdings[symbol]=int(np.floor(shares*multiplier))
                if multiplier>0 and symbol in costs:costs[symbol]/=multiplier
            pending_actions.pop(key,None)
    for day in sessions:
        day = pd.Timestamp(day)
        day_actions = actions[actions.date == day] if len(actions) else actions
        for action in day_actions.itertuples():
            if action.symbol in holdings:
                shares = holdings[action.symbol]
                cash += shares * float(action.cash_per_share)
                multiplier = float(action.share_multiplier)
                holdings[action.symbol] = int(np.floor(shares * multiplier))
                if multiplier > 0 and action.symbol in costs:
                    costs[action.symbol] /= multiplier
            pending_actions.pop((str(action.symbol),str(day.date())),None)
        if pending and pd.Timestamp(pending["trade_date"]).normalize() == day:
            market = _market_for_day(folder, day)
            before = dict(holdings)
            holdings, cash, logs = execute_plan(
                pending, market, holdings, cash, cfg, fees,
                sellable=before,
                action_symbols=(set(day_actions.symbol) if len(day_actions) else set()) | {key[0] for key in pending_actions},
            )
            for log in logs:
                log["paper_recorded_at"] = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
                if log.get("filled_shares", 0) > 0 and log["side"] == "BUY":
                    symbol, quantity = log["symbol"], int(log["filled_shares"])
                    old = before.get(symbol, 0)
                    costs[symbol] = (costs.get(symbol, 0.) * old + float(log["fill_price"]) * quantity) / (old + quantity)
                if log.get("filled_shares", 0) > 0 and log["side"] == "SELL" and log["symbol"] not in holdings:
                    costs.pop(log["symbol"], None)
            execution_rows.extend(logs)
            pending = None
        elif pending and pd.Timestamp(pending["trade_date"]).normalize() < day:
            # An unobserved/expired trade date is never rolled forward.
            pending = None
    state.update(cash=cash, holdings=holdings, cost_prices=costs, as_of=str(latest.date()),
                 pending_corporate_actions=list(pending_actions.values()),
                 pending_plan=pending, executions=int(state.get("executions", 0)) + sum(int(x.get("filled_shares", 0) > 0) for x in execution_rows))
    _append_executions(live_root, execution_rows)
    market = _market_for_day(folder, latest)
    account = _write_account_files(folder, state, latest, market)
    _write_json(_state_path(live_root), state)
    return {"account": account, "executions": execution_rows, "state": state}


def register_pending_plan(report: dict, live_root=LIVE_ROOT) -> None:
    cfg = Config(**report["config"])
    state = load_paper_state(cfg, live_root, report.get("historical_research_cutoff"))
    plan = report["plan"]
    state["pending_plan"] = plan if plan.get("orders") and plan.get("status") not in {"DATA_BLOCKED", "STALE_BLOCKED"} else None
    state["last_plan_run"] = report["run_id"]
    _write_json(_state_path(live_root), state)


def verify_and_finalize_run(run_id: str, cfg: Config, live_root=LIVE_ROOT) -> dict:
    """A second source must confirm every order-sensitive closing price."""
    run_dir = ROOT / "runs" / Path(run_id).name
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text())
    plan = report["plan"]
    order_symbols = [row["symbol"] for row in plan.get("orders", [])]
    check_symbols = order_symbols or [row["symbol"] for row in report.get("rankings", [])[:5]]
    verification = crosscheck_symbols(
        check_symbols, report["as_of"], tolerance_bps=cfg.live_crosscheck_tolerance_bps,
        raw_output=run_dir / "market_crosscheck.parquet",
    )
    # Missing cross-checks block actual conditional orders.  A monitor-only day
    # remains visible with a warning because no order can be acted upon.
    if order_symbols and not verification["passed"]:
        plan["blocked_orders"] = plan.get("orders", [])
        plan["orders"] = []
        plan["status"] = "DATA_BLOCKED"
        report["warnings"].append("独立行情源未确认全部计划标的，订单已自动阻断。")
    elif not verification["passed"]:
        report["warnings"].append("独立行情源本次未完整响应；今日没有订单，评分仍保留观察。")
    plan["source_verification"] = verification
    report["source_verification"] = verification
    report["plan"] = plan
    _write_json(report_path, report)
    _write_json(run_dir / "plan.json", plan)
    columns = list((plan.get("orders") or plan.get("blocked_orders") or [{"signal_date": None, "trade_date": None, "symbol": None}])[0])
    pd.DataFrame(plan.get("orders", []), columns=columns).to_csv(run_dir / "orders.csv", index=False)
    register_pending_plan(report, live_root)
    return report
