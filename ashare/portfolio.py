"""Long-only, integer-share, next-session conditional execution simulator."""
import math
import numpy as np
import pandas as pd

def should_rebalance(signal_date, next_date, cfg):
    """Run every day, but trade only at the configured calendar boundary."""
    if cfg.rebalance_frequency == "daily":
        return True
    signal_iso = pd.Timestamp(signal_date).isocalendar()
    next_iso = pd.Timestamp(next_date).isocalendar()
    return (signal_iso.year, signal_iso.week) != (next_iso.year, next_iso.week)

def round_order(quantity, min_order, step, side="BUY", owned=0):
    quantity = max(0, int(math.floor(quantity)))
    if side == "SELL" and quantity >= owned > 0:
        return int(owned)  # full liquidation includes residual odd lots
    if quantity < min_order:
        return 0
    return int(min_order + ((quantity-min_order)//step)*step)

def trade_fee(notional, side, date, cfg, fees):
    row = fees[(fees.start_date <= pd.Timestamp(date)) & (fees.end_date >= pd.Timestamp(date))]
    if len(row) != 1:
        raise ValueError(f"{date}: 费率区间缺失或重复")
    r = row.iloc[0]
    return max(cfg.min_commission, notional*cfg.commission_rate) + notional*float(r.transfer_rate) + (notional*float(r.sell_stamp_rate) if side == "SELL" else 0)

def allocate_weights(selected, cfg, reserved_weight=0., reserved_sector_weights=None):
    weights = {s: 0. for s in selected.symbol}
    sectors = dict(zip(selected.symbol, selected.industry))
    reserved_sector_weights = reserved_sector_weights or {}
    inv = {r.symbol: 1/max(float(r.volatility_20), .10) if np.isfinite(r.volatility_20) else 1/.3 for r in selected.itertuples()}
    for _ in range(50):
        remaining = 1-cfg.cash_reserve-reserved_weight-sum(weights.values())
        sector_sums = {v: reserved_sector_weights.get(v,0.)+sum(w for s,w in weights.items() if sectors[s] == v) for v in set(sectors.values())}
        active = [s for s,w in weights.items() if w < cfg.max_weight-1e-10 and sector_sums[sectors[s]] < cfg.max_sector_weight-1e-10]
        if remaining < 1e-8 or not active:
            break
        denominator = sum(inv[s] for s in active)
        for s in active:
            room = cfg.max_sector_weight-sector_sums[sectors[s]]
            addition = max(0., min(remaining*inv[s]/denominator, cfg.max_weight-weights[s], room))
            weights[s] += addition
            sector_sums[sectors[s]] += addition
    return weights

def make_plan(scored, holdings, cash, market, next_date, cfg, fees):
    date = pd.Timestamp(scored.date.max())
    prices = market.set_index("symbol")
    missing = set(holdings)-set(prices.index)
    if missing:
        raise ValueError(f"持仓缺少估值日线：{sorted(missing)}；需要包含历史退出成分的后续行情")
    nav = cash + sum(q*float(prices.loc[s,"close"]) for s,q in holdings.items())
    locked = {s:q for s,q in holdings.items() if "suspended" in prices and bool(prices.loc[s,"suspended"])}
    locked_weight = sum(q*float(prices.loc[s,"close"])/nav for s,q in locked.items()) if nav>0 else 0.
    locked_sector_weights = {}
    for symbol,quantity in locked.items():
        sector=str(prices.loc[symbol,"industry"])
        locked_sector_weights[sector]=locked_sector_weights.get(sector,0.)+quantity*float(prices.loc[symbol,"close"])/nav
    ranked = scored.sort_values("rank")
    eligible = ranked[ranked.eligible].copy()
    excluded = []
    for r in eligible.itertuples():
        minimum = r.close*(1+cfg.max_open_gap)*r.min_order
        if minimum > nav*cfg.max_weight:
            excluded.append({"symbol": r.symbol, "rank": int(r.rank), "reason": "最低申报金额超过单股仓位上限", "minimum_notional": round(minimum,2)})
    blocked = {r["symbol"] for r in excluded}
    viable = eligible[~eligible.symbol.isin(blocked)]
    # Selection is the exact eligible top K.  The order cap below may delay a
    # replacement, but it never substitutes rank K+1 for a deferred top-K name.
    selected = viable[~viable.symbol.isin(locked)].head(max(0,cfg.top_k-len(locked)))
    weights = allocate_weights(selected, cfg, locked_weight, locked_sector_weights)
    ranks = dict(zip(ranked.symbol, ranked["rank"]))
    adv = dict(zip(scored.symbol, scored.adv20))
    candidates = []
    target_quantities = {}
    rounded_weights = {}
    for symbol in sorted(set(weights)|set(holdings)):
        r = prices.loc[symbol]
        owned = int(holdings.get(symbol,0))
        if symbol in locked:
            target_quantities[symbol] = owned
            rounded_weights[symbol] = owned*float(r.close)/nav
            excluded.append({"symbol": symbol, "rank": int(ranks.get(symbol,999)),
                             "reason": "停牌或公司行动明细待补，仓位冻结且不生成买卖订单",
                             "minimum_notional": round(float(r.close)*int(r.min_order),2)})
            continue
        target_value = weights.get(symbol,0)*nav
        # Budget at the upper conditional price, not an unknown tomorrow open.
        buy_limit = math.floor(float(r.close)*(1+cfg.max_open_gap)*100)/100
        sell_limit = math.ceil(float(r.close)*(1-cfg.max_open_gap)*100)/100
        target_shares = round_order(target_value/max(buy_limit,.01), int(r.min_order), int(r.lot_step))
        if target_shares > 0:
            target_quantities[symbol] = target_shares
            rounded_weights[symbol] = target_shares*float(r.close)/nav
        delta = target_shares-owned
        if delta == 0:
            continue
        side = "BUY" if delta > 0 else "SELL"
        limit = buy_limit if side == "BUY" else sell_limit
        qty = round_order(abs(delta), int(r.min_order), int(r.lot_step), side, owned)
        if qty == 0 or (qty*float(r.close) < nav*cfg.min_trade_weight and not (side == "SELL" and target_shares == 0)):
            continue
        candidates.append({"signal_date": str(date.date()), "trade_date": str(next_date)[:10], "symbol": symbol, "side": side, "shares": qty, "reference_price": float(r.close), "limit_price": limit, "target_weight": weights.get(symbol,0), "rank": int(ranks.get(symbol,999)), "industry": str(r.industry), "adv20": float(adv.get(symbol, np.nan)), "min_order": int(r.min_order), "lot_step": int(r.lot_step), "reason": "排名缓冲/仓位调整" if symbol in weights else "退出候选池或不满足筛选条件", "full_liquidation": side == "SELL" and target_shares == 0, "status": "CONDITIONAL"})
    def priority(order):
        if order["full_liquidation"]:
            return (0, -order["rank"])  # free cash and remove the worst ranks first
        if order["side"] == "BUY" and order["symbol"] not in holdings:
            return (1, order["rank"])   # enter new top-K names from best to worst
        if order["side"] == "SELL":
            return (2, -order["rank"])
        return (3, order["rank"])
    candidates.sort(key=priority)
    turnover_left = cfg.max_daily_turnover*nav
    projected_cash = cash
    orders = []
    deferred_by_order_cap = 0
    for index,o in enumerate(candidates):
        if len(orders) >= cfg.max_daily_orders:
            deferred = candidates[index:]
            deferred_by_order_cap = len(deferred)
            excluded.extend({"symbol": row["symbol"], "rank": row["rank"],
                             "reason": f"达到每日{cfg.max_daily_orders}笔买卖上限，顺延后重新评分",
                             "minimum_notional": round(row["limit_price"]*row["min_order"],2)}
                            for row in deferred)
            break
        amount_cap = turnover_left
        if math.isfinite(o["adv20"]):
            amount_cap = min(amount_cap, o["adv20"]*cfg.max_adv_participation)
        else:
            # Missing ADV for an exited constituent: use market-provided trailing history if available.
            amount_cap = min(amount_cap, float(prices.loc[o["symbol"]].get("adv20",0))*cfg.max_adv_participation)
        if o["side"] == "BUY":
            amount_cap = min(amount_cap, max(0,projected_cash-cfg.cash_reserve*nav-cfg.min_commission)/(1+cfg.commission_rate+.002))
        qty = round_order(min(o["shares"], amount_cap/o["limit_price"]), o["min_order"], o["lot_step"], o["side"], holdings.get(o["symbol"],0))
        if qty <= 0:
            excluded.append({"symbol": o["symbol"], "rank": o["rank"], "reason": "换手/流动性/现金预算不足以完成最低申报", "minimum_notional": round(o["limit_price"]*o["min_order"],2)})
            continue
        amount = qty*o["limit_price"]
        full_exit = o["full_liquidation"] and qty >= holdings.get(o["symbol"], 0)
        if amount < nav*cfg.min_trade_weight and not full_exit:
            excluded.append({"symbol": o["symbol"], "rank": o["rank"], "reason": "预算截断后的订单低于最小交易占比", "minimum_notional": round(nav*cfg.min_trade_weight,2)})
            continue
        fee = trade_fee(amount, o["side"], next_date, cfg, fees)
        o.update(shares=qty, estimated_notional=round(amount,2), estimated_fee=round(fee,2), fee_bps=round(fee/amount*1e4,1))
        projected_cash += amount-fee if o["side"] == "SELL" else -amount-fee
        turnover_left -= amount
        orders.append(o)
    top = eligible.head(cfg.top_k)
    min_capital = max((float((top.close*top.min_order/cfg.max_weight).max()) if len(top) else 0), float((top.close*top.min_order).sum())/(1-cfg.cash_reserve))
    return {"signal_date": str(date.date()), "trade_date": str(next_date)[:10], "nav": round(nav,2), "cash": round(cash,2), "target_weights": rounded_weights, "continuous_target_weights": weights, "target_shares": target_quantities, "orders": orders, "excluded": excluded, "selected": list(target_quantities), "theoretical_top": top.symbol.tolist(), "minimum_capital_for_topk": round(min_capital*(1+cfg.max_open_gap),2), "projected_cash_if_all_fill": round(projected_cash,2), "max_daily_orders": cfg.max_daily_orders, "deferred_by_order_cap": deferred_by_order_cap, "status": "CONDITIONAL_RESEARCH_PLAN", "conditions": ["仅在下一交易日有效；开盘预检通过后才可使用", f"每日买卖指令合计不超过{cfg.max_daily_orders}笔；超出部分不追单，下一交易日重新评分", "买卖价格须在计划限价和当日交易范围内", "买入预算依赖卖单成交；未成交订单当日失效", "实际账户可卖数量、板块权限和公司行动须复核"]}

def execute_plan(plan, market, holdings, cash, cfg, fees, sellable=None, action_symbols=None):
    """Simulate an opening fill; all eligibility decisions use signal-day data.

    Opening prices and daily limits determine fills only, never ranking or quantities.
    """
    p = market.set_index("symbol")
    sellable = dict(holdings) if sellable is None else dict(sellable)
    holdings = dict(holdings)
    action_symbols = action_symbols or set()
    logs = []
    nav_open = cash+sum(q*float(p.loc[s,"open"] if p.loc[s,"open"] > 0 else p.loc[s,"close"]) for s,q in holdings.items())
    turnover_remaining = plan["nav"]*cfg.max_daily_turnover
    for o in plan["orders"]:
        s, side = o["symbol"], o["side"]
        log = dict(o, filled_shares=0, fill_price=None, actual_fee=0., fill_status="REJECTED")
        reason = None
        if s not in p.index:
            reason = "缺少交易日行情"
        elif s in action_symbols:
            reason = "除权除息日取消旧价格订单，收盘后重新生成"
        elif p.loc[s,"suspended"] or p.loc[s,"open"] <= 0:
            reason = "停牌/无有效开盘价"
        if reason:
            logs.append(log | {"fill_reason": reason}); continue
        r = p.loc[s]
        if (side == "BUY" and pd.notna(r.upper_limit) and r.open >= r.upper_limit-.001) or (side == "SELL" and pd.notna(r.lower_limit) and r.open <= r.lower_limit+.001):
            logs.append(log | {"fill_reason": "开盘触及方向性涨跌停，保守假设无法成交"}); continue
        price = round(float(r.open)*(1+cfg.slippage_bps/1e4*(1 if side == "BUY" else -1)),2)
        if (side == "BUY" and price > o["limit_price"]) or (side == "SELL" and price < o["limit_price"]):
            logs.append(log | {"fill_reason": "开盘跳空/滑点超出计划限价"}); continue
        if (pd.notna(r.upper_limit) and price > r.upper_limit) or (pd.notna(r.lower_limit) and price < r.lower_limit):
            logs.append(log | {"fill_reason": "模拟成交价超出当日价格限制"}); continue
        qty = round_order(min(o["shares"],turnover_remaining/price),o["min_order"],o["lot_step"],side,holdings.get(s,0))
        if side == "SELL":
            qty = round_order(min(qty,sellable.get(s,0)),o["min_order"],o["lot_step"],side,holdings.get(s,0))
        else:
            reserve = cfg.cash_reserve*nav_open
            own_value = holdings.get(s,0)*float(r.open)
            sector_value = sum(q*float(p.loc[k,"open"] if p.loc[k,"open"] > 0 else p.loc[k,"close"]) for k,q in holdings.items() if p.loc[k,"industry"] == r.industry)
            budget = min(max(0,cash-reserve-cfg.min_commission)/(1+cfg.commission_rate+.002), max(0,cfg.max_weight*nav_open-own_value), max(0,cfg.max_sector_weight*nav_open-sector_value))
            qty = round_order(min(qty,budget/price),o["min_order"],o["lot_step"])
        if qty <= 0:
            logs.append(log | {"fill_reason": "可用现金、T+1可卖数量或仓位上限不足"}); continue
        notional = price*qty
        full_exit = bool(o.get("full_liquidation")) and side == "SELL" and qty >= holdings.get(s, 0)
        if notional < plan["nav"]*cfg.min_trade_weight and not full_exit:
            logs.append(log | {"fill_reason": "实际可成交金额低于最小交易占比"}); continue
        fee = trade_fee(notional,side,plan["trade_date"],cfg,fees)
        if side == "BUY":
            if notional+fee > cash+1e-6:
                logs.append(log | {"fill_reason": "现金不足"}); continue
            cash -= notional+fee
            holdings[s] = holdings.get(s,0)+qty
            # Newly bought shares are deliberately not added to sellable (T+1).
        else:
            cash += notional-fee
            holdings[s] -= qty
            sellable[s] = sellable.get(s,0)-qty
            if holdings[s] == 0:
                del holdings[s]
        logs.append(log | {"filled_shares": qty, "fill_price": price, "actual_fee": fee, "fill_status": "FILLED" if qty == o["shares"] else "PARTIAL", "fill_reason": "次日开盘模拟成交"})
        turnover_remaining -= notional
    return holdings, cash, logs

def backtest(scores, ds, cfg, score_column="score", include_plans=False):
    holdings, cash, pending = {}, cfg.initial_cash, None
    curve, executions, plans = [], [], []
    sessions = sorted(ds["prices"].date.unique())
    by_score = {d: g for d,g in scores.groupby("date")}
    by_price = {d: g for d,g in ds["prices"].groupby("date")}
    # trailing liquidity for constituents after they exit the index
    raw = ds["prices"].sort_values(["symbol","date"]).copy()
    raw["adv20"] = raw.groupby("symbol").amount.transform(lambda x:x.rolling(20,min_periods=15).mean())
    by_price = {d:g for d,g in raw.groupby("date")}
    benchmark = ds["benchmark"].set_index("date")
    start = pd.Timestamp(scores.date.min())
    base_bench = float(benchmark.loc[start,"total_return_close"])
    previous_nav = cfg.initial_cash
    for i, day in enumerate(sessions):
        day = pd.Timestamp(day)
        if day < start:
            continue
        p = by_price[day]
        actions = ds["actions"]
        actions = actions[actions.date == day] if len(actions) else actions
        for a in actions.itertuples():
            if a.symbol in holdings:
                old = holdings[a.symbol]
                cash += old*a.cash_per_share
                holdings[a.symbol] = int(math.floor(old*a.share_multiplier))
        turnover, costs = 0.,0.
        if pending is not None and pending["trade_date"] == str(day.date()):
            holdings,cash,logs = execute_plan(pending,p,holdings,cash,cfg,ds["fees"],action_symbols=set(actions.symbol))
            executions += logs
            turnover = sum(l["filled_shares"]*(l["fill_price"] or 0) for l in logs)/previous_nav
            costs = sum(l["actual_fee"] for l in logs)
        marks = p.set_index("symbol")
        if set(holdings)-set(marks.index):
            raise ValueError("回测持仓丢失行情；请补齐退出成分/退市证券后续数据")
        nav = cash+sum(q*float(marks.loc[s,"close"]) for s,q in holdings.items())
        curve.append({"date": str(day.date()), "nav": nav, "equity": nav/cfg.initial_cash, "benchmark": float(benchmark.loc[day,"total_return_close"])/base_bench, "cash": cash, "exposure": 1-cash/nav, "turnover": turnover, "fees": costs, "positions": len(holdings)})
        previous_nav = nav
        pending = None
        if day in by_score and i+1 < len(sessions) and should_rebalance(day, sessions[i+1], cfg):
            scored = by_score[day].copy()
            if score_column != "score":
                scored["score"] = scored[score_column].rank(pct=True)*100
                scored["rank"] = scored.score.rank(ascending=False,method="first").astype(int)
            pending = make_plan(scored,holdings,cash,p,pd.Timestamp(sessions[i+1]),cfg,ds["fees"])
            plans.append(pending)
    result = (pd.DataFrame(curve), pd.DataFrame(executions), holdings, cash)
    return result + (plans,) if include_plans else result

def auction_precheck(plan, auction, sellable=None):
    """Checks only; does not send or mutate broker orders."""
    result=[]
    for o in plan["orders"]:
        rows=auction[(auction.symbol==o["symbol"]) & (auction.date==pd.Timestamp(plan["trade_date"]))]
        issues=[]
        if len(rows)!=1:
            issues.append("缺少目标交易日竞价数据")
        else:
            r=rows.iloc[0]
            expected=pd.Timestamp(plan["trade_date"]+" 09:30:00",tz="Asia/Shanghai")
            if not (expected-pd.Timedelta(minutes=10) <= r.available_at < expected):
                issues.append("竞价数据时间不在 09:20–09:30")
            if r.suspended: issues.append("停牌")
            if o["side"]=="BUY" and (r.indicative_open>o["limit_price"] or (pd.notna(r.upper_limit) and r.indicative_open>=r.upper_limit)):
                issues.append("买入价格超限或涨停")
            if o["side"]=="SELL" and (r.indicative_open<o["limit_price"] or (pd.notna(r.lower_limit) and r.indicative_open<=r.lower_limit)):
                issues.append("卖出价格超限或跌停")
        if o["side"]=="SELL" and (sellable is None or sellable.get(o["symbol"],0)<o["shares"]):
            issues.append("尚未确认账户可卖股数")
        result.append(o|{"precheck":"BLOCKED" if issues else "PRICE_CHECK_PASSED", "issues":issues, "account_check_required":True})
    return result
