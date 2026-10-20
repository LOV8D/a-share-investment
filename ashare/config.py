from dataclasses import asdict, dataclass
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Config:
    initial_cash: float = 100_000
    top_k: int = 10
    hold_buffer: int = 0
    max_weight: float = .10
    max_sector_weight: float = .30
    cash_reserve: float = .03
    max_daily_turnover: float = 2.0  # permits an exact daily top-ten rotation
    max_daily_orders: int = 15
    min_trade_weight: float = .025
    max_adv_participation: float = .005
    commission_rate: float = .00025
    min_commission: float = 5.
    slippage_bps: float = 10.
    max_open_gap: float = .03
    label_horizon: int = 5
    min_train_days: int = 252
    train_window_days: int = 756
    refit_every: int = 20
    embargo_days: int = 5
    decision_time: str = "23:59:59"
    schedule_time: str = "00:00:00"
    paper_replay_start: str = "2026-09-15"
    event_half_life_days: float = 3.
    enable_text_alpha: bool = True
    text_alpha_weight: float = .20
    worldquant_weight: float = .50
    min_text_coverage: float = .05
    min_liquidity: float = 20_000_000
    min_listed_days: int = 120
    seed: int = 42
    daily_schedule_enabled: bool = True
    rebalance_frequency: str = "daily"
    live_price_coverage_min: float = .995
    live_crosscheck_tolerance_bps: float = 20.

    def __post_init__(self):
        import math
        for key,value in asdict(self).items():
            if isinstance(value,(float,int)) and not isinstance(value,bool) and not math.isfinite(value):
                raise ValueError(f"{key} 必须为有限数值")
        if not (1 <= self.top_k <= 300 and self.initial_cash > 0):
            raise ValueError("top_k 必须为 1–300，资金必须大于零")
        if not (0 < self.max_weight <= 1 and self.max_weight <= self.max_sector_weight <= 1):
            raise ValueError("仓位约束必须满足 0 < 个股上限 <= 行业上限 <= 1")
        if not (0 <= self.cash_reserve < 1 and 0 < self.max_daily_turnover <= 2):
            raise ValueError("现金/换手约束不合法")
        if not (self.label_horizon >= 1 and self.min_train_days >= 40 and self.refit_every >= 1 and self.embargo_days >= 0):
            raise ValueError("训练窗口/预测周期不合法")
        if self.train_window_days < self.min_train_days:
            raise ValueError("训练窗口不能小于最小训练期")
        if not (0 <= self.text_alpha_weight <= 1 and 0 <= self.worldquant_weight <= 1 and self.event_half_life_days > 0):
            raise ValueError("文本参数不合法")
        if any(x < 0 for x in [self.commission_rate, self.min_commission, self.slippage_bps, self.max_open_gap, self.min_trade_weight]):
            raise ValueError("费用、价差和最小交易额不能为负")
        if not (0 < self.max_adv_participation <= 1):
            raise ValueError("成交参与率必须在 (0,1] 内")
        integer_fields=[self.top_k,self.max_daily_orders,self.refit_every,self.hold_buffer,self.label_horizon,self.min_train_days,self.train_window_days,self.embargo_days,self.min_listed_days,self.seed]
        if not (all(type(v) is int for v in integer_fields) and self.hold_buffer>=0 and 1<=self.max_daily_orders<=600):
            raise ValueError("持股数、每日订单上限、重训间隔和缓冲名次必须为合法整数")
        if not 0<=self.min_text_coverage<=1 or self.min_liquidity<0 or self.min_listed_days<0:
            raise ValueError("数据覆盖率/筛选阈值不合法")
        if self.rebalance_frequency not in {"daily", "weekly"}:
            raise ValueError("调仓频率只支持 daily 或 weekly")
        if not (.95 <= self.live_price_coverage_min <= 1):
            raise ValueError("实时行情覆盖率阈值必须在 0.95–1 之间")
        if not (0 <= self.live_crosscheck_tolerance_bps <= 200):
            raise ValueError("双源收盘价容差必须在 0–200bp 之间")
        import datetime
        datetime.time.fromisoformat(self.decision_time)
        datetime.time.fromisoformat(self.schedule_time)
        datetime.date.fromisoformat(self.paper_replay_start)

    def dict(self):
        return asdict(self)

def load_config(path=None, overrides=None):
    p = Path(path) if path else ROOT / "config.json"
    data = json.loads(p.read_text()) if p.exists() else {}
    return Config(**(data | (overrides or {})))
