# -*- coding: utf-8 -*-
"""策略配置 v2：多因子 + 择时 + 风控，面向全A。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict, replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output")

for _d in (DATA_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "market.sqlite")


@dataclass
class CostModel:
    """A股真实成本。"""
    commission_rate: float = 0.00025
    commission_min: float = 5.0
    stamp_tax: float = 0.0005
    transfer_fee: float = 0.00001
    slippage: float = 0.0010

    def buy_cost(self, amount: float) -> float:
        if amount <= 0:
            return 0.0
        return max(amount * self.commission_rate, self.commission_min) + amount * self.transfer_fee

    def sell_cost(self, amount: float) -> float:
        if amount <= 0:
            return 0.0
        return (max(amount * self.commission_rate, self.commission_min)
                + amount * self.stamp_tax + amount * self.transfer_fee)


@dataclass
class StrategyConfig:
    # 股票池
    min_list_days: int = 250
    min_avg_amount: float = 8e7
    min_price: float = 2.0
    max_price: float = 500.0
    exclude_st: bool = True
    exclude_bj: bool = True
    exclude_kcb: bool = False
    exclude_cyb: bool = False
    min_turnover_amt: float = 1e6

    # 因子权重
    w_mom20: float = 0.28
    w_mom60: float = 0.18
    w_pos52: float = 0.16
    w_ma_gap: float = 0.14
    w_rev5: float = 0.12
    w_vol20: float = 0.12
    w_amount_z: float = 0.10

    # 组合
    max_positions: int = 8
    rebalance_days: int = 10
    max_weight: float = 0.22
    min_score: float = -1.0
    keep_rank_mult: float = 1.8

    # IC 因子模型
    is_start: str = "2017-01-01"     # 样本内起点
    is_end: str = "2019-12-31"       # 样本内终点（样本外从 2020 起）
    horizon: int = 20                # 因子预测周期（交易日）
    top_k: int = 6                   # 保留因子数

    # 择时
    timing_index: str = "399006"
    timing_fast: int = 15
    timing_slow: int = 45
    timing_bear_ma: int = 150
    pos_full: float = 1.00
    pos_half: float = 0.50
    pos_weak: float = 0.30
    pos_empty: float = 0.00
    use_breadth: bool = True
    breadth_hi: float = 0.55
    breadth_lo: float = 0.25

    # 个股风控
    stop_loss: float = -0.08
    trail_stop: float = -0.15
    take_profit: float = 0.30
    take_profit_trail: float = -0.08
    max_hold_days: int = 30
    atr_stop_mult: float = 2.5


@dataclass
class AccountConfig:
    initial_cash: float = 100_000.0
    max_total_position: float = 0.95
    cash_buffer: float = 0.05
    lot_size: int = 100


@dataclass
class Config:
    cost: CostModel = field(default_factory=CostModel)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    account: AccountConfig = field(default_factory=AccountConfig)

    def to_dict(self):
        return asdict(self)

    def copy_with(self, **kw):
        st = replace(self.strategy, **{k: v for k, v in kw.items()
                                      if hasattr(self.strategy, k)})
        acc = replace(self.account, **{k: v for k, v in kw.items()
                                       if hasattr(self.account, k)})
        cst = replace(self.cost, **{k: v for k, v in kw.items()
                                    if hasattr(self.cost, k)})
        return Config(cost=cst, strategy=st, account=acc)


CFG = Config()

FETCH = dict(
    workers=10,
    timeout=20,
    retries=4,
    qps=12.0,
    beg="20170101",
    hfq=2,
)
