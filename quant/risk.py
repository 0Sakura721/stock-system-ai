# -*- coding: utf-8 -*-
"""风控模块：仓位、单票、止损、涨跌停、可交易性、资金校验。

所有下单动作必须先过 RiskEngine.check_order()，返回 (ok, reason)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class RiskVerdict:
    ok: bool
    reason: str = ""
    adjusted_shares: int | None = None


class RiskEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.s = cfg.strategy
        self.a = cfg.account

    # ---------------------------------------------------------- 账户级
    def position_ratio(self, equity: float, pos_value: float) -> float:
        return pos_value / equity if equity > 0 else 0.0

    def can_increase(self, equity: float, pos_value: float, add_value: float,
                     target_pos: float) -> RiskVerdict:
        cap = min(target_pos, self.a.max_total_position)
        new_ratio = (pos_value + add_value) / equity if equity > 0 else 0
        if new_ratio > cap + 1e-6:
            over = (pos_value + add_value) - equity * cap
            return RiskVerdict(False, f"总仓位将达 {new_ratio:.0%}，超上限 {cap:.0%}（需减 {over:,.0f} 元）")
        return RiskVerdict(True)

    # ---------------------------------------------------------- 单票级
    def check_single_weight(self, equity: float, stock_value: float) -> RiskVerdict:
        w = stock_value / equity if equity > 0 else 0
        if w > self.s.max_weight + 1e-6:
            return RiskVerdict(False, f"单票权重 {w:.1%} 超上限 {self.s.max_weight:.0%}")
        return RiskVerdict(True)

    def check_lot(self, shares: int, lot: int = 100) -> RiskVerdict:
        if shares < lot:
            return RiskVerdict(False, f"不足一手（{shares} < {lot}）")
        if shares % lot != 0:
            adj = (shares // lot) * lot
            return RiskVerdict(True, f"取整为 {adj} 股", adj)
        return RiskVerdict(True)

    def check_cash(self, cash: float, amount: float, fee: float) -> RiskVerdict:
        if amount + fee > cash:
            return RiskVerdict(False, f"资金不足：需 {amount+fee:,.0f}，可用 {cash:,.0f}")
        return RiskVerdict(True)

    # ---------------------------------------------------------- 交易性
    def check_buy_tradable(self, price: float, pre_close: float, code: str,
                           amount_today: float = None) -> RiskVerdict:
        if not price or not np.isfinite(price) or price <= 0:
            return RiskVerdict(False, "无有效价格（停牌或数据缺失）")
        if not pre_close or pre_close <= 0:
            return RiskVerdict(True, "无昨收，跳过涨停校验")
        pct = price / pre_close - 1
        lim = self._limit(code)
        if pct >= lim - 0.002:
            return RiskVerdict(False, f"开盘涨停 {pct:+.1%}，无法买入")
        if amount_today is not None and amount_today <= 0:
            return RiskVerdict(False, "当日无成交（停牌）")
        return RiskVerdict(True)

    def check_sell_tradable(self, price: float, pre_close: float, code: str) -> RiskVerdict:
        if not price or not np.isfinite(price) or price <= 0:
            return RiskVerdict(False, "无有效价格（停牌）")
        if not pre_close or pre_close <= 0:
            return RiskVerdict(True)
        pct = price / pre_close - 1
        lim = self._limit(code)
        if pct <= -lim + 0.002:
            return RiskVerdict(False, f"跌停 {pct:+.1%}，卖不出，次日再试")
        return RiskVerdict(True)

    @staticmethod
    def _limit(code: str) -> float:
        c = str(code).zfill(6)
        if c.startswith(("300", "301", "688")):
            return 0.20
        if c.startswith(("8", "4", "9")):
            return 0.30
        return 0.10

    # ---------------------------------------------------------- 止损
    def stop_price(self, cost: float, peak: float, atr: float = None) -> tuple[float, str]:
        """返回建议止损价与依据。"""
        hard = cost * (1 + self.s.stop_loss)
        trail = peak * (1 + self.s.trail_stop) if peak > 0 else None
        atr_s = peak - self.s.atr_stop_mult * atr if (atr and peak > 0) else None
        picks = [x for x in (hard, trail, atr_s) if x and np.isfinite(x) and x > 0]
        p = max(picks) if picks else hard
        basis = []
        if p == hard:
            basis.append(f"成本止损 {self.s.stop_loss:.0%}")
        if trail and p == trail:
            basis.append(f"回撤止损 {self.s.trail_stop:.0%}")
        if atr_s and p == atr_s:
            basis.append(f"ATR×{self.s.atr_stop_mult}")
        return round(p, 2), " / ".join(basis)

    def should_stop(self, cost: float, price: float, peak: float, days: int,
                    atr: float = None) -> tuple[bool, str]:
        ret = price / cost - 1 if cost > 0 else 0
        trail = price / peak - 1 if peak > 0 else 0
        if ret <= self.s.stop_loss:
            return True, f"触发成本止损 {ret:.1%}"
        if peak / cost - 1 >= self.s.take_profit and trail <= self.s.take_profit_trail:
            return True, f"触发移动止盈（峰值+{peak/cost-1:.0%}，回撤{trail:.1%}）"
        if trail <= self.s.trail_stop and ret > 0.02:
            return True, f"触发回撤止盈 {trail:.1%}"
        if days >= self.s.max_hold_days:
            return True, f"持有到期 {days} 日"
        return False, ""

    # ---------------------------------------------------------- 汇总检查
    def full_check(self, side: str, code: str, price: float, shares: int,
                   pre_close: float, cash: float, equity: float, pos_value: float,
                   target_pos: float, cur_weight: float = 0.0,
                   amount_today: float = None) -> RiskVerdict:
        if side == "BUY":
            v = self.check_buy_tradable(price, pre_close, code, amount_today)
            if not v.ok:
                return v
            lot = self.check_lot(shares)
            if not lot.ok:
                return lot
            sh = lot.adjusted_shares or shares
            amt = price * sh
            fee = self.cfg.cost.buy_cost(amt)
            c = self.check_cash(cash, amt, fee)
            if not c.ok:
                return c
            p = self.can_increase(equity, pos_value, amt + fee, target_pos)
            if not p.ok:
                return p
            w = self.check_single_weight(equity, cur_weight * equity + amt)
            if not w.ok:
                return w
            return RiskVerdict(True, "", sh)
        else:
            v = self.check_sell_tradable(price, pre_close, code)
            if not v.ok:
                return v
            lot = self.check_lot(shares)
            return RiskVerdict(True, "", lot.adjusted_shares or shares)
