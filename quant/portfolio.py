# -*- coding: utf-8 -*-
"""组合构建：目标持仓的稳定性控制（低换手版）。

为什么需要这个模块：
  回放测试发现，若每日按「目标权重 vs 当前持仓」逐股 diff，
  10 万本金 6 年会成交 2189 笔、换手 110 倍本金、手续费 1.4 万，
  把 +78% 的等权收益吃成 -36%。

  问题不在执行引擎，在目标持仓的定义：每天重算 top-N，
  排序稍变就引发整笔卖出再买入。

  解决：给目标组合加三个稳定器
    1. 排名容忍带 —— 已持有的股票，排名掉到 1.6 倍名单之外才卖
    2. 权重漂移容忍 —— 实际权重与目标偏离 < 25% 不动
    3. 最小交易额 —— 单笔低于 2% 权益的调整不做（整手门槛之上再设一层）

  这样「必要调整」和「噪声调整」被区分开，换手从 110 倍降到 20 倍以内。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig


def build_target(scores: pd.DataFrame, day: str, current_secids: list[str],
                 cfg: StrategyConfig, target_pos: float,
                 prev_targets: dict[str, float] | None = None) -> dict[str, float]:
    """目标权重（带粘性）。

    参数
      scores        当日打分面板
      day           信号日
      current_secids 当前持仓的 secid 列表
      target_pos    目标总仓位
      prev_targets  上一轮的目标权重（用于抑制来回摆动）

    返回 {secid: weight}，weight 是占总权益的比例（已乘 target_pos）
    """
    if target_pos <= 0.02:
        return {}

    row = scores.loc[day].dropna().sort_values(ascending=False)
    if row.empty:
        return {}

    max_w = cfg.max_weight
    # 需要多少只才能装下 target_pos
    n_need = max(1, int(np.ceil(target_pos / max_w)))
    n_max = max(cfg.max_positions, n_need)
    n_max = min(n_max, len(row))

    # ---- 第一步：保留已有的（排名容忍带 1.6 倍）
    keep_band = int(n_max * 1.6)
    picks: list[str] = []
    for sec in current_secids:
        if sec not in row.index:
            continue
        rank = int(row.index.get_loc(sec))
        if rank < keep_band:
            picks.append(sec)

    # ---- 第二步：新票补足到 n_max
    for sec in row.index:
        if len(picks) >= n_max:
            break
        if sec in picks:
            continue
        picks.append(sec)

    if not picks:
        return {}

    # ---- 第三步：等权分配，单票封顶
    per = min(target_pos / len(picks), max_w)
    weights = {s: per for s in picks}

    # ---- 第四步：抑制摆动——上一轮已持有且权重接近的，保持原权重
    if prev_targets:
        for s in list(weights):
            old = prev_targets.get(s, 0.0)
            if old > 0 and abs(old - weights[s]) / max(weights[s], 1e-9) < 0.25:
                weights[s] = old

    tot = sum(weights.values())
    if tot > target_pos:
        weights = {k: v * target_pos / tot for k, v in weights.items()}

    return weights


def diff_orders(target: dict[str, float], holdings: dict[str, dict],
                equity: float, prices: dict[str, float],
                cfg: StrategyConfig, min_trade_value: float = None) -> list[dict]:
    """目标 vs 持仓 → 订单清单（带最小交易额阈值）。

    holdings: {secid: {shares, cost, name}}
    prices:   {secid: 现价}
    """
    if min_trade_value is None:
        min_trade_value = equity * 0.02          # 单笔低于 2% 权益不动

    orders: list[dict] = []

    # ---- 卖
    for secid, h in holdings.items():
        px = prices.get(secid, np.nan)
        if not np.isfinite(px) or px <= 0:
            continue
        cur_val = h["shares"] * px
        tgt_val = target.get(secid, 0.0) * equity
        delta = cur_val - tgt_val

        if delta <= 0:
            continue
        # 全清（目标为 0）时必须卖，不受阈值限制
        if tgt_val <= 0:
            sh = int(h["shares"])
            reason = "调出组合"
        else:
            if delta < min_trade_value:
                continue
            sh = int(delta / px / 100) * 100
            reason = "减仓"
        if sh <= 0:
            continue
        orders.append(dict(side="SELL", secid=secid, shares=sh,
                           reason=reason, est_value=sh * px))

    # ---- 买（现金先加上卖出所得）
    avail = equity - sum(h["shares"] * prices.get(s, 0)
                         for s, h in holdings.items()) \
        + sum(o["est_value"] for o in orders)

    todo = {s: w * equity for s, w in target.items() if s not in holdings}
    for secid in sorted(todo, key=lambda s: -todo[s]):
        if avail <= 0:
            break
        need = todo[secid]
        if need < min_trade_value:
            continue
        px = prices.get(secid, np.nan)
        if not np.isfinite(px) or px <= 0:
            continue
        money = min(need, avail * 0.98)
        if money < px * 100:
            continue
        sh = int(money / px / 100) * 100
        if sh < 100:
            continue
        if sh * px < min_trade_value * 0.5:
            continue
        orders.append(dict(side="BUY", secid=secid, shares=sh,
                           reason=f"建仓 {target[secid]:.1%}",
                           est_value=sh * px))
        avail -= sh * px

    return orders
