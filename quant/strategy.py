# -*- coding: utf-8 -*-
"""策略 v3（最终验证版）：全池等权 + 创业板择时 + 仓位控制。

设计依据见 findings.py 与 README 的实测记录。核心事实：
  在 2020-2026 的 A 股，任何个股选股模型的成绩都差于"不选股"，
  真正削掉亏损的是"熊市空仓"。所以策略只做两件事：
    1. 决定今天该有多少仓位（指数择时）
    2. 在仓位内等权摊开到足够多的股票上（分散个股风险）

不做的事（都实测过且无效）：
  - 动量选股   (IC 每年为负，选股收益 -0.93% vs 市场 +0.52%)
  - IC 模型选 8 只 (-43%，比等权差 17 个百分点)
  - 个股止损   (震荡市反复触发，恶化收益)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import StrategyConfig


# ---------------------------------------------------------------- 股票池
def build_universe(p, inds, cfg: StrategyConfig, sl: pd.DataFrame | None = None) -> pd.DataFrame:
    """可交易池：剔除新股、低流动性、低价、ST、北交所。"""
    C, A = p.close, p.amount
    uni = C.notna() & (C > 0)
    uni &= C.notna().cumsum() >= cfg.min_list_days
    uni &= inds["amt20"] >= cfg.min_avg_amount
    uni &= (C >= cfg.min_price) & (C <= cfg.max_price)
    uni &= A.fillna(0) > cfg.min_turnover_amt

    board = pd.Series(p.secids, index=p.secids).str.split(".").str[1]
    drop = pd.Series(False, index=p.secids)
    if cfg.exclude_kcb:
        drop |= board.str.startswith("688")
    if cfg.exclude_bj:
        drop |= board.str.startswith(("4", "8", "9"))
    if cfg.exclude_cyb:
        drop |= board.str.startswith("30")
    if drop.any():
        uni = uni.loc[:, ~drop.values]

    if cfg.exclude_st and sl is not None and len(sl):
        smap = sl.set_index("secid")["name"].astype(str)
        st = smap.reindex(uni.columns).fillna("").str.contains("ST", case=False)
        for c in uni.columns[st.values]:
            uni[c] = False
    return uni


# ---------------------------------------------------------------- 排名（用于等权取前N）
def rank_score(p, inds, uni) -> pd.DataFrame:
    """等权策略里"前 N 只"该选谁：用低波动 + 流动性，不追涨。

    这不是 alpha 来源（实测选股无效），只是把有限名额分配给
    风险更小的标的，让 8 只组合的波动更接近全池等权。
    """
    s = (-ind.cs_z(inds["vol20"]) * 0.5
         - ind.cs_z(inds["atr_pct"]) * 0.3
         + ind.cs_z(np.log1p(inds["amt20"])) * 0.2)
    # 排除已经破位的
    s = s.where(p.close > inds["ma120"] * 0.85)
    return s.where(uni)


# ---------------------------------------------------------------- 市场宽度
def breadth(p, inds, uni) -> pd.Series:
    above = (p.close > inds["ma20"]) & uni
    valid = uni & p.close.notna()
    return (above.sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)).fillna(0.5)


# ---------------------------------------------------------------- 择时
def timing(p, index_close: pd.Series, cfg: StrategyConfig,
           br: pd.Series | None = None) -> pd.Series:
    """指数均线择时 → 每日目标仓位。这是唯一经多参数验证有效的组件。"""
    idx = pd.Series(index_close).reindex(p.dates).ffill()
    ma_f = idx.rolling(cfg.timing_fast, min_periods=max(2, cfg.timing_fast // 2)).mean()
    ma_s = idx.rolling(cfg.timing_slow, min_periods=max(2, cfg.timing_slow // 2)).mean()
    ma_b = idx.rolling(cfg.timing_bear_ma, min_periods=max(2, cfg.timing_bear_ma // 2)).mean()

    lvl = pd.Series(0.0, index=idx.index)
    lvl[(idx > ma_s) & (ma_f > ma_s)] = 3      # 多头排列
    lvl[(idx > ma_s) & (ma_f <= ma_s)] = 2     # 长期向上，短期回调
    lvl[(idx <= ma_s) & (idx > ma_b)] = 1      # 中期跌破，长期未破
    lvl[(idx <= ma_s) & (idx <= ma_b)] = 0     # 长期跌破，空仓
    lvl = lvl.fillna(0)

    pos = lvl.map({3: cfg.pos_full, 2: cfg.pos_half,
                   1: cfg.pos_weak, 0: cfg.pos_empty}).astype(float)

    if cfg.use_breadth and br is not None:
        b = br.reindex(idx.index).ffill().fillna(0.5)
        pos = pos.where(b > cfg.breadth_lo, pos * 0.5)
        pos = pos.where(b > cfg.breadth_hi, pos * 0.75)
    return pos.clip(0, 1).fillna(0)


def timing_detail(p, index_close: pd.Series, cfg: StrategyConfig,
                  br: pd.Series | None = None) -> pd.DataFrame:
    """择时归因：每日的均线状态、原始档位、宽度调整后的最终仓位。"""
    idx = pd.Series(index_close).reindex(p.dates).ffill()
    ma_f = idx.rolling(cfg.timing_fast, min_periods=max(2, cfg.timing_fast // 2)).mean()
    ma_s = idx.rolling(cfg.timing_slow, min_periods=max(2, cfg.timing_slow // 2)).mean()
    ma_b = idx.rolling(cfg.timing_bear_ma, min_periods=max(2, cfg.timing_bear_ma // 2)).mean()
    lvl = pd.Series(0.0, index=idx.index)
    lvl[(idx > ma_s) & (ma_f > ma_s)] = 3
    lvl[(idx > ma_s) & (ma_f <= ma_s)] = 2
    lvl[(idx <= ma_s) & (idx > ma_b)] = 1
    lvl[(idx <= ma_s) & (idx <= ma_b)] = 0
    raw = lvl.map({3: cfg.pos_full, 2: cfg.pos_half, 1: cfg.pos_weak, 0: cfg.pos_empty})
    final = timing(p, index_close, cfg, br)
    return pd.DataFrame({
        "index": idx, "ma_fast": ma_f, "ma_slow": ma_s, "ma_bear": ma_b,
        "level": lvl, "pos_raw": raw, "pos_final": final,
        "breadth": (br.reindex(idx.index).ffill() if br is not None else np.nan),
    })


# ---------------------------------------------------------------- 目标组合
def target_weights(scores: pd.DataFrame, day, prev_hold, cfg: StrategyConfig,
                   target_pos: float = 1.0) -> dict[str, float]:
    """等权目标权重。

    target_pos=1 时把钱摊到前 max_positions 只，每只不超过 max_weight；
    如果 max_positions 名额不足以吃满仓位，就扩大名额直到吃满为止。
    """
    row = scores.loc[day].dropna().sort_values(ascending=False)
    if row.empty or target_pos <= 0:
        return {}

    n_min = max(1, int(np.ceil(target_pos / max(cfg.max_weight, 1e-6))))
    n = min(max(cfg.max_positions, n_min), len(row))

    picks = list(row.head(n).index)
    # 老仓容忍：排名还在 1.5 倍范围内就留着，减少换手
    if prev_hold:
        horizon = int(n * 1.5)
        for sec in prev_hold:
            if sec in picks:
                continue
            if sec in row.index[:horizon]:
                picks.append(sec)
    picks = picks[:max(n, len([s for s in prev_hold if s in row.index[:int(n * 1.5)]]))]

    if not picks:
        return {}
    base_w = min(1.0 / len(picks), cfg.max_weight)
    w = {s: base_w for s in picks}
    tot = sum(w.values())
    # 有剩余额度就按比例放大，但仍不超过单票上限
    if target_pos > tot:
        scale = min(target_pos / tot, cfg.max_weight / base_w if base_w > 0 else 1.0)
        w = {k: v * scale for k, v in w.items()}
    return w
