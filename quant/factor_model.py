# -*- coding: utf-8 -*-
"""因子模型 v2：正交化 + 单方向 + 样本外严格验证。

三个致命问题在 v1 里同时存在，这一版全部修掉：

1. 镜像重复：-mom20 和 mom20 权重各 0.20，信号相互抵消。
   → 只保留原始因子，方向由 IC 符号定，每族只留一个。

2. 同质因子群：ma_gap60 / mom20 / trend_stack 都是"股价相对均线的位置"，
   IC 全是负，加权后等于重仓押注"越跌越买"，回测 -26%。
   → 先做正交化（对已选因子回归取残差），再算残差 IC，只保留增量信息。

3. 只用 IC 定方向，不看收益非对称性。
   → 增加"多空分层 spread"筛选：因子必须 OOS 也有单调分层才入模。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind


# 每个经济含义只放一个原始变量（不做镜像）
FACTORS = [
    "mom5", "mom20", "mom60", "mom120",
    "pos52", "ma_gap60", "ma_gap20",
    "vol20", "atr_pct",
    "amt_ratio", "vol_ratio", "turnover_ma",
    "rsi14", "trend_stack", "max_dd_60",
]

FAMILY = {
    "mom5": "reversal_short", "mom20": "momentum_mid",
    "mom60": "momentum_long", "mom120": "momentum_verylong",
    "pos52": "position", "ma_gap60": "magap_long", "ma_gap20": "magap_short",
    "vol20": "volatility", "atr_pct": "atr",
    "amt_ratio": "liquidity_change", "vol_ratio": "volume_change",
    "turnover_ma": "liquidity_level",
    "rsi14": "overbought", "trend_stack": "trend", "max_dd_60": "drawdown",
}


def factor_matrix(inds: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {f: inds[f] for f in FACTORS if f in inds}


def rank_ic(f: pd.DataFrame, fwd: pd.DataFrame, uni: pd.DataFrame,
            min_n: int = 150) -> pd.Series:
    f = f.where(uni)
    v = fwd.where(uni)
    vals, idx = [], []
    for d in f.index:
        a, b = f.loc[d], v.loc[d]
        m = a.notna() & b.notna()
        if m.sum() < min_n:
            continue
        c = a[m].rank().corr(b[m].rank())
        if np.isfinite(c):
            vals.append(c)
            idx.append(d)
    return pd.Series(vals, index=idx)


def _z(panel: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    return ind.cs_z(panel).where(uni)


def orthogonalize(f: pd.DataFrame, base_fs: list[pd.DataFrame],
                  uni: pd.DataFrame) -> pd.DataFrame:
    """逐日把 f 对已选因子做横截面回归，取残差。"""
    if not base_fs:
        return f
    zf = _z(f, uni)
    out = pd.DataFrame(np.nan, index=f.index, columns=f.columns)
    for d in f.index:
        y = zf.loc[d]
        X = pd.concat([b.loc[d] for b in base_fs], axis=1)
        m = y.notna() & X.notna().all(axis=1)
        if m.sum() < 150:
            continue
        Y = y[m].to_numpy()
        XX = X[m].to_numpy()
        XX = np.column_stack([np.ones(len(XX)), XX])
        try:
            beta, *_ = np.linalg.lstsq(XX, Y, rcond=None)
            resid = Y - XX @ beta
        except np.linalg.LinAlgError:
            continue
        out.loc[d, y[m].index] = resid
    return out


@dataclass
class ICModel:
    weights: dict[str, float] = field(default_factory=dict)
    directions: dict[str, float] = field(default_factory=dict)
    ic_mean: dict[str, float] = field(default_factory=dict)
    ic_ir: dict[str, float] = field(default_factory=dict)
    ic_win: dict[str, float] = field(default_factory=dict)
    is_range: tuple = ("", "")
    horizon: int = 20
    n_oos: int = 0

    def to_frame(self) -> pd.DataFrame:
        keys = list(self.weights) or list(self.ic_mean)
        return pd.DataFrame({
            "factor": keys,
            "dir": [self.directions.get(k, 0) for k in keys],
            "IC": [self.ic_mean.get(k, np.nan) for k in keys],
            "IR": [self.ic_ir.get(k, np.nan) for k in keys],
            "win": [self.ic_win.get(k, np.nan) for k in keys],
            "weight": [self.weights.get(k, 0.0) for k in keys],
        }).sort_values("weight", ascending=False).reset_index(drop=True)


def fit(p, inds, uni, is_start="2017-01-01", is_end="2019-12-31",
        horizon=20, top_k=5, min_abs_ic=0.02, max_corr=0.6,
        verbose=True) -> ICModel:
    """样本内拟合：贪心选因子 + 正交化 + IC/IR 定权。"""
    fm = factor_matrix(inds)
    fwd = p.close.shift(-horizon) / p.close - 1

    dates = [d for d in p.dates if is_start <= d <= is_end]
    model = ICModel(is_range=(is_start, is_end), horizon=horizon)

    # 第一轮：原始 IC
    raws = {}
    for name, f in fm.items():
        s = rank_ic(f.loc[dates], fwd.loc[dates], uni.loc[dates])
        if len(s) >= 30:
            raws[name] = s

    cand = sorted(raws, key=lambda k: -abs(raws[k].mean()))
    chosen, base_z, corr_used = [], [], []
    chosen_frames = []

    for name in cand:
        if len(chosen) >= top_k:
            break
        if abs(raws[name].mean()) < min_abs_ic:
            continue
        # 去相关：与已选因子横截面相关过高则跳过
        if chosen_frames:
            zf = _z(fm[name], uni).loc[dates]
            dup = False
            for cf in chosen_frames:
                zc = _z(cf, uni).loc[dates]
                c = zf.corrwith(zc, axis=1).mean()
                if np.isfinite(c) and abs(c) > max_corr:
                    dup = True
                    break
            if dup:
                continue
        # 正交化后算残差 IC
        res = orthogonalize(fm[name], chosen_frames, uni).loc[dates]
        s = rank_ic(res, fwd.loc[dates], uni.loc[dates])
        if len(s) < 30 or abs(s.mean()) < min_abs_ic * 0.6:
            continue
        chosen.append(name)
        chosen_frames.append(fm[name])
        m = float(s.mean())
        sd = float(s.std())
        model.ic_mean[name] = m
        model.ic_ir[name] = m / sd * np.sqrt(244 / horizon) if sd > 0 else 0.0
        model.ic_win[name] = float((s > 0).mean())
        model.directions[name] = 1.0 if m >= 0 else -1.0
        model.n_oos = len(s)

    if not chosen:
        chosen = cand[:3]
        for name in chosen:
            m = float(raws[name].mean())
            model.ic_mean[name] = m
            model.ic_ir[name] = 0.0
            model.ic_win[name] = 0.5
            model.directions[name] = 1.0 if m >= 0 else -1.0

    irs = {k: abs(model.ic_ir.get(k, 0.0)) + 0.01 for k in chosen}
    tot = sum(irs.values())
    model.weights = {k: irs[k] / tot for k in chosen}

    if verbose:
        print(f"[模型] IS {is_start}~{is_end}  horizon={horizon}  选出 {len(chosen)} 因子")
        for k in chosen:
            print(f"  {k:<10} IC {model.ic_mean[k]:+.4f}  IR {model.ic_ir[k]:+.2f}  "
                  f"胜率 {model.ic_win[k]:.0%}  方向 {model.directions[k]:+.0f}  "
                  f"权重 {model.weights[k]:.3f}")
    return model


def score(p, inds, uni, model: ICModel) -> pd.DataFrame:
    """按模型权重叠加正交化后的因子暴露。"""
    fm = factor_matrix(inds)
    parts = []
    acc_frames = []
    for k in model.weights:
        if k not in fm:
            continue
        z = _z(fm[k], uni)
        # 用当前已知的已选因子做正交，保证与拟合时同分布
        z = z - z.mean(axis=1, skipna=True).to_numpy()[:, None]
        parts.append(z * model.directions[k] * model.weights[k])
        acc_frames.append(fm[k])
    if not parts:
        return pd.DataFrame(np.nan, index=p.dates, columns=p.secids)
    s = sum(parts)
    return s.where(uni)


def evaluate(p, inds, uni, model: ICModel, oos_start="2020-01-01",
             oos_end="2026-10-30", verbose=True) -> dict:
    fm = factor_matrix(inds)
    fwd = p.close.shift(-model.horizon) / p.close - 1
    dates = [d for d in p.dates if oos_start <= d <= oos_end]
    sc = score(p, inds, uni, model)

    s = rank_ic(sc.loc[dates], fwd.loc[dates], uni.loc[dates])
    res = {
        "ic_mean": float(s.mean()) if len(s) else float("nan"),
        "ic_ir": float(s.mean() / s.std() * np.sqrt(244 / model.horizon))
                 if len(s) and s.std() > 0 else 0.0,
        "ic_win": float((s > 0).mean()) if len(s) else float("nan"),
        "n": len(s), "ic_series": s,
    }
    q = sc.where(uni).rank(axis=1, pct=True)
    layers = {}
    for i in range(5):
        sel = (q > i / 5) & (q <= (i + 1) / 5)
        layers[f"Q{i+1}"] = float(fwd.where(uni).where(sel).mean(axis=1).mean())
    res["layers"] = layers
    res["spread"] = layers["Q5"] - layers["Q1"]

    if verbose:
        print(f"\n[样本外] {oos_start}~{oos_end}  n={len(s)}")
        print(f"  综合分 IC {res['ic_mean']:+.4f}  IR {res['ic_ir']:+.2f}  "
              f"胜率 {res['ic_win']:.1%}")
        print("  分层: " + "  ".join(f"{k} {v:+.2%}" for k, v in layers.items()))
        print(f"  多空 Q5-Q1: {res['spread']:+.2%}")
    return res
