# -*- coding: utf-8 -*-
"""指标与因子（宽表面板，全向量化）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Panel:
    """行情面板容器。"""

    __slots__ = ("open", "close", "high", "low", "volume", "amount", "dates", "secids")

    def __init__(self, df: pd.DataFrame):
        self.open = df.pivot(index="date", columns="secid", values="open").sort_index()
        self.close = df.pivot(index="date", columns="secid", values="close").sort_index()
        self.high = df.pivot(index="date", columns="secid", values="high").sort_index()
        self.low = df.pivot(index="date", columns="secid", values="low").sort_index()
        self.volume = df.pivot(index="date", columns="secid", values="volume").sort_index()
        self.amount = df.pivot(index="date", columns="secid", values="amount").sort_index()
        self.dates = self.close.index
        self.secids = list(self.close.columns)

    def __repr__(self):
        return (f"<Panel {len(self.dates)}日 × {len(self.secids)}只 "
                f"{self.dates[0]}~{self.dates[-1]}>")

    def last_date(self) -> str:
        return str(self.dates[-1])


def compute(p: Panel) -> dict[str, pd.DataFrame]:
    C, O, H, L, V, A = p.close, p.open, p.high, p.low, p.volume, p.amount

    ma5 = C.rolling(5, min_periods=3).mean()
    ma10 = C.rolling(10, min_periods=5).mean()
    ma20 = C.rolling(20, min_periods=10).mean()
    ma60 = C.rolling(60, min_periods=30).mean()
    ma120 = C.rolling(120, min_periods=60).mean()
    ma250 = C.rolling(250, min_periods=120).mean()

    ret1 = C.pct_change()
    out = {
        "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
        "ma120": ma120, "ma250": ma250, "ret1": ret1,
        "mom5": C.pct_change(5), "mom20": C.pct_change(20),
        "mom60": C.pct_change(60), "mom120": C.pct_change(120),
        "vol20": ret1.rolling(20, min_periods=10).std(),
        "vol60": ret1.rolling(60, min_periods=30).std(),
    }

    prev_c = C.shift(1)
    tr = pd.concat([
        (H - L).stack(),
        (H - prev_c).abs().stack(),
        (L - prev_c).abs().stack(),
    ], axis=1).max(axis=1).unstack()
    out["atr14"] = tr.rolling(14, min_periods=7).mean()
    out["atr_pct"] = out["atr14"] / C

    out["amt20"] = A.rolling(20, min_periods=10).mean()
    out["amt60"] = A.rolling(60, min_periods=30).mean()
    out["amt_ratio"] = A / out["amt20"].replace(0, np.nan)
    out["vol_ratio"] = V / V.rolling(20, min_periods=10).mean().replace(0, np.nan)

    hi252 = H.rolling(250, min_periods=120).max()
    lo252 = L.rolling(250, min_periods=120).min()
    out["hi252"], out["lo252"] = hi252, lo252
    out["pos52"] = (C - lo252) / (hi252 - lo252).replace(0, np.nan)
    out["to_hi52"] = C / hi252.replace(0, np.nan) - 1

    out["ma_gap60"] = C / ma60 - 1
    out["ma_gap20"] = C / ma20 - 1
    out["trend_stack"] = ((ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)).astype(float)

    up = C.diff().clip(lower=0)
    dn = (-C.diff()).clip(lower=0)
    au = up.rolling(14, min_periods=7).mean()
    ad = dn.rolling(14, min_periods=7).mean()
    out["rsi14"] = 100 - 100 / (1 + au / ad.replace(0, np.nan))

    out["max_dd_60"] = C / C.rolling(60, min_periods=30).max() - 1
    out["turnover_ma"] = out["amt60"] / (out["amt60"].rolling(120, min_periods=60).mean()
                                          .replace(0, np.nan))
    out["gap_open"] = O / prev_c - 1
    return out


def cs_z(panel: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """横截面稳健 z（中位数 + MAD），逐行。"""
    med = panel.median(axis=1)
    mad = panel.sub(med, axis=0).abs().median(axis=1) * 1.4826
    z = panel.sub(med, axis=0).div(mad.replace(0, np.nan), axis=0)
    return z.clip(-clip, clip)


def cs_rank(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.rank(axis=1, pct=True)
