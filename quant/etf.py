# -*- coding: utf-8 -*-
"""ETF 择时策略：小资金（10 万级）唯一可执行的形式。

为什么需要这个：
  全池等权需要 1000 万本金才能表达（A 股一手 100 股 + 5 元最低佣金）。
  10 万本金只能买 8-18 只，等价于主动选股，而选股已证明无 alpha。
  用 ETF 单标的 + 择时，可以做到「零个股风险 + 100% 仓位可表达」。

ETF 不用一手 100 股限制的例外：ETF 最小 1 份，10 万可精确按比例配置。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import strategy as st
from .config import CFG
from . import findings as FN


# 可选 ETF 池（代码, 名称, 与哪个指数择时最匹配）
ETF_POOL = {
    "159915": ("创业板ETF", "399006"),
    "510300": ("沪深300ETF", "000300"),
    "510500": ("中证500ETF", "000905"),
    "512100": ("中证1000ETF", "000852"),
    "159949": ("创业板50ETF", "399006"),
}


def fetch_etf_bars(code: str, start: str = "2016-01-01") -> pd.DataFrame:
    """ETF 日线（用新浪源，ETF 与股票同一接口）。"""
    df = D._sina_kline(code, 2400)
    if len(df) == 0:
        return df
    return df[df["date"] >= start].reset_index(drop=True)


def backtest_etf(code: str = "159915", cfg=None, start="2020-01-01", cash: float = 1e5,
                 verbose: bool = True) -> dict:
    """单 ETF 择时回测：信号 T 日收盘 → T+1 开盘调仓。"""
    cfg = FN.apply_to(cfg or CFG)
    s = cfg.strategy
    etf = fetch_etf_bars(code, "2016-01-01")
    if len(etf) == 0:
        raise RuntimeError(f"ETF {code} 无数据")

    idx_code = ETF_POOL.get(code, (None, s.timing_index))[1]
    idx = D.fetch_index_close(idx_code, "2015-01-01")

    etf = etf.set_index("date")
    idx_s = pd.Series(idx.values, index=pd.Index(idx.index))
    dates = [d for d in etf.index if d >= start]
    if len(dates) < 60:
        raise RuntimeError(f"ETF {code} 区间过短")

    # 择时：用指数算仓位，映射到 ETF 日期
    p_like = type("P", (), {"dates": pd.Index(dates)})()
    pos = st.timing(p_like, idx_s, s)

    close = etf["close"].reindex(dates)
    open_ = etf["open"].reindex(dates)
    ret = close.pct_change().fillna(0)

    # 仓位：T 日信号，T+1 生效
    pos_lag = pos.shift(1).fillna(0)
    strat_ret = pos_lag * ret - pos_lag.diff().abs().fillna(0) * cfg.cost.slippage

    eq = (1 + strat_ret).cumprod() * cash
    total = eq.iloc[-1] / cash - 1
    years = len(dates) / 244
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1
    dd = (eq / eq.cummax() - 1)
    vol = strat_ret.std() * np.sqrt(244)
    sharpe = strat_ret.mean() * 244 / vol if vol > 0 else 0
    bh = (1 + ret).cumprod() * cash

    res = {
        "code": code, "name": ETF_POOL.get(code, ("?", ""))[0],
        "start": dates[0], "end": dates[-1], "days": len(dates),
        "total_return": float(total), "cagr": float(cagr),
        "max_drawdown": float(dd.min()), "sharpe": float(sharpe),
        "exposure": float(pos_lag.mean()),
        "bh_return": float(bh.iloc[-1] / cash - 1),
        "bh_dd": float((bh / bh.cummax() - 1).min()),
        "equity": eq, "bench": bh, "position": pos_lag, "returns": strat_ret,
    }
    if verbose:
        print(f"[ETF {code} {res['name']}] {dates[0]}~{dates[-1]}  {len(dates)}日")
        print(f"  择时策略：总 {total:+.1%}  年化 {cagr:+.2%}  回撤 {dd.min():.1%}  "
              f"Sharpe {sharpe:.2f}  平均仓位 {pos_lag.mean():.0%}")
        print(f"  买入持有：总 {res['bh_return']:+.1%}  回撤 {res['bh_dd']:.1%}")
    return res


def compare_etfs(start="2020-01-01") -> pd.DataFrame:
    rows = []
    for code in ETF_POOL:
        try:
            r = backtest_etf(code, start=start, verbose=False)
            rows.append({k: r[k] for k in
                         ("code", "name", "total_return", "cagr", "max_drawdown",
                          "sharpe", "exposure", "bh_return", "bh_dd")})
        except Exception as e:                       # noqa: BLE001
            rows.append({"code": code, "name": ETF_POOL[code][0], "error": str(e)[:60]})
    return pd.DataFrame(rows)


def plot_etf(res: dict, path: str = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams["axes.unicode_minus"] = False
    for f in ("Microsoft YaHei", "SimHei", "DengXian"):
        try:
            font_manager.findfont(f, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [f]
            break
        except Exception:                            # noqa: BLE001
            continue

    eq = res["equity"]
    bh = res["bench"]
    pos = res["position"]
    dates = pd.to_datetime(eq.index)

    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1, 1]})
    ax[0].plot(dates, eq.values, lw=1.6, color="#c0392b",
               label=f"择时策略 (+{res['total_return']:.0%})")
    ax[0].plot(dates, bh.values, lw=1.1, color="#7f8c8d", alpha=.85,
               label=f"买入持有 ({res['bh_return']:+.0%})")
    ax[0].set_title(f"{res['code']} {res['name']} · ETF 择时回测", fontsize=13)
    ax[0].set_ylabel("资产 (元)")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=.3)

    dd = (eq / eq.cummax() - 1) * 100
    ax[1].fill_between(dates, dd.values, 0, color="#2980b9", alpha=.65)
    ax[1].set_ylabel("回撤 %")
    ax[1].grid(alpha=.3)

    ax[2].fill_between(dates, pos.values * 100, 0, color="#27ae60", alpha=.55, step="mid")
    ax[2].set_ylabel("仓位 %")
    ax[2].set_ylim(0, 105)
    ax[2].grid(alpha=.3)

    plt.tight_layout()
    path = path or "output/etf_backtest.png"
    plt.savefig(path, dpi=135)
    plt.close()
    return path


if __name__ == "__main__":
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "159915"
    r = backtest_etf(code)
    print("图表 ->", plot_etf(r, f"output/etf_{code}.png"))
