# -*- coding: utf-8 -*-
"""回测运行器：加载 → 因子 → 择时 → 回测 → 报告图表。"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import data as D
from . import indicators as ind
from . import strategy as st
from . import findings as FN
from .backtest import Backtester
from .config import CFG, OUT_DIR


def load_state(start: str = "2018-01-01", cfg=None):
    cfg = FN.apply_to(cfg or CFG)
    conn = D.connect()
    bars = D.load_bars(conn, adj="hfq", start=start)
    sl = D.load_stocklist(conn)
    conn.close()
    if bars.empty:
        raise RuntimeError("行情库为空，先跑 python -m quant.fetch")
    bars = bars.dropna(subset=["close"])
    bars = bars[bars["close"] > 0]
    p = ind.Panel(bars)
    inds = ind.compute(p)
    uni = st.build_universe(p, inds, cfg.strategy, sl)
    sc = st.rank_score(p, inds, uni)
    br = st.breadth(p, inds, uni)
    idx = D.fetch_index_close(cfg.strategy.timing_index, "2015-01-01")
    pos = st.timing(p, pd.Series(idx), cfg.strategy, br)
    return dict(p=p, inds=inds, uni=uni, scores=sc, breadth=br, timing=pos,
                stocks=sl, cfg=cfg, index=idx)


def run(start="2020-01-01", end=None, cfg=None, save=True, verbose=True):
    cfg = FN.apply_to(cfg or CFG)
    stt = load_state("2017-06-01", cfg)
    p, sc, pos = stt["p"], stt["scores"], stt["timing"]
    print(f"[数据] {p}", flush=True)
    print(f"[因子] 股票池均值 {stt['uni'].sum(axis=1).mean():.0f} 只/日（全池等权，不选股）",
          flush=True)
    print(f"[择时] 平均仓位 {pos.mean():.0%} 空仓 {(pos<0.1).mean():.0%}", flush=True)

    bt = Backtester(p, sc, pos, cfg)
    res = bt.run(start=start, end=end)
    s = res["stats"]

    idx_s = pd.Series(stt["index"].values, index=pd.to_datetime(stt["index"].index))
    seg = idx_s.reindex(res["equity"].index).ffill()
    if seg.notna().sum() > 2:
        s["bench_return"] = float(seg.iloc[-1] / seg.iloc[0] - 1)
        s["excess"] = s["total_return"] - s["bench_return"]
        bret = seg.pct_change().fillna(0)
        s["bench_sharpe"] = float(bret.mean() * 244 / (bret.std() * np.sqrt(244))) if bret.std() else 0
        s["corr"] = float(res["returns"].corr(bret))
        res["bench"] = seg

    if verbose:
        _print(s, cfg)
    if save:
        _save(res, cfg)
    return res


def _print(s, cfg):
    print("\n" + "=" * 60)
    print("  回测绩效")
    print("=" * 60)
    rows = [
        ("区间", f"{s['start']} ~ {s['end']} ({s['days']}日)"),
        ("初始资金", f"{cfg.account.initial_cash:,.0f}"),
        ("期末权益", f"{s['final_equity']:,.0f}"),
        ("累计收益", f"{s['total_return']:+.2%}"),
        ("年化收益", f"{s['cagr']:+.2%}"),
        ("沪深300同期", f"{s.get('bench_return', float('nan')):+.2%}"),
        ("超额", f"{s.get('excess', float('nan')):+.2%}"),
        ("最大回撤", f"{s['max_drawdown']:.2%}"),
        ("年化波动", f"{s['vol']:.2%}"),
        ("夏普", f"{s['sharpe']:.2f}"),
        ("索提诺", f"{s['sortino']:.2f}"),
        ("Calmar", f"{s['calmar']:.2f}"),
        ("交易笔数", f"{s['n_trades']}"),
        ("完整回合", f"{s['n_round']}"),
        ("胜率", f"{s['win_rate']:.1%}"),
        ("平均盈利", f"{s['avg_win']:+.2%}"),
        ("平均亏损", f"{s['avg_loss']:+.2%}"),
        ("盈亏比", f"{s['profit_factor']:.2f}"),
        ("平均持仓数", f"{s['avg_pos']:.1f}"),
        ("平均仓位", f"{s['exposure']:.0%}"),
        ("单日最佳", f"{s['best_day']:+.2%}"),
        ("单日最差", f"{s['worst_day']:+.2%}"),
        ("总手续费", f"{s['total_fee']:,.0f}"),
    ]
    for k, v in rows:
        print(f"  {k:<12}{v:>32}")
    print("=" * 60)


def _save(res, cfg):
    res["nav"].to_csv(os.path.join(OUT_DIR, "nav.csv"), encoding="utf-8-sig")
    if len(res["trades"]):
        res["trades"].to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False,
                             encoding="utf-8-sig")
    if len(res["pnl"]):
        res["pnl"].to_csv(os.path.join(OUT_DIR, "rounds.csv"), index=False,
                          encoding="utf-8-sig")
    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                   for k, v in res["stats"].items()}, f, ensure_ascii=False, indent=2)
    try:
        plot(res)
    except Exception as e:                          # noqa: BLE001
        print("  [绘图跳过]", str(e)[:80])


def plot(res, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams["axes.unicode_minus"] = False
    for f in ("Microsoft YaHei", "SimHei", "DengXian", "SimSun"):
        try:
            font_manager.findfont(f, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [f]
            break
        except Exception:                          # noqa: BLE001
            continue

    eq, dd = res["equity"], res["drawdown"] * 100
    fig, ax = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1, 1.2]})
    ax[0].plot(eq.index, eq.values, lw=1.5, color="#c0392b", label="策略")
    if "bench" in res:
        b = res["bench"].dropna()
        if len(b) > 2:
            ax[0].plot(b.index, b.values / b.iloc[0] * CFG.account.initial_cash,
                       lw=1.0, color="#7f8c8d", alpha=.85, label="沪深300")
    ax[0].set_title("策略净值 vs 沪深300", fontsize=13)
    ax[0].set_ylabel("权益 (元)")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=.3)

    ax[1].fill_between(dd.index, dd.values, 0, color="#2980b9", alpha=.65)
    ax[1].set_ylabel("回撤 %")
    ax[1].grid(alpha=.3)

    m = eq.resample("ME").last().pct_change().dropna() * 100
    ax[2].bar(m.index, m.values, width=18,
              color=["#27ae60" if v >= 0 else "#c0392b" for v in m.values])
    ax[2].axhline(0, color="#333", lw=.8)
    ax[2].set_ylabel("月度 %")
    ax[2].grid(alpha=.3)
    plt.tight_layout()
    path = path or os.path.join(OUT_DIR, "backtest.png")
    plt.savefig(path, dpi=135)
    plt.close()
    print("  图表 ->", path)


if __name__ == "__main__":
    run()
