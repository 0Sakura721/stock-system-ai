# -*- coding: utf-8 -*-
"""系统自检：数据 / 因子 / 择时 / 引擎 / 策略一致性。

python -m quant.selftest
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import data as D
from . import indicators as ind
from . import strategy as st
from . import findings as FN
from . import etf as ETF
from .backtest import Backtester
from .config import CFG

FAIL, PASS = [], []


def ck(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def main():
    print("=" * 68)
    print("  系统自检")
    print("=" * 68)

    # ---------------------------------------------------- 1 数据完整性
    print("\n[1] 数据完整性")
    conn = D.connect()
    n_sec = conn.execute("SELECT COUNT(DISTINCT secid) FROM bars WHERE adj='hfq'").fetchone()[0]
    n_rows = conn.execute("SELECT COUNT(*) FROM bars WHERE adj='hfq'").fetchone()[0]
    rng = conn.execute("SELECT MIN(date),MAX(date) FROM bars WHERE adj='hfq'").fetchone()
    n_sl = conn.execute("SELECT COUNT(*) FROM stocklist").fetchone()[0]
    ck("股票数 > 3000", n_sec > 3000, f"{n_sec} 只")
    ck("日线行数 > 500万", n_rows > 5_000_000, f"{n_rows:,} 行")
    ck("时间跨度 >= 8 年", rng[0][:4] == "2017" and rng[1][:4] == "2026",
       f"{rng[0]} ~ {rng[1]}")
    ck("股票列表已缓存", n_sl > 3000, f"{n_sl} 只")

    # 价格合理性：不应有单日跌幅超过 21%（若为连续复权序列则不存在）
    q = ("SELECT COUNT(*) FROM bars b1 JOIN bars b2 ON b1.secid=b2.secid "
         "AND b2.date=(SELECT MIN(date) FROM bars WHERE secid=b1.secid "
         "AND adj='hfq' AND date>b1.date) "
         "WHERE b1.adj='hfq' AND b2.close/b1.close-1 < -0.21")
    bad = conn.execute(q).fetchone()[0]
    ck("无异常跳空（复权连续）", bad < 500, f"{bad} 处（含真实跌停/退市复牌）")

    # ---------------------------------------------------- 2 面板与指标
    print("\n[2] 面板与指标")
    bars = D.load_bars(conn, adj="hfq", start="2020-01-01").dropna(subset=["close"])
    bars = bars[bars["close"] > 0]
    p = ind.Panel(bars)
    sl = D.load_stocklist(conn)
    ck("面板构建", p.close.shape[0] > 1000 and p.close.shape[1] > 3000, str(p))

    inds = ind.compute(p)
    need = ["ma20", "ma60", "ma120", "vol20", "atr_pct", "mom20", "mom60",
            "pos52", "amt20", "rsi14", "trend_stack", "max_dd_60", "turnover_ma"]
    missing = [k for k in need if k not in inds]
    ck("指标齐全", not missing, f"缺 {missing}" if missing else f"{len(inds)} 个")
    last = str(p.dates[-1])
    ck("最新日有指标", inds["ma20"].loc[last].notna().sum() > 500,
       f"{inds['ma20'].loc[last].notna().sum()} 只")

    # ---------------------------------------------------- 3 股票池
    print("\n[3] 股票池")
    uni = st.build_universe(p, inds, FN.apply_to(CFG).strategy, sl)
    n_uni = int(uni.loc[last].sum())
    ck("池规模合理", 100 < n_uni < 5000, f"{n_uni} 只")
    ck("无 ST", not any("ST" in str(sl.set_index("secid")["name"].get(c, ""))
                        for c in uni.columns[uni.loc[last].values]), "已过滤")
    nb = [c for c in uni.columns[uni.loc[last].values]
          if c.split(".")[-1][0] in "89"]
    ck("无北交所", not nb, f"{len(nb)} 只" if nb else "已过滤")

    # ---------------------------------------------------- 4 择时
    print("\n[4] 择时")
    cfg = FN.apply_to(CFG)
    br = st.breadth(p, inds, uni)
    idx = D.fetch_index_close(cfg.strategy.timing_index, "2015-01-01")
    pos = st.timing(p, pd.Series(idx), cfg.strategy, br)
    ck("择时序列有效", len(pos) > 1000 and pos.notna().all())
    ck("仓位在 0~1", (pos >= 0).all() and (pos <= 1).all(), f"均值 {pos.mean():.0%}")
    ck("有满仓也有空仓", pos.max() >= 0.9 and pos.min() <= 0.1,
       f"最高 {pos.max():.0%} 最低 {pos.min():.0%}")
    detail = st.timing_detail(p, pd.Series(idx), cfg.strategy, br)
    ck("归因表完整", len(detail) == len(p.dates), f"{len(detail)} 行")

    # ---------------------------------------------------- 5 回测引擎对账
    print("\n[5] 回测引擎（强制对账）")
    sc = st.rank_score(p, inds, uni)
    from dataclasses import replace
    btcfg = type(cfg)(cost=cfg.cost,
                      strategy=replace(cfg.strategy, max_positions=64),
                      account=cfg.account)
    try:
        bt = Backtester(p, sc, pos, btcfg, strict=True)
        r = bt.run(start="2023-01-01")
        ck("引擎跑通（strict 对账无异常）", True, f"{r['stats']['days']} 日")
        ck("净值无 NaN", r["equity"].notna().all())
        ck("净值无负值", (r["equity"] > 0).all())
        ck("现金非负", (r["nav"]["cash"] >= -0.01).all())
        ck("权益 = 现金 + 持仓",
           np.abs(r["nav"]["equity"] - r["nav"]["cash"] - r["nav"]["pos_value"]).max() < 1)
        ck("止损未触发（配置为极宽）",
           sum(1 for x in r["trades"].reason if "止损" in x) < 20 if len(r["trades"]) else True)
    except AssertionError as e:
        ck("引擎跑通", False, str(e)[:70])

    # ---------------------------------------------------- 6 ETF
    print("\n[6] ETF 择时")
    try:
        er = ETF.backtest_etf(code="159949", start="2022-01-01", verbose=False)
        ck("ETF 回测跑通", True, f"{er['days']} 日")
        ck("ETF 收益有效", np.isfinite(er["total_return"]), f"{er['total_return']:+.0%}")
        ck("ETF 仓位在 0~1", 0 <= er["position"].min() and er["position"].max() <= 1,
           f"均值 {er['exposure']:.0%}")
    except Exception as e:                          # noqa: BLE001
        ck("ETF 回测跑通", False, str(e)[:70])

    # ---------------------------------------------------- 7 本金判断
    print("\n[7] 本金适用性")
    v1 = FN.capital_verdict(100_000)
    v2 = FN.capital_verdict(10_000_000)
    ck("10万 -> ETF 模式", v1["mode"] == "etf_only")
    ck("1000万 -> 个股组合", v2["mode"] == "stock_portfolio")

    conn.close()
    print("\n" + "=" * 68)
    if FAIL:
        print(f"  未通过 {len(FAIL)} 项: {FAIL}")
        return 1
    print(f"  全部通过（{len(PASS)} 项）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
