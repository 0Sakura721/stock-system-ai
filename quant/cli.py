# -*- coding: utf-8 -*-
"""命令行入口：一条命令跑完整流程。

用法:
  python -m quant.cli init --cash 100000        # 初始化模拟账户
  python -m quant.cli fetch [--full]            # 更新行情
  python -m quant.cli backtest [--start ...]    # 回测
  python -m quant.cli plan [--save]             # 生成次日计划
  python -m quant.cli execute --price-csv p.csv # 按开盘价执行计划（模拟）
  python -m quant.cli status                    # 账户与持仓
  python -m quant.cli daily                     # 一体化：fetch→plan→execute→status
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from . import data as D
from . import engine as E
from .account import Account
from .config import CFG, OUT_DIR
from .risk import RiskEngine


def cmd_init(a):
    acc = Account(initial_cash=a.cash)
    acc.reset(a.cash)
    print(f"账户已初始化：现金 {a.cash:,.0f} 元")
    return 0


def cmd_fetch(a):
    from .fetch import update
    update(full=a.full, workers=a.workers)
    return 0


def cmd_backtest(a):
    from . import runner
    runner.run(start=a.start, end=a.end)
    return 0


def cmd_plan(a):
    conn = D.connect()
    st = E.build_state(conn, cfg=CFG)
    acc = Account(conn, initial_cash=a.cash)
    risk = RiskEngine(CFG)
    plan = E.make_plan(st, acc, risk, top_n=a.top_n, exec_date=a.exec_date)
    md = E.render_md(plan)
    print(md)
    if a.save:
        tag = plan["signal_date"].replace("-", "")
        p1 = os.path.join(OUT_DIR, f"plan_{tag}.md")
        p2 = os.path.join(OUT_DIR, f"plan_{tag}.json")
        with open(p1, "w", encoding="utf-8") as f:
            f.write(md)
        with open(p2, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2, default=float)
        print(f"\n已保存 → {p1}")
    return 0


def cmd_execute(a):
    conn = D.connect()
    acc = Account(conn, initial_cash=a.cash)
    pend = acc.pending_orders(a.exec_date)
    if pend.empty:
        print("没有待执行委托。先跑 plan 生成并入库。")
        return 0
    print(pend.to_string(index=False))
    return 0


def cmd_status(a):
    conn = D.connect()
    acc = Account(conn, initial_cash=a.cash)
    hold = acc.positions
    codes = list(hold.code) if len(hold) else []
    rt = D.fetch_realtime(codes) if codes else pd.DataFrame()
    prices = {r.code: float(r.price) for r in rt.itertuples(index=False)} if len(rt) else {}
    snap = acc.snapshot(prices)
    print("=" * 56)
    print(f"  权益 {snap['equity']:>12,.2f}   现金 {snap['cash']:>12,.2f}")
    print(f"  持仓市值 {snap['pos_value']:>9,.2f}   累计 {snap['total_return']:+.2%}")
    print("=" * 56)
    if snap["positions"]:
        for p in snap["positions"]:
            print(f"  {p['code']} {p['name'][:6]:<7} {p['shares']:>6} 股  "
                  f"成本{p['cost']:>8.2f} 现价{p['price']:>8.2f} "
                  f"盈亏{p['pnl']:>+7.1%} 权重{p['weight']:>6.1%}")
    else:
        print("  空仓")
    return 0


def cmd_daily(a):
    from .fetch import update
    update(full=False, workers=a.workers)
    conn = D.connect()
    st = E.build_state(conn)
    acc = Account(conn, initial_cash=a.cash)
    risk = RiskEngine(CFG)
    plan = E.make_plan(st, acc, risk, top_n=a.top_n)
    md = E.render_md(plan)
    print(md)
    tag = plan["signal_date"].replace("-", "")
    with open(os.path.join(OUT_DIR, f"plan_{tag}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(OUT_DIR, f"plan_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2, default=float)
    for s in plan["sells"]:
        acc.queue_order(plan["signal_date"], plan["exec_date"], "SELL",
                        s["code"], s["name"], s["price"], s["shares"], s["reason"])
    for b in plan["buys"]:
        acc.queue_order(plan["signal_date"], plan["exec_date"], "BUY",
                        b["code"], b["name"], b["price"], b["shares"], b["reason"])
    print(f"\n已入库 {(len(plan['sells'])+len(plan['buys']))} 笔委托")
    return 0


def cmd_fill(a):
    """记录实际成交，更新账本。"""
    conn = D.connect()
    acc = Account(conn, initial_cash=a.cash)
    if not a.code:
        print("用法: --code 600519 --side BUY --price 12.5 --shares 500 [--name 名称]")
        pend = acc.pending_orders()
        if not pend.empty:
            print("\n待执行委托:")
            print(pend.to_string(index=False))
        return 0
    acc.apply_fill(a.side.upper(), a.code, a.name or a.code, a.price, a.shares)
    print(f"已记录: {a.side.upper()} {a.code} {a.shares}股 @ {a.price}")
    return 0


def cmd_etf(a):
    from . import etf as ETF
    r = ETF.backtest_etf(code=a.code, start=a.start)
    print("图表 ->", ETF.plot_etf(r, f"output/etf_{a.code}.png"))
    return 0


def cmd_signal(a):
    """只输出今天的择时信号与仓位建议，不下单。"""
    from . import findings as FN
    from . import strategy as st
    from . import indicators as ind
    from dataclasses import replace
    cfg = FN.apply_to(CFG)
    conn = D.connect()
    bars = D.load_bars(conn, adj="hfq", start="2024-01-01")
    conn.close()
    bars = bars.dropna(subset=["close"])
    p = ind.Panel(bars)
    inds = ind.compute(p)
    sl = D.load_stocklist(D.connect())
    uni = st.build_universe(p, inds, cfg.strategy, sl)
    br = st.breadth(p, inds, uni)
    idx = D.fetch_index_close(cfg.strategy.timing_index, "2015-01-01")
    detail = st.timing_detail(p, pd.Series(idx), cfg.strategy, br)
    row = detail.iloc[-1]
    print(f"信号日 {p.dates[-1]}")
    print(f"  指数 {cfg.strategy.timing_index}  收盘 {row['index']:.0f}")
    print(f"  MA{cfg.strategy.timing_fast}={row['ma_fast']:.0f}  "
          f"MA{cfg.strategy.timing_slow}={row['ma_slow']:.0f}  "
          f"MA{cfg.strategy.timing_bear_ma}={row['ma_bear']:.0f}")
    print(f"  档位 {int(row['level'])}/3  原始仓位 {row['pos_raw']:.0%}  "
          f"宽度 {row['breadth']:.0%}")
    print(f"  >>> 目标仓位 {row['pos_final']:.0%}")
    v = FN.capital_verdict(a.cash)
    print(f"\n本金 {a.cash:,.0f} -> {v['mode']}: {v['message']}")
    return 0


def cmd_report(a):
    from . import report as R
    print(R.build_report())
    print(f"\n已保存 -> {R.save()}")
    return 0


def cmd_selftest(a):
    from . import selftest as S
    return S.main()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="quant")
    ap.add_argument("--cash", type=float, default=CFG.account.initial_cash)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("fetch")
    p.add_argument("--full", action="store_true")
    p.add_argument("--workers", type=int, default=10)
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("backtest")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser("plan")
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--exec-date", default=None)
    p.add_argument("--save", action="store_true")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("execute")
    p.add_argument("--exec-date", default=None)
    p.add_argument("--price-csv", default=None)
    p.set_defaults(fn=cmd_execute)

    p = sub.add_parser("status")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("fill")
    p.add_argument("--code", default=None)
    p.add_argument("--side", default="BUY")
    p.add_argument("--price", type=float, default=0)
    p.add_argument("--shares", type=int, default=0)
    p.add_argument("--name", default=None)
    p.set_defaults(fn=cmd_fill)

    p = sub.add_parser("etf")
    p.add_argument("--code", default="159949")
    p.add_argument("--start", default="2020-01-01")
    p.set_defaults(fn=cmd_etf)

    p = sub.add_parser("signal")
    p.set_defaults(fn=cmd_signal)

    p = sub.add_parser("report")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("selftest")
    p.set_defaults(fn=cmd_selftest)

    p = sub.add_parser("daily")
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--workers", type=int, default=10)
    p.set_defaults(fn=cmd_daily)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
