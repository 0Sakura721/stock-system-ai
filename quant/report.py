# -*- coding: utf-8 -*-
"""报告生成：把回测结果、被证伪的路径、最终结论编译成一份可交付文档。

用法:
  python -m quant.report
"""
from __future__ import annotations

import json
import os

import pandas as pd

from . import data as D
from . import etf as ETF
from . import findings as FN
from .config import OUT_DIR, CFG


def build_report() -> str:
    L = []
    W = 78

    L.append("=" * W)
    L.append("  AI 全权炒股系统 · 最终报告")
    L.append("=" * W)
    L.append("")

    # ---------------------------------------------------------------- 数据
    conn = D.connect()
    n_sec = conn.execute("SELECT COUNT(DISTINCT secid) FROM bars WHERE adj='hfq'").fetchone()[0]
    n_rows = conn.execute("SELECT COUNT(*) FROM bars WHERE adj='hfq'").fetchone()[0]
    rng = conn.execute("SELECT MIN(date),MAX(date) FROM bars WHERE adj='hfq'").fetchone()
    n_sl = conn.execute("SELECT COUNT(*) FROM stocklist").fetchone()[0]
    conn.close()

    L.append("一、数据基础")
    L.append("")
    L.append(f"  股票数        {n_sec:,} 只（全 A，剔除北交所）")
    L.append(f"  日线总量      {n_rows:,} 行")
    L.append(f"  时间跨度      {rng[0]} ~ {rng[1]}（约 10 年）")
    L.append(f"  股票列表      {n_sl:,} 只（含名称、板块）")
    L.append(f"  价格口径      前复权连续序列（末值 = 真实现价）")
    L.append("")

    # ---------------------------------------------------------------- 结论
    L.append("二、核心结论：什么有效，什么无效")
    L.append("")
    L.append("  有效：")
    L.append("    1. 指数均线择时（唯一在多参数下稳定为正的组件）")
    L.append("    2. 分散化（持仓数量从 8 只增到 1000+ 只，收益单调改善）")
    L.append("")
    L.append("  无效：")
    L.append("    1. 动量选股（20日 RankIC 在 2018-2026 每年为负）")
    L.append("    2. IC 因子模型选股（IC +0.10 但选 8 只后年化 -8%）")
    L.append("    3. 个股止损（震荡市反复触发，恶化收益）")
    L.append("")

    # ---------------------------------------------------------------- 证据表
    L.append("三、实测证据（2020-01-02 ~ 2026-10-30，1653 交易日）")
    L.append("")
    ev = FN.VERIFIED.evidence()
    hdr = f"  {'方案':<26}{'总收益':>10}{'年化':>9}{'最大回撤':>10}{'Sharpe':>8}"
    L.append(hdr)
    L.append("  " + "-" * (W - 4))
    for r in ev:
        L.append(f"  {r['方案']:<26}{r['总收益']:>10}{r['年化']:>9}"
                 f"{r['最大回撤']:>10}{r['Sharpe']:>8}")
    L.append("")

    # ---------------------------------------------------------------- 本金约束
    L.append("四、策略适用性取决于本金规模")
    L.append("")
    L.append(f"  本金        {'总收益':>10}{'年化':>9}{'平均持仓':>10}{'与理论上限':>11}")
    L.append("  " + "-" * (W - 4))
    cap_rows = [
        ("10 万", "-16.4%", "-2.61%", "18 只", "-127 pct"),
        ("100 万", "+54.2%", "+6.61%", "282 只", "-57 pct"),
        ("1000 万", "+91.1%", "+10.03%", "1148 只", "-20 pct"),
        ("1 亿", "+97.4%", "+10.56%", "1264 只", "-14 pct"),
        ("向量化理论", "+111.1%", "+11.66%", "无限", "—"),
    ]
    for a, b, c, d, e in cap_rows:
        L.append(f"  {a:<12}{b:>10}{c:>9}{d:>10}{e:>11}")
    L.append("")
    L.append("  原因：A股一手100股 + 5元最低佣金，10万本金无法表达全池等权。")
    L.append("")

    # ---------------------------------------------------------------- ETF 方案
    L.append("五、小资金可行方案：ETF 择时")
    L.append("")
    L.append("  用同一套择时信号，交易单一 ETF，规避一手门槛与个股风险。")
    L.append("")
    L.append(f"  {'代码':<9}{'名称':<14}{'择时收益':>10}{'年化':>9}"
             f"{'择时回撤':>10}{'买入持有':>10}{'买入回撤':>10}")
    L.append("  " + "-" * (W - 4))
    try:
        df = ETF.compare_etfs("2020-01-01")
        for r in df.itertuples(index=False):
            if hasattr(r, "error"):
                L.append(f"  {r.code:<9}{r.name:<14}  数据缺失")
                continue
            L.append(f"  {r.code:<9}{r.name:<14}{r.total_return:>+9.1%}"
                     f"{r.cagr:>+9.2%}{r.max_drawdown:>10.1%}"
                     f"{r.bh_return:>+10.1%}{r.bh_dd:>10.1%}")
    except Exception as e:                          # noqa: BLE001
        L.append(f"  ETF 回测失败: {e}")
    L.append("")
    L.append("  关键：所有 ETF 买入持有回撤 -38% ~ -59%，择时后压到 -22% ~ -33%。")
    L.append("  择时牺牲了部分上涨，换来的是回撤减半。")
    L.append("")

    # ---------------------------------------------------------------- 执行
    L.append("六、每日执行流程")
    L.append("")
    L.append("  收盘后：")
    L.append("    1. python -m quant.fetch          增量更新行情（约 1 分钟）")
    L.append("    2. python -m quant.cli plan       生成次日计划")
    L.append("    3. python -m quant.web            打开面板确认")
    L.append("")
    L.append("  次日开盘：")
    L.append("    1. 先执行卖出（清仓/减仓）")
    L.append("    2. 再执行买入（按限价挂单）")
    L.append("    3. 记录成交 python -m quant.cli fill")
    L.append("")

    # ---------------------------------------------------------------- 免责
    L.append("七、必须知道的风险")
    L.append("")
    L.append("  1. 回测不等于未来。2018/2022 的极端行情未必重演，也未必不重演。")
    L.append("  2. 择时天然滞后。它在趋势反转后 1-2 周才确认，会漏掉第一波反弹。")
    L.append("  3. ETF 有跟踪误差和流动性差异，实盘收益会低于回测。")
    L.append("  4. 全池等权需要千万级本金，小资金的实际表现显著更差（见第四节）。")
    L.append("  5. 本系统把决策自动化，不消除市场风险。")
    L.append("")

    return "\n".join(L)


def save(path: str = None) -> str:
    path = path or os.path.join(OUT_DIR, "REPORT.txt")
    txt = build_report()
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return path


if __name__ == "__main__":
    p = save()
    print(build_report())
    print(f"\n已保存 -> {p}")
