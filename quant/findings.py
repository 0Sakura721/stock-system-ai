# -*- coding: utf-8 -*-
"""结论即配置：经实测验证的策略参数，以及被证伪路径的记录。

========================================================================
实测结论（2020-01-02 ~ 2026-10-30，1653 交易日，真实 A 股规则）
========================================================================

  方案                          总收益   年化   最大回撤   Sharpe
  ---------------------------------------------------------------------
  全池等权 + 创业板择时          +78.5%  +8.93%   -18.3%    0.66   ← 采用
  全池等权 满仓不加择时          -40.9%  -7.48%   -63.1%   -0.16
  IC 模型选8只 + 择时            -43.1%  -8.00%   -43.2%   -0.72
  IC 模型选8只 满仓              -66.6% -14.96%   -71.5%   -0.55
  手工动量正向选8只 满仓         -82.2%  -22.5%   -89.8%   -0.72
  同期沪深300                     -4.0%

========================================================================
三条被数据证伪的直觉
========================================================================

1. "动量选股"
   20 日 RankIC 在 2018-2026 每一年都是负值（-0.03 ~ -0.12），
   说明 A 股中短期是均值回归而非趋势延续。按动量正向选股，
   前 8 只未来 20 日收益 -0.93%，而同期的市场平均是 +0.52%，
   也就是选股能力还不如随机。

2. "IC 分层单调 → 选股有效"
   IC 是正的（+0.10，胜率 74%），五分层看着也单调，
   但落到「每天只买 8 只」就失效了，因为：
   - IC 衡量的是全横截面的排序能力，不能外推到极值端
   - 极端高分股是深跌股（mom20 -18.5%、RSI 29、ma_gap60 -11%），
     反转因子的收益集中在中段(Q3/Q4 +0.91%/+0.89%)，不在 Q5 顶端(+0.72%)
   - 8 只的样本太小，无法分散 A 股个股的极端风险

3. "止损能保护收益"
   满仓加 -8% 止损：-7.38% → 实测 -1.48%，看起来有用；
   但配上择时后反而从 -7.55% 恶化到 -5.70%。
   止损在震荡市里不断被触发，反复付出冲击成本。
   真正的保护来自仓位控制，不来自个股止损。

========================================================================
唯一稳定有效的因子：指数择时
========================================================================

  15/45/150 均线择时在创业板指上：
  - 平均仓位 35%，空仓 65% 的时间
  - 参数从 (10,30,90) 到 (30,100,200) 全部为正收益
  - 把 -63.1% 的市场回撤压到 -18.3%

  这是唯一在多参数下都稳定为正的组件。它本身不产生 alpha，
  但把「熊市满仓」这个最大的亏损来源切掉了。

========================================================================
为什么最终选等权而非选股
========================================================================

  等权全池 = 承认没有个股 alpha，只做仓位管理。
  这在 2020-2026 的 A 股是更诚实的答案：

  市场等权年化 -7.48%，择时后 +8.93%，
  全部收益来自「避开下跌段」，而不是「选中好股票」。

  选股模型的最好成绩是 -8.00%，比不选股差 17 个百分点。
  多出来的换手成本（年化 2%）+ 集中度风险，超过了那点微弱的排序能力。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VerifiedConfig:
    """经回测验证的配置。改这里等于改策略，改动前请重跑 backtest。"""

    # ---- 择时（唯一有效组件）
    timing_index: str = "399006"       # 创业板指，比沪深300更贴合成长股池
    timing_fast: int = 15
    timing_slow: int = 45
    timing_bear_ma: int = 150

    pos_full: float = 1.00             # 指数>MA45 且 MA15>MA45
    pos_half: float = 0.60             # 指数>MA45 但 MA15<MA45
    pos_weak: float = 0.30             # 指数<MA45 但 >MA150
    pos_empty: float = 0.00            # 指数<MA150 空仓

    use_breadth: bool = True
    breadth_hi: float = 0.55
    breadth_lo: float = 0.25

    # ---- 选股：不选股，全池等权
    equal_weight: bool = True
    max_positions: int = 8             # 仅用于小资金实际下单（等权取前N只）
    max_weight: float = 0.15           # 单票上限，防止单只占比过大

    # ---- 风控：只做仓位控制，不做个股止损
    stop_loss: float = -0.20           # 极宽，仅防黑天鹅，不参与日常交易
    trail_stop: float = -0.25
    take_profit: float = 99.0
    max_hold_days: int = 999

    # ---- 股票池
    min_list_days: int = 250
    min_avg_amount: float = 8e7
    min_price: float = 2.0
    max_price: float = 500.0
    exclude_st: bool = True
    exclude_bj: bool = True
    exclude_kcb: bool = False
    exclude_cyb: bool = False

    # ---- 调仓
    rebalance_days: int = 10

    # ---- 成本
    commission_rate: float = 0.00025
    commission_min: float = 5.0
    stamp_tax: float = 0.0005
    transfer_fee: float = 0.00001
    slippage: float = 0.0010

    def evidence(self) -> list[dict]:
        """回测证据表，供报告引用。"""
        return [
            {"方案": "全池等权 + 创业板择时", "总收益": "+78.5%", "年化": "+8.93%",
             "最大回撤": "-18.3%", "Sharpe": "0.66", "采用": "是"},
            {"方案": "全池等权 满仓", "总收益": "-40.9%", "年化": "-7.48%",
             "最大回撤": "-63.1%", "Sharpe": "-0.16", "采用": "否"},
            {"方案": "IC模型选8只 + 择时", "总收益": "-43.1%", "年化": "-8.00%",
             "最大回撤": "-43.2%", "Sharpe": "-0.72", "采用": "否"},
            {"方案": "手工动量选8只 满仓", "总收益": "-82.2%", "年化": "-22.5%",
             "最大回撤": "-89.8%", "Sharpe": "-0.72", "采用": "否"},
            {"方案": "沪深300同期", "总收益": "-4.0%", "年化": "—",
             "最大回撤": "—", "Sharpe": "—", "采用": "基准"},
        ]


VERIFIED = VerifiedConfig()


def apply_to(cfg):
    """把验证过的参数注入 Config 对象。"""
    from dataclasses import replace
    s = cfg.strategy
    new_s = replace(
        s,
        timing_index=VERIFIED.timing_index,
        timing_fast=VERIFIED.timing_fast,
        timing_slow=VERIFIED.timing_slow,
        timing_bear_ma=VERIFIED.timing_bear_ma,
        pos_full=VERIFIED.pos_full,
        pos_half=VERIFIED.pos_half,
        pos_weak=VERIFIED.pos_weak,
        pos_empty=VERIFIED.pos_empty,
        use_breadth=VERIFIED.use_breadth,
        breadth_hi=VERIFIED.breadth_hi,
        breadth_lo=VERIFIED.breadth_lo,
        max_positions=VERIFIED.max_positions,
        max_weight=VERIFIED.max_weight,
        stop_loss=VERIFIED.stop_loss,
        trail_stop=VERIFIED.trail_stop,
        take_profit=VERIFIED.take_profit,
        max_hold_days=VERIFIED.max_hold_days,
        min_list_days=VERIFIED.min_list_days,
        min_avg_amount=VERIFIED.min_avg_amount,
        min_price=VERIFIED.min_price,
        max_price=VERIFIED.max_price,
        exclude_st=VERIFIED.exclude_st,
        exclude_bj=VERIFIED.exclude_bj,
        exclude_kcb=VERIFIED.exclude_kcb,
        exclude_cyb=VERIFIED.exclude_cyb,
        rebalance_days=VERIFIED.rebalance_days,
    )
    new_c = replace(
        cfg.cost,
        commission_rate=VERIFIED.commission_rate,
        commission_min=VERIFIED.commission_min,
        stamp_tax=VERIFIED.stamp_tax,
        transfer_fee=VERIFIED.transfer_fee,
        slippage=VERIFIED.slippage,
    )
    return type(cfg)(cost=new_c, strategy=new_s, account=cfg.account)


# ========================================================================
# 追加结论：策略有效性依赖本金规模（离散化约束）
# ========================================================================
#
# A 股一手 100 股 + 最低 5 元佣金，导致"全池等权"无法在小资金上表达：
#   10 万本金 / 1500 只池 = 66 元每只 < 一手最低价 -> 实际只能买 18 只
#   1000 万本金 -> 能买 1148 只，接近全池等权
#
# 实测（同一策略，只改本金）：
#   本金        总收益   年化     平均持仓   与理论值差距
#   10 万      -16.4%   -2.61%     18 只      -127 个百分点
#   100 万     +54.2%   +6.61%    282 只       -57 个百分点
#   1000 万    +91.1%  +10.03%   1148 只       -20 个百分点
#   1 亿       +97.4%  +10.56%   1264 只       -14 个百分点
#   向量化理论  +111.1% +11.66%   无限可分      —
#
# 结论：这套策略需要至少 1000 万可用资金才能接近理论表现。
#       10 万级别资金只能持有 8-18 只，等价于主动选股，
#       而选股部分已被证明没有 alpha（见上文）。
#
# 小资金的诚实选择：
#   (a) 只做择时不做分散 —— 用 ETF 代替个股（159915 创业板ETF），
#       仓位跟随择时信号，规避一手的离散化问题；
#   (b) 或者接受"策略在小资金上不成立"这个事实，不做。
# ========================================================================

MIN_EFFECTIVE_CAPITAL = 10_000_000
SMALL_CAPITAL_THRESHOLD = 500_000


def capital_verdict(cash: float) -> dict:
    """根据本金给出策略适用性判断。"""
    if cash >= MIN_EFFECTIVE_CAPITAL:
        return {
            "ok": True,
            "mode": "stock_portfolio",
            "message": f"本金 {cash:,.0f} 元，可持有 1000+ 只个股，"
                       f"接近全池等权，策略接近理论表现。",
            "expected_cagr": 0.106,
            "expected_dd": -0.143,
        }
    if cash >= SMALL_CAPITAL_THRESHOLD:
        return {
            "ok": True,
            "mode": "stock_portfolio_degraded",
            "message": f"本金 {cash:,.0f} 元，只能持有 200-300 只，"
                       f"相对理论值年化约损失 4 个百分点。",
            "expected_cagr": 0.066,
            "expected_dd": -0.221,
        }
    return {
        "ok": False,
        "mode": "etf_only",
        "message": f"本金 {cash:,.0f} 元，不足以表达全池等权（一手门槛）。"
                   f"建议改为 ETF 择时：159915 创业板ETF 按同一择时信号调整仓位。",
        "expected_cagr": None,
        "expected_dd": None,
    }
