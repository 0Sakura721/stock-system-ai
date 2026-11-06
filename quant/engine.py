# -*- coding: utf-8 -*-
"""每日决策引擎：把行情 → 因子 → 目标组合 → 具体委托单，含全部风控校验。

输出结构（可直接给执行器或人）：
{
  "signal_date": "2026-10-30",
  "exec_date":   "2026-11-02",
  "target_position": 0.5,
  "sells": [{code, name, shares, price, reason, urgency}],
  "buys":  [{code, name, shares, price, amount, weight, score, reason}],
  "risk":  {...},
  "notes": [...]
}
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import data as D
from . import indicators as ind
from . import strategy as st
from . import findings as FN
from .account import Account
from .config import CFG
from .risk import RiskEngine

OUT = None


def _name_map(conn) -> dict[str, str]:
    sl = D.load_stocklist(conn)
    if not len(sl):
        return {}
    return dict(zip(sl.secid, sl.name))


def _ind_map(conn) -> dict[str, str]:
    sl = D.load_stocklist(conn)
    if not len(sl):
        return {}
    return dict(zip(sl.secid, sl.industry))


def build_state(conn, start: str = "2018-01-01", cfg=None):
    """加载面板 + 因子 + 打分 + 择时。"""
    cfg = FN.apply_to(cfg or CFG)
    bars = D.load_bars(conn, adj="hfq", start=start)
    if bars.empty:
        raise RuntimeError("行情库为空，先跑 python -m quant.fetch")
    bars = bars.dropna(subset=["close"])
    bars = bars[bars["close"] > 0]
    p = ind.Panel(bars)
    sl = D.load_stocklist(conn)
    inds = ind.compute(p)
    uni = st.build_universe(p, inds, cfg.strategy, sl)
    sc = st.rank_score(p, inds, uni)
    br = st.breadth(p, inds, uni)
    idx = D.fetch_index_close(cfg.strategy.timing_index, "2016-01-01")
    pos = st.timing(p, pd.Series(idx), cfg.strategy, br)
    return dict(p=p, inds=inds, uni=uni, score=sc, breadth=br,
                timing=pos, stocks=sl, cfg=cfg)


def make_plan(state, account: Account, risk: RiskEngine, top_n: int = None,
              exec_date: str = None, verbose: bool = True) -> dict:
    p, inds, sc = state["p"], state["inds"], state["score"]
    cfg = state["cfg"]
    day = str(p.dates[-1])
    eq_cfg = cfg.account

    # 实时报价（持仓 + 候选）
    hold = account.positions
    day_scores = sc.loc[day].dropna().sort_values(ascending=False)
    cand_codes = [s.split(".")[-1] for s in day_scores.head(80).index]
    rt_codes = list(set(list(hold.code) + cand_codes)) if len(hold) else cand_codes

    rt = pd.DataFrame()
    try:
        rt = D.fetch_realtime(rt_codes)
    except Exception as e:                        # noqa: BLE001
        if verbose:
            print("[警告] 实时报价失败，使用收盘价:", str(e)[:60])
    rt_map = {r.code: r for r in rt.itertuples(index=False)} if len(rt) else {}

    # 现价：优先实时，其次最新收盘（不复权），最后后复权价
    def price_of(secid: str) -> float:
        code = secid.split(".")[-1]
        if code in rt_map and np.isfinite(rt_map[code].price):
            return float(rt_map[code].price)
        v = p.close.at[day, secid]
        return float(v) if np.isfinite(v) else 0.0

    def pre_of(secid: str) -> float:
        code = secid.split(".")[-1]
        if code in rt_map and np.isfinite(rt_map[code].pre_close):
            return float(rt_map[code].pre_close)
        return 0.0

    def amt_of(secid: str) -> float:
        code = secid.split(".")[-1]
        if code in rt_map and np.isfinite(rt_map[code].amount):
            return float(rt_map[code].amount)
        return float(p.amount.at[day, secid] or 0)

    # 账户快照
    prices = {}
    for r in hold.itertuples(index=False):
        prices[r.code] = price_of(str(r.code))
    snap = account.snapshot(prices)
    equity = snap["equity"]

    target = float(state["timing"].reindex(p.dates).ffill().iloc[-1])

    # ---------------- 卖出判断
    sells = []
    name_map = _name_map(account.conn)
    held_codes = set(hold.code) if len(hold) else set()
    keep_rank = int(cfg.strategy.max_positions * cfg.strategy.keep_rank_mult)
    top_codes = [s.split(".")[-1] for s in day_scores.head(keep_rank).index]
    secid_of = {r.code: f"{'1' if r.code[0] in '56' else '0'}.{r.code}"
                for r in (hold.itertuples(index=False) if len(hold) else [])}

    for r in (hold.itertuples(index=False) if len(hold) else []):
        secid = secid_of[r.code]
        px = prices[r.code]
        cost = float(r.cost)
        peak = float(r.peak or px)
        days = int(r.hold_days or 0)
        ret = px / cost - 1 if cost else 0
        reason, urgency = "", "normal"

        if target <= 0.05:
            reason, urgency = "择时信号为空仓，全部清仓", "high"
        else:
            stop_hit, why = risk.should_stop(cost, px, peak, days)
            if stop_hit:
                reason, urgency = why, "high"
            elif r.code not in top_codes:
                reason = f"综合分跌出前 {keep_rank} 名，调出"
                urgency = "normal"
            elif ret > cfg.strategy.take_profit * 0.8 and target < 0.6:
                reason = f"浮盈 {ret:.0%} 且仓位转弱，减半锁定"
                urgency = "normal"

        if reason:
            shares = int(r.shares)
            if "减半" in reason:
                shares = int(shares / 2 / 100) * 100
            v = risk.check_sell_tradable(px, pre_of(secid), r.code)
            if not v.ok:
                sells.append(dict(code=r.code, name=r.name, shares=shares,
                                  price=round(px, 2), action="SELL",
                                  reason=f"{reason}｜但{v.reason}", blocked=True,
                                  pnl=round(ret, 4), urgency=urgency))
            else:
                sells.append(dict(code=r.code, name=r.name, shares=shares,
                                  price=round(px, 2), action="SELL",
                                  reason=reason, blocked=False,
                                  pnl=round(ret, 4), urgency=urgency))

    # ---------------- 买入判断
    keep = [c for c in held_codes if c not in {s["code"] for s in sells}]
    n = top_n or cfg.strategy.max_positions
    slots = max(n - len(keep), 0)

    realized = sum(s["price"] * s["shares"] for s in sells if not s["blocked"])
    cash_avail = snap["cash"] + realized
    budget_total = equity * min(target, eq_cfg.max_total_position) * (1 - eq_cfg.cash_buffer)
    pos_keep_val = sum(pr["value"] for pr in snap["positions"]
                       if pr["code"] in keep)
    room = max(budget_total - pos_keep_val, 0)
    per = room / max(slots, 1) if slots else 0

    buys = []
    for secid, s in day_scores.items():
        if slots <= 0 or per <= 0:
            break
        code = secid.split(".")[-1]
        if code in held_codes or s < cfg.strategy.min_score:
            if s < cfg.strategy.min_score:
                continue
            continue
        px = price_of(secid)
        if px <= 0:
            continue
        shares = int(min(per, cash_avail) / px / eq_cfg.lot_size) * eq_cfg.lot_size
        if shares < eq_cfg.lot_size:
            continue
        v = risk.full_check("BUY", code, px, shares, pre_of(secid), cash_avail,
                            equity, snap["pos_value"], target, 0.0, amt_of(secid))
        if not v.ok:
            continue
        shares = v.adjusted_shares or shares
        amt = px * shares
        # 现金扣除
        cash_avail -= amt + cfg.cost.buy_cost(amt)
        if cash_avail < 0:
            continue
        mom20 = float(inds["mom20"].at[day, secid]) if secid in inds["mom20"].columns else np.nan
        atr = float(inds["atr_pct"].at[day, secid]) if secid in inds["atr_pct"].columns else np.nan
        buys.append(dict(
            code=code, name=name_map.get(secid, ""), action="BUY",
            shares=shares, price=round(px, 2), amount=round(amt, 2),
            weight=round(amt / equity, 4) if equity else 0,
            score=round(float(s), 3),
            mom20=round(mom20, 4) if np.isfinite(mom20) else None,
            atr_pct=round(atr, 4) if np.isfinite(atr) else None,
            stop=risk.stop_price(px, px, px * atr if np.isfinite(atr) else None)[0],
            reason=f"综合分{s:.2f}｜20日动量{(mom20 or 0):+.1%}"))
        slots -= 1

    cur_pct = snap["pos_value"] / equity if equity else 0
    plan = dict(
        signal_date=day,
        exec_date=exec_date or "下一交易日",
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        target_position=round(target, 4),
        current_position=round(cur_pct, 4),
        breadth=round(float(state["breadth"].iloc[-1]), 4),
        equity=round(equity, 2),
        cash=round(snap["cash"], 2),
        pos_value=round(snap["pos_value"], 2),
        total_return=round(snap["total_return"], 4),
        sells=sells,
        buys=buys,
        holdings=snap["positions"],
        config=dict(
            max_positions=cfg.strategy.max_positions,
            max_weight=cfg.strategy.max_weight,
            stop_loss=cfg.strategy.stop_loss,
            trail_stop=cfg.strategy.trail_stop,
            take_profit=cfg.strategy.take_profit,
            max_hold_days=cfg.strategy.max_hold_days,
            rebalance_days=cfg.strategy.rebalance_days),
        market=dict(
            index=cfg.strategy.timing_index,
            timing_signal=round(target, 2),
        ),
    )

    # 候选池（供人工参考）
    plan["candidates"] = [
        dict(code=s.split(".")[-1], name=name_map.get(s, ""),
             score=round(float(v), 3))
        for s, v in day_scores.head(20).items()
    ]
    plan["notes"] = _notes(plan)
    return plan


def _notes(plan: dict) -> list[str]:
    out = []
    t, cur = plan["target_position"], plan["current_position"]
    if t >= 0.99:
        out.append("择时信号：满仓。指数在MA45上方且MA15上穿MA45，趋势确认。")
    elif t >= 0.55:
        out.append("择时信号：六成仓。指数在MA45上方，但短期均线回调。")
    elif t >= 0.25:
        out.append("择时信号：三成仓。指数跌破MA45但仍在MA150上方，控制风险。")
    else:
        out.append("择时信号：空仓。指数跌破MA150，深度弱势，等待右侧信号。")
    if abs(t - cur) > 0.15:
        out.append(f"当前仓位 {cur:.0%} 与目标 {t:.0%} 偏离较大，需要调整。")
    blocked = [s for s in plan["sells"] if s.get("blocked")]
    if blocked:
        out.append(f"有 {len(blocked)} 只卖单因跌停/停牌无法成交，次日重试。")
    out.append("买入挂限价：参考价×1.005~1.01，未成交当日撤单，不追高。")
    out.append("个股止损设置极宽（-20%），仅防黑天鹅；日常保护来自仓位控制。")
    return out


# ------------------------------------------------------------------ 输出
def render_md(plan: dict) -> str:
    L = [f"# 交易计划 · 信号日 {plan['signal_date']}", ""]
    L.append(f"**执行日** {plan['exec_date']} ｜ 生成 {plan['generated_at']}")
    L.append("")
    L.append("## 一、大盘与仓位")
    L.append("")
    L.append("| 项目 | 数值 |")
    L.append("|---|---|")
    L.append(f"| 择时信号（目标仓位） | **{plan['target_position']:.0%}** |")
    L.append(f"| 当前仓位 | {plan['current_position']:.0%} |")
    L.append(f"| 市场宽度（站上MA20占比） | {plan['breadth']:.0%} |")
    L.append(f"| 账户权益 | {plan['equity']:,.0f} 元 |")
    L.append(f"| 可用现金 | {plan['cash']:,.0f} 元 |")
    L.append(f"| 累计收益 | {plan['total_return']:+.2%} |")
    L.append("")
    for nt in plan["notes"]:
        L.append(f"- {nt}")
    L.append("")

    L.append("## 二、卖出（先执行）")
    L.append("")
    if plan["sells"]:
        L.append("| 代码 | 名称 | 股数 | 参考价 | 浮动盈亏 | 原因 |")
        L.append("|---|---|---|---|---|---|")
        for s in plan["sells"]:
            flag = " ⚠️" if s.get("blocked") else ""
            L.append(f"| {s['code']} | {s['name']} | {s['shares']} | "
                     f"{s['price']} | {s['pnl']:+.1%} | {s['reason']}{flag} |")
    else:
        L.append("无。持仓全部保留。")
    L.append("")

    L.append("## 三、买入（后执行）")
    L.append("")
    if plan["buys"]:
        L.append("| 代码 | 名称 | 股数 | 限价 | 金额 | 权重 | 综合分 | 止损价 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for b in plan["buys"]:
            L.append(f"| {b['code']} | {b['name']} | {b['shares']} | {b['price']} | "
                     f"{b['amount']:,.0f} | {b['weight']:.1%} | {b['score']} | {b['stop']} |")
        L.append("")
        L.append(f"合计买入 **{sum(b['amount'] for b in plan['buys']):,.0f}** 元")
    else:
        L.append("无。保持现金或维持现仓。")
    L.append("")

    if plan.get("candidates"):
        L.append("## 四、候选池（备选，未下单）")
        L.append("")
        L.append("| 代码 | 名称 | 综合分 |")
        L.append("|---|---|---|")
        for c in plan["candidates"][:10]:
            L.append(f"| {c['code']} | {c['name']} | {c['score']} |")
        L.append("")

    L.append("## 五、执行纪律")
    L.append("")
    L.append("1. **卖出先于买入**，A股卖出资金当日可用。")
    L.append(f"2. 买入限价 = 参考价 × 1.005（不打市价追高）；开盘一字涨停的直接放弃。")
    L.append(f"3. 单票止损 {plan['config']['stop_loss']:.0%}、回撤止损 {plan['config']['trail_stop']:.0%}，触发即卖。")
    L.append(f"4. 最长持有 {plan['config']['max_hold_days']} 个交易日，到期无条件调出。")
    L.append(f"5. 每 {plan['config']['rebalance_days']} 个交易日调仓一次，其余时间只做风控。")
    return "\n".join(L)
