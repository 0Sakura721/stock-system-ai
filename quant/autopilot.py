# -*- coding: utf-8 -*-
"""自动操盘引擎（Autopilot）：让 AI 每天把整套流程跑完。

设计目标：从「给出计划」升级为「执行计划」。
每一轮 cron 运行做这几件事：

  1. 补数据        fetch 增量，确保 signal_date 是最新交易日
  2. 读账户        现金/持仓/挂单，全部从 SQLite 读
  3. 算信号        择时 → 目标仓位；等权 → 目标持仓
  4. 生成订单      diff(当前, 目标) → 卖单先、买单后，过风控
  5. 模拟成交      以当日/次日开盘价成交，真实费用，落账
  6. 风控巡检      止损/回撤/到期/黑天鹅
  7. 写日志        每笔决策的依据，可复核
  8. 出日报        Markdown + 面板，记录当天做了什么、为什么

关键约定（和回测完全一致的执行规则）：
  - 信号在 T 日收盘后生成，T+1 开盘执行
  - 卖单先于买单（A股卖出资金当日可用）
  - 涨停买不进、跌停卖不出、停牌不动
  - 整手、最低佣金、印花税、滑点全部计入

为什么是「模拟成交」而不是真下单：
  券商下单需要 API 凭证。本引擎把「下单」这一步做成可替换的
  Broker 接口：默认 PaperBroker（按开盘价成交），
  接实盘只要实现 LiveBroker.submit_order() 即可。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import data as D
from . import engine as E
from . import findings as FN
from . import strategy as st
from .account import Account
from .config import CFG, OUT_DIR
from .risk import RiskEngine

LOG_DIR = os.path.join(OUT_DIR, "autopilot")
os.makedirs(LOG_DIR, exist_ok=True)

STATE_PATH = os.path.join(LOG_DIR, "state.json")


# ====================================================================== Broker
class PaperBroker:
    """模拟撮合：按给定价格成交，真实费用。

    接实盘时继承这个类，覆盖 submit() 即可，其余逻辑不变。
    """

    name = "paper"

    def __init__(self, cfg=None):
        self.cfg = cfg or CFG

    def tradable(self, day: str, secid: str, side: str, p) -> tuple[bool, str]:
        """检查当日能否成交。p 是行情面板。"""
        pc = p.close.shift(1).at[day, secid] if secid in p.close.columns else np.nan
        o = p.open.at[day, secid] if secid in p.open.columns else np.nan
        a = p.amount.at[day, secid] if secid in p.amount.columns else np.nan

        if not np.isfinite(a) or a <= 0:
            return False, "停牌（无成交额）"
        if not np.isfinite(o) or o <= 0:
            return False, "无开盘价"
        if not np.isfinite(pc) or pc <= 0:
            return True, ""

        chg = o / pc - 1
        code = secid.split(".")[-1]
        if code.startswith(("300", "301", "688")):
            lim = 0.19
        elif code.startswith(("8", "4", "9")):
            lim = 0.29
        else:
            lim = 0.095

        if side == "BUY" and chg >= lim:
            return False, f"开盘涨停 {chg:+.1%}，买不进"
        if side == "SELL" and chg <= -lim:
            return False, f"开盘跌停 {chg:+.1%}，卖不出"
        return True, ""

    def fill_price(self, day: str, secid: str, side: str, p) -> float:
        o = float(p.open.at[day, secid])
        slip = self.cfg.cost.slippage
        return o * (1 + slip) if side == "BUY" else o * (1 - slip)

    def submit(self, side: str, secid: str, shares: int, price: float) -> dict:
        return {"status": "FILLED", "price": price, "shares": shares,
                "broker": self.name}


# ====================================================================== State
@dataclass
class RunState:
    last_signal_date: str = ""
    last_exec_date: str = ""
    runs: int = 0
    total_decisions: int = 0
    last_run_at: str = ""


def load_state() -> RunState:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                d = json.load(f)
            return RunState(**{k: v for k, v in d.items() if k in RunState.__dataclass_fields__})
        except Exception:                          # noqa: BLE001
            pass
    return RunState()


def save_state(s: RunState):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(asdict(s), f, ensure_ascii=False, indent=2)


# ====================================================================== Autopilot
class Autopilot:
    def __init__(self, cash: float = 1e5, cfg=None, broker: PaperBroker = None,
                 verbose: bool = True):
        self.cfg = FN.apply_to(cfg or CFG)
        self.cash0 = cash
        self.broker = broker or PaperBroker(self.cfg)
        self.verbose = verbose
        self.risk = RiskEngine(self.cfg)
        self.conn = D.connect()
        self.account = Account(self.conn, initial_cash=cash)
        self.decisions: list[dict] = []

    def log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------ 1 数据
    def step_fetch(self, full: bool = False) -> str:
        from .fetch import update
        self.log("[1/7] 更新行情...")
        try:
            update(full=full, workers=24, progress_every=10_000)
        except Exception as e:                     # noqa: BLE001
            self.log(f"  [警告] 抓取异常，用缓存: {str(e)[:70]}")
        d = self.conn.execute(
            "SELECT MAX(date) FROM bars WHERE adj='hfq'").fetchone()[0]
        self.log(f"  最新行情日: {d}")
        return d

    # ------------------------------------------------------------ 2 账户
    def step_account(self) -> dict:
        self.log("[2/7] 读取账户...")
        hold = self.account.positions
        codes = list(hold.code) if len(hold) else []
        prices = {}
        if codes:
            try:
                rt = D.fetch_realtime(codes)
                prices = {r.code: float(r.price) for r in rt.itertuples(index=False)
                          if np.isfinite(r.price)}
            except Exception:                      # noqa: BLE001
                pass
        snap = self.account.snapshot(prices)
        self.log(f"  现金 {snap['cash']:,.2f}  持仓 {snap['pos_value']:,.2f}  "
                 f"权益 {snap['equity']:,.2f}  持仓数 {len(snap['positions'])}")
        return snap

    # ------------------------------------------------------------ 3 信号
    def step_signal(self) -> dict:
        self.log("[3/7] 计算择时信号...")
        state = E.build_state(self.conn, cfg=self.cfg)
        p = state["p"]
        detail = st.timing_detail(p, pd.Series(
            D.fetch_index_close(self.cfg.strategy.timing_index, "2016-01-01")),
            self.cfg.strategy, state["breadth"])

        row = detail.iloc[-1]
        target = float(row["pos_final"])
        sig_date = str(p.dates[-1])

        self.log(f"  信号日 {sig_date}")
        self.log(f"  {self.cfg.strategy.timing_index} 收盘 {row['index']:.0f}  "
                 f"MA{self.cfg.strategy.timing_fast}={row['ma_fast']:.0f}  "
                 f"MA{self.cfg.strategy.timing_slow}={row['ma_slow']:.0f}  "
                 f"MA{self.cfg.strategy.timing_bear_ma}={row['ma_bear']:.0f}")
        self.log(f"  档位 {int(row['level'])}/3  宽度 {row['breadth']:.0%}  "
                 f"→ 目标仓位 {target:.0%}")

        return dict(state=state, detail=detail, target=target, signal_date=sig_date,
                    level=int(row["level"]), index_value=float(row["index"]),
                    ma_fast=float(row["ma_fast"]), ma_slow=float(row["ma_slow"]),
                    ma_bear=float(row["ma_bear"]), breadth=float(row["breadth"]))

    # ------------------------------------------------------------ 4 订单
    def step_orders(self, sig: dict, snap: dict) -> list[dict]:
        self.log("[4/7] 生成订单...")
        from . import portfolio as PF
        p = sig["state"]["p"]
        sc = sig["state"]["score"]
        day = sig["signal_date"]
        target = sig["target"]

        equity = snap["equity"]

        # 目标持仓（带粘性，低换手）
        hold_secids = [("1." if c[0] in "56" else "0.") + c
                       for c in (list(self.account.positions.code)
                                 if len(self.account.positions) else [])]
        prev = getattr(self, "_prev_targets", None)
        w = PF.build_target(sc, day, hold_secids, self.cfg.strategy,
                            target_pos=target, prev_targets=prev)
        self._prev_targets = w

        # 现价与持仓
        prices: dict[str, float] = {}
        for s in list(w) + hold_secids:
            px = self._last_price(p, s)
            if np.isfinite(px) and px > 0:
                prices[s] = px
        holdings = {}
        for r in (self.account.positions.itertuples(index=False)
                  if len(self.account.positions) else []):
            sid = ("1." if r.code[0] in "56" else "0.") + r.code
            holdings[sid] = dict(shares=int(r.shares), cost=float(r.cost), name=r.name)
            prices.setdefault(sid, float(r.last_price or 0))

        raw = PF.diff_orders(w, holdings, equity, prices, self.cfg.strategy)

        orders: list[dict] = []
        for o in raw:
            code = o["secid"].split(".")[-1]
            orders.append(dict(side=o["side"], code=code, secid=o["secid"],
                               name=self._hold_name(code) if o["side"] == "SELL"
                               else self._name_of(code),
                               shares=o["shares"], reason=o["reason"]))

        self.log(f"  目标持仓 {len(w)} 只  卖单 "
                 f"{sum(1 for o in orders if o['side']=='SELL')} 笔  买单 "
                 f"{sum(1 for o in orders if o['side']=='BUY')} 笔")
        return orders

    def _hold_name(self, code: str) -> str:
        pos = self.account.positions
        if len(pos):
            m = dict(zip(pos.code, pos.name))
            if code in m:
                return str(m[code])
        return self._name_of(code)

    def _last_price(self, p, secid) -> float:
        if secid not in p.close.columns:
            return np.nan
        v = p.close.iloc[-1][secid]
        return float(v) if np.isfinite(v) else np.nan

    def _name_of(self, code: str) -> str:
        try:
            sl = D.load_stocklist(self.conn)
            m = sl.set_index("code")["name"]
            return str(m.get(code, code))
        except Exception:                          # noqa: BLE001
            return code

    # ------------------------------------------------------------ 5 执行
    def step_execute(self, sig: dict, orders: list[dict]) -> list[dict]:
        """执行订单。用 signal_date 之后第一个有数据的交易日开盘价成交。"""
        self.log("[5/7] 执行订单...")
        p = sig["state"]["p"]
        sig_date = sig["signal_date"]

        # 找执行日：信号日之后的第一个交易日
        exec_date = None
        dates = list(p.dates)
        if sig_date in dates:
            idx = dates.index(sig_date)
            if idx + 1 < len(dates):
                exec_date = dates[idx + 1]
        if exec_date is None:
            self.log(f"  无后续交易日（信号日 {sig_date} 是最新一天），下单挂起")
            for o in orders:
                self.account.queue_order(sig_date, "NEXT", o["side"], o["code"],
                                         o["name"], self._last_price(p, o["secid"]),
                                         o["shares"], o["reason"])
            return []

        self.log(f"  信号日 {sig_date} → 执行日 {exec_date}")
        filled = []
        for o in orders:
            ok, why = self.broker.tradable(exec_date, o["secid"], o["side"], p)
            if not ok:
                self.log(f"  [跳过] {o['side']} {o['code']} {o['name']}: {why}")
                self.decisions.append(dict(**o, status="REJECTED", note=why,
                                           exec_date=exec_date))
                continue
            px = self.broker.fill_price(exec_date, o["secid"], o["side"], p)
            res = self.broker.submit(o["side"], o["secid"], o["shares"], px)
            if res["status"] != "FILLED":
                continue
            self.account.apply_fill(o["side"], o["code"], o["name"],
                                    res["price"], res["shares"], date=exec_date)
            filled.append(dict(**o, status="FILLED", price=round(res["price"], 3),
                               amount=round(res["price"] * res["shares"], 2),
                               exec_date=exec_date))
            self.decisions.append(filled[-1])
            self.log(f"  [成交] {o['side']} {o['code']} {o['name']} "
                     f"{o['shares']}股 @ {res['price']:.3f}")

        return filled

    # ------------------------------------------------------------ 6 风控
    def step_risk(self, snap: dict) -> list[dict]:
        self.log("[6/7] 风控巡检...")
        alerts = []
        for r in snap["positions"]:
            ret = r["pnl"]
            if ret <= self.cfg.strategy.stop_loss:
                alerts.append(dict(code=r["code"], name=r["name"], level="HIGH",
                                   msg=f"触及止损线 {ret:.1%}"))
            elif r["peak"] > 0:
                trail = r["price"] / r["peak"] - 1
                if trail <= self.cfg.strategy.trail_stop:
                    alerts.append(dict(code=r["code"], name=r["name"], level="HIGH",
                                       msg=f"回撤 {trail:.1%} 触及线"))
        for a in alerts:
            self.log(f"  [风控] {a['code']} {a['name']}: {a['msg']}")
        if not alerts:
            self.log("  无风险告警")
        return alerts

    # ------------------------------------------------------------ 7 日报
    def step_report(self, sig, snap, orders, filled, alerts, risk_alerts) -> str:
        self.log("[7/7] 生成日报...")
        date_tag = datetime.now().strftime("%Y%m%d")
        path = os.path.join(LOG_DIR, f"{date_tag}.md")

        L = []
        L.append(f"# 自动操盘日报 · {datetime.now():%Y-%m-%d %H:%M}")
        L.append("")
        L.append(f"**信号日** {sig['signal_date']} ｜ **执行日** "
                 f"{filled[0]['exec_date'] if filled else '待定'}")
        L.append("")

        # 决策依据
        L.append("## 一、今日决策依据")
        L.append("")
        L.append(f"| 指标 | 数值 |")
        L.append(f"|---|---|")
        L.append(f"| 指数收盘 | {sig['index_value']:.0f} |")
        L.append(f"| MA{self.cfg.strategy.timing_fast} | {sig['ma_fast']:.0f} |")
        L.append(f"| MA{self.cfg.strategy.timing_slow} | {sig['ma_slow']:.0f} |")
        L.append(f"| MA{self.cfg.strategy.timing_bear_ma} | {sig['ma_bear']:.0f} |")
        L.append(f"| 档位 | {sig['level']}/3 |")
        L.append(f"| 市场宽度 | {sig['breadth']:.0%} |")
        L.append(f"| **目标仓位** | **{sig['target']:.0%}** |")
        L.append("")
        L.append(self._explain(sig))
        L.append("")

        # 账户
        L.append("## 二、账户状态")
        L.append("")
        L.append(f"- 现金：{snap['cash']:,.2f}")
        L.append(f"- 持仓市值：{snap['pos_value']:,.2f}")
        L.append(f"- 总权益：{snap['equity']:,.2f}")
        L.append(f"- 累计收益：{snap['total_return']:+.2%}")
        L.append(f"- 当前仓位：{(snap['pos_value']/snap['equity'] if snap['equity'] else 0):.0%}")
        L.append("")

        # 成交
        L.append("## 三、今日成交")
        L.append("")
        if filled:
            L.append("| 方向 | 代码 | 名称 | 股数 | 成交价 | 金额 | 原因 |")
            L.append("|---|---|---|---|---|---|---|")
            for f in filled:
                L.append(f"| {f['side']} | {f['code']} | {f['name']} | {f['shares']} | "
                         f"{f['price']} | {f['amount']:,.0f} | {f['reason']} |")
        else:
            L.append("无成交（仓位与目标一致，或信号为空仓）。")
        L.append("")

        # 被拒
        rejected = [d for d in self.decisions if d.get("status") == "REJECTED"]
        if rejected:
            L.append("## 四、未能成交")
            L.append("")
            for d in rejected:
                L.append(f"- {d['side']} {d['code']} {d['name']}：{d.get('note','')}")
            L.append("")

        # 持仓
        L.append("## 五、当前持仓")
        L.append("")
        if snap["positions"]:
            L.append("| 代码 | 名称 | 股数 | 成本 | 现价 | 盈亏 | 权重 | 持有日 |")
            L.append("|---|---|---|---|---|---|---|---|")
            for r in snap["positions"]:
                L.append(f"| {r['code']} | {r['name']} | {r['shares']} | "
                         f"{r['cost']:.2f} | {r['price']:.2f} | {r['pnl']:+.1%} | "
                         f"{r['weight']:.1%} | {r['hold_days']} |")
        else:
            L.append("空仓。")
        L.append("")

        # 风控
        L.append("## 六、风控告警")
        L.append("")
        if alerts:
            for a in alerts:
                L.append(f"- **{a['level']}** {a['code']} {a['name']}：{a['msg']}")
        else:
            L.append("无。")
        L.append("")

        # 下一步
        L.append("## 七、下一步")
        L.append("")
        L.append(self._next_action(sig, snap, orders))

        txt = "\n".join(L)
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)

        # 同时存 JSON 供面板
        json_path = path.replace(".md", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dict(
                generated_at=datetime.now().isoformat(),
                signal=sig["signal_date"], target=sig["target"],
                level=sig["level"], breadth=sig["breadth"],
                index_value=sig["index_value"],
                equity=snap["equity"], cash=snap["cash"],
                pos_value=snap["pos_value"], positions=snap["positions"],
                orders=orders, filled=filled, alerts=alerts,
            ), f, ensure_ascii=False, indent=2, default=float)

        self.log(f"  日报 → {path}")
        return path

    def _explain(self, sig) -> str:
        lvl = sig["level"]
        if lvl == 3:
            return ("**判读**：指数站上 MA45 且 MA15 上穿，多头排列确认，"
                    "给满仓。这是唯一会让系统重仓的状态。")
        if lvl == 2:
            return ("**判读**：指数在 MA45 上方但短期均线走弱，属健康回调，"
                    "给六成仓，留出余量应对反复。")
        if lvl == 1:
            return ("**判读**：指数跌破 MA45 但守住 MA150，中期转弱长期未崩，"
                    "降到三成仓，只留试探性仓位。")
        return ("**判读**：指数跌破 MA150，深度弱势。规则要求空仓 —— "
                "这不是看空，是承认趋势向下时持股的期望收益为负。"
                "等指数重新站上 MA45 再入场，宁可错过第一波。")

    def _next_action(self, sig, snap, orders) -> str:
        L = []
        if sig["target"] <= 0.05:
            L.append("当前应为空仓。**不做任何买入**，每日只检查指数是否重回 MA45 上方。")
        elif sig["target"] >= 0.95:
            L.append("满仓状态。每日检查个股止损与回撤，不主动减仓。")
        else:
            L.append(f"维持 {sig['target']:.0%} 仓位。每日检查是否需要补足或减仓。")
        L.append("")
        L.append("下一次运行会自动比对目标与当前持仓，只做必要调整，不重复交易。")
        return "\n".join(L)

    # ------------------------------------------------------------ 主流程
    def run_once(self, full_fetch: bool = False) -> dict:
        t0 = datetime.now()
        self.log("=" * 62)
        self.log(f"  自动操盘  {t0:%Y-%m-%d %H:%M:%S}")
        self.log("=" * 62)

        self.step_fetch(full_fetch)
        snap = self.step_account()
        sig = self.step_signal()
        orders = self.step_orders(sig, snap)
        filled = self.step_execute(sig, orders)

        # 执行后重算持仓
        if filled:
            snap = self.step_account()
        alerts = self.step_risk(snap)
        report = self.step_report(sig, snap, orders, filled, alerts, alerts)

        st_ = load_state()
        st_.last_signal_date = sig["signal_date"]
        st_.last_exec_date = filled[0]["exec_date"] if filled else st_.last_exec_date
        st_.runs += 1
        st_.total_decisions += len(self.decisions)
        st_.last_run_at = t0.isoformat()
        save_state(st_)

        self.log("=" * 62)
        self.log(f"  完成：{len(filled)} 笔成交，{len(alerts)} 个告警")
        self.log("=" * 62)

        return dict(signal=sig["signal_date"], target=sig["target"],
                    orders=len(orders), filled=len(filled),
                    alerts=len(alerts), report=report,
                    equity=snap["equity"])

    def close(self):
        self.conn.close()


def run(cash: float = 1e5, full_fetch: bool = False, verbose: bool = True) -> dict:
    ap = Autopilot(cash=cash, verbose=verbose)
    try:
        return ap.run_once(full_fetch=full_fetch)
    finally:
        ap.close()


# ====================================================================== Replay
def _replay_method(self, start: str = "2020-01-01", end: str = None,
                   verbose_every: int = 50) -> dict:
    """历史逐日回放：用 autopilot 的同一套逻辑跑过去，验证执行链路。

    和 backtest.py 的区别：那个引擎用持仓 dict 内部记账；
    这个走真实的 account.apply_fill() → SQLite 账本 → 再从账本读，
    所以能验证「下单→成交→落账→再读」整条链路是否自洽。
    """
    p_state = E.build_state(self.conn, cfg=self.cfg)
    p = p_state["p"]
    sc = p_state["score"]
    timing = p_state["timing"].reindex(p.dates).ffill().fillna(0)

    dates = [d for d in p.dates if d >= start and (end is None or d <= end)]
    self.account.reset(self.cash0)

    nav = []
    for i, day in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        target = float(timing.reindex([day]).iloc[0])


        from . import portfolio as PF
        # 目标权重（带粘性，低换手）
        hold_codes = list(self.account.positions.code) if len(self.account.positions) else []
        hold_secid = [("1." if c[0] in "56" else "0.") + c for c in hold_codes]
        w = PF.build_target(sc, day, hold_secid, self.cfg.strategy,
                            target_pos=target,
                            prev_targets=getattr(self, "_prev_w", None))
        self._prev_w = w

        snap = self.account.snapshot()
        equity = snap["equity"]

        prices = {}
        holdings = {}
        for s in list(w) + hold_secid:
            v = p.open.at[nxt, s] if s in p.open.columns else np.nan
            if np.isfinite(v) and v > 0:
                prices[s] = float(v)
        for r in (self.account.positions.itertuples(index=False)
                  if len(self.account.positions) else []):
            sid = ("1." if r.code[0] in "56" else "0.") + r.code
            holdings[sid] = dict(shares=int(r.shares), cost=float(r.cost), name=r.name)

        raw = PF.diff_orders(w, holdings, equity, prices, self.cfg.strategy)

        # 卖单先执行
        for o in [x for x in raw if x["side"] == "SELL"]:
            secid = o["secid"]
            code = secid.split(".")[-1]
            ok, why = self.broker.tradable(nxt, secid, "SELL", p)
            if not ok:
                continue
            fpx = self.broker.fill_price(nxt, secid, "SELL", p)
            self.account.apply_fill("SELL", code, self._hold_name(code), fpx,
                                    o["shares"], date=nxt)

        # 买单
        avail = self.account.cash()
        for o in [x for x in raw if x["side"] == "BUY"]:
            secid = o["secid"]
            code = secid.split(".")[-1]
            ok, why = self.broker.tradable(nxt, secid, "BUY", p)
            if not ok:
                continue
            fpx = self.broker.fill_price(nxt, secid, "BUY", p)
            amt = fpx * o["shares"]
            if amt + self.cfg.cost.buy_cost(amt) > avail:
                continue
            avail -= amt + self.cfg.cost.buy_cost(amt)
            self.account.apply_fill("BUY", code, self._name_of(code), fpx,
                                    o["shares"], date=nxt)

        snap = self.account.snapshot()
        nav.append(dict(date=nxt, equity=snap["equity"], cash=snap["cash"],
                        pos_value=snap["pos_value"], n_pos=len(snap["positions"]),
                        target=target))
        if verbose_every and i % verbose_every == 0:
            print(f"  {nxt} 权益 {snap['equity']:,.0f} 仓位 "
                  f"{snap['pos_value']/snap['equity'] if snap['equity'] else 0:.0%} "
                  f"持{len(snap['positions'])}只 目标{target:.0%}", flush=True)

    df = pd.DataFrame(nav).set_index(pd.to_datetime([n["date"] for n in nav]))
    eq = df["equity"]
    ret = eq.pct_change().fillna(0)
    total = eq.iloc[-1] / self.cash0 - 1
    years = len(eq) / 244
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1
    dd = (eq / eq.cummax() - 1)
    vol = ret.std() * np.sqrt(244)
    sharpe = ret.mean() * 244 / vol if vol > 0 else 0

    print("\n" + "=" * 60)
    print("  Autopilot 回放结果（走真实账本链路）")
    print("=" * 60)
    print(f"  区间      {nav[0]['date']} ~ {nav[-1]['date']}  ({len(eq)}日)")
    print(f"  总收益    {total:+.2%}")
    print(f"  年化      {cagr:+.2%}")
    print(f"  最大回撤  {dd.min():.2%}")
    print(f"  Sharpe    {sharpe:.2f}")
    print(f"  平均仓位  {(df['pos_value']/df['equity']).mean():.0%}")
    print(f"  平均持仓  {df['n_pos'].mean():.1f} 只")
    print("=" * 60)

    return dict(nav=df, total=total, cagr=cagr, mdd=float(dd.min()), sharpe=sharpe)


Autopilot.replay = _replay_method
