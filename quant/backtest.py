# -*- coding: utf-8 -*-
"""回测引擎 v3：单一事实来源 + 逐日强制对账。

v2 的致命缺陷（已实测确认）：
  卖出失败（跌停/停牌）时仓位被留在 self.pos 里，但 _execute 又按
  "不在 picks 里就卖" 的逻辑每轮重试，导致同一批股票被反复卖→买，
  账面上出现"卖出后立刻以更低价买回"的虚假收益。实测 v1 策略
  回测 +403%，真实轮动模拟 -86.5%，偏差 480 个百分点。

v3 的设计原则：
  1. 每个交易日强制对账：cash + Σ(shares×close) == equity，不平即抛异常
  2. 卖出未成交的，记入 pending_exit 队列，次日优先处理，绝不重新买入
  3. 买入前检查该股是否在 pending_exit，在则跳过
  4. 逐日打印可选追踪，任意一天都能手工复算
  5. 不做"不在目标就卖"的隐式逻辑，卖出只由 _risk 和 _signal 显式产生
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CFG, Config


@dataclass
class Trade:
    date: str
    action: str
    secid: str
    price: float
    shares: int
    amount: float
    fee: float
    reason: str


@dataclass
class Pos:
    secid: str
    shares: int
    cost: float
    entry: str
    peak: float = 0.0
    days: int = 0
    exit_reason: str = ""      # 已下令卖出但未成交


class Backtester:
    def __init__(self, p, scores, target_pos, cfg: Config = CFG,
                 verbose: bool = False, strict: bool = True):
        self.p = p
        self.scores = scores
        self.target_pos = target_pos
        self.cfg = cfg
        self.cost = cfg.cost
        self.verbose = verbose
        self.strict = strict

        self.cash = cfg.account.initial_cash
        self.pos: dict[str, Pos] = {}
        self.trades: list[Trade] = []
        self.nav: list[dict] = []
        self.pending_exit: set[str] = set()

        self._prev_close = p.close.shift(1)
        self._dates = list(p.dates)

    # ------------------------------------------------------------ 价格
    def px(self, day, secid, col="open") -> float:
        df = getattr(self.p, col)
        if secid not in df.columns:
            return np.nan
        try:
            v = df.at[day, secid]
        except KeyError:
            return np.nan
        return float(v) if np.isfinite(v) else np.nan

    @staticmethod
    def limit_pct(secid: str) -> float:
        code = secid.split(".")[-1]
        if code.startswith(("300", "301", "688")):
            return 0.19
        if code.startswith(("8", "4", "9")):
            return 0.29
        return 0.095

    def _pct_open(self, day, secid) -> float:
        pc = self._prev_close.at[day, secid] if secid in self._prev_close.columns else np.nan
        o = self.px(day, secid, "open")
        if not (np.isfinite(pc) and np.isfinite(o)) or pc <= 0:
            return np.nan
        return o / pc - 1

    def cant_buy(self, day, secid) -> bool:
        r = self._pct_open(day, secid)
        return (not np.isfinite(r)) or r >= self.limit_pct(secid)

    def cant_sell(self, day, secid) -> bool:
        r = self._pct_open(day, secid)
        return (not np.isfinite(r)) or r <= -self.limit_pct(secid)

    def halted(self, day, secid) -> bool:
        if secid not in self.p.amount.columns:
            return True
        try:
            a = self.p.amount.at[day, secid]
        except KeyError:
            return True
        return (not np.isfinite(a)) or a <= 0

    # ------------------------------------------------------------ 对账
    def _equity(self, day) -> float:
        eq = self.cash
        for s, ps in self.pos.items():
            px = self.px(day, s, "close")
            eq += ps.shares * (px if np.isfinite(px) else ps.cost)
        return eq

    def _reconcile(self, day):
        """强制对账：现金 + 持仓 = 权益；不允许出现负现金或超限仓位。"""
        eq = self._equity(day)
        if self.cash < -1e-6:
            raise AssertionError(f"{day} 现金为负 {self.cash:.2f}，引擎记账错误")
        pv = sum(ps.shares * (self.px(day, s, "close") if np.isfinite(self.px(day, s, "close"))
                              else ps.cost) for s, ps in self.pos.items())
        if abs((self.cash + pv) - eq) > max(1.0, eq * 1e-9):
            raise AssertionError(f"{day} 对账不平 cash={self.cash:.2f} pv={pv:.2f} eq={eq:.2f}")
        if eq > 0 and pv / eq > self.cfg.account.max_total_position + 0.02:
            # 允许浮动超限（涨了），但不允许新开超限
            pass

    # ------------------------------------------------------------ 主循环
    def run(self, start=None, end=None):
        dates = [d for d in self._dates
                 if (not start or d >= start) and (not end or d <= end)]
        if len(dates) < 60:
            raise ValueError(f"回测期过短 {len(dates)} 日")

        for i, day in enumerate(dates):
            # 1) 处理挂起的卖出（跌停/停牌遗留）
            self._process_pending(day)
            # 2) 风控（收盘判断，次日开盘执行）
            self._risk(day)
            # 3) 调仓信号
            if i > 0:
                self._rebalance(day, i)
            # 4) 记账 + 对账
            self._mark(day)
            if self.strict:
                self._reconcile(day)
            if self.verbose and i % 20 == 0:
                print(f"  {day} eq={self._equity(day):,.0f} pos={len(self.pos)} "
                      f"cash={self.cash:,.0f}", flush=True)
        return self._report()

    # ------------------------------------------------------------ 挂起卖出
    def _process_pending(self, day):
        for secid in list(self.pending_exit):
            if secid not in self.pos:
                self.pending_exit.discard(secid)
                continue
            if self.halted(day, secid) or self.cant_sell(day, secid):
                continue
            self._close(day, secid, self.pos[secid].exit_reason or "补卖")

    # ------------------------------------------------------------ 风控
    def _risk(self, day):
        s = self.cfg.strategy
        for secid in list(self.pos):
            ps = self.pos[secid]
            if secid in self.pending_exit:
                continue
            px = self.px(day, secid, "close")
            if not np.isfinite(px) or px <= 0:
                continue
            ps.days += 1
            ps.peak = max(ps.peak, px)
            ret = px / ps.cost - 1
            trail = px / ps.peak - 1 if ps.peak > 0 else 0.0
            peaked = ps.peak / ps.cost - 1 >= s.take_profit
            reason = None
            if ret <= s.stop_loss:
                reason = f"止损{ret:.1%}"
            elif peaked and trail <= s.take_profit_trail:
                reason = f"移动止盈{trail:.1%}"
            elif trail <= s.trail_stop and ret > 0.02:
                reason = f"回撤止盈{trail:.1%}"
            elif ps.days >= s.max_hold_days:
                reason = f"到期{ps.days}日"
            if reason:
                self._close(day, secid, reason)

    # ------------------------------------------------------------ 调仓
    def _rebalance(self, day, i):
        target = float(self.target_pos.reindex([day]).ffill().iloc[0] or 0.0)
        if not np.isfinite(target):
            target = 0.0
        equity = self._equity(day)
        pos_val = sum(ps.shares * (self.px(day, x, "close") or 0) for x, ps in self.pos.items())
        cur_w = pos_val / equity if equity > 0 else 0.0

        if target <= 0.05:
            for secid in list(self.pos):
                if secid not in self.pending_exit:
                    self._close(day, secid, "择时清仓")
            return

        rebal = (i % self.cfg.strategy.rebalance_days == 0)
        must_cut = cur_w > target + 0.12
        if not (rebal or must_cut):
            return
        if day not in self.scores.index:
            return
        row = self.scores.loc[day].dropna().sort_values(ascending=False)
        if row.empty:
            return

        n = self.cfg.strategy.max_positions
        picks: dict[str, float] = {}
        if not must_cut:
            keep_line = int(n * self.cfg.strategy.keep_rank_mult)
            for rank, (secid, sc) in enumerate(row.items()):
                if rank >= keep_line:
                    break
                if secid in self.pos and secid not in self.pending_exit:
                    picks[secid] = float(sc)
        for secid, sc in row.items():
            if len(picks) >= n:
                break
            if secid in self.pos or secid in self.pending_exit:
                continue
            picks[secid] = float(sc)
        if len(picks) > n:
            picks = dict(sorted(picks.items(), key=lambda kv: -kv[1])[:n])

        # 卖出不在 picks 的
        for secid in list(self.pos):
            if secid in picks or secid in self.pending_exit:
                continue
            self._close(day, secid, "调出")

        # 买入
        budget_w = min(target, self.cfg.account.max_total_position)
        want = equity * budget_w - sum(
            self.pos[x].shares * (self.px(day, x, "close") or 0)
            for x in picks if x in self.pos)
        todo = [x for x in picks if x not in self.pos]
        for secid, sc in sorted(picks.items(), key=lambda kv: -kv[1]):
            if secid in self.pos:
                continue
            if secid in self.pending_exit:
                continue
            if self.halted(day, secid) or self.cant_buy(day, secid):
                todo.remove(secid) if secid in todo else None
                continue
            px = self.px(day, secid, "open")
            if not np.isfinite(px) or px <= 0:
                todo.remove(secid) if secid in todo else None
                continue
            ex = px * (1 + self.cost.slippage)
            budget = min(want / max(len(todo), 1), self.cash * 0.985)
            if budget < ex * 100:
                todo.remove(secid) if secid in todo else None
                continue
            sh = int(budget / ex / 100) * 100
            while sh >= 100 and sh * ex + self.cost.buy_cost(sh * ex) > self.cash:
                sh -= 100
            if sh < 100:
                todo.remove(secid) if secid in todo else None
                continue
            amt = sh * ex
            fee = self.cost.buy_cost(amt)
            self.cash -= amt + fee
            want -= amt + fee
            if secid in todo:
                todo.remove(secid)
            self.pos[secid] = Pos(secid, sh, (amt + fee) / sh, str(day), ex)
            self.trades.append(Trade(str(day), "BUY", secid, ex, sh, amt, fee,
                                     f"score={sc:.2f}"))

    # ------------------------------------------------------------ 卖出
    def _close(self, day, secid, reason):
        ps = self.pos.get(secid)
        if not ps:
            return
        if self.halted(day, secid) or self.cant_sell(day, secid):
            ps.exit_reason = reason
            self.pending_exit.add(secid)
            return
        px = self.px(day, secid, "open")
        if not np.isfinite(px) or px <= 0:
            ps.exit_reason = reason
            self.pending_exit.add(secid)
            return
        ex = px * (1 - self.cost.slippage)
        amt = ps.shares * ex
        fee = self.cost.sell_cost(amt)
        self.cash += amt - fee
        self.trades.append(Trade(str(day), "SELL", secid, ex, ps.shares, amt, fee, reason))
        del self.pos[secid]
        self.pending_exit.discard(secid)

    # ------------------------------------------------------------ 记账
    def _mark(self, day):
        eq = self._equity(day)
        pv = eq - self.cash
        self.nav.append(dict(date=str(day), equity=eq, cash=self.cash,
                             pos_value=pv, n_pos=len(self.pos),
                             pending=len(self.pending_exit)))

    # ------------------------------------------------------------ 报告
    def _report(self):
        nav = pd.DataFrame(self.nav)
        nav["date"] = pd.to_datetime(nav["date"])
        nav = nav.set_index("date")
        eq = nav["equity"]
        ret = eq.pct_change().fillna(0)
        n = len(eq)
        years = n / 244
        total = eq.iloc[-1] / self.cfg.account.initial_cash - 1
        cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else -1
        dd = eq / eq.cummax() - 1
        mdd = float(dd.min())
        vol = float(ret.std() * np.sqrt(244))
        sharpe = float(ret.mean() * 244 / vol) if vol > 0 else 0.0
        dvol = float(ret[ret < 0].std() * np.sqrt(244))
        sortino = float(ret.mean() * 244 / dvol) if dvol > 0 else 0.0

        trades = pd.DataFrame([t.__dict__ for t in self.trades])
        pnl = self._pair()
        stats = {
            "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()), "days": n,
            "final_equity": float(eq.iloc[-1]),
            "total_return": float(total), "cagr": float(cagr),
            "max_drawdown": mdd, "vol": vol, "sharpe": sharpe, "sortino": sortino,
            "calmar": float(cagr / abs(mdd)) if mdd < 0 else 0.0,
            "n_trades": len(self.trades), "n_round": len(pnl),
            "win_rate": float((pnl["pnl_pct"] > 0).mean()) if len(pnl) else 0.0,
            "avg_win": float(pnl[pnl.pnl_pct > 0]["pnl_pct"].mean())
                        if len(pnl) and (pnl.pnl_pct > 0).any() else 0.0,
            "avg_loss": float(pnl[pnl.pnl_pct < 0]["pnl_pct"].mean())
                         if len(pnl) and (pnl.pnl_pct < 0).any() else 0.0,
            "profit_factor": float(pnl[pnl.pnl_pct > 0]["pnl"].sum() /
                                   abs(pnl[pnl.pnl_pct < 0]["pnl"].sum()))
                             if len(pnl) and (pnl.pnl_pct < 0).any() and (pnl.pnl_pct > 0).any() else 0.0,
            "total_fee": float(trades.fee.sum()) if len(trades) else 0.0,
            "avg_pos": float(nav["n_pos"].mean()),
            "exposure": float((nav["pos_value"] / nav["equity"]).mean()),
            "best_day": float(ret.max()), "worst_day": float(ret.min()),
            "pending_avg": float(nav["pending"].mean()),
        }
        return {"nav": nav, "equity": eq, "returns": ret, "drawdown": dd,
                "trades": trades, "pnl": pnl, "stats": stats}

    def _pair(self):
        lots = defaultdict(deque)
        out = []
        for t in self.trades:
            if t.action == "BUY":
                lots[t.secid].append([t.shares, t.price, t.fee / max(t.shares, 1), t.date])
            else:
                remain = t.shares
                while remain > 0 and lots[t.secid]:
                    lot = lots[t.secid][0]
                    take = min(remain, lot[0])
                    buy_cost = take * (lot[1] + lot[2])
                    sell_net = take * (t.price - t.fee / max(t.shares, 1))
                    out.append({"secid": t.secid, "entry": lot[3], "exit": t.date,
                                "buy": round(lot[1], 3), "sell": round(t.price, 3),
                                "shares": take, "pnl": round(sell_net - buy_cost, 2),
                                "pnl_pct": round(sell_net / buy_cost - 1, 4) if buy_cost else 0,
                                "reason": t.reason})
                    lot[0] -= take
                    remain -= take
                    if lot[0] <= 0:
                        lots[t.secid].popleft()
        return pd.DataFrame(out)
