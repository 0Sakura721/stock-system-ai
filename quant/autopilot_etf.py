# -*- coding: utf-8 -*-
"""ETF 自动操盘：小资金（10 万级）的完整闭环。

为什么小资金必须用这个而不是个股组合：
  实测 10 万本金做全池等权只能持 4.8 只（一手门槛 + 最低佣金），
  6 年回放 -31%，而同样择时用 ETF 是 +115%。
  差别不在择时，在于「能不能把钱摊开」。

流程和个股 autopilot 完全同构：
  1 补行情 → 2 读账户 → 3 算择时 → 4 算目标份额
  → 5 模拟成交 → 6 风控 → 7 日报

和个股版的唯一区别：标的是单一 ETF，份额精确到 1 份（ETF 无一手限制）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd

from . import data as D
from . import etf as ETF
from . import strategy as st
from .account import Account
from .config import CFG, OUT_DIR
from . import findings as FN

LOG_DIR = os.path.join(OUT_DIR, "autopilot")
os.makedirs(LOG_DIR, exist_ok=True)

# ETF 账户用独立的现金键，避免和个股账本混
ETF_CASH_KEY = "etf_cash"
ETF_CODE_KEY = "etf_code"
ETF_SHARES_KEY = "etf_shares"
ETF_COST_KEY = "etf_cost"


@dataclass
class EtfPosition:
    code: str
    name: str
    shares: int
    cost: float
    price: float
    entry_date: str = ""


class EtfAutopilot:
    """单 ETF 择时自动操盘。"""

    def __init__(self, code: str = "159949", cash: float = 1e5, cfg=None,
                 verbose: bool = True):
        self.cfg = FN.apply_to(cfg or CFG)
        self.code = code
        self.cash0 = cash
        self.verbose = verbose
        self.conn = D.connect()
        self._init_account()

    def _init_account(self):
        if D.get_meta(self.conn, ETF_CASH_KEY) is None:
            D.set_meta(self.conn, ETF_CASH_KEY, f"{self.cash0:.2f}")
            D.set_meta(self.conn, ETF_CODE_KEY, self.code)
            D.set_meta(self.conn, ETF_SHARES_KEY, "0")
            D.set_meta(self.conn, ETF_COST_KEY, "0")
            self.conn.commit()

    def log(self, m: str):
        if self.verbose:
            print(m, flush=True)

    # ------------------------------------------------------------ 状态
    def cash(self) -> float:
        return float(D.get_meta(self.conn, ETF_CASH_KEY, self.cash0))

    def shares(self) -> int:
        return int(float(D.get_meta(self.conn, ETF_SHARES_KEY, 0)))

    def cost(self) -> float:
        return float(D.get_meta(self.conn, ETF_COST_KEY, 0))

    def _set(self, cash=None, shares=None, cost=None):
        if cash is not None:
            D.set_meta(self.conn, ETF_CASH_KEY, f"{cash:.2f}")
        if shares is not None:
            D.set_meta(self.conn, ETF_SHARES_KEY, str(int(shares)))
        if cost is not None:
            D.set_meta(self.conn, ETF_COST_KEY, f"{cost:.4f}")
        self.conn.commit()

    def record(self, action, price, shares, fee, note=""):
        self.conn.execute(
            "INSERT INTO account(ts,action,code,name,price,shares,amount,fee,note)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action, self.code,
             ETF.ETF_POOL.get(self.code, ("ETF", ""))[0], price, shares,
             price * shares, fee, note))
        self.conn.commit()

    # ------------------------------------------------------------ 1 行情
    def step_fetch(self) -> pd.DataFrame:
        self.log("[1/7] 更新 ETF 与指数行情...")
        df = ETF.fetch_etf_bars(self.code, "2016-01-01")
        if len(df) == 0:
            raise RuntimeError(f"ETF {self.code} 无数据")
        idx_code = ETF.ETF_POOL.get(self.code, (None, self.cfg.strategy.timing_index))[1]
        idx = D.fetch_index_close(idx_code, "2015-01-01")
        self.log(f"  ETF {len(df)} 根  最新 {df['date'].iloc[-1]}  "
                 f"收 {df['close'].iloc[-1]:.3f}")
        return df, idx

    # ------------------------------------------------------------ 3 信号
    def step_signal(self, df: pd.DataFrame, idx: pd.Series) -> dict:
        self.log("[3/7] 计算择时...")
        dates = df["date"].tolist()
        p_like = type("P", (), {"dates": pd.Index(dates)})()
        idx_s = pd.Series(idx.values, index=pd.Index(idx.index))
        pos = st.timing(p_like, idx_s, self.cfg.strategy)

        row_idx = idx_s.reindex(dates).ffill()
        target = float(pos.iloc[-1])
        sig_date = dates[-1]

        s = self.cfg.strategy
        ma_f = row_idx.rolling(s.timing_fast, min_periods=max(2, s.timing_fast // 2)).mean().iloc[-1]
        ma_s = row_idx.rolling(s.timing_slow, min_periods=max(2, s.timing_slow // 2)).mean().iloc[-1]
        ma_b = row_idx.rolling(s.timing_bear_ma, min_periods=max(2, s.timing_bear_ma // 2)).mean().iloc[-1]
        lvl = 0
        cur = float(row_idx.iloc[-1])
        if cur > ma_s and ma_f > ma_s:
            lvl = 3
        elif cur > ma_s:
            lvl = 2
        elif cur > ma_b:
            lvl = 1

        px = float(df["close"].iloc[-1])
        self.log(f"  信号日 {sig_date}  指数 {cur:.0f}  MA{s.timing_fast}={ma_f:.0f}  "
                 f"MA{s.timing_slow}={ma_s:.0f}  MA{s.timing_bear_ma}={ma_b:.0f}")
        self.log(f"  档位 {lvl}/3  → 目标仓位 {target:.0%}  ETF 价 {px:.3f}")
        return dict(signal_date=sig_date, target=target, level=lvl,
                    index_value=cur, ma_fast=float(ma_f), ma_slow=float(ma_s),
                    ma_bear=float(ma_b), etf_price=px)

    # ------------------------------------------------------------ 4/5 交易
    def step_trade(self, df: pd.DataFrame, sig: dict) -> list[dict]:
        self.log("[4/7][5/7] 计算目标并成交...")
        px = sig["etf_price"]
        shares_now = self.shares()
        cash = self.cash()
        equity = cash + shares_now * px
        target_val = equity * min(sig["target"], self.cfg.account.max_total_position)
        target_shares = int(target_val / px) if px > 0 else 0

        delta = target_shares - shares_now
        if abs(delta) * px < equity * 0.03:
            self.log(f"  偏离 {(delta*px/equity if equity else 0):.1%} < 3%，不调仓")
            return []

        filled = []
        if delta > 0:
            # 买
            money = min(delta * px, cash * 0.995)
            sh = int(money / (px * (1 + self.cfg.cost.slippage)))
            if sh <= 0:
                return []
            fpx = px * (1 + self.cfg.cost.slippage)
            amt = sh * fpx
            fee = max(amt * 0.00025, 5.0)
            if amt + fee > cash:
                sh = int((cash - 5) / fpx)
                if sh <= 0:
                    return []
                amt = sh * fpx
                fee = max(amt * 0.00025, 5.0)
            new_cost = (shares_now * self.cost() + amt + fee) / (shares_now + sh) \
                if shares_now + sh > 0 else (amt + fee) / sh
            self._set(cash=cash - amt - fee, shares=shares_now + sh, cost=new_cost)
            self.record("BUY", fpx, sh, fee, f"择时建仓 {sig['target']:.0%}")
            filled.append(dict(side="BUY", shares=sh, price=round(fpx, 4),
                               amount=round(amt, 2), fee=round(fee, 2)))
            self.log(f"  [买入] {sh} 份 @ {fpx:.3f}  费 {fee:.2f}")
        else:
            # 卖
            sh = min(-delta, shares_now)
            if sh <= 0:
                return []
            fpx = px * (1 - self.cfg.cost.slippage)
            amt = sh * fpx
            fee = max(amt * 0.00025, 5.0) + amt * 0.00001
            self._set(cash=cash + amt - fee, shares=shares_now - sh)
            self.record("SELL", fpx, sh, fee, f"择时减仓 {sig['target']:.0%}")
            filled.append(dict(side="SELL", shares=sh, price=round(fpx, 4),
                               amount=round(amt, 2), fee=round(fee, 2)))
            self.log(f"  [卖出] {sh} 份 @ {fpx:.3f}  费 {fee:.2f}")

        return filled

    # ------------------------------------------------------------ 6 风控
    def step_risk(self, sig: dict) -> list[dict]:
        self.log("[6/7] 风控巡检...")
        px = sig["etf_price"]
        alerts = []
        cost = self.cost()
        if self.shares() > 0 and cost > 0:
            ret = px / cost - 1
            if ret <= self.cfg.strategy.stop_loss:
                alerts.append(dict(level="HIGH", msg=f"ETF 浮亏 {ret:.1%} 触及止损"))
            elif sig["level"] <= 1:
                alerts.append(dict(level="MED", msg=f"择时转弱（档位 {sig['level']}/3），减仓"))
        self.log(f"  {len(alerts)} 个告警")
        return alerts

    # ------------------------------------------------------------ 7 日报
    def step_report(self, sig, filled, alerts) -> str:
        self.log("[7/7] 生成日报...")
        px = sig["etf_price"]
        cash = self.cash()
        sh = self.shares()
        eq = cash + sh * px
        cost = self.cost()
        name = ETF.ETF_POOL.get(self.code, ("ETF", ""))[0]

        L = []
        L.append(f"# ETF 自动操盘日报 · {datetime.now():%Y-%m-%d %H:%M}")
        L.append("")
        L.append(f"**标的** {self.code} {name} ｜ **信号日** {sig['signal_date']}")
        L.append("")
        L.append("## 一、择时决策")
        L.append("")
        L.append("| 指标 | 数值 |")
        L.append("|---|---|")
        L.append(f"| 指数收盘 | {sig['index_value']:.0f} |")
        L.append(f"| MA{self.cfg.strategy.timing_fast} | {sig['ma_fast']:.0f} |")
        L.append(f"| MA{self.cfg.strategy.timing_slow} | {sig['ma_slow']:.0f} |")
        L.append(f"| MA{self.cfg.strategy.timing_bear_ma} | {sig['ma_bear']:.0f} |")
        L.append(f"| 档位 | {sig['level']}/3 |")
        L.append(f"| **目标仓位** | **{sig['target']:.0%}** |")
        L.append(f"| ETF 现价 | {px:.3f} |")
        L.append("")
        L.append(self._explain(sig))
        L.append("")
        L.append("## 二、账户")
        L.append("")
        L.append(f"- 现金 {cash:,.2f}　持仓 {sh} 份　市值 {sh*px:,.2f}")
        L.append(f"- 总权益 {eq:,.2f}　累计 {eq/self.cash0-1:+.2%}")
        L.append(f"- 当前仓位 {(sh*px/eq if eq else 0):.0%}")
        if sh > 0 and cost > 0:
            L.append(f"- 持仓成本 {cost:.3f}　浮动 {px/cost-1:+.2%}")
        L.append("")
        L.append("## 三、今日成交")
        L.append("")
        if filled:
            L.append("| 方向 | 份数 | 价格 | 金额 | 费用 |")
            L.append("|---|---|---|---|---|")
            for f in filled:
                L.append(f"| {f['side']} | {f['shares']} | {f['price']} | "
                         f"{f['amount']:,.0f} | {f['fee']:.2f} |")
        else:
            L.append("无成交（偏离小于 3%，不调仓）。")
        L.append("")
        L.append("## 四、风控")
        L.append("")
        if alerts:
            for a in alerts:
                L.append(f"- **{a['level']}** {a['msg']}")
        else:
            L.append("无告警。")
        L.append("")
        L.append("## 五、下一步")
        L.append("")
        if sig["target"] <= 0.02:
            L.append("空仓。每日只检查指数是否重回 MA45 上方，不主动抄底。")
        elif sig["target"] >= 0.95:
            L.append("满仓。每日检查指数是否跌破 MA45。")
        else:
            L.append(f"维持 {sig['target']:.0%} 仓位。")
        L.append("")
        L.append("下一次运行自动比对目标份额与当前份额，偏离超 3% 才动手。")

        path = os.path.join(LOG_DIR, f"etf_{datetime.now():%Y%m%d}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(L))
        with open(path.replace(".md", ".json"), "w", encoding="utf-8") as f:
            json.dump(dict(code=self.code, signal=sig, filled=filled, alerts=alerts,
                           cash=cash, shares=sh, equity=eq, price=px),
                      f, ensure_ascii=False, indent=2, default=float)
        self.log(f"  日报 → {path}")
        return path

    def _explain(self, sig) -> str:
        lvl = sig["level"]
        if lvl == 3:
            return "**判读**：创业板指站上 MA45 且 MA15 上穿，多头排列，满仓持有 ETF。"
        if lvl == 2:
            return "**判读**：指数在 MA45 上方但短期走弱，健康回调，保持六成仓。"
        if lvl == 1:
            return "**判读**：跌破 MA45 但守住 MA150，中期转弱，降到三成仓。"
        return ("**判读**：跌破 MA150，深度弱势。规则要求空仓 —— "
                "趋势向下时持有 ETF 的期望收益为负，等重回 MA45 再入场。")

    # ------------------------------------------------------------ 主流程
    def run_once(self) -> dict:
        t0 = datetime.now()
        self.log("=" * 62)
        self.log(f"  ETF 自动操盘  {t0:%Y-%m-%d %H:%M:%S}")
        self.log("=" * 62)
        df, idx = self.step_fetch()
        sig = self.step_signal(df, idx)
        filled = self.step_trade(df, sig)
        alerts = self.step_risk(sig)
        report = self.step_report(sig, filled, alerts)
        self.log("=" * 62)
        px = sig["etf_price"]
        eq = self.cash() + self.shares() * px
        self.log(f"  完成：{len(filled)} 笔成交  权益 {eq:,.2f}  "
                 f"仓位 {(self.shares()*px/eq if eq else 0):.0%}")
        self.log("=" * 62)
        return dict(mode="etf", code=self.code, signal=sig["signal_date"],
                    target=sig["target"], filled=len(filled),
                    equity=eq, report=report, alerts=len(alerts))

    def replay(self, start: str = "2020-01-01") -> dict:
        """历史回放验证：走同样的现金/份额账本。"""
        df = ETF.fetch_etf_bars(self.code, "2016-01-01")
        idx_code = ETF.ETF_POOL.get(self.code, (None, self.cfg.strategy.timing_index))[1]
        idx = D.fetch_index_close(idx_code, "2015-01-01")
        dates = [d for d in df["date"] if d >= start]

        p_like = type("P", (), {"dates": pd.Index(dates)})()
        idx_s = pd.Series(idx.values, index=pd.Index(idx.index))
        pos = st.timing(p_like, idx_s, self.cfg.strategy)
        pos = pos.reindex(dates).ffill().fillna(0)

        cash = self.cash0
        sh = 0
        nav = []
        sub = df.set_index("date").reindex(dates)
        closes = sub["close"].to_numpy()
        targets = pos.shift(1).fillna(0).to_numpy()

        for i, d in enumerate(dates):
            px = float(closes[i])
            tv = (cash + sh * px) * min(targets[i], 0.95)
            ts = int(tv / px) if px > 0 else 0
            delta = ts - sh
            equity = cash + sh * px
            if abs(delta) * px >= equity * 0.03:
                if delta > 0:
                    fpx = px * (1 + self.cfg.cost.slippage)
                    buy_sh = int(min(delta * px, cash * 0.995) / fpx)
                    if buy_sh > 0:
                        amt = buy_sh * fpx
                        fee = max(amt * 0.00025, 5.0)
                        if amt + fee <= cash:
                            cash -= amt + fee
                            sh += buy_sh
                else:
                    sell_sh = min(-delta, sh)
                    if sell_sh > 0:
                        fpx = px * (1 - self.cfg.cost.slippage)
                        amt = sell_sh * fpx
                        fee = max(amt * 0.00025, 5.0) + amt * 0.00001
                        cash += amt - fee
                        sh -= sell_sh
            nav.append(dict(date=d, equity=cash + sh * px, cash=cash,
                            pos_value=sh * px, shares=sh))

        nd = pd.DataFrame(nav).set_index(pd.to_datetime([n["date"] for n in nav]))
        eq = nd["equity"]
        ret = eq.pct_change().fillna(0)
        total = eq.iloc[-1] / self.cash0 - 1
        years = len(eq) / 244
        cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1
        dd = (eq / eq.cummax() - 1)
        vol = ret.std() * np.sqrt(244)
        sharpe = ret.mean() * 244 / vol if vol > 0 else 0

        print("\n" + "=" * 60)
        print(f"  ETF Autopilot 回放  {self.code}")
        print("=" * 60)
        print(f"  区间      {dates[0]} ~ {dates[-1]}  ({len(eq)}日)")
        print(f"  总收益    {total:+.2%}")
        print(f"  年化      {cagr:+.2%}")
        print(f"  最大回撤  {dd.min():.2%}")
        print(f"  Sharpe    {sharpe:.2f}")
        print(f"  平均仓位  {(nd['pos_value']/nd['equity']).mean():.0%}")
        print("=" * 60)
        return dict(nav=nd, total=total, cagr=cagr, mdd=float(dd.min()), sharpe=float(sharpe))

    def close(self):
        self.conn.close()


def auto_mode(cash: float) -> str:
    """按本金决定用哪种模式。"""
    v = FN.capital_verdict(cash)
    return "etf" if v["mode"] == "etf_only" else "stock"
