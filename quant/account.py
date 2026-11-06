# -*- coding: utf-8 -*-
"""账户账本：持仓、成交、净值、下单队列，全部 SQLite 持久化。"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import data as D
from .config import DB_PATH, CFG


class Account:
    def __init__(self, conn=None, initial_cash: float = None):
        self.conn = conn or D.connect()
        self.initial_cash = initial_cash or CFG.account.initial_cash
        self._ensure_init()

    def _ensure_init(self):
        n = self.conn.execute("SELECT COUNT(*) FROM meta WHERE key='initial_cash'").fetchone()[0]
        if not n:
            D.set_meta(self.conn, "initial_cash", self.initial_cash)
            self.conn.commit()
        self.equity0 = float(D.get_meta(self.conn, "initial_cash", self.initial_cash))

    # ---------------------------------------------------------- 持仓
    @property
    def positions(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM positions", self.conn)

    @property
    def trades(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM account ORDER BY id", self.conn)

    def cash(self) -> float:
        v = D.get_meta(self.conn, "cash")
        return float(v) if v is not None else self.equity0

    def set_cash(self, v: float):
        D.set_meta(self.conn, "cash", f"{v:.2f}")
        self.conn.commit()

    # ---------------------------------------------------------- 成交
    def apply_fill(self, action: str, code: str, name: str, price: float,
                   shares: int, fee: float = None, note: str = "",
                   date: str = None):
        fee = self._fee(action, price * shares) if fee is None else fee
        amount = price * shares
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date = date or datetime.now().strftime("%Y-%m-%d")

        self.conn.execute(
            "INSERT INTO account(ts,action,code,name,price,shares,amount,fee,note)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (ts, action, code, name, price, shares, amount, fee, note))

        row = self.conn.execute(
            "SELECT shares,cost,peak,entry_date,hold_days FROM positions WHERE code=?",
            (code,)).fetchone()

        if action == "BUY":
            cash = self.cash() - amount - fee
            self.set_cash(cash)
            if row:
                tot = row[0] + shares
                new_cost = (row[0] * row[1] + amount + fee) / tot
                self.conn.execute(
                    "UPDATE positions SET shares=?, cost=?, last_price=?,"
                    " peak=MAX(peak,?) WHERE code=?",
                    (tot, new_cost, price, price, code))
            else:
                self.conn.execute(
                    "INSERT INTO positions(code,name,shares,cost,last_price,peak,"
                    "entry_date,hold_days,stop,note) VALUES(?,?,?,?,?,?,?,0,NULL,?)",
                    (code, name, shares, (amount + fee) / shares, price, price, date, note))
        else:
            cash = self.cash() + amount - fee
            self.set_cash(cash)
            if row:
                left = row[0] - shares
                if left <= 0:
                    self.conn.execute("DELETE FROM positions WHERE code=?", (code,))
                else:
                    self.conn.execute(
                        "UPDATE positions SET shares=?, last_price=? WHERE code=?",
                        (left, price, code))
        self.conn.commit()

    def _fee(self, action: str, amount: float) -> float:
        return CFG.cost.buy_cost(amount) if action == "BUY" else CFG.cost.sell_cost(amount)

    def update_marks(self, prices: dict[str, float]):
        for code, px in prices.items():
            row = self.conn.execute("SELECT peak FROM positions WHERE code=?", (code,)).fetchone()
            if row:
                peak = max(row[0] or 0, px)
                self.conn.execute(
                    "UPDATE positions SET last_price=?, peak=? WHERE code=?",
                    (px, peak, code))
        self.conn.commit()

    def tick_hold_days(self):
        self.conn.execute("UPDATE positions SET hold_days = hold_days + 1")
        self.conn.commit()

    # ---------------------------------------------------------- 快照
    def snapshot(self, prices: dict[str, float] | None = None) -> dict:
        pos = self.positions
        prices = prices or {}
        rows, total = [], 0.0
        for r in pos.itertuples(index=False):
            px = prices.get(r.code, r.last_price)
            val = px * r.shares
            total += val
            rows.append(dict(
                code=r.code, name=r.name, shares=int(r.shares), cost=float(r.cost),
                price=float(px), value=float(val),
                pnl=float(px / r.cost - 1) if r.cost else 0.0,
                pnl_amount=float(val - r.shares * r.cost),
                peak=float(r.peak or px), hold_days=int(r.hold_days or 0),
                weight=0.0, stop=r.stop))
        cash = self.cash()
        equity = cash + total
        for r in rows:
            r["weight"] = r["value"] / equity if equity else 0.0
        return dict(cash=cash, pos_value=total, equity=equity,
                    positions=rows, initial=self.equity0,
                    total_return=equity / self.equity0 - 1 if self.equity0 else 0)

    def record_nav(self, date: str, snap: dict):
        self.conn.execute(
            "INSERT OR REPLACE INTO nav(date,equity,cash,pos_value,n_pos) VALUES(?,?,?,?,?)",
            (date, snap["equity"], snap["cash"], snap["pos_value"],
             len(snap["positions"])))
        self.conn.commit()

    def nav_history(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM nav ORDER BY date", self.conn)

    # ---------------------------------------------------------- 下单队列
    def queue_order(self, plan_date: str, exec_date: str, side: str, code: str,
                    name: str, price: float, shares: int, reason: str, status="PENDING"):
        self.conn.execute(
            "INSERT INTO orders(plan_date,exec_date,side,code,name,plan_price,"
            "plan_shares,status,reason,created) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (plan_date, exec_date, side, code, name, price, shares, status,
             reason, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.conn.commit()

    def pending_orders(self, exec_date: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM orders WHERE status='PENDING'"
        if exec_date:
            q += f" AND exec_date='{exec_date}'"
        return pd.read_sql_query(q, self.conn)

    def mark_order(self, oid: int, status: str, filled_price=None, filled_shares=None):
        self.conn.execute(
            "UPDATE orders SET status=?, filled_price=?, filled_shares=? WHERE id=?",
            (status, filled_price, filled_shares, oid))
        self.conn.commit()

    def reset(self, cash: float = None):
        """清空账本，重新开始（模拟盘初始化）。"""
        for t in ("account", "positions", "nav", "orders"):
            self.conn.execute(f"DELETE FROM {t}")
        cash = cash or self.initial_cash
        D.set_meta(self.conn, "initial_cash", cash)
        D.set_meta(self.conn, "cash", f"{cash:.2f}")
        self.equity0 = cash
        self.conn.commit()

    def close(self):
        self.conn.close()
