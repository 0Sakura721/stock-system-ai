# -*- coding: utf-8 -*-
"""Web 面板：本地 Flask 风格极简 HTTP 服务（只用标准库，无外部依赖）。

启动:
  python -m quant.web            # http://127.0.0.1:8770
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import pandas as pd

from . import data as D
from .account import Account
from .config import CFG, OUT_DIR
from . import engine as E
from .risk import RiskEngine

_state = {"plan": None, "state": None, "built_at": None}
_lock = threading.Lock()


def build(force=False):
    with _lock:
        if _state["plan"] is not None and not force:
            return _state["plan"]
        conn = D.connect()
        stt = E.build_state(conn, cfg=CFG)
        acc = Account(conn, initial_cash=CFG.account.initial_cash)
        plan = E.make_plan(stt, acc, risk=RiskEngine(CFG), verbose=False)
        _state["plan"] = plan
        _state["state"] = stt
        return plan


PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>AI 炒股系统</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,"Microsoft YaHei",sans-serif;background:#0f1117;color:#e6e8ee;margin:0;padding:24px}}
h1{{font-size:22px;margin:0 0 4px}}
.sub{{color:#8b93a7;font-size:13px;margin-bottom:20px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:24px}}
.card{{background:#181b24;border:1px solid #232734;border-radius:10px;padding:14px}}
.card .k{{color:#8b93a7;font-size:12px;margin-bottom:6px}}
.card .v{{font-size:20px;font-weight:600}}
.up{{color:#2ecc71}}.down{{color:#e74c3c}}
table{{width:100%;border-collapse:collapse;background:#181b24;border-radius:10px;overflow:hidden;margin-bottom:24px}}
th{{background:#1f2330;color:#8b93a7;font-size:12px;padding:10px;text-align:left}}
td{{padding:10px;border-top:1px solid #232734;font-size:13px}}
h2{{font-size:16px;margin:22px 0 10px;color:#aeb6c8}}
.note{{background:#181b24;border-left:3px solid #4a6cf7;padding:12px 14px;border-radius:6px;color:#b9c0d0;font-size:13px;line-height:1.7}}
.btn{{display:inline-block;padding:7px 14px;background:#4a6cf7;color:#fff;border-radius:6px;text-decoration:none;font-size:13px;margin-right:8px}}
</style></head><body>
<h1>AI 全权炒股系统</h1>
<div class="sub">信号日 {signal_date} · 生成 {generated_at}</div>
<a class="btn" href="/api/plan">JSON</a><a class="btn" href="/refresh">刷新数据</a>
<div class="cards">
  <div class="card"><div class="k">目标仓位</div><div class="v">{target_position}</div></div>
  <div class="card"><div class="k">当前仓位</div><div class="v">{current_position}</div></div>
  <div class="card"><div class="k">账户权益</div><div class="v">{equity}</div></div>
  <div class="card"><div class="k">可用现金</div><div class="v">{cash}</div></div>
  <div class="card"><div class="k">累计收益</div><div class="v {pnl_cls}">{total_return}</div></div>
  <div class="card"><div class="k">市场宽度</div><div class="v">{breadth}</div></div>
</div>
<h2>卖出（先执行）</h2>{sell_table}
<h2>买入（后执行）</h2>{buy_table}
<h2>当前持仓</h2>{hold_table}
<h2>执行纪律</h2><div class="note">{notes}</div>
</body></html>"""


def _tbl(headers, rows):
    if not rows:
        return '<table><tr><td style="color:#8b93a7">无</td></tr></table>'
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{body}</table>"


def render_html(plan: dict) -> str:
    sells = [[s["code"], s["name"], s["shares"], s["price"],
              f'<span class="down">{s["pnl"]:+.1%}</span>', s["reason"],
              "⚠️" if s.get("blocked") else "可执行"] for s in plan["sells"]]
    buys = [[b["code"], b["name"], b["shares"], b["price"], f"{b['amount']:,.0f}",
             f"{b['weight']:.1%}", b["score"], b.get("stop", "-")] for b in plan["buys"]]
    holds = [[p["code"], p["name"], p["shares"], f"{p['cost']:.2f}", f"{p['price']:.2f}",
              f'<span class="{"up" if p["pnl"]>=0 else "down"}">{p["pnl"]:+.1%}</span>',
              f"{p['weight']:.1%}", p["hold_days"]] for p in plan["holdings"]]
    return PAGE.format(
        signal_date=plan["signal_date"], generated_at=plan["generated_at"],
        target_position=f"{plan['target_position']:.0%}",
        current_position=f"{plan['current_position']:.0%}",
        equity=f"{plan['equity']:,.0f}", cash=f"{plan['cash']:,.0f}",
        total_return=f"{plan['total_return']:+.2%}",
        pnl_cls="up" if plan["total_return"] >= 0 else "down",
        breadth=f"{plan['breadth']:.0%}",
        sell_table=_tbl(["代码", "名称", "股数", "参考价", "盈亏", "原因", "状态"], sells),
        buy_table=_tbl(["代码", "名称", "股数", "限价", "金额", "权重", "综合分", "止损价"], buys),
        hold_table=_tbl(["代码", "名称", "股数", "成本", "现价", "盈亏", "权重", "持有日"], holds),
        notes="<br>".join(plan["notes"]),
    )


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body.encode("utf-8")))
                         if isinstance(body, str) else str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode("utf-8") if isinstance(body, str) else body)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/":
                self._send(200, render_html(build()))
            elif u.path == "/api/plan":
                body = json.dumps(build(), ensure_ascii=False, indent=2, default=float)
                self._send(200, body, "application/json; charset=utf-8")
            elif u.path == "/refresh":
                from .fetch import update
                try:
                    update(full=False, workers=24)
                except Exception as e:              # noqa: BLE001
                    print("refresh fail", e, flush=True)
                self._send(200, render_html(build(force=True)))
            elif u.path == "/health":
                self._send(200, "ok", "text/plain")
            else:
                self._send(404, "not found", "text/plain")
        except Exception as e:                     # noqa: BLE001
            self._send(500, f"error: {e}", "text/plain; charset=utf-8")

    def log_message(self, *a):
        pass


def serve(port: int = 8770, host: str = "127.0.0.1"):
    print(f"面板 http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
    serve(port)
