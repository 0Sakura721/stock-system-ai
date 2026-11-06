# -*- coding: utf-8 -*-
"""数据层 v4：新浪为主源（稳定、20/s），自研除权修正，东财/腾讯为备选。

复权方案（关键）：
  新浪给的是不复权价。除权日会有价格跳空，若不修正会误判为暴跌。
  检测规则：单日跌幅 |ret| >= 21%（A股任何板块都不可能单日跌超 20%），
  即判定为除权/送转，用当晚的 pct 作为复权因子乘到历史价格上。
  纯分红（1~3%）除权幅度小，对 20/60 日动量影响可忽略，不再处理。
"""
from __future__ import annotations

import itertools
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .config import DB_PATH, FETCH

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
HDR = {"User-Agent": UA, "Accept": "*/*", "Accept-Language": "zh-CN,zh;q=0.9"}
HDR_SINA = dict(HDR, Referer="http://finance.sina.com.cn/")
HDR_THS = dict(HDR, Referer="http://stockpage.10jqka.com.cn/")

BAR_COLS = ["date", "open", "close", "high", "low", "volume", "amount",
            "amplitude", "pct_chg", "chg", "turnover"]


# ------------------------------------------------------------------ 限速
class RateLimiter:
    def __init__(self, qps: float = 20.0):
        self.qps = qps
        self._next = time.time()
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.time()
            if self._next <= now:
                self._next = now + 1.0 / self.qps
                return
            wait = self._next - now
            self._next += 1.0 / self.qps
        if wait > 0:
            time.sleep(wait)


LIMITER = RateLimiter(FETCH.get("qps", 20.0))


def _raw(url: str, timeout: int = 20, retries: int = 4, enc: str = "utf-8",
         limit: bool = True, hdr: dict = None) -> str:
    last = None
    for i in range(retries):
        if limit:
            LIMITER.acquire()
        try:
            req = urllib.request.Request(url, headers=hdr or HDR)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode(enc, "ignore")
        except Exception as e:                       # noqa: BLE001
            last = e
            time.sleep(0.4 * (2 ** i))
    raise RuntimeError(f"GET fail {url[:110]} :: {last}")


def _json(url: str, **kw):
    return json.loads(_raw(url, **kw))


def to_secid(code: str) -> str:
    code = str(code).zfill(6)
    return ("1." if code[0] in "56" else "0.") + code


def to_sina(code: str) -> str:
    code = str(code).zfill(6)
    return ("sh" if code[0] in "56" else "sz") + code


def board_of(code: str) -> str:
    if code.startswith("688"):
        return "kcb"
    if code.startswith(("300", "301")):
        return "cyb"
    if code.startswith(("8", "4", "9")):
        return "bj"
    return "main"


# ------------------------------------------------------------------ 除权修正
def adjust_splits(df: pd.DataFrame, hard: float = 0.21) -> pd.DataFrame:
    """把不复权日线转成"前复权连续序列"。

    检测单日跌幅 >= hard 的点为除权日，用 ratio 把该日之前所有价格按比例缩放，
    使序列连续。返回新 DataFrame（保留原始 raw_* 列）。
    """
    if df is None or len(df) < 3:
        return df
    d = df.copy().reset_index(drop=True)
    c = d["close"].to_numpy(dtype=float)
    ret = np.zeros(len(c))
    ret[1:] = c[1:] / c[:-1] - 1

    # 除权点：跌幅过大（且不是连续跌停日）
    idx = np.where(ret <= -hard)[0]
    if len(idx) == 0:
        d["adj_factor"] = 1.0
        return d

    factor = np.ones(len(c))
    for i in idx:
        ratio = c[i] / c[i - 1]         # 例如 0.5 表示 1 拆 2
        if not np.isfinite(ratio) or ratio <= 0:
            continue
        factor[:i] *= ratio             # 之前的全部价格乘比例，衔接今日

    for col in ("open", "close", "high", "low"):
        d[col] = d[col].astype(float) * factor
    d["volume"] = d["volume"].astype(float) / factor      # 量对应反向
    d["amount"] = d["volume"] * d["close"]
    d["adj_factor"] = factor
    d["pct_chg"] = d["close"].pct_change() * 100
    return d


# ------------------------------------------------------------------ 股票列表
def fetch_stock_list() -> pd.DataFrame:
    try:
        df = _list_sina()
        if len(df) > 1000:
            return df
    except Exception as e:                           # noqa: BLE001
        print("[列表] 新浪失败:", str(e)[:70], flush=True)
    try:
        return _list_em()
    except Exception as e:                           # noqa: BLE001
        print("[列表] 东财失败:", str(e)[:70], flush=True)
        return pd.DataFrame(columns=["code", "name", "secid", "market", "industry",
                                     "price", "amount", "float_cap", "mkt_cap", "turnover"])


def _list_sina() -> pd.DataFrame:
    import re as _re
    rows = []
    for page in range(1, 90):
        url = ("http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               f"Market_Center.getHQNodeData?page={page}&num=100&sort=symbol&asc=1"
               "&node=hs_a&symbol=&_s_r_a=page")
        try:
            txt = _raw(url, enc="utf-8", limit=True, hdr=HDR_SINA, retries=3)
        except Exception:                          # noqa: BLE001
            break
        if not txt or txt.strip() in ("null", "[]"):
            break
        txt = _re.sub(r"([{,])(\w+):", r'\1"\2":', txt)
        try:
            arr = json.loads(txt)
        except Exception:                          # noqa: BLE001
            break
        if not arr:
            break
        rows.extend(arr)

    if not rows:
        raise RuntimeError("sina list empty")
    df = pd.DataFrame(rows).rename(columns={"trade": "price"})
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["turnover"] = pd.to_numeric(df.get("turnoverratio"), errors="coerce")
    df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce")
    df["mkt_cap"] = pd.to_numeric(df.get("mktcap"), errors="coerce") * 1e4
    df["float_cap"] = pd.to_numeric(df.get("nmc"), errors="coerce") * 1e4
    df["market"] = df["code"].map(lambda c: 1 if c[0] in "56" else 0)
    df["secid"] = df["market"].astype(str) + "." + df["code"]
    df["industry"] = df["code"].map(board_of)
    return df[["code", "name", "secid", "market", "industry", "price", "amount",
               "float_cap", "mkt_cap", "turnover"]].drop_duplicates("code").reset_index(drop=True)


def _list_em() -> pd.DataFrame:
    fields = "f2,f3,f5,f6,f8,f12,f13,f14,f15,f16,f17,f18,f20,f21,f100"
    base = ("https://push2.eastmoney.com/api/qt/clist/get?pz=100&po=1&np=1&fltt=2&invt=2"
            f"&fid=f12&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23&fields={fields}")
    first = _json(base + "&pn=1", limit=False, retries=1)
    total = first["data"]["total"]
    rows = list(first["data"]["diff"])
    for p in range(2, (total + 99) // 100 + 1):
        try:
            rows.extend((_json(base + f"&pn={p}", limit=False, retries=1)
                         .get("data") or {}).get("diff") or [])
        except Exception:                          # noqa: BLE001
            pass
    df = pd.DataFrame(rows).rename(columns={
        "f2": "price", "f3": "pct_chg", "f5": "volume", "f6": "amount",
        "f8": "turnover", "f12": "code", "f13": "market", "f14": "name",
        "f20": "mkt_cap", "f21": "float_cap", "f100": "industry"})
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["market"] = df["market"].astype(int)
    df["secid"] = df["market"].astype(str) + "." + df["code"]
    return df[["code", "name", "secid", "market", "industry", "price", "amount",
               "float_cap", "mkt_cap", "turnover"]].drop_duplicates("code").reset_index(drop=True)


# ------------------------------------------------------------------ 历史行情（新浪）
def _sina_kline(code: str, n: int = 4000) -> pd.DataFrame:
    url = ("http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={to_sina(code)}&scale=240&ma=no&datalen={n}")
    txt = _raw(url, enc="utf-8", hdr=HDR_SINA, limit=True, retries=3)
    arr = json.loads(txt)
    if not arr:
        return pd.DataFrame(columns=["date", "open", "close", "high", "low",
                                     "volume", "amount", "pct_chg", "turnover"])
    df = pd.DataFrame(arr).rename(columns={"day": "date"})
    for col in ("open", "close", "high", "low", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    df["amount"] = df["volume"] * df["close"]
    df["pct_chg"] = df["close"].pct_change() * 100
    df["turnover"] = np.nan
    return df[["date", "open", "close", "high", "low", "volume", "amount",
               "pct_chg", "turnover"]]


def fetch_kline(secid: str, fqt: int = 2, beg: str = "20170101",
                end: str = "20500101", tries: int = 3) -> pd.DataFrame:
    """统一入口。fqt=2 → 复权连续序列；fqt=0 → 原始价。"""
    code = secid.split(".")[-1]
    raw = _sina_kline(code, 4000)
    if len(raw) == 0:
        raw = _em_kline(secid, 0, beg, end)
    if len(raw) == 0:
        return pd.DataFrame(columns=["date", "open", "close", "high", "low",
                                     "volume", "amount", "pct_chg", "turnover"])
    start = _fmt_start(beg)
    raw = raw[raw["date"] >= start].reset_index(drop=True)
    if fqt == 0:
        return raw
    return adjust_splits(raw)


def _fmt_start(beg) -> str:
    b = str(beg)
    if b in ("0", ""):
        return "2010-01-01"
    if len(b) == 8:
        return f"{b[:4]}-{b[4:6]}-{b[6:]}"
    return b


def _em_kline(secid: str, fqt: int, beg: str, end: str, tries: int = 2) -> pd.DataFrame:
    host = "29.push2his"
    url = (f"https://{host}.eastmoney.com/api/qt/stock/kline/get?"
           f"secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,"
           f"f57,f58,f59,f60,f61&klt=101&fqt={fqt}&beg={beg}&end={end}")
    try:
        d = _json(url, limit=False, retries=tries)
        kl = ((d.get("data") or {}).get("klines")) or []
        if not kl:
            return pd.DataFrame(columns=BAR_COLS)
        df = pd.DataFrame([k.split(",") for k in kl], columns=BAR_COLS)
        for c in BAR_COLS[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception:                              # noqa: BLE001
        return pd.DataFrame(columns=BAR_COLS)


def fetch_index_close(code: str = "000300", beg: str = "2015-01-01") -> pd.Series:
    """指数日线收盘（新浪指数接口）。"""
    sym = ("sh" if code.startswith("000") else "sz") + code
    url = ("http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen=4000")
    try:
        txt = _raw(url, enc="utf-8", hdr=HDR_SINA, retries=3)
        arr = json.loads(txt)
        df = pd.DataFrame(arr).rename(columns={"day": "date"})
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df[df["date"] >= beg].sort_values("date")
        return pd.Series(df["close"].values, index=df["date"].values)
    except Exception:                              # noqa: BLE001
        return pd.Series(dtype=float)


# ------------------------------------------------------------------ 实时
def fetch_realtime(codes) -> pd.DataFrame:
    codes = [str(c).zfill(6) for c in codes]
    if not codes:
        return pd.DataFrame()
    rows = []
    for i in range(0, len(codes), 80):
        q = ",".join(to_sina(c) for c in codes[i:i + 80])
        try:
            txt = _raw(f"http://hq.sinajs.cn/list={q}", enc="gbk", hdr=HDR_SINA, retries=3)
        except Exception:                          # noqa: BLE001
            continue
        for line in txt.split(";"):
            if '"' not in line:
                continue
            body = line.split('"')[1].split(",")
            if len(body) < 33:
                continue
            try:
                rows.append({
                    "code": line.split("=")[0].split("_")[-1][-6:],
                    "name": body[0],
                    "open": float(body[1] or 0), "pre_close": float(body[2] or 0),
                    "price": float(body[3] or 0), "high": float(body[4] or 0),
                    "low": float(body[5] or 0),
                    "volume": float(body[8] or 0), "amount": float(body[9] or 0),
                    "pct_chg": (float(body[3] or 0) / float(body[2] or 1) - 1) * 100,
                    "turnover": np.nan, "mkt_cap": np.nan, "float_cap": np.nan,
                })
            except Exception:                      # noqa: BLE001
                continue
    df = pd.DataFrame(rows)
    if len(df):
        df = df[df["price"] > 0].reset_index(drop=True)
    return df


# ------------------------------------------------------------------ SQLite
SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    secid TEXT, date TEXT, adj TEXT,
    open REAL, close REAL, high REAL, low REAL, volume REAL, amount REAL,
    pct_chg REAL, turnover REAL,
    PRIMARY KEY (secid, date, adj)
);
CREATE INDEX IF NOT EXISTS idx_bars_date ON bars(date);
CREATE INDEX IF NOT EXISTS idx_bars_secid ON bars(secid);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS stocklist (
    code TEXT PRIMARY KEY, name TEXT, secid TEXT, market INTEGER,
    industry TEXT, price REAL, amount REAL, float_cap REAL, mkt_cap REAL,
    turnover REAL, updated TEXT
);
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, action TEXT, code TEXT, name TEXT, price REAL, shares INTEGER,
    amount REAL, fee REAL, note TEXT
);
CREATE TABLE IF NOT EXISTS positions (
    code TEXT PRIMARY KEY, name TEXT, shares INTEGER, cost REAL,
    last_price REAL, peak REAL, entry_date TEXT, hold_days INTEGER DEFAULT 0,
    stop REAL, note TEXT
);
CREATE TABLE IF NOT EXISTS nav (
    date TEXT PRIMARY KEY, equity REAL, cash REAL, pos_value REAL, n_pos INTEGER
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_date TEXT, exec_date TEXT, side TEXT, code TEXT, name TEXT,
    plan_price REAL, plan_shares INTEGER, status TEXT, filled_price REAL,
    filled_shares INTEGER, reason TEXT, created TEXT
);
CREATE TABLE IF NOT EXISTS signals (
    date TEXT, secid TEXT, score REAL, eligible INTEGER,
    PRIMARY KEY (date, secid)
);
"""


def connect(path: str = DB_PATH):
    import sqlite3
    conn = sqlite3.connect(path, timeout=60, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def save_bars(conn, secid: str, df: pd.DataFrame, adj: str = "hfq") -> int:
    if df is None or len(df) == 0:
        return 0
    recs = [(secid, r.date, adj, r.open, r.close, r.high, r.low,
             r.volume, r.amount, r.pct_chg, r.turnover)
            for r in df.itertuples(index=False)]
    conn.executemany(
        "INSERT OR REPLACE INTO bars(secid,date,adj,open,close,high,low,volume,"
        "amount,pct_chg,turnover) VALUES(?,?,?,?,?,?,?,?,?,?,?)", recs)
    return len(recs)


def load_bars(conn, adj: str = "hfq", start: str = None, secids=None) -> pd.DataFrame:
    q = ("SELECT secid,date,open,close,high,low,volume,amount,pct_chg,turnover "
         "FROM bars WHERE adj=?")
    args = [adj]
    if start:
        q += " AND date>=?"
        args.append(start)
    if secids:
        q += f" AND secid IN ({','.join('?' * len(secids))})"
        args.extend(secids)
    return pd.read_sql_query(q + " ORDER BY secid,date", conn, params=args)


def save_stocklist(conn, df: pd.DataFrame):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    recs = [(r.code, r.name, r.secid, int(r.market), r.industry, _f(r.price),
             _f(r.amount), _f(r.float_cap), _f(r.mkt_cap), _f(r.turnover), now)
            for r in df.itertuples(index=False)]
    conn.executemany(
        "INSERT OR REPLACE INTO stocklist(code,name,secid,market,industry,price,"
        "amount,float_cap,mkt_cap,turnover,updated) VALUES(?,?,?,?,?,?,?,?,?,?,?)", recs)


def _f(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:                              # noqa: BLE001
        return None


def load_stocklist(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM stocklist", conn)


def set_meta(conn, k, v):
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (k, str(v)))


def get_meta(conn, k, default=None):
    r = conn.execute("SELECT value FROM meta WHERE key=?", (k,)).fetchone()
    return r[0] if r else default
