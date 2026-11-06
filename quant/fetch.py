# -*- coding: utf-8 -*-
"""全市场抓取（新浪源）。生产者-消费者：N 线程抓取，1 线程批量落库。

新浪单只约 0.7s（4000根），24 线程并发可达 20+/s。
落库集中在写线程，避免 SQLite 写锁竞争。

用法:
  python -m quant.fetch --full    全量
  python -m quant.fetch           增量（从最新日期重抓当月）
  python -m quant.fetch --test    30只冒烟
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from . import data as D

BEG = "20170101"
SKIP_BJ = True
BARS_PER_STOCK = 2400          # 约回溯到 2016-11，覆盖 2017 起


def _fetch(secid: str, start: str, tries: int = 3):
    for i in range(tries):
        try:
            raw = D._sina_kline(secid.split(".")[-1], BARS_PER_STOCK)
            if len(raw) == 0:
                return secid, None
            adj = D.adjust_splits(raw)
            adj = adj[adj["date"] >= start].reset_index(drop=True)
            raw = raw[raw["date"] >= start].reset_index(drop=True)
            return secid, (raw, adj)
        except Exception:                          # noqa: BLE001
            time.sleep(0.3 * (2 ** i))
    return secid, None


def _writer(conn, q: queue.Queue, stats: dict):
    while True:
        item = q.get()
        if item is None:
            break
        secid, raw, adj = item
        try:
            D.save_bars(conn, secid, raw, "raw")
            D.save_bars(conn, secid, adj, "hfq")
            stats["saved"] += 1
        except Exception as e:                     # noqa: BLE001
            stats["write_fail"] += 1
            stats.setdefault("last_err", str(e)[:200])
        q.task_done()


def update(full: bool = False, codes=None, workers: int = 24, start: str = BEG,
           progress_every: int = 400, commit_every: int = 600):
    conn = D.connect()
    t0 = time.time()

    sl = D.fetch_stock_list()
    D.save_stocklist(conn, sl)
    conn.commit()
    print(f"[列表] {len(sl)} 只  {time.time()-t0:.0f}s", flush=True)

    if SKIP_BJ:
        n0 = len(sl)
        sl = sl[~sl.code.str.startswith(("4", "8", "9"))].reset_index(drop=True)
        print(f"[过滤] 北交所 -{n0-len(sl)} -> {len(sl)}", flush=True)
    if codes:
        sl = sl[sl.code.isin(codes)].reset_index(drop=True)

    start_s = D._fmt_start(start)
    tasks = [(r.secid, start_s) for r in sl.itertuples(index=False)]
    total = len(tasks)

    q: queue.Queue = queue.Queue(maxsize=200)
    stats = {"saved": 0, "write_fail": 0}
    wt = threading.Thread(target=_writer, args=(conn, q, stats), daemon=True)
    wt.start()

    done, fail = 0, []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch, s, st): s for s, st in tasks}
        for fut in as_completed(futs):
            secid, res = fut.result()
            done += 1
            if res is None:
                fail.append(secid)
            else:
                raw, adj = res
                q.put((secid, raw, adj))
            if done % progress_every == 0:
                el = time.time() - t_start
                eta = (total - done) / max(done / el, 0.01)
                print(f"  {done}/{total}  {el:.0f}s  {done/el:.1f}/s  "
                      f"落库{stats['saved']} 失败{len(fail)}  ETA {eta/60:.1f}min",
                      flush=True)
            if done % commit_every == 0:
                conn.commit()
    q.put(None)
    wt.join(timeout=120)
    conn.commit()

    n = conn.execute("SELECT COUNT(DISTINCT secid) FROM bars WHERE adj='hfq'").fetchone()[0]
    rng = conn.execute("SELECT MIN(date),MAX(date) FROM bars WHERE adj='hfq'").fetchone()
    D.set_meta(conn, "last_update", time.strftime("%Y-%m-%d %H:%M:%S"))
    conn.commit()
    conn.close()
    print(f"[完成] {n} 只  {rng[0]}~{rng[1]}  {time.time()-t0:.0f}s  "
          f"写失败{stats['write_fail']}  抓取失败{len(fail)}")
    if fail:
        print("  失败:", fail[:20])
    return fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    codes = [c.strip() for c in a.only.split(",")] if a.only else None
    if a.test:
        sl = D.fetch_stock_list()
        codes = sl[~sl.code.str.startswith(("4", "8", "9"))].code.head(30).tolist()
        print("测试:", codes, flush=True)
    update(full=a.full, codes=codes, workers=a.workers)


if __name__ == "__main__":
    main()
