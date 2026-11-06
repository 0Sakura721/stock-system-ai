# -*- coding: utf-8 -*-
"""回测命令行入口：python -m quant.backtest_cli [--start 2020-01-01]"""
from __future__ import annotations

import argparse
import sys

from . import runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    res = runner.run(start=a.start, end=a.end, save=not a.no_save, verbose=not a.quiet)
    s = res["stats"]
    return 0 if s["sharpe"] > -5 else 1


if __name__ == "__main__":
    sys.exit(main())
