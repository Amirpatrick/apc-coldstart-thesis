#!/usr/bin/env python
"""
Entry point for the five pre-commitment data checks.

    python run_checks.py inventory        # what's in the shared folder
    python run_checks.py build            # JSON -> parquet caches (once)
    python run_checks.py all              # run checks 1-5
    python run_checks.py 1 3 4 5          # metadata checks only (fast, no CF)
    python run_checks.py 2 --force-cf     # retrain the CF teacher

Reports land in reports/ inside the repo and are meant to be committed.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

from src import config as C
from src import prepare
from src.checks_cf import check2
from src.checks_meta import check1, check3, check4, check5

CHECKS = {"1": check1, "2": check2, "3": check3, "4": check4, "5": check5}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="APC cold-start data checks")
    ap.add_argument(
        "targets",
        nargs="+",
        help="inventory | build | all | one or more of 1 2 3 4 5",
    )
    ap.add_argument("--force", action="store_true", help="rebuild parquet caches")
    ap.add_argument("--force-cf", action="store_true", help="retrain the CF teacher")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    print(f"Melon root : {C.MELON_ROOT}")
    print(f"Workspace  : {C.WORKSPACE}")
    print(f"Reports    : {C.REPORT_DIR}")

    if "inventory" in args.targets:
        prepare.inventory()
        return 0

    if "build" in args.targets:
        prepare.build(force=args.force)
        if len(args.targets) == 1:
            return 0

    targets = list(CHECKS) if "all" in args.targets else [
        t for t in args.targets if t in CHECKS
    ]
    if not targets:
        print("Nothing to run.")
        return 1

    failed = []
    for t in targets:
        t0 = time.time()
        try:
            if t == "2":
                CHECKS[t](force_cf=args.force_cf)
            else:
                CHECKS[t]()
            print(f"  check {t} done in {time.time() - t0:.1f}s")
        except Exception:  # noqa: BLE001
            failed.append(t)
            print(f"  check {t} FAILED:")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Completed: {[t for t in targets if t not in failed]}")
    if failed:
        print(f"Failed:    {failed}")
    print(f"Reports in: {C.REPORT_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
