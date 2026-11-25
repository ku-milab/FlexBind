"""Table-level multiple-comparison correction of the bootstrap p-values (Benjamini-Hochberg FDR).

scripts/bootstrap_test.py compares FlexBind with ONE other method / configuration and reports raw
(unadjusted) p-values. A results table holds many such tests (benchmark x task x metric), so the
p-values belonging to one table are corrected together here. The bootstrap itself is not re-run:
this script only reads the JSON files written by `bootstrap_test.py --out`.

Input: one JSON per (block of the table, competitor). `--group NAME FILE [FILE ...]` collects the
files of one block (e.g. one benchmark); every file of a group must compare the same FlexBind runs
on the same test proteins with a different competitor.

For every cell (group, task, metric), FlexBind is compared with the strongest competitor of that
cell, i.e. the one with the highest seed-averaged metric among the files of the group (the second-best
method wherever FlexBind ranks first). All cells passed in one call form one family (= one table),
including the cells in which FlexBind is not the best, and their two-sided p-values are adjusted with
the Benjamini-Hochberg step-up procedure. `*` marks cells in which FlexBind is better and the
adjusted p-value is below alpha.

Use --tasks / --metrics to restrict the family to the cells that are actually shown in the table.

Example (Table 2: three benchmarks, three binding tasks, six metrics -> 54 tests in one family):
    python scripts/adjust_pvalues.py \
        --group DP81 results/DP81_vs_*.json \
        --group DP93 results/DP93_vs_*.json \
        --group DP94 results/DP94_vs_*.json \
        --tasks protein rna dna --out results/table2_bh.json
"""
import argparse
import json
import math
import os

import numpy as np


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg adjusted p-values, returned in the input order."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    scaled = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)   # enforce monotonicity
    out = np.empty(m, dtype=np.float64)
    out[order] = adjusted
    return out


def _ordered_union(sequences):
    seen = []
    for seq in sequences:
        for x in seq:
            if x not in seen:
                seen.append(x)
    return seen


def select_cells(groups, tasks=None, metrics=None, tol=1e-6):
    """One cell per (group, task, metric): FlexBind (A) against the strongest competitor (B).

    groups: list of (name, [json paths]). Returns (cells, warnings).
    """
    cells, warnings = [], []
    found_tasks, found_metrics = set(), set()
    for name, paths in groups:
        reports = []
        for path in paths:
            with open(path) as f:
                reports.append((os.path.splitext(os.path.basename(path))[0], json.load(f)))
        group_tasks = _ordered_union(rep.keys() for _, rep in reports)
        group_metrics = _ordered_union(res.keys() for _, rep in reports for res in rep.values())
        found_tasks.update(group_tasks)
        found_metrics.update(group_metrics)
        for task in group_tasks:
            if tasks and task not in tasks:
                continue
            for metric in group_metrics:
                if metrics and metric not in metrics:
                    continue
                cand = [(label, rep[task][metric]) for label, rep in reports
                        if metric in rep.get(task, {}) and math.isfinite(rep[task][metric]["B"])]
                if not cand:
                    continue
                a_values = [r["A"] for _, r in cand]
                if max(a_values) - min(a_values) > tol:
                    warnings.append(f"[{name}] {task}/{metric}: the FlexBind value differs between files "
                                    f"({min(a_values):.4f} to {max(a_values):.4f}); the files of a group must "
                                    f"use the same FlexBind runs and the same test proteins")
                label, r = max(cand, key=lambda c: c[1]["B"])
                cells.append({"group": name, "task": task, "metric": metric, "competitor": label,
                              **{k: r[k] for k in ("A", "B", "diff", "ci_low", "ci_high", "p_value")}})
    for requested, found, kind in ((tasks, found_tasks, "task"), (metrics, found_metrics, "metric")):
        missing = [x for x in (requested or []) if x not in found]
        if missing:
            raise ValueError(f"Unknown {kind}(s) {missing}; available: {sorted(found)}")
    return cells, warnings


def adjust(cells, alpha=0.05):
    """Adds p_adjusted / significant_raw / significant to every cell; cells without a p-value stay untested."""
    tested = [c for c in cells if math.isfinite(c["p_value"])]
    for c, q in zip(tested, benjamini_hochberg([c["p_value"] for c in tested])):
        c["p_adjusted"] = float(q)
    for c in cells:
        c.setdefault("p_adjusted", float("nan"))
        better = c["diff"] > 0
        c["significant_raw"] = bool(better and c["p_value"] < alpha)
        c["significant"] = bool(better and c["p_adjusted"] < alpha)
    return len(tested)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--group", required=True, action="append", nargs="+", metavar=("NAME", "JSON"),
                   help="block name followed by the bootstrap_test.py JSON files of that block "
                        "(FlexBind vs each competitor); repeat for every block of the table")
    p.add_argument("--tasks", nargs="+", default=None, help="tasks shown in the table (default: all)")
    p.add_argument("--metrics", nargs="+", default=None, help="metrics shown in the table (default: all)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    groups = []
    for g in args.group:
        if len(g) < 2:
            p.error(f"--group {g[0]}: no JSON file given")
        groups.append((g[0], g[1:]))

    try:
        cells, warnings = select_cells(groups, args.tasks, args.metrics)
    except ValueError as e:
        p.error(str(e))
    if not cells:
        raise SystemExit("No comparison found in the given files")
    n_tests = adjust(cells, args.alpha)

    for w in warnings:
        print("WARNING:", w)
    print(f"Family: {n_tests} tests in {len(groups)} group(s)   Benjamini-Hochberg, alpha = {args.alpha}")
    width = max(len(c["competitor"]) for c in cells)
    block = None
    for c in cells:
        if (c["group"], c["task"]) != block:
            block = (c["group"], c["task"])
            print(f"[{c['group']}] {c['task']}")
        if c["significant"]:
            mark = "*"
        elif c["significant_raw"]:
            mark = "(raw p < alpha only)"
        elif not math.isfinite(c["p_value"]):
            mark = "(not tested)"
        else:
            mark = ""
        print(f"  {c['metric']:<10} A={c['A']:.4f}  B={c['B']:.4f} ({c['competitor']:<{width}})  "
              f"diff={c['diff']:+.4f}  CI=[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  "
              f"p={c['p_value']:.4f}  p_adj={c['p_adjusted']:.4f} {mark}")
    n_raw = sum(c["significant_raw"] for c in cells)
    n_adj = sum(c["significant"] for c in cells)
    print(f"Significant cells (FlexBind better): {n_raw} at raw p < {args.alpha} -> {n_adj} after correction")
    for c in cells:
        if c["significant_raw"] and not c["significant"]:
            print(f"  no longer significant: [{c['group']}] {c['task']} {c['metric']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"method": "benjamini-hochberg", "alpha": args.alpha, "n_tests": n_tests,
                       "n_significant_raw": n_raw, "n_significant": n_adj, "cells": cells}, f, indent=2)


if __name__ == "__main__":
    main()
