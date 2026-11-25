"""Protein-level paired bootstrap significance test (Section "Experimental Settings").

Every method is represented by all of its runs: the 10 training seeds of a re-trained model, or a
single run for web-server / pretrained methods. In each of N (=1000) iterations, test proteins are
sampled with replacement and the SAME resampled protein set is applied to every seed of both
methods (paired). Residues of the sampled proteins are pooled, micro-averaged metrics are computed
for each seed, and the metrics are averaged over the seeds of each method (a single-run method
contributes one value). The test statistic is the difference of the seed-averaged metrics,
    d = mean_seeds(metric_A) - mean_seeds(metric_B).
Reported per metric: the observed difference, the 95% percentile bootstrap confidence interval of
d, and the two-sided p-value p = min(1, 2 * min(P(d <= 0), P(d >= 0))); * marks p < alpha.
These p-values are unadjusted. A table contains many such tests, so the significance marks of the
tables are obtained by correcting the p-values of each table together (Benjamini-Hochberg): save the
results with --out and pass the JSON files of a table to scripts/adjust_pvalues.py.
Threshold-dependent metrics (Precision, Recall, MCC) use each run's own pre-specified threshold
(the F1-maximising threshold on the validation set); F1max is recomputed on every resample.

The trained models are kept fixed, so the test quantifies the uncertainty due to the composition
of the test set, not the uncertainty due to new training seeds.

Prediction files are pickles in the format written by scripts/train.py / scripts/evaluate.py:
    {"tasks": [...], "proteins": {id: {"prob": (L, T), "label": (L, T)}}, "thresholds": {task: thr}}
Baselines (web servers, other tools) must be converted to this format; tasks a baseline does not
predict can be filled with NaN probabilities and are skipped.

Example (all 10 seeds of both methods; pass one file for a single-run method):
    python scripts/bootstrap_test.py \
        --pred_a checkpoints/DP81/seed*/test_predictions.pkl \
        --pred_b baselines/DP81/DisoFLAG/seed*/test_predictions.pkl \
        --out results/DP81_vs_DisoFLAG.json
"""
import argparse
import json
import pickle

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.evaluate import METRIC_NAMES, best_f1_threshold, binary_metrics  # noqa: E402


def _metrics(y, p, thr):
    out = {m: np.nan for m in METRIC_NAMES}
    if len(y) == 0 or y.min() == y.max():
        return out
    out["AUPR"] = average_precision_score(y, p)
    out["AUROC"] = roc_auc_score(y, p)
    out["F1max"] = best_f1_threshold(y, p)[0]
    out.update({k: v for k, v in binary_metrics(y, p, thr).items() if k in METRIC_NAMES})
    return out


def _per_protein(preds, task, ids):
    j = preds["tasks"].index(task)
    ys, ps = [], []
    for pid in ids:
        rec = preds["proteins"][pid]
        y = np.asarray(rec["label"])[:, j]
        pr = np.asarray(rec["prob"], dtype=np.float64)[:, j]
        keep = y >= 0
        ys.append(y[keep].astype(np.int64))
        ps.append(pr[keep])
    return ys, ps


def _seed_mean(y, probs, thresholds, idx=None):
    """Seed-averaged metrics of one method on the proteins `idx` (all proteins if None).

    A metric is NaN if it is undefined for any seed, so that every seed always contributes.
    """
    cat = (lambda xs: np.concatenate(xs)) if idx is None else (lambda xs: np.concatenate([xs[i] for i in idx]))
    yy = cat(y)
    runs = [_metrics(yy, cat(p), t) for p, t in zip(probs, thresholds)]
    return {m: float(np.mean([r[m] for r in runs])) for m in METRIC_NAMES}


def paired_bootstrap(preds_a, preds_b, task, n_iter=1000, seed=0, ci=95.0):
    """preds_a / preds_b: lists of prediction dicts (one per seed; a single element for single-run methods)."""
    runs = list(preds_a) + list(preds_b)
    ids = sorted(set.intersection(*[set(r["proteins"]) for r in runs]))
    if not ids:
        raise ValueError("No protein is shared by all prediction files")
    per_run = [_per_protein(r, task, ids) for r in runs]
    if any(np.isnan(x).any() for _, ps in per_run for x in ps):
        return None
    y = per_run[0][0]
    for ys, _ in per_run[1:]:
        for a, b in zip(y, ys):
            if not np.array_equal(a, b):
                raise ValueError(f"Label mismatch between prediction files for task {task}")
    na = len(preds_a)
    pa = [ps for _, ps in per_run[:na]]
    pb = [ps for _, ps in per_run[na:]]
    ta = [r["thresholds"][task] for r in preds_a]
    tb = [r["thresholds"][task] for r in preds_b]

    obs_a = _seed_mean(y, pa, ta)
    obs_b = _seed_mean(y, pb, tb)

    rng = np.random.RandomState(seed)
    diffs = {m: [] for m in METRIC_NAMES}
    for _ in range(n_iter):
        idx = rng.randint(0, len(ids), size=len(ids))   # same resampled proteins for every seed of both methods
        ma = _seed_mean(y, pa, ta, idx)
        mb = _seed_mean(y, pb, tb, idx)
        for m in METRIC_NAMES:
            d = ma[m] - mb[m]
            if np.isfinite(d):
                diffs[m].append(d)

    lo_q, hi_q = (100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0
    out = {}
    for m in METRIC_NAMES:
        d = np.asarray(diffs[m], dtype=np.float64)
        if len(d):
            lo, hi = np.percentile(d, [lo_q, hi_q])
            p = min(1.0, 2.0 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0))))
        else:
            lo = hi = p = np.nan
        out[m] = {"A": obs_a[m], "B": obs_b[m], "diff": obs_a[m] - obs_b[m],
                  "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p),
                  "n_valid_iter": int(len(d)), "n_seeds_A": na, "n_seeds_B": len(preds_b)}
    return out


def _load(paths):
    preds = []
    for path in paths:
        with open(path, "rb") as f:
            preds.append(pickle.load(f))
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_a", required=True, nargs="+",
                   help="FlexBind predictions (.pkl), one file per seed")
    p.add_argument("--pred_b", required=True, nargs="+",
                   help="second-best method / configuration (.pkl), one file per seed (one file if single-run)")
    p.add_argument("--n_iter", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    A, B = _load(args.pred_a), _load(args.pred_b)
    print(f"A: {len(A)} run(s)   B: {len(B)} run(s)   iterations: {args.n_iter}")

    report = {}
    for task in A[0]["tasks"]:
        if any(task not in r["tasks"] for r in A + B):
            continue
        res = paired_bootstrap(A, B, task, args.n_iter, args.seed, ci=100.0 * (1.0 - args.alpha))
        if res is None:
            print(f"[{task}] skipped (missing predictions)")
            continue
        report[task] = res
        print(f"[{task}]")
        for m, r in res.items():
            star = "*" if np.isfinite(r["p_value"]) and r["p_value"] < args.alpha else ""
            print(f"  {m:<10} A={r['A']:.4f}  B={r['B']:.4f}  diff={r['diff']:+.4f}  "
                  f"CI=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  p={r['p_value']:.4f} {star}")
    print("p-values are unadjusted; correct them per table with scripts/adjust_pvalues.py")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
