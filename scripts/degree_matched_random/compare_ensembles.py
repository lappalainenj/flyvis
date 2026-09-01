"""Compare two ensembles on one protocol, testing equivalence and difference.

"Statistically indistinguishable" is not the same as "we failed to find a
difference": a non-significant t-test is also what an underpowered comparison
produces. So this reports both

  * a difference test (Welch t, Mann-Whitney), which can reject equality, and
  * an equivalence test (TOST -- two one-sided tests), which can affirmatively
    bound the difference below a margin you specify.

For validating that a reimplemented training pipeline reproduces a published
ensemble, the equivalence test is the one that matters, and the margin should be
smaller than the effect the pipeline is later used to measure.

Example:
    python compare_ensembles.py --a flow/0310 --b flow/0000 \
        --results eval.json --margin-l2norm 13.5 --margin-epe 0.078
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def members(results, prefix):
    return {
        k: v
        for k, v in results.items()
        if isinstance(v, dict) and k.startswith(prefix) and "@init" not in k
    }


def tost(x, y, margin):
    """Two one-sided tests. Equivalent if both one-sided tests reject."""
    diff = x.mean() - y.mean()
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    df = (x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)) ** 2 / (
        (x.var(ddof=1) / len(x)) ** 2 / (len(x) - 1)
        + (y.var(ddof=1) / len(y)) ** 2 / (len(y) - 1)
    )
    # H0 lower: diff <= -margin ; H0 upper: diff >= +margin
    t_lower = (diff + margin) / se
    t_upper = (diff - margin) / se
    p_lower = stats.t.sf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    return {
        "diff": float(diff),
        "se": float(se),
        "margin": float(margin),
        "ci90": [float(diff - 1.645 * se), float(diff + 1.645 * se)],
        "p_tost": float(max(p_lower, p_upper)),
        "equivalent": bool(max(p_lower, p_upper) < 0.05),
    }


def report(name, x, y, margin, label_a, label_b):
    print(f"\n== {name} ==")
    print(
        f"  {label_a} (n={len(x)}): mean {x.mean():.4f}  sd {x.std(ddof=1):.4f}  "
        f"min {x.min():.4f}  max {x.max():.4f}"
    )
    print(
        f"  {label_b} (n={len(y)}): mean {y.mean():.4f}  sd {y.std(ddof=1):.4f}  "
        f"min {y.min():.4f}  max {y.max():.4f}"
    )
    t, p = stats.ttest_ind(x, y, equal_var=False)
    _, pu = stats.mannwhitneyu(x, y, alternative="two-sided")
    print(f"  difference test : Welch t={t:.2f} p={p:.4f} | Mann-Whitney p={pu:.4f}")
    if margin is None:
        print("  equivalence test: skipped (no margin given)")
        return {"welch_p": float(p), "mwu_p": float(pu)}
    eq = tost(x, y, margin)
    verdict = "EQUIVALENT" if eq["equivalent"] else "NOT SHOWN EQUIVALENT"
    print(
        f"  equivalence test: diff {eq['diff']:+.4f} +- {eq['se']:.4f}, "
        f"90% CI [{eq['ci90'][0]:+.4f}, {eq['ci90'][1]:+.4f}], "
        f"margin +-{margin:g}"
    )
    print(f"                    TOST p={eq['p_tost']:.4f}  -> {verdict}")
    return {"welch_p": float(p), "mwu_p": float(pu), **eq}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="e.g. flow/0310")
    p.add_argument("--b", default="flow/0000")
    p.add_argument("--results", required=True, help="json from evaluate_ensemble.py")
    p.add_argument(
        "--margin-l2norm",
        type=float,
        default=None,
        help="equivalence margin; a natural choice is the effect size the "
        "pipeline is later used to measure",
    )
    p.add_argument("--margin-epe", type=float, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    results = json.loads(Path(args.results).read_text())
    A, B = members(results, args.a), members(results, args.b)
    if not A or not B:
        raise SystemExit(f"missing members: {args.a}={len(A)}, {args.b}={len(B)}")

    out = {"a": args.a, "b": args.b, "n_a": len(A), "n_b": len(B)}
    for metric, margin in [("l2norm", args.margin_l2norm), ("epe", args.margin_epe)]:
        x = np.array([v[metric] for v in A.values()])
        y = np.array([v[metric] for v in B.values()])
        out[metric] = report(metric, x, y, margin, args.a, args.b)

    for metric in ["corr_pred_gt", "max_abs_activity"]:
        x = np.array([v[metric] for v in A.values()])
        y = np.array([v[metric] for v in B.values()])
        print(
            f"\n  {metric}: {args.a} {np.nanmean(x):.3f} vs {args.b} {np.nanmean(y):.3f}"
        )

    if "zero" in results:
        z = results["zero"]
        xa = np.array([v["epe"] for v in A.values()]).mean()
        xb = np.array([v["epe"] for v in B.values()]).mean()
        print(
            f"\n  EPE gain over zero prediction ({z['epe']:.4f}): "
            f"{args.a} {z['epe'] - xa:.4f}, {args.b} {z['epe'] - xb:.4f} "
            f"-> ratio {(z['epe'] - xa) / (z['epe'] - xb):.3f}"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
