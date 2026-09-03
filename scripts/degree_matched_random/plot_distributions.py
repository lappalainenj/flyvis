"""Histograms of per-network EPE and dynamic range across ensembles.

All values come from one common-protocol evaluation, so the same EPE function,
dataset and held-out split produced every number.

Example:
    python plot_distributions.py --results eval.json --out figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "flow/0000": "published (0000)",
    "flow/0310": "our reference (0310)",
    "flow/0301": "locality-preserving null (0301)",
}
COLORS = {"flow/0000": "#33691e", "flow/0310": "#1565c0", "flow/0301": "#c62828"}


def members(results, prefix):
    return [
        v
        for k, v in results.items()
        if isinstance(v, dict) and k.startswith(prefix) and "@init" not in k
    ]


def panel(ax, data, key, log=False, bins=20):
    values = {name: np.array([m[key] for m in ms]) for name, ms in data.items() if ms}
    allv = np.concatenate(list(values.values()))
    edges = (
        np.geomspace(max(allv.min(), 1e-3), allv.max(), bins)
        if log
        else np.linspace(allv.min(), allv.max(), bins)
    )
    for name, v in values.items():
        ax.hist(
            v,
            bins=edges,
            alpha=0.55,
            label=f"{LABELS.get(name, name)}  n={len(v)}",
            color=COLORS.get(name),
            edgecolor="white",
            linewidth=0.4,
        )
        ax.axvline(np.median(v), color=COLORS.get(name), lw=1.6, ls="--")
    if log:
        ax.set_xscale("log")
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", default="figures")
    p.add_argument(
        "--ensembles", nargs="+", default=["flow/0000", "flow/0310", "flow/0301"]
    )
    args = p.parse_args()

    results = json.loads(Path(args.results).read_text())
    data = {e: members(results, e) for e in args.ensembles}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    v = panel(axes[0], data, "epe")
    if "zero" in results:
        axes[0].axvline(
            results["zero"]["epe"], color="k", lw=1.2, ls=":", label="zero prediction"
        )
    axes[0].set(xlabel="EPE (common protocol)", ylabel="networks")
    axes[0].set_title("task error", loc="left")
    axes[0].legend(fontsize=7, frameon=False)

    a = panel(axes[1], data, "max_abs_activity", log=True)
    axes[1].set(xlabel="max |activity| (log scale)", ylabel="networks")
    axes[1].set_title("dynamic range", loc="left")
    axes[1].legend(fontsize=7, frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "ensemble_distributions.pdf", bbox_inches="tight")
    fig.savefig(out / "ensemble_distributions.png", dpi=170, bbox_inches="tight")

    print(f"{'ensemble':<34} {'n':>3} {'EPE median':>11} {'EPE mean':>9} "
          f"{'max|act| median':>16} {'max|act| max':>13}")
    for name in args.ensembles:
        if not data[name]:
            continue
        e = np.array([m["epe"] for m in data[name]])
        m = np.array([m["max_abs_activity"] for m in data[name]])
        print(f"{LABELS.get(name, name):<34} {len(e):>3} {np.median(e):>11.4f} "
              f"{e.mean():>9.4f} {np.median(m):>16.1f} {m.max():>13.1f}")
    print(f"\nwrote {out}/ensemble_distributions.{{pdf,png}}")
    return v, a


if __name__ == "__main__":
    main()
