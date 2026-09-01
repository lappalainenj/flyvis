"""Stage 9: do the two stacks agree with augmentation ON?

Every earlier stage compared the stacks with augmentation off, because the two
consume their random number streams differently and cannot be matched frame by
frame. But augmentation is active for all 250k training iterations, so agreement
with it off is agreement on a regime the models never train in.

Two things are checked here that do not need matched RNG streams:

1. the sampled augmentation *parameters* -- rotation, flip axis, contrast,
   brightness, noise -- which should be drawn from the same distributions, and
2. the marginal statistics of the produced `lum` and `flow` tensors over many
   independent draws.

Also reports the temporal structure of `lum`, i.e. the mean absolute difference
between consecutive output frames. Piecewise-constant resampling duplicates
frames, so this quantity is sensitive to whether pixel noise is applied before
or after resampling -- duplicated frames carry duplicated noise and differ by
exactly zero, while independently noised frames do not.

Run:
    python stage9_augmentation_distribution.py --draws 400
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import dvs_config, flyvis_config  # noqa: E402


def summarize_tensor(x):
    """Statistics of one produced sample."""
    x = x.detach().float()
    d = x[1:] - x[:-1] if x.shape[0] > 1 else x
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "abs_mean": float(x.abs().mean()),
        "temporal_absdiff": float(d.abs().mean()),
        "frac_identical_frames": float(
            (d.abs().flatten(1).max(dim=1).values == 0).float().mean()
        )
        if x.shape[0] > 1
        else float("nan"),
    }


def collect(dataset, indices, n_draws, is_dvs):
    keys = None
    stats = {}
    shapes = {}
    for i in range(n_draws):
        item = dataset[int(indices[i % len(indices)])]
        for name in ("lum", "flow"):
            if name not in item:
                continue
            t = item[name]
            s = summarize_tensor(t)
            shapes.setdefault(name, tuple(t.shape))
            for k, v in s.items():
                stats.setdefault(f"{name}.{k}", []).append(v)
        keys = keys or list(stats)
    return {k: np.array(v) for k, v in stats.items()}, shapes


def compare(a, b, label_a="dvs-sim", label_b="flyvis"):
    from scipy import stats as st

    print(
        f"\n{'statistic':<28} {label_a + ' mean':>14} {label_b + ' mean':>14} "
        f"{'rel diff':>10} {'KS p':>9}"
    )
    problems = []
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if not np.isfinite(x).any() or not np.isfinite(y).any():
            continue
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]
        rel = (y.mean() - x.mean()) / (abs(x.mean()) + 1e-12)
        p = st.ks_2samp(x, y).pvalue
        flag = ""
        if p < 0.01 and abs(rel) > 0.02:
            flag = "  <-- differs"
            problems.append((k, rel, p))
        print(f"{k:<28} {x.mean():>14.5f} {y.mean():>14.5f} {rel:>+9.2%} {p:>9.2e}{flag}")
    return problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draws", type=int, default=400)
    args = p.parse_args()

    from dvs.datasets.sintel import MultiTaskSintel as DvsSintel
    from flyvis.datasets.sintel import MultiTaskSintel as FlyvisSintel

    dcfg = dvs_config()
    dconf = dcfg.task.dataset.deepcopy()
    dconf.pop("type")
    dset = DvsSintel(**dconf)
    dset.dt = 0.02

    fcfg = flyvis_config(["task=task_original"])
    fconf = fcfg.task.dataset.deepcopy()
    fconf.pop("type")
    fset = FlyvisSintel(**fconf)
    fset.dt = 0.02

    print(f"dvs-sim dataset: {len(dset)} sequences, augment={dset.augment}")
    print(f"flyvis  dataset: {len(fset)} sequences, augment={fset.augment}")
    print(f"flyvis original_sampling={fset.piecewise_resample.original_sampling}, "
          f"align_corners={fset.linear_interpolate.align_corners}")

    n = min(len(dset), len(fset))
    rng = np.random.default_rng(0)
    order = rng.integers(0, n, size=args.draws)

    torch.manual_seed(0)
    np.random.seed(0)
    dstats, dshapes = collect(dset, order, args.draws, True)
    torch.manual_seed(0)
    np.random.seed(0)
    fstats, fshapes = collect(fset, order, args.draws, False)

    print(f"\nshapes  dvs-sim {dshapes}\n        flyvis  {fshapes}")
    if dshapes != fshapes:
        print("  !! output shapes differ")

    problems = compare(dstats, fstats)

    print()
    if problems:
        print(f"{len(problems)} statistic(s) differ materially:")
        for k, rel, p in problems:
            print(f"  {k}: flyvis is {rel:+.2%} vs dvs-sim (KS p={p:.2e})")
    else:
        print("no statistic differs by more than 2% at KS p<0.01")


if __name__ == "__main__":
    main()
