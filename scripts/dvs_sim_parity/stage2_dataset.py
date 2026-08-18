"""Stage 2: do the two repos feed the network the same movies and targets?

Compares the rendered (pre-augmentation) sequences, then the dataset items with
augmentation switched off, which removes all randomness and leaves only the
deterministic temporal resampling and interpolation.

Run:
    python stage2_dataset.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import dvs_config, flyvis_config, print_report, summarize  # noqa: E402

import dvs  # noqa: E402
import flyvis  # noqa: E402


def build_datasets():
    dcfg, fcfg = dvs_config(), flyvis_config()
    from dvs.datasets.sintel import MultiTaskSintel as DvsSintel
    from flyvis.datasets.sintel import MultiTaskSintel as FlyvisSintel

    dconf = dcfg.task.dataset.deepcopy()
    dconf.pop("type")
    dset = DvsSintel(**dconf)
    dset.dt = dcfg.scheduler.dt.start

    fconf = fcfg.task.dataset.deepcopy()
    fconf.pop("type")
    fset = FlyvisSintel(**fconf)
    return dset, fset, dcfg, fcfg


def main():
    dset, fset, dcfg, fcfg = build_datasets()

    print("== dataset inventory ==")
    print(f"  n sequences dvs={len(dset.arg_df)} flyvis={len(fset.arg_df)}")
    print(f"  dt dvs={dset.dt} flyvis={fset.dt}")
    dnames = list(dset.arg_df.name)
    fnames = list(fset.arg_df.name)
    print(f"  sequence names identical: {dnames == fnames}")
    if dnames != fnames:
        print("   dvs   :", dnames[:4])
        print("   flyvis:", fnames[:4])

    print("\n== rendered (cached, un-augmented) sequences ==")
    rows = []
    for seq in (0, 1, 17, 40):
        d = dset.cached_samples[seq]
        f = fset.cached_sequences[seq]
        for key in ("lum", "flow"):
            rows.append(summarize(d[key], f[key], f"seq{seq}.{key}"))
    print_report(rows)

    print("\n== dataset items, augmentation off ==")
    rows = []
    with dset.augmentation(False), fset.augmentation(False):
        for seq in (0, 1, 17, 40):
            d = dset[seq]
            f = fset[seq]
            for key in ("lum", "flow"):
                rows.append(summarize(d[key], f[key], f"seq{seq}.{key}"))
    print_report(rows)

    print("\n== temporal sampling detail (sequence 0, augmentation off) ==")
    with dset.augmentation(False), fset.augmentation(False):
        d = dset[0]
        f = fset[0]
    print(f"  dvs    lum {tuple(d['lum'].shape)}  flow {tuple(d['flow'].shape)}")
    print(f"  flyvis lum {tuple(f['lum'].shape)}  flow {tuple(f['flow'].shape)}")

    raw = dset.cached_samples[0]["lum"]
    n_frames = dset.n_frames
    dvs_idx = dset.get_temporal_sample_indices(n_frames, raw.shape[0], False)
    fly_idx = fset.piecewise_resample.piecewise_constant_indices(n_frames)
    print(f"  dvs    frame indices ({len(dvs_idx)}): {dvs_idx.tolist()}")
    print(f"  flyvis frame indices ({len(fly_idx)}): {fly_idx.tolist()}")
    print(f"  identical: {dvs_idx.shape == fly_idx.shape and bool((dvs_idx == fly_idx).all())}")

    print("\n== target interpolation ==")
    print(f"  dvs    linear interp align_corners=False (dvs/datasets/sintel.py interp)")
    print(
        f"  flyvis linear interp align_corners="
        f"{fset.linear_interpolate.align_corners}"
    )
    # isolate the interpolation on identical input
    tgt = dset.cached_samples[0]["flow"][:n_frames]
    import torch.nn.functional as nnf

    size = len(fly_idx)
    dvs_interp = dset.interp(tgt, size)
    fly_interp = fset.linear_interpolate(tgt)
    print_report([summarize(dvs_interp, fly_interp, f"interp(flow[:19] -> {size})")])

    print("\n== augmentation parameter spaces ==")
    print(f"  dvs    flip axis draw: np.random.randint(0, 3) -> 0,1,2 (None = no flip)")
    print(f"  flyvis flip axes: {fset.flip.flip_axes}, draw randint(1, {max(fset.flip.flip_axes) + 1}) (0 = no flip)")
    print(f"  dvs    rotation draw: randint(1, 6) if p_rot > rand")
    print(f"  flyvis rotation draw: randint(1, 6) if p_rot > rand")
    print(f"  dvs    gamma: applied at render time as lum ** {dcfg.task.dataset.gamma}")
    print(f"  flyvis gamma_std: {fcfg.task.dataset.gamma_std} (augmentation-time)")

    print("\n== hex permutation / rotation tables ==")
    rows = []
    for n in range(1, 6):
        rows.append(
            summarize(
                torch.as_tensor(dset.rotate._cached_indices[n]),
                torch.as_tensor(fset.rotate.permutation_indices[n]),
                f"rotate.perm[{n}]",
            )
        )
        rows.append(
            summarize(
                dset.rotate._cached_matrices[n]["2d"],
                fset.rotate.rotation_matrices[n][0],
                f"rotate.matrix2d[{n}]",
            )
        )
    for n in (0, 1, 2):
        rows.append(
            summarize(
                torch.as_tensor(dset.flip._cached_indices[n]),
                torch.as_tensor(fset.flip.permutation_indices[n + 1]),
                f"flip.perm dvs[{n}] vs flyvis[{n + 1}]",
            )
        )
        rows.append(
            summarize(
                dset.flip._cached_matrices[n]["2d"],
                fset.flip.rotation_matrices[n + 1][0],
                f"flip.matrix2d dvs[{n}] vs flyvis[{n + 1}]",
            )
        )
    print_report(rows)


if __name__ == "__main__":
    main()
