"""Stage 8: does the new `original_sampling` flag reproduce dvs-sim exactly?

Stage 2b established the two operations by monkeypatching. This checks the real
implementation: a flyvis dataset built with `original_sampling=True` through the
public API, and the `task=task_original` config profile, against dvs-sim.

Run:
    python stage8_verify_flag.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import dvs_config, flyvis_config, print_report, summarize  # noqa: E402

import dvs  # noqa: E402
import flyvis  # noqa: E402


def main():
    dcfg = dvs_config()
    from dvs.datasets.sintel import MultiTaskSintel as DvsSintel
    from flyvis.datasets.sintel import MultiTaskSintel as FlyvisSintel

    dconf = dcfg.task.dataset.deepcopy()
    dconf.pop("type")
    dset = DvsSintel(**dconf)
    dset.dt = 0.02

    print("== config profile task=task_original ==")
    cfg = flyvis_config(["task=task_original"])
    print(f"  dataset.original_sampling = {cfg.task.dataset.original_sampling}")
    fconf = cfg.task.dataset.deepcopy()
    fconf.pop("type")
    fset = FlyvisSintel(**fconf)

    print(f"  flyvis piecewise_resample.original_sampling ="
          f" {fset.piecewise_resample.original_sampling}")
    print(f"  flyvis linear_interpolate.align_corners ="
          f" {fset.linear_interpolate.align_corners}")

    print("\n== frame indices ==")
    dvs_idx = dset.get_temporal_sample_indices(fset.n_frames, 50, False)
    fly_idx = fset.piecewise_resample.piecewise_constant_indices(fset.n_frames)
    print(f"  dvs   : {dvs_idx.tolist()}")
    print(f"  flyvis: {fly_idx.tolist()}")
    print(f"  identical: {bool((dvs_idx == fly_idx).all())}")

    print("\n== items, augmentation off, all 69 sequences ==")
    rows = []
    with dset.augmentation(False), fset.augmentation(False):
        for seq in range(69):
            d, f = dset[seq], fset[seq]
            for key in ("lum", "flow"):
                rows.append(summarize(d[key], f[key], f"seq{seq}.{key}"))
    n_equal = sum(r["status"] == "equal" for r in rows)
    print(f"  {n_equal}/{len(rows)} tensors bit-identical")
    for r in rows:
        if r["status"] != "equal":
            print_report([r])

    print("\n== items, augmentation on with matched parameters ==")
    rows = []
    for seq, n_rot, flip, contrast, brightness in [
        (0, 0, None, 1.0, 0.0),
        (5, 3, 0, 1.15, 0.03),
        (23, 4, 2, 0.85, -0.04),
        (61, 1, 1, 1.05, 0.01),
    ]:
        for ds, is_dvs in ((dset, True), (fset, False)):
            ds.augment = True
            ds.fix_augmentation_params = True
            ds.rotate.n_rot = n_rot
            ds.jitter.contrast_factor = contrast
            ds.jitter.brightness_factor = brightness
            if is_dvs:
                ds.flip.axis = flip
                ds.noise_std = 0.0
                ds.noise.std = 0.0
                ds.t_window_augment = False
            else:
                ds.flip.axis = 0 if flip is None else flip + 1
                ds.noise.std = 0.0
                ds.temporal_crop.random = False
        d, f = dset[seq], fset[seq]
        for key in ("lum", "flow"):
            rows.append(summarize(d[key], f[key], f"seq{seq}.rot{n_rot}.flip{flip}.{key}"))
    print_report(rows)

    print("\n== public flyvis default is unchanged ==")
    default = FlyvisSintel(**{
        k: v for k, v in flyvis_config().task.dataset.deepcopy().items() if k != "type"
    })
    print(f"  default original_sampling = {default.piecewise_resample.original_sampling}")
    print(f"  default align_corners     = {default.linear_interpolate.align_corners}")
    with default.augmentation(False), dset.augmentation(False):
        d, f = dset[0], default[0]
    print_report([
        summarize(d["lum"], f["lum"], "default seq0.lum vs dvs"),
        summarize(d["flow"], f["flow"], "default seq0.flow vs dvs"),
    ])


if __name__ == "__main__":
    main()
