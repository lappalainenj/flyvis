"""Stage 2b: is the dataset difference fully explained by two operations?

Hypothesis: flyvis differs from dvs-sim only in

  (1) the piecewise-constant temporal sampling pattern of the input movie
      (dvs: floor(arange(0, n_frames, dt*framerate)); flyvis: nearest-exact
      interpolation to ceil(n_frames/dt/framerate) samples), and
  (2) align_corners in the linear interpolation of the targets
      (dvs: False; flyvis: True).

This patches those two operations in the flyvis dataset instance and checks
whether the resulting items become bit-identical to the dvs-sim items.

Run:
    python stage2b_dataset_fix.py
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as nnf

sys.path.insert(0, str(Path(__file__).parent))
from common import print_report, summarize  # noqa: E402
from stage2_dataset import build_datasets  # noqa: E402


def patch_flyvis_to_dvs_semantics(fset, framerate=24):
    """Give the flyvis dataset dvs-sim's temporal sampling and interpolation."""

    def dvs_indices(length, dt):
        return torch.arange(0, length - 1e-6, dt * framerate).long()

    piecewise = fset.piecewise_resample
    linear = fset.linear_interpolate

    def piecewise_transform(sequence, dim=0):
        assert dim == 0
        idx = dvs_indices(sequence.shape[0], 1 / piecewise.target_framerate)
        return sequence[idx]

    def linear_transform(sequence, dim=0):
        assert dim == 0
        size = len(dvs_indices(sequence.shape[0], 1 / linear.target_framerate))
        return nnf.interpolate(
            sequence.permute(2, 1, 0), size=size, mode="linear", align_corners=False
        ).permute(2, 1, 0)

    piecewise.transform = piecewise_transform
    linear.transform = linear_transform
    return fset


def main():
    dset, fset, dcfg, fcfg = build_datasets()

    print("== before patch, augmentation off ==")
    rows = []
    with dset.augmentation(False), fset.augmentation(False):
        for seq in (0, 1, 17, 40, 68):
            d, f = dset[seq], fset[seq]
            for key in ("lum", "flow"):
                rows.append(summarize(d[key], f[key], f"seq{seq}.{key}"))
    print_report(rows)

    patch_flyvis_to_dvs_semantics(fset)

    print("\n== after patch, augmentation off ==")
    rows = []
    with dset.augmentation(False), fset.augmentation(False):
        for seq in range(0, 69, 7):
            d, f = dset[seq], fset[seq]
            for key in ("lum", "flow"):
                rows.append(summarize(d[key], f[key], f"seq{seq}.{key}"))
    print_report(rows)

    print("\n== after patch, augmentation on with matched parameters ==")
    # drive both with the same augmentation parameters and temporal window
    rows = []
    for seq, n_rot, flip, contrast, brightness, start in [
        (0, 0, None, 1.0, 0.0, 0),
        (3, 2, 1, 1.1, 0.05, 5),
        (11, 5, 2, 0.9, -0.05, 12),
    ]:
        dset.augment = True
        fset.augment = True
        dset.fix_augmentation_params = True
        fset.fix_augmentation_params = True
        # dvs: augmentation parameters are attributes, window start comes from
        # get_temporal_sample_indices, so pin it by disabling window augmentation
        dset.rotate.n_rot = n_rot
        dset.flip.axis = flip
        dset.jitter.contrast_factor = contrast
        dset.jitter.brightness_factor = brightness
        # note: dvs re-applies self.noise_std to the callable on every get_item,
        # so the field, not the callable, has to be zeroed
        dset.noise_std = 0.0
        dset.noise.std = 0.0
        dset.t_window_augment = False
        d = dset[seq]

        fset.rotate.n_rot = n_rot
        fset.flip.axis = 0 if flip is None else flip + 1
        fset.jitter.contrast_factor = contrast
        fset.jitter.brightness_factor = brightness
        fset.noise.std = 0.0
        fset.temporal_crop.random = False
        f = fset[seq]
        for key in ("lum", "flow"):
            rows.append(summarize(d[key], f[key], f"seq{seq}.rot{n_rot}.flip{flip}.{key}"))
    print_report(rows)


if __name__ == "__main__":
    main()
