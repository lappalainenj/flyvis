"""Is the gap to the published models a training deficit or a selection artifact?

The published checkpoints were chosen by `argmin` EPE in a retrospective
re-validation pass -- see `validation_loss_fn = "epe"` in
/Users/janne.lappalainen/Projects/dvs-sim/scripts/cleanup/ensemble_to_flyvision.py.
Ensembles trained here record only the training loss (l2norm) during validation,
so flyvis's best-checkpoint function falls back to it. Both are then compared on
EPE, which favours the ensemble that was selected on EPE.

This scores every checkpoint of a sample of networks on the common protocol and
reports, per network:

  * the EPE of the checkpoint the l2norm-based selection picks (what we compare)
  * the EPE of the checkpoint an EPE-based selection would pick
  * the difference, i.e. what the selection metric costs

If that difference accounts for the ensemble-level gap, the residual is a
selection artifact and the fix is to record EPE during validation.

Example:
    python checkpoint_selection.py --ensemble flow/0310 --networks 12
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from evaluate_ensemble import common_dataloader, evaluate

import flyvis
from flyvis.utils.chkpt_utils import recover_decoder, recover_network


def read_h5(path):
    with h5py.File(path, "r") as h:
        return np.atleast_1d(h["data"][()])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble", default="flow/0310")
    p.add_argument("--networks", type=int, default=12, help="how many members")
    p.add_argument("--stride", type=int, default=1, help="evaluate every Nth checkpoint")
    p.add_argument("--val-set", default="original")
    p.add_argument(
        "--reference-gap",
        type=float,
        default=None,
        help="ensemble-level EPE gap to the published models, to express the "
        "selection cost as a fraction of",
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    root = Path(flyvis.results_dir) / args.ensemble
    names = sorted(d.name for d in root.iterdir() if d.name.isdigit())
    names = [
        n
        for n in names
        if (root / n / "loss.h5").exists()
        and read_h5(root / n / "loss.h5").size >= 250000
    ][: args.networks]

    loader, dt, val_index = common_dataloader(f"{args.ensemble}/{names[0]}", args.val_set)
    print(f"protocol: val_set={args.val_set}, {len(val_index)} sequences\n")

    rows = []
    print(
        f"{'net':>5} {'chkpts':>6} {'l2-selected':>12} {'EPE-selected':>13} "
        f"{'cost':>8} {'l2 idx':>7} {'epe idx':>8}"
    )
    for name in names:
        view = flyvis.NetworkView(f"{args.ensemble}/{name}")
        net, dec = view.init_network(), view.init_decoder()
        chkpts = sorted((root / name / "chkpts").glob("chkpt_*"))
        val = read_h5(root / name / "validation" / "loss.h5")

        indices = list(range(0, len(chkpts), args.stride))
        epes = {}
        for i in indices:
            chkpt = torch.load(chkpts[i], map_location="cpu", weights_only=False)
            recover_network(net, chkpt)
            recover_decoder(dec, chkpt)
            epes[i] = evaluate(net, dec, loader, dt)["epe"]

        # what the l2norm-based selection picks, restricted to evaluated indices
        l2_idx = min(indices, key=lambda i: val[i] if i < len(val) else np.inf)
        epe_idx = min(epes, key=epes.get)
        cost = epes[l2_idx] - epes[epe_idx]
        rows.append({
            "network": name,
            "n_checkpoints_scored": len(indices),
            "l2_selected_index": int(l2_idx),
            "epe_selected_index": int(epe_idx),
            "epe_at_l2_selected": epes[l2_idx],
            "epe_at_epe_selected": epes[epe_idx],
            "selection_cost": cost,
        })
        print(
            f"{name:>5} {len(indices):>6} {epes[l2_idx]:>12.4f} {epes[epe_idx]:>13.4f} "
            f"{cost:>8.4f} {l2_idx:>7} {epe_idx:>8}"
        )

    c = np.array([r["selection_cost"] for r in rows])
    a = np.array([r["epe_at_l2_selected"] for r in rows])
    b = np.array([r["epe_at_epe_selected"] for r in rows])
    print(
        f"\nselection cost: mean {c.mean():.4f}  sd {c.std(ddof=1):.4f}  "
        f"median {np.median(c):.4f}  max {c.max():.4f}"
    )
    print(f"mean EPE  l2-selected {a.mean():.4f}   EPE-selected {b.mean():.4f}")
    if args.reference_gap:
        print(
            f"\nensemble-level gap to the published models: "
            f"+{args.reference_gap:.4f} EPE"
        )
        print(
            f"selecting on EPE instead would recover {c.mean():.4f} of it "
            f"({100 * c.mean() / args.reference_gap:.0f}%)"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
