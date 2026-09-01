"""Evaluate ensembles on one common protocol.

The validation losses stored with a model are not comparable across ensembles:
the published models ship a single value on a different normalization from what
training records. This script instead runs every network through the *same*
dataset and the same held-out sequences, so the numbers can be compared.

Includes two reference points that decide whether a network learned anything:

  * `zero` -- predicting no flow at all. Any network that does not beat this has
    learned nothing useful.
  * `init` -- a network of the ensemble at its first checkpoint, i.e. before
    training.

Example:
    python evaluate_ensemble.py --ensembles flow/0300 flow/0000 --limit 6
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import flyvis
from flyvis.datasets import MultiTaskDataset
from flyvis.task.objectives import epe, l2norm
from flyvis.utils.class_utils import forward_subclass
from flyvis.utils.dataset_utils import IndexSampler

T_PRE = 0.5


def common_dataloader(reference_network, val_set="config"):
    """One dataset and one held-out split, used for every network compared.

    Which sequences are held out is not a detail here. The published models'
    stored config has no `original_split` key, so it defaults to the fold split,
    whereas ensembles trained here set `original_split=True`. Those two splits
    share only 6 sequences: 10 of the original split's 16 validation sequences
    are *training* data under the fold split, and 11 of the fold split's 17 are
    training data under the original split. Evaluating on either one alone
    therefore favours whichever ensemble did not train on it.

    `intersection` is the only set held out under both, and so the only fair one
    when the two ensembles were trained under different splits.
    """
    task_config = flyvis.NetworkView(reference_network).dir.config.task
    dataset = forward_subclass(MultiTaskDataset, task_config.dataset)
    _, original = dataset.original_train_and_validation_indices()
    _, fold = dataset.get_random_data_split(
        task_config.fold, task_config.n_folds, task_config.seed
    )
    original, fold = [int(i) for i in original], [int(i) for i in fold]
    if val_set == "original":
        val_index = original
    elif val_set == "fold":
        val_index = fold
    elif val_set == "intersection":
        val_index = sorted(set(original) & set(fold))
    elif val_set == "config":
        val_index = original if task_config.get("original_split", False) else fold
    else:
        raise ValueError(f"unknown val_set {val_set!r}")
    loader = DataLoader(dataset, batch_size=1, sampler=IndexSampler(val_index))
    return loader, dataset.dt, val_index


@torch.no_grad()
def evaluate(net, decoder, dataloader, dt):
    dataset = dataloader.dataset
    net.eval()
    for d in decoder.values():
        d.eval()
    steady_state = net.steady_state(
        t_pre=T_PRE, dt=dt, batch_size=1, value=0.5, state=None, grad=False
    )
    l2s, epes, preds, gts = [], [], [], []
    max_abs_activity = 0.0
    with dataset.augmentation(False):
        for data in dataloader:
            n_samples, n_frames = data["lum"].shape[:2]
            net.stimulus.zero(n_samples, n_frames)
            net.stimulus.add_input(data["lum"])
            activity = net(net.stimulus(), dt, state=steady_state)
            max_abs_activity = max(max_abs_activity, float(activity.abs().max()))
            y_est = decoder["flow"](activity)
            l2s.append(l2norm(y_est, data["flow"]).item())
            epes.append(epe(y_est, data["flow"]).item())
            preds.append(y_est.cpu().numpy())
            gts.append(data["flow"].cpu().numpy())
    e = np.concatenate([p.ravel() for p in preds])
    g = np.concatenate([p.ravel() for p in gts])
    return {
        "l2norm": float(np.mean(l2s)),
        "epe": float(np.mean(epes)),
        "epe_sem": float(np.std(epes, ddof=1) / np.sqrt(len(epes))),
        "corr_pred_gt": float(np.corrcoef(e, g)[0, 1]),
        "pred_std": float(e.std()),
        "gt_std": float(g.std()),
        "max_abs_activity": max_abs_activity,
    }


@torch.no_grad()
def zero_baseline(dataloader):
    """What predicting no flow at all scores."""
    dataset = dataloader.dataset
    l2s, epes, gts = [], [], []
    with dataset.augmentation(False):
        for data in dataloader:
            y = torch.zeros_like(data["flow"])
            l2s.append(l2norm(y, data["flow"]).item())
            epes.append(epe(y, data["flow"]).item())
            gts.append(data["flow"].cpu().numpy())
    g = np.concatenate([p.ravel() for p in gts])
    return {
        "l2norm": float(np.mean(l2s)),
        "epe": float(np.mean(epes)),
        "epe_sem": float(np.std(epes, ddof=1) / np.sqrt(len(epes))),
        "corr_pred_gt": float("nan"),
        "pred_std": 0.0,
        "gt_std": float(g.std()),
        "max_abs_activity": float("nan"),
    }


def networks_of(ensemble, limit=None, min_iterations=None):
    """Members of an ensemble, optionally only those that finished training.

    An ensemble still training has members with partial checkpoints; including
    them would silently mix half-trained networks into the statistics.

    The filter only applies where training records exist. The published models
    ship no `loss.h5` at all -- only the best checkpoint -- so for them the
    absence of a training record means "not recorded", not "not finished", and
    excluding them would silently drop the entire comparison group.
    """
    root = Path(flyvis.results_dir) / ensemble
    nets = sorted(d.name for d in root.iterdir() if d.name.isdigit())
    if min_iterations:
        import h5py

        recorded = [n for n in nets if (root / n / "loss.h5").exists()]
        if not recorded:
            print(
                f"  {ensemble}: no training records, keeping all {len(nets)} "
                f"member(s) (published models ship only the best checkpoint)"
            )
            return nets[:limit] if limit else nets

        complete = []
        for name in nets:
            loss = root / name / "loss.h5"
            if not loss.exists():
                # mixed ensemble: a member with no record cannot be verified
                print(f"  {ensemble}/{name}: no training record, skipping")
                continue
            with h5py.File(loss, "r") as h:
                if np.atleast_1d(h["data"][()]).size >= min_iterations:
                    complete.append(name)
        skipped = len(nets) - len(complete)
        if skipped:
            print(f"  {ensemble}: skipping {skipped} member(s) still training")
        nets = complete
    return nets[:limit] if limit else nets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensembles", nargs="+", default=["flow/0300", "flow/0000"])
    p.add_argument(
        "--protocol-from",
        default=None,
        help="network whose task config defines the common dataset and split; "
        "defaults to the first network of the first ensemble",
    )
    p.add_argument(
        "--val-set",
        default="config",
        choices=["config", "original", "fold", "intersection"],
        help="which sequences to hold out; `intersection` is the only set held "
        "out under both the original and the fold split",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--min-iterations",
        type=int,
        default=None,
        help="only evaluate members whose training reached this many "
        "iterations, so a partly finished ensemble is not mixed in",
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    complete = networks_of(args.ensembles[0], min_iterations=args.min_iterations)
    first = f"{args.ensembles[0]}/{complete[0]}"
    protocol_from = args.protocol_from or first
    loader, dt, val_index = common_dataloader(protocol_from, args.val_set)
    print(
        f"protocol from {protocol_from}: val_set={args.val_set}, "
        f"{len(val_index)} sequences, dt={dt}"
    )
    print(f"validation indices: {list(val_index)}\n")

    results = {
        "protocol_from": protocol_from,
        "val_set": args.val_set,
        "n_val_sequences": len(val_index),
        "val_indices": list(val_index),
    }

    base = zero_baseline(loader)
    print(
        f"{'model':>22}  {'l2norm':>9}  {'EPE':>7}  {'corr':>6}  "
        f"{'pred_std':>8}  {'max|act|':>9}"
    )
    print(
        f"{'zero prediction':>22}  {base['l2norm']:>9.2f}  {base['epe']:>7.4f}  "
        f"{'--':>6}  {base['pred_std']:>8.4f}  {'--':>9}"
    )
    results["zero"] = base

    for ensemble in args.ensembles:
        for name in networks_of(ensemble, args.limit, args.min_iterations):
            full = f"{ensemble}/{name}"
            view = flyvis.NetworkView(full)
            net, dec = view.init_network(), view.init_decoder()
            res = evaluate(net, dec, loader, dt)
            results[full] = res
            print(
                f"{full:>22}  {res['l2norm']:>9.2f}  {res['epe']:>7.4f}  "
                f"{res['corr_pred_gt']:>6.3f}  {res['pred_std']:>8.4f}  "
                f"{res['max_abs_activity']:>9.1f}"
            )

    # -- untrained reference from the first ensemble ------------------------
    name = complete[0]
    full = f"{args.ensembles[0]}/{name}"
    view = flyvis.NetworkView(full)
    chkpt0 = sorted((Path(view.dir.path) / "chkpts").glob("chkpt_*"))[0]
    net, dec = view.init_network(), view.init_decoder()
    chkpt = torch.load(chkpt0, map_location="cpu", weights_only=False)
    net.load_state_dict(chkpt["network"])
    for task, sd in chkpt["decoder"].items():
        dec[task].load_state_dict(sd)
    res = evaluate(net, dec, loader, dt)
    results[f"{full}@init"] = res
    print(
        f"{full + '@init':>22}  {res['l2norm']:>9.2f}  {res['epe']:>7.4f}  "
        f"{res['corr_pred_gt']:>6.3f}  {res['pred_std']:>8.4f}  "
        f"{res['max_abs_activity']:>9.1f}"
    )

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()
