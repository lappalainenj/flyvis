"""Stage 6: end-to-end training trajectory, dvs-sim vs flyvis.

Runs the real training loops of both repos for a handful of iterations and
compares the per-iteration loss and the parameters afterwards. Augmentation is
switched off and the sampler order is pinned, so the trajectory is deterministic
and any residual difference is a difference in the training code, not in a random
draw.

Compares three configurations:
  * flyvis with public defaults,
  * flyvis with `task=task_original scheduler=scheduler_original`,
  * dvs-sim.

Run:
    python stage6_trajectory.py [n_iterations]
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import dvs_config, flyvis_config, param_tensors, print_report, summarize  # noqa: E402

import dvs  # noqa: E402
import flyvis  # noqa: E402

WORK = Path("/tmp/dvscmp")
N_ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 24


def force_no_augmentation(dataset):
    """Both train loops re-enable augmentation themselves, so pin it here."""
    from contextlib import contextmanager

    dataset.augment = False

    @contextmanager
    def never_augment(abool):
        dataset.augment = False
        yield

    dataset.augmentation = never_augment
    return dataset


def pin_loader(task):
    """Replace the shuffling train sampler by a deterministic one."""
    from torch.utils.data import DataLoader

    from flyvis.utils.dataset_utils import IndexSampler

    index = list(task.train_seq_index)
    task.train_data = DataLoader(
        task.dataset, batch_size=4, sampler=IndexSampler(index), drop_last=True
    )
    return task


def run_flyvis(overrides, tag, n_iters):
    from flyvis.solver import MultiTaskSolver

    cfg = flyvis_config([
        *overrides,
        f"task.n_iters={n_iters}",
        # dropout is the only remaining rng consumer once augmentation is off;
        # disabling it makes the trajectory fully deterministic
        "task.decoder.flow.p_dropout=0",
    ])
    cfg["network_name"] = f"parity/{tag}"
    if "sched_stop_iter" in cfg.scheduler:
        # the 200k schedule stop only makes sense for a full run; for the short
        # trajectory check both stacks stop scheduling at n_iters
        cfg.scheduler.sched_stop_iter = n_iters
    solver = MultiTaskSolver(f"parity/{tag}", cfg, delete_if_exists=True)
    # validation at every checkpoint is not what is compared here
    solver.checkpoint = lambda *a, **k: None
    pin_loader(solver.task)
    force_no_augmentation(solver.task.dataset)
    solver.task.dataset.dt = 0.02
    torch.manual_seed(0)
    solver.train(initial_checkpoint=False)
    losses = np.array(solver.dir.loss[:])
    return losses, param_tensors(solver.network), param_tensors(solver.decoder["flow"]), solver


def run_dvs(tag, n_iters):
    import dvs.solver as dsolver

    cfg = dvs_config([
        f"solver.task.n_iters={n_iters}",
        f"solver.sched_stop_iter={n_iters}",
        "solver.task.decoder.flow.p_dropout=0",
    ])
    with dvs.root_dir_context(WORK / "exp"):
        solver = dsolver.MultiTaskSolver(
            name=f"parity/{tag}", config=cfg, delete_if_exists=True
        )
        solver.checkpoint = lambda *a, **k: None
        pin_loader_dvs(solver.task)
        force_no_augmentation(solver.task.dataset)
        solver.task.dataset.dt = 0.02
        torch.manual_seed(0)
        solver.train(initial_checkpoint=False)
        losses = np.array([float(x) for x in solver.wrap.loss[:]])
    return losses, param_tensors(solver.network), param_tensors(solver.decoder["flow"]), solver


def pin_loader_dvs(task):
    from torch.utils.data import DataLoader

    from dvs import datasets as dvs_datasets

    index = list(task.train_seq_index)
    task.train_data = DataLoader(
        task.dataset,
        batch_size=4,
        sampler=dvs_datasets.IndexSampler(index),
        drop_last=True,
    )
    return task


def main():
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"== running {N_ITERS} iterations, augmentation off, deterministic sampler ==")
    print("\n-- dvs-sim --")
    d_loss, d_net, d_dec, dsolv = run_dvs("dvs", N_ITERS)
    print("\n-- flyvis public defaults --")
    f_loss, f_net, f_dec, fsolv = run_flyvis([], "flyvis_default", N_ITERS)
    print("\n-- flyvis with original task/scheduler --")
    c_loss, c_net, c_dec, csolv = run_flyvis(
        ["task=task_original", "scheduler=scheduler_original"], "flyvis_original", N_ITERS
    )

    print("\n== per-iteration loss ==")
    print("  iter        dvs-sim   flyvis default   flyvis original")
    for i in range(min(len(d_loss), len(f_loss), len(c_loss))):
        print(f"  {i:>4}   {d_loss[i]:>12.6f}   {f_loss[i]:>12.6f}   {c_loss[i]:>12.6f}")

    print("\n== trajectory agreement with dvs-sim ==")
    n = min(len(d_loss), len(f_loss), len(c_loss))
    print_report([
        summarize(torch.tensor(d_loss[:n]), torch.tensor(f_loss[:n]), "loss: flyvis default"),
        summarize(torch.tensor(d_loss[:n]), torch.tensor(c_loss[:n]), "loss: flyvis original"),
    ])

    print("\n== network parameters after training ==")
    print(" flyvis default vs dvs-sim:")
    print_report([summarize(d_net[k], f_net[k], k) for k in d_net])
    print(" flyvis original vs dvs-sim:")
    print_report([summarize(d_net[k], c_net[k], k) for k in d_net])

    print("\n== decoder parameters after training ==")
    print(" flyvis original vs dvs-sim:")
    print_report([summarize(d_dec[k], c_dec[k], k) for k in d_dec])


if __name__ == "__main__":
    main()
