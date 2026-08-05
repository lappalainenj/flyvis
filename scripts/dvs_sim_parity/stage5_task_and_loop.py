"""Stage 5: task/dataloader construction and training-loop control flow.

Compares the train and validation index sets, batch composition, loader options,
checkpoint cadence, and validation semantics -- everything that decides *which*
data the run sees and *which* checkpoint is finally selected.

Run:
    python stage5_task_and_loop.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import dvs_config, flyvis_config  # noqa: E402

import dvs  # noqa: E402
import flyvis  # noqa: E402


def main():
    dcfg, fcfg = dvs_config(), flyvis_config()

    from dvs.solver import _chkpt_every_epoch, _init_task
    from flyvis.task.tasks import Task as FlyvisTask

    dtask = _init_task(dcfg.task)
    ftask = FlyvisTask(**fcfg.task)

    print("== train / validation indices ==")
    dtr, dva = list(dtask.train_seq_index), list(dtask.val_seq_index)
    ftr, fva = list(ftask.train_seq_index), list(ftask.val_seq_index)
    print(f"  train: dvs n={len(dtr)} flyvis n={len(ftr)} identical={dtr == ftr}")
    if dtr != ftr:
        print(f"   dvs   : {dtr}")
        print(f"   flyvis: {ftr}")
    print(f"  val  : dvs n={len(dva)} flyvis n={len(fva)} identical={dva == fva}")
    if dva != fva:
        print(f"   dvs   : {dva}")
        print(f"   flyvis: {fva}")
        print(f"   set difference: dvs-only={sorted(set(dva) - set(fva))}"
              f" flyvis-only={sorted(set(fva) - set(dva))}")

    print("\n== dataloaders ==")
    for label, dl, fl in [
        ("train", dtask.train_data, ftask.train_data),
        ("val", dtask.val_data, ftask.val_data),
    ]:
        print(
            f"  {label}: batch_size dvs={dl.batch_size} flyvis={fl.batch_size}"
            f" | drop_last dvs={dl.drop_last} flyvis={fl.drop_last}"
            f" | n_batches dvs={len(dl)} flyvis={len(fl)}"
            f" | sampler dvs={type(dl.sampler).__name__} flyvis={type(fl.sampler).__name__}"
        )
    print(f"  tracked train batch: dvs={type(dtask.tracked_train_batch.sampler).__name__}"
          f" n={len(dtask.tracked_train_batch)}"
          f" | flyvis train_batch n={len(ftask.train_batch)}"
          f", val_batch n={len(ftask.val_batch)}")

    print("\n== iterations and epochs ==")
    n_iters_d, n_iters_f = dcfg.task.n_iters, fcfg.task.n_iters
    epochs_d = int(np.ceil(n_iters_d / len(dtask.train_data)))
    epochs_f = int(np.ceil(n_iters_f / len(ftask.train_data)))
    print(f"  dvs   : n_iters={n_iters_d} len(loader)={len(dtask.train_data)} epochs={epochs_d}")
    print(f"  flyvis: n_iters={n_iters_f} len(loader)={len(ftask.train_data)} epochs={epochs_f}")

    print("\n== checkpoint cadence ==")
    dvs_every = _chkpt_every_epoch(n_iters_d, len(dtask.train_data))
    fly_every = fcfg.scheduler.chkpt_every_epoch
    print(f"  dvs   : every {dvs_every} epochs -> {epochs_d // dvs_every + 1} checkpoints")
    print(f"  flyvis: every {fly_every} epochs -> {epochs_f // fly_every + 1} checkpoints")

    print("\n== loss wiring ==")
    print(f"  dvs    loss fn: {dtask.dataset.losses}")
    print(f"  dvs    task weights: {dtask.dataset.task_weights},"
          f" sum={dtask.dataset.task_weights_sum}")
    print(f"  flyvis loss fn: {dict(ftask.losses)}")
    print(f"  flyvis task weights: {ftask.task_weights}, sum={ftask.task_weights_sum}")

    print("\n== validation semantics ==")
    print("  dvs    solver.test(): t_pre default")
    import inspect

    from dvs.solver import MultiTaskSolver as DvsSolver
    from flyvis.solver import MultiTaskSolver as FlyvisSolver

    print(f"   dvs    test signature: {inspect.signature(DvsSolver.test)}")
    print(f"   flyvis test signature: {inspect.signature(FlyvisSolver.test)}")
    print(f"   dvs    train signature: {inspect.signature(DvsSolver.train)}")
    print(f"   flyvis train signature: {inspect.signature(FlyvisSolver.train)}")

    print("\n== t_pre_train ==")
    print(f"  dvs   : {dcfg.get('t_pre_train', None)}")
    print(f"  flyvis: {fcfg.get('t_pre_train', 'absent -> solver default 0.5')}")


if __name__ == "__main__":
    main()
