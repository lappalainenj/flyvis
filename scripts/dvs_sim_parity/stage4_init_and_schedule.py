"""Stage 4: parameter initialization draws and hyperparameter schedules.

Stage 3 showed the compute path agrees once parameters and data match. What is
left is where the two repos *start* (random draws for the bias) and how they move
the learning rate over the run.

Run:
    python stage4_init_and_schedule.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import dvs_config, flyvis_config, print_report, summarize  # noqa: E402

import dvs  # noqa: E402
import flyvis  # noqa: E402


def main():
    dcfg, fcfg = dvs_config(), flyvis_config()
    from dvs.networks import Network as DvsNetwork
    from flyvis.network.network import Network as FlyvisNetwork

    print("== bias initialization (seed 0, N(0.5, 0.05) per cell type) ==")
    dnet = DvsNetwork(**dcfg.network)
    fnet = FlyvisNetwork(**fcfg.network)
    db = dnet.nodes_bias.detach()
    fb = fnet.nodes_bias.detach()
    print_report([summarize(db, fb, "nodes_bias")])
    print(f"  dvs    first 6: {db[:6].tolist()}")
    print(f"  flyvis first 6: {fb[:6].tolist()}")
    print(f"  dvs    mean={db.mean():.6f} std={db.std():.6f}")
    print(f"  flyvis mean={fb.mean():.6f} std={fb.std():.6f}")

    # is the flyvis draw reproducible from dvs's rng and vice versa?
    torch.manual_seed(0)
    global_draw = torch.normal(
        torch.full((65,), 0.5), torch.full((65,), 0.05)
    )
    gen = torch.Generator()
    gen.manual_seed(0)
    gen_draw = torch.normal(
        torch.full((65,), 0.5), torch.full((65,), 0.05), generator=gen
    )
    print(f"  matches global-RNG draw:    dvs={torch.allclose(db, global_draw)}"
          f" flyvis={torch.allclose(fb, global_draw)}")
    print(f"  matches Generator(0) draw:  dvs={torch.allclose(db, gen_draw)}"
          f" flyvis={torch.allclose(fb, gen_draw)}")

    print("\n== repeated construction (is the draw seeded at all?) ==")
    d2 = DvsNetwork(**dvs_config().network).nodes_bias.detach()
    f2 = FlyvisNetwork(**flyvis_config().network).nodes_bias.detach()
    print(f"  dvs    deterministic across constructions: {torch.allclose(db, d2)}")
    print(f"  flyvis deterministic across constructions: {torch.allclose(fb, f2)}")

    print("\n== scheduler configuration ==")
    print(f"  dvs    n_iters={dcfg.task.n_iters}  sched_stop_iter={dcfg.get('sched_stop_iter', None)}")
    print(f"  flyvis n_iters={fcfg.task.n_iters}  sched_stop_iter="
          f"{fcfg.scheduler.get('sched_stop_iter', None)}")
    print(f"  dvs    lr_net: {dict(dcfg.scheduler.lr_net)}")
    print(f"  flyvis lr_net: {dict(fcfg.scheduler.lr_net)}")
    print(f"  dvs    extra scheduled: relu_leak={dict(dcfg.scheduler.get('relu_leak', {}))}")
    print(f"  flyvis extra scheduled: relu_leak="
          f"{dict(fcfg.scheduler.get('relu_leak', {})) if 'relu_leak' in fcfg.scheduler else 'absent'}")
    print(f"  dvs    activity_penalty schedule: {dict(dcfg.scheduler.get('activity_penalty', {}))}")
    print(f"  flyvis activity_penalty schedule: "
          f"{dict(fcfg.scheduler.get('activity_penalty', {})) if 'activity_penalty' in fcfg.scheduler else 'absent'}")
    print(f"  dvs    chkpt_every={dcfg.get('chkpt_every', None)} (epochs, recomputed in solver)")
    print(f"  flyvis chkpt_every_epoch={fcfg.scheduler.chkpt_every_epoch}")

    print("\n== resulting learning-rate trajectory ==")
    from flyvis.solver import HyperParamScheduler

    def stepwise(stop_iter, n_iterations, start, stop, steps):
        f = np.linspace(start, stop, steps).repeat(stop_iter / steps)
        return np.pad(f, (0, n_iterations - len(f) + 1), constant_values=stop)

    n_iters = dcfg.task.n_iters
    dvs_lr = stepwise(dcfg.sched_stop_iter, n_iters, 5e-5, 5e-6, 10)
    fly_lr = stepwise(fcfg.task.n_iters, fcfg.task.n_iters, 5e-5, 5e-6, 10)
    probes = [0, 20_000, 25_000, 50_000, 100_000, 150_000, 199_999, 200_001, 249_999]
    print("  iteration      dvs lr      flyvis lr")
    for it in probes:
        print(f"  {it:>9}  {dvs_lr[it]:.3e}   {fly_lr[it]:.3e}")
    print(f"  identical trajectory: {np.allclose(dvs_lr[:n_iters], fly_lr[:n_iters])}")
    print(f"  mean lr over run: dvs={dvs_lr[:n_iters].mean():.4e}"
          f" flyvis={fly_lr[:n_iters].mean():.4e}")
    print(f"  total lr mass (sum): dvs={dvs_lr[:n_iters].sum():.4f}"
          f" flyvis={fly_lr[:n_iters].sum():.4f}")

    print("\n== validation split ==")
    from dvs.datasets.sintel import MultiTaskSintel as DvsSintel
    from flyvis.datasets.sintel import MultiTaskSintel as FlyvisSintel

    print(f"  dvs    original_validation={dcfg.task.get('original_validation', None)}")
    print(f"  flyvis original_split={fcfg.task.original_split}")


if __name__ == "__main__":
    main()
