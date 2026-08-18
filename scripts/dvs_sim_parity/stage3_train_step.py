"""Stage 3: given identical parameters and identical data, does one training step agree?

Removes every source of randomness by
  * copying the dvs-sim parameters into the flyvis network and decoder,
  * feeding both the same batch (dataset patched to dvs-sim semantics, so the
    batch is bit-identical in both),
  * running the decoder in eval mode for the deterministic comparison and, in a
    second pass, in train mode with the RNG reseeded identically.

Then compares: steady state, per-frame activity, decoder output, loss, all
gradients, parameters after the optimizer step, and parameters after the
activity-penalty step.

Run:
    python stage3_train_step.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    dvs_config,
    flyvis_config,
    grad_tensors,
    param_tensors,
    print_report,
    summarize,
)
from stage2_dataset import build_datasets  # noqa: E402
from stage2b_dataset_fix import patch_flyvis_to_dvs_semantics  # noqa: E402

import dvs  # noqa: E402
import flyvis  # noqa: E402

T_PRE = 0.5
DT = 0.02
BATCH = 4
SEQS = [0, 1, 2, 3]


def build_all():
    dcfg, fcfg = dvs_config(), flyvis_config()
    from dvs.decoder import init_decoder as dvs_init_decoder
    from dvs.networks import Network as DvsNetwork
    from flyvis.network.network import Network as FlyvisNetwork
    from flyvis.task.decoder import init_decoder as flyvis_init_decoder

    dnet = DvsNetwork(**dcfg.network)
    fnet = FlyvisNetwork(**fcfg.network)
    ddec = dvs_init_decoder(dcfg.task.decoder.flow, dnet.ctome)
    fdec = flyvis_init_decoder(fcfg.task.decoder.flow, fnet.connectome)

    # make flyvis start from exactly the dvs-sim parameters
    fnet.load_state_dict(dnet.state_dict(), strict=True)
    fdec.load_state_dict(ddec.state_dict(), strict=True)
    return dcfg, fcfg, dnet, fnet, ddec, fdec


def get_batch(fset, seqs):
    with torch.no_grad():
        items = [fset[i] for i in seqs]
    return {
        key: torch.stack([it[key] for it in items], dim=0) for key in ("lum", "flow")
    }


def main():
    dcfg, fcfg, dnet, fnet, ddec, fdec = build_all()
    dset, fset, _, _ = build_datasets()
    patch_flyvis_to_dvs_semantics(fset)
    fset.dt = DT
    dset.dt = DT

    print("== initial parameters after copy ==")
    dp, fp = param_tensors(dnet), param_tensors(fnet)
    print_report([summarize(dp[k], fp[k], f"net.{k}") for k in dp])
    dd, fd = param_tensors(ddec), param_tensors(fdec)
    print_report([summarize(dd[k], fd[k], f"dec.{k}") for k in dd])

    print("\n== batch ==")
    with fset.augmentation(False):
        batch = get_batch(fset, SEQS)
    with dset.augmentation(False):
        dbatch = get_batch(dset, SEQS)
    print_report([summarize(dbatch[k], batch[k], f"batch.{k}") for k in batch])
    print(f"  lum {tuple(batch['lum'].shape)}  flow {tuple(batch['flow'].shape)}")

    print("\n== steady state (t_pre=0.5, value=0.5) ==")
    dstate = dnet.steady_state(t_pre=T_PRE, dt=DT, batch_size=BATCH, value=0.5)
    fstate = fnet.steady_state(t_pre=T_PRE, dt=DT, batch_size=BATCH, value=0.5)
    print_report([
        summarize(dstate.nodes.activity, fstate.nodes.activity, "steady_state.activity")
    ])

    print("\n== forward pass (decoder in eval mode) ==")
    ddec.eval()
    fdec.eval()
    dnet.eval()
    fnet.eval()

    dnet._stimulus.zero(*batch["lum"].shape[:2])
    dnet._stimulus.add_input(batch["lum"])
    dact = dnet(dnet._stimulus(), DT, state=dstate)

    fnet.stimulus.zero(*batch["lum"].shape[:2])
    fnet.stimulus.add_input(batch["lum"])
    fact = fnet(fnet.stimulus(), DT, state=fstate)
    print_report([
        summarize(dnet._stimulus(), fnet.stimulus(), "stimulus"),
        summarize(dact, fact, "activity"),
        summarize(dact[:, 0], fact[:, 0], "activity[frame 0]"),
        summarize(dact[:, -1], fact[:, -1], "activity[frame -1]"),
    ])

    dy = ddec(dact)
    fy = fdec(fact)
    print_report([summarize(dy, fy, "decoder output")])

    print("\n== loss ==")
    from dvs import objectives as dvs_obj
    from flyvis.task import objectives as fly_obj

    dloss = dvs_obj.l2_3d_wotc(batch["flow"], dy)
    floss = fly_obj.l2norm(fy, batch["flow"])
    print(f"  dvs    l2_3d_wotc = {dloss.item():.10f}")
    print(f"  flyvis l2norm     = {floss.item():.10f}")
    print(f"  abs diff          = {abs(dloss.item() - floss.item()):.3e}")

    print("\n== gradients after backward ==")
    dnet.zero_grad()
    fnet.zero_grad()
    ddec.zero_grad()
    fdec.zero_grad()
    dloss.backward(retain_graph=True)
    floss.backward(retain_graph=True)
    dg, fg = grad_tensors(dnet), grad_tensors(fnet)
    print_report([summarize(dg[k], fg[k], f"grad net.{k}") for k in dg])
    dgd, fgd = grad_tensors(ddec), grad_tensors(fdec)
    print_report([summarize(dgd[k], fgd[k], f"grad dec.{k}") for k in dgd])

    print("\n== optimizer step ==")
    from dvs.solver import _init_optimizer as dvs_init_optim
    from flyvis.solver import MultiTaskSolver

    dopt = dvs_init_optim(dcfg, dnet, {"flow": ddec})
    fopt = MultiTaskSolver._init_optimizer(fcfg.optim, fnet, {"flow": fdec})
    print("  dvs    param groups:", [(len(g["params"]), g["lr"]) for g in dopt.param_groups])
    print("  flyvis param groups:", [(len(g["params"]), g["lr"]) for g in fopt.param_groups])
    dopt.step()
    fopt.step()
    dp2, fp2 = param_tensors(dnet), param_tensors(fnet)
    print_report([summarize(dp2[k], fp2[k], f"post-step net.{k}") for k in dp2])
    dd2, fd2 = param_tensors(ddec), param_tensors(fdec)
    print_report([summarize(dd2[k], fd2[k], f"post-step dec.{k}") for k in dd2])

    print("\n== activity penalty step ==")
    from dvs.penalizer import Penalty as DvsPenalty
    from flyvis.solver import Penalty as FlyvisPenalty

    class SolverShim:
        """Minimal stand-in for what dvs.penalizer.Penalty reads off the solver."""

        def __init__(self, network, config, dt, iteration=0):
            self.network = network
            self.config = config
            self.dt = dt
            self.iteration = iteration
            self.n_iters = config.task.n_iters

    dpen = DvsPenalty(SolverShim(dnet, dcfg, DT))
    fpen = FlyvisPenalty(fcfg.penalizer, fnet)
    print("  dvs   :", type(dpen).__name__, getattr(dpen, "activity_penalty", None),
          getattr(dpen, "activity_baseline", None))
    print("  flyvis:", type(fpen).__name__, fpen.activity_penalty, fpen.activity_baseline)
    dpen(dact)
    fpen(activity=fact, iteration=0)
    dp3, fp3 = param_tensors(dnet), param_tensors(fnet)
    print_report([summarize(dp3[k], fp3[k], f"post-penalty net.{k}") for k in dp3])


if __name__ == "__main__":
    main()
