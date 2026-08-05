"""Stage 7: how big are the two data differences, in units the reader cares about?

Reports, over the whole dataset:
  * how much the input movie differs frame by frame,
  * how much the flow target differs, relative to the target's own scale,
  * what the loss of one and the same network is on the two versions of the data,
  * how much the gradient direction changes.

Run:
    python stage7_effect_size.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import dvs_config, flyvis_config  # noqa: E402
from stage2_dataset import build_datasets  # noqa: E402
from stage2b_dataset_fix import patch_flyvis_to_dvs_semantics  # noqa: E402

import flyvis  # noqa: E402

SEQS = list(range(0, 69))
BATCH = [0, 1, 2, 3]


def main():
    dset, fset, dcfg, fcfg = build_datasets()

    print("== input and target differences over all 69 sequences (augmentation off) ==")
    lum_rel, flow_rel, flow_abs, tgt_scale = [], [], [], []
    with dset.augmentation(False), fset.augmentation(False):
        for seq in SEQS:
            d, f = dset[seq], fset[seq]
            lum_rel.append(
                ((d["lum"] - f["lum"]).norm() / d["lum"].norm()).item()
            )
            flow_rel.append(
                ((d["flow"] - f["flow"]).norm() / d["flow"].norm()).item()
            )
            flow_abs.append((d["flow"] - f["flow"]).abs().max().item())
            tgt_scale.append(d["flow"].abs().mean().item())
    print(f"  input  relative L2 difference: median {np.median(lum_rel):.4f}"
          f"  max {np.max(lum_rel):.4f}")
    print(f"  target relative L2 difference: median {np.median(flow_rel):.4f}"
          f"  max {np.max(flow_rel):.4f}")
    print(f"  target max abs difference:     median {np.median(flow_abs):.4f}"
          f"  max {np.max(flow_abs):.4f}")
    print(f"  target mean abs value:         median {np.median(tgt_scale):.4f}")
    frac = np.median(flow_abs) / np.median(tgt_scale)
    print(f"  target worst-frame error / typical target magnitude: {frac:.2f}x")

    print("\n== same network, two versions of the data ==")
    from flyvis.network.network import Network as FlyvisNetwork
    from flyvis.task.decoder import init_decoder
    from flyvis.task import objectives

    net = FlyvisNetwork(**fcfg.network)
    dec = init_decoder(fcfg.task.decoder.flow, net.connectome)
    net.eval()
    dec.eval()

    def loss_and_grad(batch):
        net.zero_grad()
        dec.zero_grad()
        state = net.steady_state(t_pre=0.5, dt=0.02, batch_size=len(BATCH), value=0.5)
        net.stimulus.zero(*batch["lum"].shape[:2])
        net.stimulus.add_input(batch["lum"])
        activity = net(net.stimulus(), 0.02, state=state)
        loss = objectives.l2norm(dec(activity), batch["flow"])
        loss.backward()
        grad = torch.cat([
            p.grad.flatten() for p in list(net.parameters()) + list(dec.parameters())
            if p.grad is not None
        ])
        return loss.item(), grad.clone()

    def make_batch(dataset):
        with dataset.augmentation(False):
            items = [dataset[i] for i in BATCH]
        get = lambda it, k: it[k]  # noqa: E731
        return {
            k: torch.stack([get(it, k) for it in items], dim=0) for k in ("lum", "flow")
        }

    fly_batch = make_batch(fset)
    dvs_batch = make_batch(dset)
    fly_loss, fly_grad = loss_and_grad(fly_batch)
    dvs_loss, dvs_grad = loss_and_grad(dvs_batch)
    print(f"  loss on flyvis data:  {fly_loss:.6f}")
    print(f"  loss on dvs-sim data: {dvs_loss:.6f}")
    print(f"  difference:           {abs(fly_loss - dvs_loss):.6f}"
          f"  ({abs(fly_loss - dvs_loss) / dvs_loss * 100:.3f}% of the loss)")
    cos = torch.nn.functional.cosine_similarity(fly_grad, dvs_grad, dim=0).item()
    print(f"  gradient cosine similarity: {cos:.6f}")
    print(f"  gradient relative L2 difference:"
          f" {((fly_grad - dvs_grad).norm() / dvs_grad.norm()).item():.6f}")

    print("\n  for scale, the same comparison between two genuinely different batches:")
    with dset.augmentation(False):
        items = [dset[i] for i in (4, 5, 6, 7)]
    other = {
        k: torch.stack([it[k] for it in items], dim=0) for k in ("lum", "flow")
    }
    other_loss, other_grad = loss_and_grad(other)
    print(f"  loss on a different batch: {other_loss:.6f}"
          f"  (difference {abs(other_loss - dvs_loss):.6f})")
    print("  gradient cosine similarity between different batches:"
          f" {torch.nn.functional.cosine_similarity(other_grad, dvs_grad, dim=0).item():.6f}")


if __name__ == "__main__":
    main()
