"""Check that a trial ensemble trained as intended.

Verifies, for every network of an ensemble, that the stored config records the
regime that was asked for, that the connectome it trained on really is a
degree-matched randomization, that the six networks trained on six *different*
connectomes, and that training actually progressed.

Example:
    python check_trial.py flow/0399
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

import flyvis
from flyvis.connectome import init_connectome

REFERENCE = {
    "type": "ConnectomeFromAvgFilters",
    "file": "fib25-fib19_v2.2.json",
    "extent": 15,
    "n_syn_fill": 1,
}


def degrees(connectome, n_nodes):
    e = connectome.edges
    return (
        np.bincount(e.source_index[:], minlength=n_nodes),
        np.bincount(e.target_index[:], minlength=n_nodes),
    )


def edge_keys(connectome, n_nodes):
    e = connectome.edges
    keys = e.source_index[:].astype(np.int64) * n_nodes + e.target_index[:]
    return set(keys.tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ensemble", help="e.g. flow/0399")
    p.add_argument("--expect-iters", type=int, default=None)
    args = p.parse_args()

    ensemble_dir = Path(flyvis.results_dir) / args.ensemble
    networks = sorted(d for d in ensemble_dir.iterdir() if d.name.isdigit())
    if not networks:
        raise SystemExit(f"no networks found in {ensemble_dir}")

    reference = init_connectome(**REFERENCE)
    n_nodes = len(reference.nodes.type[:])
    ref_out, ref_in = degrees(reference, n_nodes)
    ref_keys = edge_keys(reference, n_nodes)

    print(f"reference: {len(ref_keys)} edges, {n_nodes} nodes")
    print(f"{len(networks)} networks in {args.ensemble}\n")

    per_net_keys = {}
    failures = []

    for net_dir in networks:
        name = f"{args.ensemble}/{net_dir.name}"
        meta = yaml.safe_load((net_dir / "_meta.yaml").read_text())
        config = meta["config"]
        cc = config["network"]["connectome"]

        problems = []

        # -- the regime that was asked for --------------------------------
        if cc.get("type") != "ConnectomeWithRewiredEdges":
            problems.append(f"connectome type is {cc.get('type')}")
        seed = cc.get("seed")
        if seed != int(net_dir.name):
            problems.append(f"seed {seed} != network id {int(net_dir.name)}")
        if not config["task"]["dataset"].get("original_sampling"):
            problems.append("original_sampling is not set")
        if config["scheduler"].get("chkpt_every_epoch") is None:
            problems.append("no chkpt_every_epoch")
        stop_iter = config["scheduler"].get("sched_stop_iter")
        n_iters = config["task"]["n_iters"]
        if stop_iter is None or stop_iter > n_iters:
            problems.append(f"sched_stop_iter {stop_iter} > n_iters {n_iters}")
        if args.expect_iters and n_iters != args.expect_iters:
            problems.append(f"n_iters {n_iters} != {args.expect_iters}")

        # -- is the connectome really degree-matched? ----------------------
        connectome = init_connectome(**cc)
        out_deg, in_deg = degrees(connectome, n_nodes)
        if not np.array_equal(in_deg, ref_in):
            problems.append("in-degree differs from the reference")
        if not np.array_equal(out_deg, ref_out):
            problems.append("out-degree differs from the reference")

        keys = edge_keys(connectome, n_nodes)
        if len(keys) != len(ref_keys):
            problems.append(f"{len(keys)} edges != {len(ref_keys)}")
        retained = len(keys & ref_keys) / len(ref_keys)
        per_net_keys[net_dir.name] = keys

        # -- did training progress? ----------------------------------------
        chkpts = sorted((net_dir / "chkpts").glob("chkpt_*"))
        loss_file = net_dir / "loss.h5"
        losses = None
        if loss_file.exists():
            import h5py

            with h5py.File(loss_file, "r") as h:
                losses = np.atleast_1d(h["data"][()])
        if not chkpts:
            problems.append("no checkpoints written")
        if losses is None or losses.size == 0:
            problems.append("no loss recorded")

        # -- do the free parameters still match FlyNet's shape? ------------
        if chkpts:
            chkpt = torch.load(chkpts[-1], map_location="cpu", weights_only=False)
            shapes = {k: tuple(v.shape) for k, v in chkpt["network"].items()}
            expected = {
                "nodes_bias": (65,),
                "nodes_time_const": (65,),
                "edges_syn_strength": (604,),
            }
            for k, want in expected.items():
                if shapes.get(k) != want:
                    problems.append(f"{k} shape {shapes.get(k)} != {want}")

        status = "OK  " if not problems else "FAIL"
        loss_str = (
            f"loss {losses[0]:.1f} -> {losses[-1]:.1f} ({losses.size} recorded)"
            if losses is not None and losses.size
            else "loss missing"
        )
        print(
            f"{status} {name}  seed {seed}  retained {retained:.5f} of reference edges"
            f"  {len(chkpts)} chkpts  {loss_str}"
        )
        for problem in problems:
            print(f"       - {problem}")
            failures.append((name, problem))

    # -- did the networks train on different connectomes? ------------------
    print("\npairwise edge overlap between networks:")
    names = list(per_net_keys)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            na, nb = names[a], names[b]
            overlap = len(per_net_keys[na] & per_net_keys[nb]) / len(ref_keys)
            flag = "" if overlap < 0.05 else "   <-- suspiciously similar"
            print(f"  {na} vs {nb}: {overlap:.5f}{flag}")
            if overlap > 0.05:
                failures.append((f"{na}/{nb}", "connectomes too similar"))

    print()
    if failures:
        print(f"{len(failures)} problem(s) found")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
