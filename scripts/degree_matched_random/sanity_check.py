"""Sanity-check a trained degree-matched random ensemble.

Asks the questions that decide whether the runs are usable at all, before any
scientific interpretation:

1. did every network run to the end and write a full validation curve?
2. are the trained parameters finite and inside their allowed range?
3. did the validation loss actually come down, or did training stall?
4. did the six networks converge to *different* solutions, as six different
   connectomes should?
5. how does the ensemble compare to the reference ensemble trained on the real
   connectome?

Example:
    python sanity_check.py --ensemble flow/0300 --reference flow/0000
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import flyvis


def read_h5(path):
    if not Path(path).exists():
        return None
    with h5py.File(path, "r") as h:
        return np.atleast_1d(h["data"][()])


def networks_of(ensemble):
    root = Path(flyvis.results_dir) / ensemble
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if d.name.isdigit())


def best_validation(net_dir):
    """Validation curve, its minimum and where it occurred."""
    curve = read_h5(net_dir / "validation" / "loss.h5")
    if curve is None or curve.size == 0:
        return None, None, None
    finite = curve[np.isfinite(curve)]
    if finite.size == 0:
        return curve, None, None
    return curve, float(finite.min()), int(np.nanargmin(curve))


def checkpoint_health(net_dir):
    """Are the trained parameters finite and within their documented range?"""
    chkpts = sorted((net_dir / "chkpts").glob("chkpt_*"))
    if not chkpts:
        return {"error": "no checkpoints"}
    chkpt = torch.load(chkpts[-1], map_location="cpu", weights_only=False)
    out = {"n_chkpts": len(chkpts), "iteration": chkpt.get("iteration")}
    problems = []
    for key, tensor in chkpt["network"].items():
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            continue
        if not torch.isfinite(tensor).all():
            problems.append(f"{key} has non-finite values")
    # syn_strength is clamped non-negative by the config
    syn = chkpt["network"].get("edges_syn_strength")
    if syn is not None:
        out["syn_strength_min"] = float(syn.min())
        out["syn_strength_max"] = float(syn.max())
        if syn.min() < 0:
            problems.append(f"edges_syn_strength has negatives (min {syn.min():.3e})")
    tau = chkpt["network"].get("nodes_time_const")
    if tau is not None:
        out["time_const_min"] = float(tau.min())
        out["time_const_max"] = float(tau.max())
        if tau.min() <= 0:
            problems.append(f"nodes_time_const not positive (min {tau.min():.3e})")
    out["problems"] = problems
    return out


def trained_parameters(net_dir):
    chkpts = sorted((net_dir / "chkpts").glob("chkpt_*"))
    best = read_h5(net_dir / "best_chkpt_index.h5")
    index = int(best[0]) if best is not None and best.size else -1
    chkpt = torch.load(chkpts[index], map_location="cpu", weights_only=False)
    return {
        k: v.detach().clone()
        for k, v in chkpt["network"].items()
        if torch.is_tensor(v) and v.is_floating_point()
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble", default="flow/0300")
    p.add_argument("--reference", default="flow/0000")
    p.add_argument("--expect-iters", type=int, default=250000)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    nets = networks_of(args.ensemble)
    if not nets:
        raise SystemExit(f"no networks in {args.ensemble}")

    report = {"ensemble": args.ensemble, "networks": {}}
    failures = []

    print(f"== {args.ensemble}: {len(nets)} networks ==\n")
    print(
        f"{'net':>5}  {'chkpts':>6}  {'best val':>10}  {'at':>4}  {'final val':>10}  "
        f"{'val[0]':>10}  {'syn_strength range':>22}"
    )

    curves = {}
    for net_dir in nets:
        curve, best, at = best_validation(net_dir)
        health = checkpoint_health(net_dir)
        iters = read_h5(net_dir / "loss.h5")

        problems = list(health.get("problems", []))
        if health.get("error"):
            problems.append(health["error"])
        if curve is None:
            problems.append("no validation curve")
        if iters is not None and args.expect_iters and iters.size < args.expect_iters:
            problems.append(f"only {iters.size} training iterations recorded")
        # did it learn anything at all relative to the first validation?
        if curve is not None and best is not None and best >= curve[0]:
            problems.append(
                f"best validation {best:.1f} is no better than the first "
                f"({curve[0]:.1f})"
            )

        curves[net_dir.name] = curve
        srange = (
            f"[{health.get('syn_strength_min', float('nan')):.2e},"
            f" {health.get('syn_strength_max', float('nan')):.2e}]"
        )
        print(
            f"{net_dir.name:>5}  {health.get('n_chkpts', 0):>6}  "
            f"{best if best is not None else float('nan'):>10.2f}  {at:>4}  "
            f"{curve[-1] if curve is not None else float('nan'):>10.2f}  "
            f"{curve[0] if curve is not None else float('nan'):>10.2f}  {srange:>22}"
        )
        for problem in problems:
            print(f"       ! {problem}")
            failures.append((net_dir.name, problem))

        report["networks"][net_dir.name] = {
            "best_validation": best,
            "best_at_checkpoint": at,
            "first_validation": float(curve[0]) if curve is not None else None,
            "final_validation": float(curve[-1]) if curve is not None else None,
            "n_checkpoints": health.get("n_chkpts"),
            "n_training_iterations": int(iters.size) if iters is not None else None,
            "syn_strength_min": health.get("syn_strength_min"),
            "syn_strength_max": health.get("syn_strength_max"),
            "time_const_min": health.get("time_const_min"),
            "time_const_max": health.get("time_const_max"),
            "problems": problems,
        }

    # -- did the six converge to different solutions? ----------------------
    print("\n== do the networks differ from each other? ==")
    params = {d.name: trained_parameters(d) for d in nets}
    names = list(params)
    for key in ["nodes_bias", "nodes_time_const", "edges_syn_strength"]:
        dists = []
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                x, y = params[names[a]].get(key), params[names[b]].get(key)
                if x is None or y is None or x.shape != y.shape:
                    continue
                rel = (x - y).norm() / max(x.norm().item(), 1e-12)
                dists.append(rel.item())
        if dists:
            print(
                f"  {key:20s} pairwise relative L2: min {min(dists):.4f} "
                f"median {np.median(dists):.4f} max {max(dists):.4f}"
            )
            report.setdefault("pairwise_parameter_distance", {})[key] = {
                "min": min(dists),
                "median": float(np.median(dists)),
                "max": max(dists),
            }
            if max(dists) < 1e-6:
                failures.append((key, "all networks have identical parameters"))

    # -- did the validation loss come down at all? -------------------------
    print("\n== did validation improve over training? ==")
    for name, curve in curves.items():
        if curve is None:
            continue
        gain = (curve[0] - np.nanmin(curve)) / curve[0]
        print(
            f"  {name}: first {curve[0]:.2f} -> best {np.nanmin(curve):.2f} "
            f"({100 * gain:.2f}% improvement)"
        )
        report["networks"][name]["relative_improvement"] = float(gain)

    # Deliberately no cross-ensemble comparison of these numbers: the published
    # models ship a single validation value on a different normalization, so the
    # stored losses of two ensembles are not comparable. Use
    # evaluate_ensemble.py, which runs every network through one protocol.
    print(
        f"\n(for a comparison against {args.reference}, run evaluate_ensemble.py --"
        " stored validation losses are not comparable across ensembles)"
    )

    print()
    if failures:
        print(f"{len(failures)} problem(s):")
        for where, what in failures:
            print(f"  {where}: {what}")
    else:
        print("no problems found")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print("wrote", args.out)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
