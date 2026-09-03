"""Compute moving-edge peak responses and T4c clustering for two ensembles.

The expensive part -- simulating moving-edge and naturalistic responses for every
member -- runs where the checkpoints are; the compact arrays it writes are small
enough to move and plot elsewhere.

Writes per ensemble:
  <out>/<tag>_peaks.nc          peak responses of the selected models
  <out>/<tag>_meta.json         model ids, task errors, cluster assignment

Example:
    python compute_tuning.py --ensembles flow/0310 flow/0000 \
        --speeds 19 --top-fraction 0.1 --out /path/to/out
"""

import argparse
import json
from pathlib import Path

import numpy as np

from flyvis import EnsembleView
from flyvis.analysis.moving_bar_responses import peak_responses
from flyvis.datasets.moving_bar import MovingEdge

# best checkpoints of ensembles trained here are recorded under `loss`, not `epe`
CHECKPOINT_KWARGS = {"validation_subdir": "validation", "loss_file_name": "loss"}


def serializable(da):
    """Make a DataArray writable to netcdf.

    `peak_responses` carries a `checkpoints` coordinate of Path objects, and
    netcdf cannot store arbitrary Python objects. Object coordinates are cast to
    string so the provenance survives the round trip.
    """
    for name in list(da.coords):
        if da[name].dtype == object:
            da = da.assign_coords({name: da[name].astype(str)})
    return da


def t4c_clusters(ensemble):
    """Cluster models by their central T4c response to naturalistic stimuli.

    This is the clustering of Lappalainen et al. (2024): a UMAP embedding of the
    central T4c voltage traces, then a Gaussian mixture over the embedding.
    """
    from flyvis.analysis.clustering import EnsembleEmbedding, get_cluster_to_indices
    from flyvis.utils.activity_utils import CentralActivity

    stims_and_resps = ensemble.naturalistic_stimuli_responses()
    responses = stims_and_resps["responses"]
    central = CentralActivity(
        responses.values, connectome=ensemble.connectome, keepref=True
    )
    embedding = EnsembleEmbedding(central)
    t4c = embedding("T4c", n_neighbors=5, min_dist=0.12, n_components=2, random_state=42)
    clustering = t4c.cluster.gaussian_mixture(
        n_clusters=3, n_init=100, random_state=1000, max_iter=1000
    )
    cluster_to_indices = get_cluster_to_indices(
        clustering.embedding.mask, clustering.labels, ensemble.task_error()
    )
    return {int(k): [int(i) for i in v] for k, v in cluster_to_indices.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensembles", nargs="+", default=["flow/0310", "flow/0000"])
    p.add_argument("--speeds", type=float, nargs="+", default=[19.0])
    p.add_argument("--top-fraction", type=float, default=0.1)
    p.add_argument("--dt", type=float, default=1 / 200)
    p.add_argument("--cluster", action="store_true", help="also run T4c clustering")
    p.add_argument(
        "--best-only",
        action="store_true",
        help="restrict every computation to the selected models. responses_norm "
        "otherwise simulates naturalistic stimuli for the whole ensemble, which "
        "dominates the runtime; the constants it stores are per model, so a "
        "later full run reuses whatever this computes",
    )
    p.add_argument("--out", default=".")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = MovingEdge(
        offsets=(-10, 11),
        intensities=[0, 1],
        speeds=list(args.speeds),
        height=9,
        dt=args.dt,
        angles=list(np.arange(0, 360, 30)),
    )
    print(f"MovingEdge: {len(dataset)} stimuli, speeds={args.speeds}")

    for name in args.ensembles:
        tag = name.replace("/", "_")
        print(f"\n=== {name} ===", flush=True)
        ensemble = EnsembleView(name, best_checkpoint_fn_kwargs=CHECKPOINT_KWARGS)
        task_error = ensemble.min_validation_losses()
        order = ensemble.argsort()
        n_top = max(1, int(round(args.top_fraction * len(ensemble))))
        model_ids = order[:n_top]
        print(
            f"  {len(ensemble)} models, best {n_top}: "
            f"{[ensemble.names[i].split('/')[-1] for i in model_ids]}",
            flush=True,
        )

        target = ensemble
        if args.best_only:
            target = ensemble[[int(i) for i in model_ids]]
            print(f"  restricted to {len(target)} models", flush=True)
        stims_and_resps = target.moving_edge_responses(dataset=dataset)
        stims_and_resps["responses"] /= target.responses_norm(rectified=True)
        peaks = peak_responses(stims_and_resps)
        if not args.best_only:
            peaks = peak_responses(stims_and_resps.sel(network_id=model_ids))
        peaks = serializable(peaks)
        peaks.to_netcdf(out / f"{tag}_peaks.nc")
        print(f"  wrote {out / f'{tag}_peaks.nc'}", flush=True)

        meta = {
            "ensemble": name,
            "n_models": len(ensemble),
            "model_ids": [int(i) for i in model_ids],
            "model_names": [ensemble.names[i] for i in model_ids],
            "task_error": [float(task_error[i]) for i in model_ids],
            "task_error_all": [float(t) for t in task_error],
            "speeds": list(args.speeds),
        }
        if args.cluster:
            try:
                meta["t4c_clusters"] = t4c_clusters(ensemble)
                print(
                    f"  clusters: "
                    f"{ {k: len(v) for k, v in meta['t4c_clusters'].items()} }",
                    flush=True,
                )
            except Exception as e:  # clustering is optional and expensive
                meta["t4c_clusters_error"] = f"{type(e).__name__}: {e}"
                print(f"  clustering failed: {e}", flush=True)
        (out / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  wrote {out / f'{tag}_meta.json'}", flush=True)


if __name__ == "__main__":
    main()
