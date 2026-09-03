"""Compare T4/T5 angular tuning between two ensembles.

Consumes the compact arrays written by
/Users/janne.lappalainen/Projects/flyvis/scripts/degree_matched_random/compute_tuning.py
so the plotting is cheap and local.

Two figures:
  tuning_best_models.pdf   T4a-d and T5a-d angular tuning of the best models of
                           each ensemble, overlaid, against the known tuning
  tuning_t4c_clusters.pdf  T4c tuning per naturalistic-response cluster, one row
                           per ensemble

Example:
    python plot_tuning.py --in /path/to/tuning --out figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from flyvis.analysis.moving_bar_responses import plot_angular_tuning

T4 = ["T4a", "T4b", "T4c", "T4d"]
T5 = ["T5a", "T5b", "T5c", "T5d"]
STYLE = {
    "flow_0000": dict(color="#33691e", label="published (0000)"),
    "flow_0310": dict(color="#1565c0", label="our reference (0310)"),
    "flow_0301": dict(color="#c62828", label="locality-preserving null (0301)"),
}


def load(indir, tag):
    peaks = xr.open_dataarray(Path(indir) / f"{tag}_peaks.nc")
    meta = json.loads((Path(indir) / f"{tag}_meta.json").read_text())
    return peaks, meta


def tuning_row(peaks_by_tag, cell_types, intensity, axes, groundtruth_on):
    """One row of polar axes, both ensembles overlaid per cell type."""
    for ax, cell_type in zip(axes, cell_types):
        # a common scale per panel, so the two ensembles stay comparable
        scale = max(
            float(
                plot_angular_tuning.__globals__["angular_tuning"](
                    peak_responses_da=p, cell_type=cell_type, intensity=intensity
                ).max()
            )
            for p in peaks_by_tag.values()
        )
        for i, (tag, peaks) in enumerate(peaks_by_tag.items()):
            plot_angular_tuning(
                dataset=None,
                cell_type=cell_type,
                intensity=intensity,
                peak_responses_da=peaks,
                average_models=True,
                model_reduction="mean",
                normalize_by=scale,
                colors=STYLE[tag]["color"],
                groundtruth=(i == 0) and groundtruth_on,
                ymin=0,
                ymax=1.0,
                ax=ax,
                zorder=10 + i,
                linewidth=1.2,
            )
        ax.set_title(cell_type, fontsize=7, pad=2)


def figure_best_models(peaks_by_tag, metas, out):
    fig, axes = plt.subplots(
        2, 4, figsize=(9.5, 5.2), subplot_kw={"projection": "polar"}
    )
    tuning_row(peaks_by_tag, T4, 1, axes[0], groundtruth_on=True)
    tuning_row(peaks_by_tag, T5, 0, axes[1], groundtruth_on=True)
    axes[0, 0].set_ylabel("ON (T4)", fontsize=8, labelpad=18)
    axes[1, 0].set_ylabel("OFF (T5)", fontsize=8, labelpad=18)

    handles = [
        plt.Line2D([], [], color=STYLE[t]["color"], lw=1.6, label=(
            f"{STYLE[t]['label']}, best {len(metas[t]['model_ids'])} "
            f"of {metas[t]['n_models']}"
        ))
        for t in peaks_by_tag
    ]
    handles.append(plt.Line2D([], [], color="k", lw=1.2, label="known tuning"))
    fig.legend(
        handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False
    )
    speeds = metas[next(iter(metas))]["speeds"]
    fig.suptitle(
        f"T4/T5 angular tuning, moving edge at {speeds} half-ommatidia/s", fontsize=9
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(out / "tuning_best_models.pdf", bbox_inches="tight")
    fig.savefig(out / "tuning_best_models.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / 'tuning_best_models.pdf'}")


def figure_clusters(peaks_full_by_tag, metas, out):
    """T4c tuning per naturalistic-response cluster, one row per ensemble."""
    tags = [t for t in peaks_full_by_tag if metas[t].get("t4c_clusters")]
    if not tags:
        print("no cluster assignments present, skipping cluster figure")
        return
    ncol = max(len(metas[t]["t4c_clusters"]) for t in tags)
    fig, axes = plt.subplots(
        len(tags), ncol, figsize=(2.4 * ncol, 2.6 * len(tags)),
        subplot_kw={"projection": "polar"}, squeeze=False,
    )
    for r, tag in enumerate(tags):
        clusters = metas[tag]["t4c_clusters"]
        peaks = peaks_full_by_tag[tag]
        for c, (label, ids) in enumerate(sorted(clusters.items(), key=lambda kv: kv[0])):
            ax = axes[r][c]
            present = [i for i in ids if i in peaks.network_id.values]
            if not present:
                ax.set_axis_off()
                continue
            plot_angular_tuning(
                dataset=None,
                cell_type="T4c",
                intensity=1,
                peak_responses_da=peaks.sel(network_id=present),
                average_models=True,
                model_reduction="mean",
                colors=STYLE[tag]["color"],
                groundtruth=True,
                ymin=0,
                ymax=1.0,
                ax=ax,
                linewidth=1.2,
            )
            ax.set_title(f"cluster {label}  (n={len(present)})", fontsize=7, pad=2)
        for c in range(len(clusters), ncol):
            axes[r][c].set_axis_off()
        axes[r][0].set_ylabel(STYLE[tag]["label"], fontsize=8, labelpad=20)
    fig.suptitle("T4c angular tuning by naturalistic-response cluster", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "tuning_t4c_clusters.pdf", bbox_inches="tight")
    fig.savefig(out / "tuning_t4c_clusters.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / 'tuning_t4c_clusters.pdf'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="indir", required=True)
    p.add_argument("--tags", nargs="+", default=["flow_0000", "flow_0310"])
    p.add_argument("--out", default="figures")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    peaks_by_tag, metas = {}, {}
    for tag in args.tags:
        try:
            peaks, meta = load(args.indir, tag)
        except FileNotFoundError as e:
            print(f"{tag}: {e}")
            continue
        peaks_by_tag[tag] = peaks
        metas[tag] = meta
        print(f"{tag}: peaks {dict(peaks.sizes)}, "
              f"best {len(meta['model_ids'])} of {meta['n_models']}")

    if not peaks_by_tag:
        raise SystemExit("nothing to plot")
    figure_best_models(peaks_by_tag, metas, out)
    figure_clusters(peaks_by_tag, metas, out)


if __name__ == "__main__":
    main()
