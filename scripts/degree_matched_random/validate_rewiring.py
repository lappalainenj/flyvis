"""Check that a rewired connectome is a degree-preserving randomization.

Asserts the invariants that make it a valid null model, and measures how far the
wiring actually moved -- a rewiring that preserved the degrees but barely changed
any edge would be useless as a control.

Example:
    python validate_rewiring.py --seeds 0 1 2 3 4 5
"""

import argparse
import json

import numpy as np
import pandas as pd

from flyvis.connectome import hex_ring_distance, init_connectome
from flyvis.utils.type_utils import byte_to_str

PARENT = {
    "type": "ConnectomeFromAvgFilters",
    "file": "fib25-fib19_v2.2.json",
    "extent": 15,
    "n_syn_fill": 1,
}


def rewired_config(seed, mode, swaps_per_edge=20):
    config = {
        "type": "ConnectomeWithRewiredEdges",
        "parent": PARENT,
        "seed": seed,
        "mode": mode,
        "preserve": ["source_type", "target_type"],
        "extent": 15,
    }
    if mode != "filter_symmetry":
        config["swaps_per_edge"] = swaps_per_edge
    return config


def edge_frame(connectome):
    e = connectome.edges
    return pd.DataFrame({
        "source_index": e.source_index[:],
        "target_index": e.target_index[:],
        "source_type": byte_to_str(e.source_type[:]),
        "target_type": byte_to_str(e.target_type[:]),
        "du": e.du[:],
        "dv": e.dv[:],
        "n_syn": e.n_syn[:],
        "sign": e.sign[:],
    })


def degrees(df, n_nodes):
    return (
        np.bincount(df.source_index, minlength=n_nodes),
        np.bincount(df.target_index, minlength=n_nodes),
    )


def rewirable_edge_fraction(parent_df):
    """Fraction of edges in blocks where rewiring is possible at all.

    A block whose sources or targets are a single cell admits only one edge set,
    so its edges can never move.
    """
    n_rewirable = 0
    for _, block in parent_df.groupby(["source_type", "target_type"], sort=False):
        if block.source_index.nunique() > 1 and block.target_index.nunique() > 1:
            n_rewirable += len(block)
    return n_rewirable / len(parent_df)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument(
        "--mode",
        default="filter_symmetry",
        choices=["filter_symmetry", "within_ring", "unconstrained"],
    )
    p.add_argument("--swaps-per-edge", type=int, default=20)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    parent = init_connectome(**PARENT)
    pdf = edge_frame(parent)
    n_nodes = len(parent.nodes.type[:])
    p_out, p_in = degrees(pdf, n_nodes)
    parent_keys = set(zip(pdf.source_index.tolist(), pdf.target_index.tolist()))
    rewirable = rewirable_edge_fraction(pdf)

    print(f"parent: {len(pdf)} edges, {n_nodes} nodes")
    print(f"  edges in rewirable blocks: {rewirable:.4f}")
    print(
        "  a fully randomized sparse block retains ~1/n_target_cells of its edges,"
        " so a low retained fraction below is the goal"
    )

    report = {
        "parent": PARENT,
        "n_edges": int(len(pdf)),
        "n_nodes": int(n_nodes),
        "rewirable_edge_fraction": rewirable,
        "seeds": {},
    }

    p_ring = hex_ring_distance(pdf.du.values, pdf.dv.values)
    p_ring_histogram = pd.Series(p_ring).value_counts().sort_index()

    per_seed_keys = {}
    for seed in args.seeds:
        cfg = rewired_config(seed, args.mode, args.swaps_per_edge)
        rdf = edge_frame(init_connectome(**cfg))

        r_out, r_in = degrees(rdf, n_nodes)
        r_ring = hex_ring_distance(rdf.du.values, rdf.dv.values)

        # -- invariants that hold in every mode --------------------------------
        assert len(rdf) == len(pdf), "edge count changed"
        assert np.array_equal(r_out, p_out), "out-degree changed"
        assert np.array_equal(rdf.source_index, pdf.source_index), "sources moved"
        assert rdf.groupby(["source_index", "target_index"]).ngroups == len(rdf), (
            "duplicate edges created"
        )
        assert np.array_equal(rdf.source_type, pdf.source_type), "source types moved"
        assert np.array_equal(rdf.target_type, pdf.target_type), "target types moved"
        assert np.array_equal(rdf.n_syn, pdf.n_syn), "synapse counts moved"
        assert np.array_equal(rdf.sign, pdf.sign), "signs moved"
        pairs = rdf.groupby(["source_type", "target_type"], sort=False).ngroups
        assert pairs == 604, f"expected 604 type pairs, got {pairs}"

        # -- mode-specific invariants ------------------------------------------
        # filter_symmetry trades exact in-degree for preserved locality; the
        # swap-based modes trade the reverse. Each mode is held to its own promise.
        in_degree_deviation = np.abs(r_in.astype(int) - p_in.astype(int))
        if args.mode == "filter_symmetry":
            assert np.array_equal(r_ring, p_ring), "an edge changed its ring distance"
            print(
                f"\nseed {seed}: every edge keeps its ring distance; "
                f"in-degree off for {int((in_degree_deviation > 0).sum())} of "
                f"{n_nodes} cells "
                f"({in_degree_deviation.sum() / len(pdf):.4f} of total in-degree, "
                f"max {in_degree_deviation.max()})"
            )
        else:
            assert np.array_equal(r_in, p_in), "in-degree changed"
            r_hist = pd.Series(r_ring).value_counts().sort_index()
            preserved = all(
                int(r_hist.get(k, 0)) == int(v) for k, v in p_ring_histogram.items()
            )
            print(
                f"\nseed {seed}: in-degree exact; ring histogram preserved: {preserved}"
            )

        # -- how much did it move? -------------------------------------------
        keys = set(zip(rdf.source_index.tolist(), rdf.target_index.tolist()))
        retained = len(keys & parent_keys) / len(pdf)
        per_seed_keys[seed] = keys

        autapses = int((rdf.source_index == rdf.target_index).sum())
        offset_groups = rdf.groupby(
            ["source_type", "target_type", "du", "dv"], sort=False
        ).ngroups
        entry = {
            "retained_parent_edge_fraction": retained,
            "n_autapses": autapses,
            "n_offset_groups": offset_groups,
            "mean_ring_distance": float(r_ring.mean()),
            "fraction_within_ring_2": float((r_ring <= 2).mean()),
            "in_degree_error_fraction": float(in_degree_deviation.sum() / len(pdf)),
            "n_cells_with_wrong_in_degree": int((in_degree_deviation > 0).sum()),
        }
        report["seeds"][seed] = entry
        p_autapses = int((pdf.source_index == pdf.target_index).sum())
        p_offset_groups = pdf.groupby(
            ["source_type", "target_type", "du", "dv"], sort=False
        ).ngroups
        print(
            f"  retained {retained:.5f} of the parent's edges, "
            f"{autapses} autapses (parent {p_autapses})"
        )
        print(
            f"  (source_type,target_type,du,dv) groups: {offset_groups} "
            f"(parent {p_offset_groups})"
        )
        print(
            f"  ring distance: mean {r_ring.mean():.2f}, within ring 2 "
            f"{(r_ring <= 2).mean():.4f} (parent mean {p_ring.mean():.2f}, "
            f"within ring 2 {(p_ring <= 2).mean():.4f})"
        )

    # -- are the seeds independent of each other? ----------------------------
    print("\npairwise edge overlap between seeds (independent draws -> ~retained):")
    overlaps = {}
    seeds = list(per_seed_keys)
    for a in range(len(seeds)):
        for b in range(a + 1, len(seeds)):
            sa, sb = seeds[a], seeds[b]
            ov = len(per_seed_keys[sa] & per_seed_keys[sb]) / len(pdf)
            overlaps[f"{sa}-{sb}"] = ov
    print("  " + "  ".join(f"{k}:{v:.5f}" for k, v in overlaps.items()))
    report["pairwise_overlap"] = overlaps

    print("\nall invariants hold for every seed")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
