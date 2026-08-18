"""Stage 1: do the two repos build the same network from the same connectome?

Compares the resolved configs, the connectome-derived tables, and every network
parameter tensor that is fully determined by the connectome (sign, synapse count)
rather than by a random draw.

Run:
    python stage1_structure.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    assert_same_connectome,
    dvs_config,
    flyvis_config,
    norm_types,
    print_report,
    summarize,
)

import dvs  # noqa: E402
import flyvis  # noqa: E402
from dvs.networks import Network as DvsNetwork  # noqa: E402
from flyvis.network.network import Network as FlyvisNetwork  # noqa: E402


def build_networks():
    dcfg = dvs_config()
    fcfg = flyvis_config()
    dnet = DvsNetwork(**dcfg.network)
    fnet = FlyvisNetwork(**fcfg.network)
    return dcfg, fcfg, dnet, fnet


def main():
    dcfg, fcfg, dnet, fnet = build_networks()

    print("== resolved configs ==")
    print("dvs-sim network:", json.dumps(json.loads(json.dumps(dcfg.network, default=str)), indent=1)[:1500])
    print("flyvis network:", json.dumps(json.loads(json.dumps(fcfg.network, default=str)), indent=1)[:1500])

    print("\n== connectome tables ==")
    assert_same_connectome(dnet, fnet)
    print(f"  identical: {dnet.n_nodes} nodes, {dnet.n_edges} edges")

    print("\n== free parameter inventory ==")
    dp = dict(dnet.named_parameters())
    fp = dict(fnet.named_parameters())
    print("  dvs-sim:", {k: (tuple(v.shape), v.requires_grad) for k, v in dp.items()})
    print("  flyvis :", {k: (tuple(v.shape), v.requires_grad) for k, v in fp.items()})
    print(f"  n params dvs={dnet.num_parameters} flyvis={fnet.num_parameters}")

    print("\n== deterministic parameters (connectome-derived) ==")
    rows = []
    for name in ("edges_sign", "edges_syn_count", "nodes_time_const", "edges_syn_strength"):
        if name in dp and name in fp:
            rows.append(summarize(dp[name], fp[name], name))
    print_report(rows)

    print("\n== derived edge weights at initialization ==")
    with torch.no_grad():
        dw = dnet._param_api().edges.weight.detach()
        fw = fnet._param_api().edges.weight.detach()
    print_report([summarize(dw, fw, "edges.weight")])

    print("\n== parameter grouping (sharing) ==")
    for pname in ("nodes_bias", "nodes_time_const", "edges_sign", "edges_syn_count",
                  "edges_syn_strength"):
        di = dnet.node_params.get(pname.replace("nodes_", ""), None) or dnet.edge_params.get(
            pname.replace("edges_", ""), None
        )
        fi = fnet.node_params.get(pname.replace("nodes_", ""), None) or fnet.edge_params.get(
            pname.replace("edges_", ""), None
        )
        if di is None or fi is None:
            print(f"  {pname}: missing in one repo")
            continue
        d_idx = di.indices.cpu().numpy()
        f_idx = fi.indices.cpu().numpy()
        same_partition = _same_partition(d_idx, f_idx)
        print(
            f"  {pname}: n_groups dvs={len(np.unique(d_idx))} flyvis={len(np.unique(f_idx))}"
            f"  identical_partition={same_partition}"
            f"  identical_index_order={bool((d_idx == f_idx).all()) if d_idx.shape == f_idx.shape else False}"
        )

    print("\n== decoder ==")
    from dvs.decoder import init_decoder as dvs_init_decoder
    from flyvis.task.decoder import init_decoder as flyvis_init_decoder

    ddec = dvs_init_decoder(dcfg.task.decoder.flow, dnet.ctome)
    fdec = flyvis_init_decoder(fcfg.task.decoder.flow, fnet.connectome)
    print("  dvs   :", type(ddec).__name__, ddec.num_parameters)
    print("  flyvis:", type(fdec).__name__, fdec.num_parameters)
    print("  dvs modules  :", [type(m).__name__ for m in ddec.modules()][:14])
    print("  flyvis modules:", [type(m).__name__ for m in fdec.modules()][:14])
    d_out = norm_types(ddec.dvs_channels.output_node_types
                       if hasattr(ddec.dvs_channels, "output_node_types")
                       else ddec.output_node_types)
    f_out = norm_types(fdec.dvs_channels.output_cell_types)
    print("  output cell types identical:", list(d_out) == list(f_out),
          len(d_out), len(f_out))
    print("  hex coords identical:",
          bool((ddec.u == fdec.u).all() and (ddec.v == fdec.v).all()),
          f"H,W dvs=({ddec.H},{ddec.W}) flyvis=({fdec.H},{fdec.W})")


def _same_partition(a, b):
    """Do two index arrays induce the same partition of positions?"""
    if a.shape != b.shape:
        return False
    remap = {}
    for x, y in zip(a.tolist(), b.tolist()):
        if x in remap:
            if remap[x] != y:
                return False
        else:
            remap[x] = y
    return len(set(remap.values())) == len(remap)


if __name__ == "__main__":
    main()
