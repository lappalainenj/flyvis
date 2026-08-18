"""Shared helpers to build matching dvs-sim and flyvis training setups.

Both packages are imported into the same process so that tensors can be compared
directly, on the same torch build, without serialization in between.

The two repos name three cell types differently (CT1 parentheses and TmY17/TmY18);
the connectome specs are otherwise byte-identical after that renaming, which
`assert_same_connectome` checks.
"""

import sys
from pathlib import Path

import numpy as np
import torch

DVS_SIM = Path("/Users/janne.lappalainen/Projects/dvs-sim")
if str(DVS_SIM) not in sys.path:
    sys.path.insert(0, str(DVS_SIM))

if not torch.cuda.is_available():
    # dvs-sim was written for a GPU box and calls .cuda() unconditionally in a few
    # places (decoder hex mask, truncation indices). On CPU those calls are no-ops
    # semantically, so redirect them instead of editing the dvs-sim checkout.
    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[assignment]

import dvs  # noqa: E402
import flyvis  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

#: dvs-sim cell type name -> flyvis cell type name
ALIAS = {"TmY17": "TmY18", "CT1Lo1": "CT1(Lo1)", "CT1M10": "CT1(M10)"}


def norm_types(names):
    """Normalize an array of cell type names to flyvis spelling."""
    out = []
    for n in names:
        n = n.decode() if isinstance(n, bytes) else str(n)
        out.append(ALIAS.get(n, n))
    return np.array(out)


# -- configs -------------------------------------------------------------------


def dvs_config(overrides=()):
    """Resolved dvs-sim solver config with cluster paths pointed at this checkout."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(DVS_SIM / "config"), version_base=None):
        cfg = compose(
            config_name="config.yaml",
            overrides=[
                "id=parity/000",
                "task=flow",
                "train=true",
                "resume=false",
                f"solver.network.connectome.path={DVS_SIM}/data/fib25-fib19_v2.2.json",
                f"solver.task.dataset.path={DVS_SIM}/data/SintelDataSet",
                "solver.comment=parity",
                *overrides,
            ],
        )
    return dvs.namespacify(OmegaConf.to_container(cfg.solver, resolve=True))


def flyvis_config(overrides=()):
    """Resolved flyvis solver config (public defaults for the flow task)."""
    from flyvis.utils.config_utils import get_default_config

    return get_default_config(
        overrides=[
            "task_name=flow",
            "ensemble_and_network_id=parity/000",
            "description=parity",
            *overrides,
        ],
        path=str(Path(flyvis.__file__).parent / "config" / "solver.yaml"),
    )


# -- structural checks ---------------------------------------------------------


def assert_same_connectome(dvs_net, flyvis_net):
    """Node and edge tables must agree element for element after renaming."""
    dn, fn = dvs_net.ctome.nodes, flyvis_net.connectome.nodes
    de, fe = dvs_net.ctome.edges, flyvis_net.connectome.edges
    assert len(dn.type) == len(fn.type), (len(dn.type), len(fn.type))
    assert (norm_types(dn.type[:]) == norm_types(fn.type[:])).all(), "node types differ"
    for key in ("u", "v"):
        assert (dn[key][:] == fn[key][:]).all(), f"node {key} differs"
    assert len(de.source_index) == len(fe.source_index), "edge count differs"
    for key in ("source_index", "target_index", "du", "dv", "n_syn"):
        a, b = de[key][:], fe[key][:]
        if a.dtype.kind == "f":
            assert np.allclose(a, b), f"edge {key} differs"
        else:
            assert (a == b).all(), f"edge {key} differs"
    return True


def param_tensors(module):
    """Named parameters as detached cpu tensors."""
    return {k: v.detach().cpu().clone() for k, v in module.named_parameters()}


def grad_tensors(module):
    return {
        k: (v.grad.detach().cpu().clone() if v.grad is not None else None)
        for k, v in module.named_parameters()
    }


def summarize(a, b, name, atol=0.0):
    """Compare two tensors and return a report row."""
    if a is None or b is None:
        return dict(name=name, status="missing", a=a is not None, b=b is not None)
    a = torch.as_tensor(a).double().flatten()
    b = torch.as_tensor(b).double().flatten()
    if a.shape != b.shape:
        return dict(name=name, status="shape", a_shape=tuple(a.shape), b_shape=tuple(b.shape))
    diff = (a - b).abs()
    denom = torch.maximum(a.abs(), b.abs()).clamp(min=1e-30)
    return dict(
        name=name,
        status="equal" if diff.max().item() <= atol else "differs",
        max_abs=diff.max().item(),
        max_rel=(diff / denom).max().item(),
        mean_abs=diff.mean().item(),
        a_norm=a.norm().item(),
        b_norm=b.norm().item(),
    )


def print_report(rows):
    width = max(len(r["name"]) for r in rows)
    for r in rows:
        if r["status"] == "equal":
            print(f"  {r['name']:<{width}}  equal")
        elif r["status"] == "differs":
            print(
                f"  {r['name']:<{width}}  DIFFERS  max_abs={r['max_abs']:.3e}"
                f"  max_rel={r['max_rel']:.3e}  |a|={r['a_norm']:.6g}  |b|={r['b_norm']:.6g}"
            )
        elif r["status"] == "shape":
            print(f"  {r['name']:<{width}}  SHAPE    {r['a_shape']} vs {r['b_shape']}")
        else:
            print(f"  {r['name']:<{width}}  MISSING  dvs={r['a']} flyvis={r['b']}")
