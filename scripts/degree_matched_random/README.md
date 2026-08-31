# Degree-matched random connectomes

An ensemble trained from scratch on random connectomes that have the *same
in-degree and out-degree per cell* as the connectome behind the published models,
and the same cell-type-level connectivity, but random wiring between individual
cells. The control for asking whether the connectome's specific wiring matters
beyond its degree statistics.

Each ensemble member gets its own connectome, drawn with its own seed. The seed
is part of the network config, so every model records the connectome it trained
on and the connectome is reproducible from the config alone -- no mask files or
side-channel artifacts.

## What is held fixed and what is randomized

The randomization is a **double-edge swap** on the edge table: the targets of two
edges are exchanged, and the swap is rejected if it would duplicate an edge that
already exists. Sources are never touched and the multiset of targets is never
changed, so:

| | |
|---|---|
| in-degree of every cell | **exactly** the reference's |
| out-degree of every cell | **exactly** the reference's |
| number of edges | 1,513,231, as the reference |
| the 604 (source_type, target_type) pairs and their edge counts | preserved |
| synapse count and sign per edge | travel with the edge |
| which individual cells are wired together | **randomized** |

Swaps may not cross (source_type, target_type) blocks. That restriction is what
makes the comparison controlled rather than merely different: `sign` and
`syn_strength` are shared per type pair and `syn_strength` is initialized as
`scale / <n_syn>` over the pair, so preserving the blocks preserves both the
number of free parameters **and their initial values**. Verified:

```
FlyNet   free params: 734  {nodes_bias: 65, nodes_time_const: 65, edges_syn_strength: 604}
rewired  free params: 734  {nodes_bias: 65, nodes_time_const: 65, edges_syn_strength: 604}
  nodes_bias         equal: True
  nodes_time_const   equal: True
  edges_syn_strength equal: True
```

So at initialization the rewired network differs from the reference in the graph
only, not in a single free parameter.

## Retinotopy is necessarily destroyed

This is a property of the null model, not a bug, and it sets what the control can
claim. Fixing each cell's in- and out-degree within a type pair says nothing
about *which* cells are connected, so a random draw connects cells that are far
apart on the hex lattice:

```
mean columnar offset radius   reference 1.33   rewired 14.20
(source_type,target_type,du,dv) groups   reference 2,355   rewired 616,899
retained fraction of reference edges     0.013
```

The reference connectome's edges are local (median offset 1 column); the rewired
ones span the array (median 13.5). The ensemble therefore tests the *full*
connectome, including its retinotopic specificity, against degree statistics plus
type-level connectivity alone. It is a strong null, not a subtle one.

One consequence worth knowing: `syn_count` is grouped by
(source_type, target_type, du, dv) and averaged, so on the reference it is a
smooth spatial filter of 2,355 values shared by 1.5M edges, whereas after
rewiring the groups hold ~2.5 edges each and the fixed per-edge scale is
effectively each edge's own synapse count, randomly repositioned. The config is
left as the reference's; grouping `syn_count` by (source_type, target_type) only
would remove that residual heterogeneity if a cleaner null is wanted.

## Training regime

The published models' regime, not the current public default -- see
`scripts/dvs_sim_parity/FINDINGS.md`:

* `task=task_original` -- the original temporal sampling of the input movie and
  `align_corners=False` for the flow targets
* `scheduler=scheduler_original` -- ten learning-rate steps over the first 200k
  of 250k iterations, then 5e-6 held; validation every 334 epochs

## Files

* `validate_rewiring.py` -- builds the connectomes and asserts the invariants
  above; also reports how far the wiring moved and whether the seeds are
  independent of each other. Doubles as the build step, since datamate caches
  each connectome on disk.
* `check_trial.py` -- after training, checks every network of an ensemble: that
  its stored config records the intended regime and seed, that its connectome
  really is degree-matched, that the six networks trained on six *different*
  connectomes, and that training progressed.
* `hpc/prepare.sbatch` -- builds and validates the connectomes. Run before the
  array so the array jobs cannot race writing the same cache directory.
* `hpc/train_array.sbatch` -- one array task per network; the array task id is
  both the network id and the rewiring seed.

## Running it

```bash
# once: build and validate the six connectomes
sbatch scripts/degree_matched_random/hpc/prepare.sbatch 6

# trial: 100 iterations, throwaway ensemble
sbatch --array=0-5 scripts/degree_matched_random/hpc/train_array.sbatch 0399 100 80
python scripts/degree_matched_random/check_trial.py flow/0399 --expect-iters 100

# full run
sbatch --array=0-5 scripts/degree_matched_random/hpc/train_array.sbatch 0300
python scripts/degree_matched_random/check_trial.py flow/0300 --expect-iters 250000
```

The equivalent single-network command, for reference:

```bash
flyvis train-single task_name=flow ensemble_and_network_id=0300/003 \
    description="degree-matched random connectome with rewiring seed 3" \
    network/connectome=connectome_degree_matched_random \
    network.connectome.seed=3 \
    task=task_original scheduler=scheduler_original
```

The sbatch scripts assume the repo at `$HOME/hpc_runs/flyvis-dmn`, data at
`$HOME/hpc_runs/flyvis-dmn-data`, conda env `flyvis-dmn` and logs in
`$HOME/hpc_runs/logs`; adjust the header and `REPO`/`FLYVIS_ROOT_DIR` if yours
differ.
