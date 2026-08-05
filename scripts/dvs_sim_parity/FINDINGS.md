# Why flyvis training differs from dvs-sim training

Everything below was established by running both stacks in **one process, on one
torch build** (2.7.1, CPU) and comparing tensors, not by reading configs. The
scripts in this directory reproduce every number.

* `stage1_structure.py` — connectome, network parameters, decoder architecture
* `stage2_dataset.py` — rendered data and dataset items
* `stage2b_dataset_fix.py` — proves the dataset delta is exactly two operations
* `stage3_train_step.py` — forward, loss, gradients, optimizer, penalty
* `stage4_init_and_schedule.py` — initialization draws and learning-rate schedule
* `stage5_task_and_loop.py` — splits, dataloaders, epochs, checkpoint cadence
* `stage6_trajectory.py` — real training loops of both repos, side by side
* `stage7_effect_size.py` — how large the data differences are
* `stage8_verify_flag.py` — the shipped flag reproduces dvs-sim through the public API

## Summary

The refactor did not change the model, the decoder, the loss, the optimizer, the
penalty, the data split, or the initialization. Those are bit-identical.

Four differences remain. Two of them change the training data at every single
iteration, and one changes the entire learning-rate trajectory:

| # | difference | dvs-sim | flyvis | effect |
|---|---|---|---|---|
| 1 | temporal sampling of the input movie | `floor(arange(0, 19, 24/50))` index sampling | `nearest-exact` interpolation to 40 samples | different time warp of every input movie |
| 2 | linear interpolation of the flow targets | `align_corners=False` | `align_corners=True` | different target at every frame, up to 0.7 in absolute flow units |
| 3 | learning-rate schedule | 10 steps over the first 200k iterations, then 5e-6 held to 250k | 10 steps spread over all 250k iterations | flyvis trains at a 20% higher mean learning rate and never reaches the long 5e-6 tail |
| 4 | validation / checkpoint cadence | every 334 epochs (derived from `n_iters`) | every 300 epochs (hardcoded) | different grid of candidate best checkpoints |

Plus one that is rounding, not semantics: `syn_strength` is initialized as
`(1/<N>) * scale` in dvs-sim and `scale/<N>` in flyvis. Evaluating both formulas
directly reproduces the observed difference exactly — 1.862645149230957e-09 max
abs, 147 of 604 groups off by one float32 ULP. No action taken.

## Verification

With augmentation off, a deterministic sampler and dropout disabled — so that the
run is reproducible rather than merely comparable in distribution —
`stage6_trajectory.py` runs 24 real training iterations in each stack:

| iteration | dvs-sim | flyvis default | flyvis original |
|---|---|---|---|
| 0 | 330.434998 | 332.113159 | 330.434998 |
| 1 | 1868.284302 | 1872.703369 | 1868.284302 |
| 3 | 2577.598145 | 2562.041504 | 2577.598145 |
| 11 | 589.933105 | 572.179810 | 589.933105 |
| 23 | 588.941589 | 571.173340 | 588.941589 |

**flyvis with `task=task_original scheduler=scheduler_original` reproduces the
dvs-sim loss trajectory bit-for-bit over all 24 iterations.** Public-default flyvis
differs by up to 3.0% per iteration. `nodes_bias` after training is bit-identical
too; `nodes_time_const` and `edges_syn_strength` carry the 1-ULP initialization
offset above (max abs 3.7e-9).

The one post-training parameter that differs more than rounding is the decoder's
`base.0.bias` (1.1e-4 absolute on a value of 3e-3). That bias sits directly in
front of a `BatchNorm2d`, which subtracts it out, so its gradient is ~0 and Adam's
update on it is dominated by `eps` — a mathematically inconsequential parameter
where 1-ULP inputs produce visible output differences. The loss trajectory being
bit-identical shows it does not affect the function.

`stage8_verify_flag.py` checks the shipped flag rather than the monkeypatch: with
`task=task_original`, all 138 tensors (69 sequences x lum/flow) are bit-identical
to dvs-sim with augmentation off, and so are matched-parameter augmented items. The
public default is unchanged.

`stage4`/`scheduler_original` gives an lr array identical to dvs-sim's over the
full 250k-iteration run (`np.array_equal` on 250,001 values).

## How much do the data differences matter per step?

From `stage7_effect_size.py`, over all 69 sequences:

* input movie: median 6.9% relative L2 difference (max 24.6%)
* flow target: median 8.3% relative L2 difference; worst frame error is 2.7x the
  typical target magnitude
* loss of one and the same network: 328.67 vs 330.33, i.e. 0.51%
* gradient cosine similarity: 0.999999, relative L2 difference 0.73%

For scale, two genuinely different batches give gradient cosine similarity 0.851
and a loss difference of 1961. So each single step is only mildly perturbed — but
it is perturbed systematically, in both input and target, at every one of 250,000
iterations, on top of a learning-rate schedule that is 20% higher on average.

## What is provably identical

Stage 1 and 3, with the dvs-sim parameters copied into the flyvis network and the
same batch fed to both:

* connectome tables: 45,669 nodes and 1,513,231 edges agree element for element
  after normalizing three cell type names (`TmY17`/`TmY18`, CT1 parentheses); the
  two JSON specs are otherwise equal
* parameter inventory, sharing partitions and index order: identical for
  `nodes_bias`, `nodes_time_const`, `edges_sign`, `edges_syn_count`,
  `edges_syn_strength` (734 free parameters both)
* decoder: same modules, same 7,427 parameters, same output cell types, same hex
  coordinates
* bias initialization at seed 0: identical values (flyvis's per-parameter
  `torch.Generator(0)` happens to produce the same stream as dvs-sim's
  `manual_seed(0)`)
* steady state, per-frame activity, decoder output: equal
* loss: `l2_3d_wotc` and `l2norm` both give 328.6682128906 on the same batch — the
  `+1e-9` stabilizer is invisible in float32
* all gradients, parameters after the Adam step, parameters after the
  activity-penalty step: equal
* train/validation split (51/16 sequences), batch sizes, `drop_last`, samplers,
  number of epochs (20,834), `t_pre_train=0.5`: identical
* dynamics: for this chemical-only connectome `StaticSynapses` and
  `PPNeuronIGRSynapses` compute the same velocity
* the `relu_leak` and `activity_penalty` schedules present in the dvs-sim config
  are inert there (`relu` has no `negative_slope`; `start`/`stop` are `None`), so
  their absence from flyvis changes nothing

## Difference 1 and 2 in detail

For a 19-frame window at `dt=0.02` and 24 fps, both stacks produce 40 frames, but
not the same 40:

```
dvs-sim  0 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 11 11 12 12 12 13 ...
flyvis   0 0 1 1 2 2 3 3 4 4 4 5 5 6 6 7 7 8 8 9 9 10 10 11 11 12 12 13 13 ...
```

dvs-sim repeats the first frame three times and triples frame 12; flyvis repeats
frames 4 and 14. The targets are then interpolated with a different corner
convention.

`stage2b_dataset_fix.py` patches exactly these two operations into the flyvis
dataset and the items become **bit-identical to dvs-sim** for all 69 sequences
with augmentation off, and for matched augmentation parameters with augmentation
on. So the dataset delta is fully characterized — nothing else in rendering,
cropping, flipping, rotating, contrast/brightness jitter or noise differs (the hex
rotation and flip permutation tables and matrices are equal, and dvs-sim's flip
axes `{None,0,1,2}` are flyvis's `{0,1,2,3}`).

## The minimal adjustment

Code (one file): `flyvis/datasets/augmentation/temporal.py` gains an
`original_sampling` flag on `Interpolate` that switches the two operations above.
`flyvis/datasets/sintel.py` passes it through as a dataset argument
(`original_sampling`, default `False`, so public behaviour is unchanged).

Config (no code): two profiles selectable on the command line

* `flyvis/config/task/task_original.yaml` — `dataset.original_sampling: true`
* `flyvis/config/scheduler/scheduler_original.yaml` — `sched_stop_iter: 200000`,
  `chkpt_every_epoch: 334`

```bash
flyvis train-single task_name=flow ensemble_and_network_id=XXXX/000 \
    task=task_original scheduler=scheduler_original
```

`HyperParamScheduler` already reads `sched_stop_iter` from the scheduler config,
so difference 3 needs no code change at all.

## Caveats

* The 250k-vs-200k question is separate from difference 3. The paper-era runs
  stopped at 200k; the later dvs-sim runs reached 250k by resuming with the
  optimizer state and `sched_stop_iter=200000`. The profile above reproduces the
  *schedule* of the resumed runs in a single continuous run; a faithful
  reproduction of the paper-era runs is `task.n_iters=200000` instead.
* Random draw order still differs during training: dvs-sim adds pixel noise to the
  whole raw sequence and then selects the temporal window, flyvis crops first and
  then adds noise, so the two consume the RNG differently. The augmentation
  distribution is the same, but two runs cannot be compared frame by frame with
  augmentation on. Trajectory comparisons therefore run with augmentation off.
* The `syn_count`/`syn_strength` grouping differs in one inert respect: dvs-sim
  groups by `(source_type, target_type, edge_type)` and flyvis by
  `(source_type, target_type)`. For this chemical-only connectome both give 604
  groups with the same partition; on a connectome with electrical edges they would
  not.

## Reproducing the comparison

The dvs-sim checkout is at `~/Projects/dvs-sim` (its `data/SintelDataSet` is a
symlink to the flyvis copy). Two local adaptations were needed, neither on the
compared path:

* the Sintel *depth* dataset is not downloaded here, so `MultiTaskSintelWrap.build`
  skips depth rendering (flyvis skips it too for the flow task);
* dvs-sim calls `.cuda()` unconditionally in the decoder; `common.py` redirects
  `Tensor.cuda` to a no-op on CPU rather than editing the checkout.

Environment: conda env `dvssim-cmp` (clone of `flyvis-gnn-mac`, plus `pynvml` and
`pymatreader`), python 3.12, torch 2.7.1, CPU.

## Recorded output

`stage6_trajectory_output.txt` and `stage7_output.txt` hold the output of the two
runs quoted above.
