# Reproducing the published ensemble with the current flyvis code

How to train an ensemble that is statistically indistinguishable from the
published `flow/0000`, and how to compare against it without introducing
artifacts. Every claim below was checked by measurement or by reading the
dvs-sim code at the commit that trained the published models; the checks live in
`/Users/janne.lappalainen/Projects/flyvis/scripts/dvs_sim_parity/` and
`/Users/janne.lappalainen/Projects/flyvis/scripts/degree_matched_random/`.

`flow/0000` is dvs-sim ensemble 92, which is ensembles `0070 + 0080 + 0089 +
0091` concatenated (4 + 20 + 20 + 6 = 50 members), all launched identically, then
resumed from 200k to 250k iterations at a flat learning rate. It was exported to
flyvis format by
`/Users/janne.lappalainen/Projects/dvs-sim/scripts/cleanup/ensemble_to_flyvision.py`.

## 1. Train

```bash
flyvis train-single \
    task_name=flow \
    ensemble_and_network_id=XXXX/NNN \
    description="reproduction of the published training regime" \
    task=task_original \
    scheduler=scheduler_original \
    task.n_iters=250000 \
    scheduler.sched_stop_iter=200000
```

or for a whole ensemble, one array task per member:

```bash
sbatch --array=0-49 \
    /Users/janne.lappalainen/Projects/flyvis/scripts/degree_matched_random/hpc/train_reference.sbatch 0310
```

**Give every member the same config.** Do not vary a seed across members. The
published 50 have byte-identical configs, network and decoder initialization are
deterministic, and nothing in the flyvis training path seeds the global RNG — so
members differ only through data order and augmentation draws. That is the same
source of ensemble diversity the published models have. Adding a per-member seed
would make your ensemble *less* comparable, not more.

### Why those flags

| flag | what it fixes |
|---|---|
| `task=task_original` | sets `dataset.original_sampling=true`, restoring the original temporal resampling of the input movie (index-based `floor(arange(0, n, 24/50))` rather than `nearest-exact` interpolation) and `align_corners=False` for the flow targets. Also sets `original_split=true`. |
| `scheduler=scheduler_original` | `sched_stop_iter: 200000` and `chkpt_every_epoch: 334` |
| `task.n_iters=250000` with `sched_stop_iter=200000` | reproduces the published learning-rate trajectory: ten stepwise drops of 20k iterations each from 5e-5 to 5e-6 over 0–200k, then 5e-6 held to 250k. The published models reached that shape as 200k of training plus a 50k resume at flat 5e-6; one continuous run with these settings gives a bit-identical lr array over all 250,001 iterations. |

Do **not** use plain `scheduler`: that spreads the ten steps over the full 250k,
which is a different trajectory.

## 2. Select the best checkpoint by EPE, not by the training loss

This is the step that is easy to miss and it is worth about 1% of task error.

The published checkpoints were chosen by `argmin` **EPE**, in a retrospective
re-validation pass after training —
`/Users/janne.lappalainen/Projects/dvs-sim/scripts/cleanup/ensemble_to_flyvision.py:203-207`:

```python
validation_subwrap = "original_validation_v2"
validation_loss_fn = "epe"
```

flyvis's `best_checkpoint_default_fn`
(`/Users/janne.lappalainen/Projects/flyvis/flyvis/utils/chkpt_utils.py:218`)
defaults to `loss_file_name="epe"`, but training records only `loss` (l2norm), so
`check_loss_name` silently falls back to it. Comparing an l2norm-selected
ensemble against an EPE-selected one, on EPE, penalises yours:

| `flow/0310` (n=48) vs published (n=50) | EPE | difference | Welch p |
|---|---|---|---|
| l2norm-selected | 5.3809 ± 0.0926 | +0.0500 ± 0.0176 | **0.0054** |
| **EPE-selected** | 5.3476 ± 0.0857 | **+0.0167 ± 0.0168** | **0.32** |
| published | 5.3309 ± 0.0805 | — | — |

Selecting on EPE recovers 67% of the gap and makes the ensembles statistically
indistinguishable. The selection cost is strongly right-skewed — median 0.0134,
mean 0.0333, max 0.2380, and zero for 9 of 48 members — so a handful of networks
where the two metrics disagree dominates the ensemble mean. **Do not estimate
this from a few models**: on a 6-member ensemble the mean cost came out at
0.0038, which led me to dismiss the effect entirely before the 48-member run
contradicted it.

Two ways to get it right:

* **Best:** record EPE during validation so the default selection uses it
  natively. Nothing in the current training path writes an `epe` validation file.
* **After the fact:** score every checkpoint by EPE on a fixed protocol and
  select on that, with
  `/Users/janne.lappalainen/Projects/flyvis/scripts/degree_matched_random/checkpoint_selection.py`
  (~0.8 s per checkpoint on a GPU, so ~50 s per 65-checkpoint member).

## 3. Compare on one protocol, never on stored values

**The published `validation/loss.h5` contains EPE, not l2norm.** The port script
wrote the EPE value under the name `loss`, which is why it reads ~5.14 while a
locally trained ensemble's `validation/loss.h5` reads ~1150. They are different
metrics, not different normalizations. Recomputing EPE for the published networks
agrees with their stored values at r = 0.998.

So evaluate every network yourself, in one process, on one dataset and one split:
`/Users/janne.lappalainen/Projects/flyvis/scripts/degree_matched_random/evaluate_ensemble.py`.
It also reports a zero-prediction baseline, which is the reference for "did this
learn anything".

### Use the original split

`original_split=True` reproduces the split the published models trained on. That
split was seed-controlled in dvs-sim at the time and later hardcoded; the
hardcoded validation sequence ids `[2, 7, 9, 12, 13, 16]` are exactly
`CrossValIndices(n_samples=23, folds=4, shuffle=True, seed=0)` at fold 1. The
launches passed `-data_seed 1`, which selected the *fold*, while the shuffle seed
stayed at its default of 0.

**Do not fall back to the fold split.** `get_random_data_split(fold=1, n_folds=4,
seed=0)` is a different algorithm and gives a different split despite the
matching nominal seed and fold — 16 vs 17 validation sequences sharing only 6.
Ten of the original split's validation sequences are *training* data under the
fold split. The published models' stored config has no `original_split` key
(the port script hardcoded `n_folds=4, fold=1, seed=0` with the comments
`# added as fixed` and `# updated seed value`), so it looks like a fold-split
model and is not.

### EnsembleView needs the loss file name

For any ensemble trained here:

```python
EnsembleView(
    "flow/XXXX",
    best_checkpoint_fn_kwargs={"validation_subdir": "validation", "loss_file_name": "loss"},
)
```

Without it, `task_error()`, `argsort()`, `training_loss()` and
`validation_loss()` raise `FileNotFoundError` on the missing `epe.h5`.

## 4. Judge equivalence with an equivalence test

A non-significant t-test does not demonstrate equivalence — it is also what an
underpowered comparison produces. Use TOST against a margin chosen as the effect
size the pipeline is later meant to resolve, with
`/Users/janne.lappalainen/Projects/flyvis/scripts/degree_matched_random/compare_ensembles.py`.
At n = 50 versus 50 the standard error is about 2.7 l2norm units and 0.018 EPE,
which is enough to bound the difference well inside a 13.5 / 0.078 margin.

## 5. What needs no attention

Verified equivalent, so do not spend time on these:

* **connectome** — 45,669 nodes and 1,513,231 edges agree element for element
  after normalizing three cell type names (`TmY17`/`TmY18`, CT1 parentheses)
* **parameters** — 734 free parameters, identical values at seed 0; `syn_strength`
  init differs by one float32 ULP in 147 of 604 groups from `(1/⟨N⟩)*scale` vs
  `scale/⟨N⟩`
* **dynamics** — `StaticSynapses` and `PPNeuronIGRSynapses` compute the same
  velocity on this chemical-only connectome
* **loss** — dvs-sim's `l2_3d_wotc` is flyvis's `l2norm` up to a `+1e-9`
  stabiliser that is invisible in float32. Note the launches used `l2_3d_wotc`,
  the variant that keeps **all** frames; the sibling `l2_3d` drops the first
  `n_frames // 4` and was not used. `l2_3d_wotc` had the correct body from the
  commit that introduced it and was never an alias of `l2_3d`.
* **augmentation** — the samplers are distributionally identical (no rotation
  w.p. ½ and each of five rotations w.p. 1/10; no flip w.p. ½ and each of three
  mirror axes w.p. 1/6; dvs-sim's flip labels `{None,0,1,2}` are flyvis's
  `{0,1,2,3}`). Over 200 draws with augmentation on, no `lum` or `flow` statistic
  differs by more than 0.4%.
* **the redundant augmentation grid** — dvs-sim's rich dataset enumerated
  `4 flip axes × 6 rotations = 24` combinations over a 12-element symmetry group,
  covering each symmetry twice; flyvis's `AugmentedSintel` uses `[0,1] × [0..5] =
  12` and is non-redundant. Uniform duplication averages to the same value, so
  this cost dvs-sim compute but changed no result. It never applied to training,
  which samples rather than enumerating.
* **GPU model** — members trained on 1080ti/titanxp/v100 and on a100 differ by
  +0.018 EPE, p = 0.55.

## 6. Known residual

After matching everything above, an EPE-selected reproduction sits **+0.0167 ±
0.0168 EPE** above the published ensemble (p = 0.32, i.e. not distinguishable at
n ≈ 50). The direction is consistent across three held-out sets, so a small real
offset cannot be excluded, but no mechanism for it survived testing: learning-rate
schedule, iteration count, frozen `syn_strength`, the train/val split, the
temporal interpolation, the augmentation distribution, and restart-versus-
continuous training were each ruled out by direct measurement. The `relu_leak`
schedule that looks like a candidate is inert, because the setter only applies
when the activation exposes `negative_slope` and a plain `ReLU` does not.

Treat a residual of this size as expected drift between a years-old codebase and
its refactor, and anchor connectome or architecture comparisons to a reference
ensemble trained with *this* pipeline rather than to `flow/0000` directly.
