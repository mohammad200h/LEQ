# MountainCar dynamics hyperparameter search

Grid-search ensemble dynamics training by launching `run_dynamics.py` once per
config combo. The winner is the run with the **lowest elite holdout loss**
(parsed from training logs). Demo trajectory delta eval is logged for monitoring
only — it is not used for selection.

## Setup

```bash
cd /workspace/LEQ
./copy_dynamic_script_to_offline_rl.sh
```

That copies `LEQ/run_dynamics.py` into `OfflineRL-Kit/run_example/` (needed so
new CLI flags like `--holdout-ratio` are available).

Search knobs live in:

| File | Role |
| --- | --- |
| `dynamic_modelf.yaml` | Comprehensive grid (LR × WD × ensemble × arch × holdout × batch × logvar × reward head) |

The full cartesian product is large (~512 runs with `--seed 1`, covering both
`twohot` and `gaussian_joint` reward heads). Prefer one seed first, then confirm
the winning region across seeds.

## Run

```bash
cd /workspace/OfflineRL-Kit
python3 ../LEQ/hyperparameter_search/MountainCar/dynamic_model/dynamic_model_search.py \
  --seed 1
```

Optional:

```bash
python3 ../LEQ/hyperparameter_search/MountainCar/dynamic_model/dynamic_model_search.py --seed 2
python3 ../LEQ/hyperparameter_search/MountainCar/dynamic_model/dynamic_model_search.py --no-track
python3 ../LEQ/hyperparameter_search/MountainCar/dynamic_model/dynamic_model_search.py --project my_wandb_project
```

Extra args after the script name are forwarded to every `run_dynamics.py` call.

After a search finishes, install a winner for `train_LEQ.py` (one reward head
at a time — both write separate `best_run_*.yaml` files):

```bash
cd /workspace/LEQ/hyperparameter_search/MountainCar/dynamic_model
python3 copy_best_model.py --reward-mode twohot --force
python3 copy_best_model.py --reward-mode gaussian_joint --force
python3 copy_best_model.py --all-reward-modes   # list winners only
```

## What is searched

Any YAML value may be a scalar or a list (list ⇒ search axis). The runner takes
the **cartesian product** of all axes and skips combos with `n_elites > n_ensemble`.

| Knob | YAML key | CLI flag | What it does |
| --- | --- | --- | --- |
| Max epochs | `dynamics_max_epochs` | `--dynamics-max-epochs` | Hard cap on dynamics epochs |
| Learning rate | `dynamics_lr` | `--dynamics-lr` | Adam LR for the ensemble |
| Weight-decay scale | `dynamics_weight_decay_scale` | (scales base → `--dynamics-weight-decay`) | Multiplies the per-layer WD schedule |
| Ensemble size | `n_ensemble` | `--n-ensemble` | Number of dynamics members |
| Elites | `n_elites` | `--n-elites` | How many members are kept as elites |
| Hidden dims | `dynamics_hidden_dims` | `--dynamics-hidden-dims` | MLP width/depth (one list, or list-of-lists) |
| Holdout ratio | `holdout_ratio` | `--holdout-ratio` | Fraction of data for validation / elite pick (capped at 1000 samples) |
| Batch size | `dynamics_batch_size` | `--dynamics-batch-size` | Mini-batch size inside dynamics `learn()` |
| Log-var coef | `logvar_loss_coef` | `--logvar-loss-coef` | Weight on Gaussian log-variance regularizer |
| Reward mode | `reward_mode` | `--reward-mode` | `twohot` (symlog + categorical) or `gaussian_joint` (legacy) |
| Reward bins | `num_reward_bins` | `--num-reward-bins` | Symlog bin count for twohot reward head |
| Reward loss wt | `reward_loss_weight` | `--reward-loss-weight` | CE weight on reward head (twohot mode) |
| Dynamics loss wt | `dynamics_loss_weight` | `--dynamics-loss-weight` | Weight on Δs Gaussian loss (twohot mode) |
| Seeds | `seeds` / `seed` | `--seed` | Reproducibility; CLI `--seed` overrides YAML |

Also fixed from YAML (not usually gridded): `early_stopping`,
`max_epochs_since_update`, `num_reward_bins`, `reward_loss_weight`,
`dynamics_loss_weight`, `eval_num_trajs`, `eval_fixed_trajs`, `eval_freq`.

When `dynamics_hidden_dims` depth changes, `dynamics_weight_decay_base` is
**resampled** to length `len(hidden_dims) + 1` (required by OfflineRL-Kit), then
scaled by `dynamics_weight_decay_scale`.

### Selection vs monitoring

- **Selection:** elite mean holdout loss, **separately per `reward_mode`**
  (twohot and gaussian_joint losses are not comparable) → written to
  `best_run_twohot.yaml` and `best_run_gaussian_joint.yaml`.
- **Monitoring:** demo delta eval (`eval_*`) → WandB / logs only.

### Out of scope for this search

- **Ray Tune parallel scheduling** (`OfflineRL-Kit/tune_example`): different
  stack; this runner is sequential subprocesses. Use multiple machines or trim
  the grid instead.
- **Policy knobs** (`real_ratio`, rollout length, CQL/expectile, etc.): tune
  those in `train_LEQ.py` / policy scripts **after** the dynamics model is
  frozen.

## Outputs

| Artifact | Path / name |
| --- | --- |
| Checkpoints | `OfflineRL-Kit/models/dynamics-ensemble/<seed>/<run_name>/` |
| Best params | `best_run_twohot.yaml` and `best_run_gaussian_joint.yaml` |
| WandB project | `dynamics_mountaincar_hyperparameter_search` (override with `--project`) |
| Run name | Encodes knobs, e.g. `dynamics_mountaincar_ep250_lr1e-3_wd2x_ens7e5_h200x4_hr0p2_bs256_lv0p01_rmtwohot_seed3` |

## Tips

1. Expand **one axis at a time** when debugging; full multi-axis grids grow fast.
2. Use `--seed 1` first, then confirm top settings with
   `seeds: [1, 2, 3, 4, 5]` in a trimmed YAML.
3. Re-copy `run_dynamics.py` after pulling changes that add CLI flags.
4. `copy_best_model.py` symlinks the winning checkpoint to
   `models/dynamics-ensemble/<seed>/<task>/` so `train_LEQ.py` finds it.

## Tips for small data (MountainCar ≈ 80 trajectories)

Human MountainCar demos are tiny by OfflineRL-Kit standards (~80 trajectories,
on the order of ~10k transitions). Defaults aimed at D4RL MuJoCo can overfit
quickly or leave too little data for training once holdout is taken. Prefer:

| Prefer | Why on ~80 trajs |
| --- | --- |
| Stronger weight decay (`dynamics_weight_decay_scale: 2.0` or higher) | Main regularizer when N is small |
| Smaller / shallower nets (`[200, 200]` before `[200]×4`) | Fewer params fit the demo support better |
| Slightly lower LR (`3e-4` vs `1e-3`) | Slower fits; early stopping has room to work |
| Early stopping on (`max_epochs_since_update: 10`, cap ~250) | Holdout picks the checkpoint; don’t force long training |
| Modest ensemble (`n_ensemble: 5–7`, `n_elites: 3–5`) | Very large ensembles add variance without more data |
| `holdout_ratio: 0.1–0.2` | Keep most transitions for training; 0.2 of ~10k is still only ~1–2k holdout (capped at 1000 anyway) |
| Smaller batches (`128` as well as `256`) | Noisier updates can act like mild regularization |
| Several seeds | With little data, ranking is noisy — confirm winners across seeds |

Practical workflow:

1. Run `dynamic_modelf.yaml` with `--seed 1` (trim axes if needed).
2. Confirm the winning region across seeds.
3. Trust **holdout loss** for checkpoint selection; use fixed-traj demo delta
   eval only to spot wild failures (e.g. exploding multi-step error).
4. After installing the best model, keep policy rollouts short at first —
   long model rollouts amplify dynamics error when the dataset is this small.
5. The LEQ search under `../leq/` loads both `best_run_*.yaml` winners via
   `--load_dir` (no install step needed). Best-run YAMLs are updated whenever
   a new mode winner is found, so a long search can be interrupted and still
   leave usable checkpoints for policy tuning.
