# Walker2D dynamics hyperparameter search

Grid-search ensemble dynamics training by launching `run_dynamics.py` once per
config combo. Task: **`walker2d-medium-replay-v2`**. The winner is the run with
the **lowest elite holdout loss** (parsed from training logs). Demo trajectory
delta eval is logged for monitoring only — it is not used for selection.

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
| `dynamic_modelf.yaml` | Default focused grid (LR × WD × seeds) |
| `dynamic_modelf_comprehensive.yaml` | All architecture / train knobs expanded |

## Run

Focused search (default config):

```bash
cd /workspace/OfflineRL-Kit
python3 ../LEQ/hyperparameter_search/Walker2D/dynamic_model_search.py
```

Comprehensive search (start with one seed — the full grid is large):

```bash
cd /workspace/OfflineRL-Kit
python3 ../LEQ/hyperparameter_search/Walker2D/dynamic_model_search.py \
  --config ../LEQ/hyperparameter_search/Walker2D/dynamic_modelf_comprehensive.yaml \
  --seed 1
```

Optional:

```bash
python3 ../LEQ/hyperparameter_search/Walker2D/dynamic_model_search.py --seed 2
python3 ../LEQ/hyperparameter_search/Walker2D/dynamic_model_search.py --no-track
python3 ../LEQ/hyperparameter_search/Walker2D/dynamic_model_search.py --project my_wandb_project
```

Extra args after the script name are forwarded to every `run_dynamics.py` call.

After a search finishes, install the winner for `train_LEQ.py`:

```bash
cd /workspace/LEQ/hyperparameter_search/Walker2D
python3 copy_best_model.py --force
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
`max_epochs_since_update`, `reward_mode`, `num_reward_bins`,
`reward_loss_weight`, `dynamics_loss_weight`, `eval_num_trajs`,
`eval_fixed_trajs`, `eval_freq`.

When `dynamics_hidden_dims` depth changes, `dynamics_weight_decay_base` is
**resampled** to length `len(hidden_dims) + 1` (required by OfflineRL-Kit), then
scaled by `dynamics_weight_decay_scale`.

### Selection vs monitoring

- **Selection:** elite mean holdout loss printed at the end of training → written
  to `best_run.yaml`.
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
| Best params | `LEQ/hyperparameter_search/Walker2D/best_run.yaml` |
| WandB project | `dynamics_walker2d_hyperparameter_search` (override with `--project`) |
| Run name | Encodes knobs, e.g. `dynamics_walker2d_ep250_lr1e-3_wd2x_ens7e5_h200x4_hr0p2_bs256_lv0p01_seed3` |

## Tips

1. Expand **one axis at a time** when debugging; full multi-axis grids grow fast.
2. Use `--seed 1` on the comprehensive config first, then confirm top settings
   with `seeds: [1, 2, 3, 4, 5]` in a smaller YAML.
3. Re-copy `run_dynamics.py` after pulling changes that add CLI flags.
4. `copy_best_model.py` symlinks the winning checkpoint to
   `models/dynamics-ensemble/<seed>/walker2d-medium-replay-v2/` so `train_LEQ.py`
   finds it.
5. After installing the best model, train LEQ with matching seed:

```bash
cd /workspace/LEQ
PYTHONPATH='.' python3 train/train_LEQ.py \
  --env_name=walker2d-medium-replay-v2 --expectile 0.5 --seed <seed>
```
