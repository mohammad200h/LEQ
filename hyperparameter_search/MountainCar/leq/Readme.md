# MountainCar LEQ hyperparameter search

Grid-search LEQ policy training by launching `train_LEQ.py` once per config
combo. For each combo the runner also sweeps **dynamics reward heads**
(`twohot` / `gaussian_joint`), loading the frozen winner from
`../dynamic_model/best_run_<reward_mode>.yaml` via `--load_dir`.

Winners are the runs with the **highest final eval return** (parsed from
`final score: ...`), tracked **separately per reward_mode**. Intermediate
`Step <n> <return>` lines are a fallback if the final line is missing.

## Prerequisites

1. Finish the dynamics search so both best-run files exist:

```bash
# while / after dynamic_model_search.py
ls ../dynamic_model/best_run_twohot.yaml
ls ../dynamic_model/best_run_gaussian_joint.yaml
```

`leq_search.py` reads those YAMLs and points `--load_dir` at

```text
OfflineRL-Kit/models/dynamics-ensemble/<dynamics_seed>/<output_model_name>/
```

You do **not** need `copy_best_model.py` before the LEQ search (that helper is
only for manual single-model training that uses the env-name path).

2. Run searches from `/workspace/LEQ` with `PYTHONPATH='.'`.

## Search config

| File | Role |
| --- | --- |
| `leq.yaml` | Comprehensive grid (expectile × mbr × rollout × horizon × λ × actor LR × pretrain × reward_mode × seeds) |

The full cartesian product is large (~1152 runs with three seeds and both
reward heads). Start with one seed and/or trim axes while exploring.

## Run

Wait until the dynamics search has finished (or at least written both
`best_run_*.yaml` files), then:

```bash
cd /workspace/LEQ
PYTHONPATH='.' python3 hyperparameter_search/MountainCar/leq/leq_search.py \
  --seed 1
```

Optional:

```bash
PYTHONPATH='.' python3 hyperparameter_search/MountainCar/leq/leq_search.py --seed 3
PYTHONPATH='.' python3 hyperparameter_search/MountainCar/leq/leq_search.py --no-track
PYTHONPATH='.' python3 hyperparameter_search/MountainCar/leq/leq_search.py --project my_wandb_project
PYTHONPATH='.' python3 hyperparameter_search/MountainCar/leq/leq_search.py --wandb-key "$WANDB_API_KEY"
```

To search only one dynamics head, set `reward_mode: twohot` (or
`gaussian_joint`) in `leq.yaml`.

Extra args after the script name are forwarded to every `train_LEQ.py` call,
e.g. `--max_steps=100000` for a smoke test.

After a search finishes, install a policy winner under the default `./tmp/EP/`
tree:

```bash
cd /workspace/LEQ/hyperparameter_search/MountainCar/leq
python3 copy_best_model.py --reward-mode twohot --force
python3 copy_best_model.py --reward-mode gaussian_joint --force
python3 copy_best_model.py --all-reward-modes   # list winners only
```

## What is searched

Any YAML value may be a scalar or a list (list ⇒ search axis). The runner takes
the **cartesian product** of all axes.

| Knob | YAML key | CLI flag | What it does |
| --- | --- | --- | --- |
| Dynamics reward head | `reward_mode` | (via `--load_dir`) | Which frozen dynamics winner to use |
| Expectile | `expectile` | `--expectile` | Conservatism of Q (lower = more pessimistic) |
| Model-batch ratio | `model_batch_ratio` | `--model_batch_ratio` | Fraction of each update from model rollouts |
| Rollout length | `rollout_length` | `--rollout_length` | Imaginary trajectory length |
| Horizon | `horizon_length` | `--horizon_length` | Multi-step / λ-return horizon |
| λ | `lamb` | `--lamb` | GAE / λ-return bias–variance |
| Actor LR | `actor_lr` | `--config.actor_lr` | Actor Adam / cosine schedule peak |
| Discount | `discount` | `--discount` | TD discount |
| Batch size | `batch_size` | `--batch_size` | Mini-batch size |
| Pretrain | `pretrain` | `--pretrain` | BC + FQE warm-start |
| Seeds | `seeds` / `seed` | `--seed` | Policy seed; CLI `--seed` overrides YAML |

Also fixed from YAML (not usually gridded): `max_steps`, `num_layers`,
`layer_size`, `value_lr`, `critic_lr`, rollout buffer knobs, `actor_update`,
`critic_update`, `video_interval`, `eval_interval`, `save_interval`.

### Selection vs monitoring

- **Selection:** final mean eval return, **separately per `reward_mode`**
  → written to `best_run_twohot.yaml` and `best_run_gaussian_joint.yaml`.
- **Monitoring:** mid-training `Step` returns and WandB curves.

### Out of scope for this search

- **Dynamics knobs:** tune those in `../dynamic_model/` and freeze the model
  before searching policy hyperparameters.
- **Ray Tune parallel scheduling:** this runner is sequential subprocesses.

## Outputs

| Artifact | Path / name |
| --- | --- |
| Per-run checkpoints | `LEQ/tmp/EP/leq_search/<run_name>/models/...` |
| Best params | `best_run_twohot.yaml` and `best_run_gaussian_joint.yaml` |
| WandB project | `leq_mountaincar_hyperparameter_search` (override with `--project`) |
| Run name | Encodes knobs + reward head, e.g. `..._pt0_rmtwohot_seed3` |

## Tips

1. Expand **one axis at a time** when debugging; full multi-axis grids grow fast
   and each combo is a full `max_steps` train.
2. Use `--seed 1` first, then confirm top settings across seeds.
3. Pass `--max_steps=50000` as an override while validating the runner.
4. Keep `video_interval: 0` during search; turn videos back on for the winner.
5. On tiny human MountainCar data, prefer shorter `rollout_length` and a lower
   `model_batch_ratio` if the dynamics are imperfect.
6. Compare the two `best_run_*.yaml` policy winners to decide which dynamics
   reward head is better for LEQ on this task.

## Tips for small data (MountainCar ≈ 80 trajectories)

| Prefer | Why on ~80 trajs |
| --- | --- |
| Modest `model_batch_ratio` (`0.1`–`0.25`) | Less trust in a dynamics model fit on tiny data |
| Short `rollout_length` (`1`–`5`) | Long imaginary rollouts amplify model error |
| `expectile` sweep (`0.3`–`0.7`) | Conservatism vs optimism is the main LEQ lever |
| Optional `--pretrain` | BC warm-start can help sparse demos |
| Several seeds | Ranking is noisy — confirm winners across seeds |

Practical workflow:

1. Let `../dynamic_model/` finish → two `best_run_*.yaml` files.
2. Run `leq.yaml` with `--seed 1` (trim axes if needed).
3. Confirm the winning region across seeds.
4. `copy_best_model.py --reward-mode <mode> --force` to expose a winner under
   `./tmp/EP/`.
