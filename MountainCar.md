# MountainCar training

Train an OfflineRL-Kit ensemble dynamics model, then a LEQ agent, on human
**MountainCarContinuous-v0** demos **without editing** `run_dynamics.py`.

Task name: `mountaincar-human-v0` (maps to `MountainCarContinuous-v0`)

## How data is loaded

`run_dynamics.py` calls `make_env(task)` → `env.get_dataset()`, which reads:


| Piece     | Value for MountainCar                     |
| --------- | ----------------------------------------- |
| Directory | `$D4RL_DATASET_DIR` or `~/.d4rl/datasets` |
| File      | `mountain_car_human.hdf5`                 |


Dropbox demos are downloaded to
`data_colllection/demostrations/mountain_car_human.hdf5`. Use
`install_mountain_car_dataset.py` to link that file into the cache above.

Trained weights are saved under OfflineRL-Kit:

```text
models/dynamics-ensemble/<seed>/mountaincar-human-v0/
```

## Setup

```bash
# OfflineRL-Kit + LEQ dynamics script
cd /workspace/OfflineRL-Kit
pip install -e .

cd /workspace/LEQ
./copy_dynamic_script_to_offline_rl.sh
```

## 1. Download demos

```bash
cd /workspace/OfflineRL-Kit
python3 data_colllection/download_from_dropbox.py
# token elsewhere:
# python3 data_colllection/download_from_dropbox.py --env /workspace/LEQ/docker/.env
```

Produces: `data_colllection/demostrations/mountain_car_human.hdf5`

## 2. Install into the D4RL dataset path

```bash
cd /workspace/OfflineRL-Kit
python3 data_colllection/install_mountain_car_dataset.py
```

By default this symlinks:

```text
~/.d4rl/datasets/mountain_car_human.hdf5
  → …/data_colllection/demostrations/mountain_car_human.hdf5
```

Useful flags:


| Flag                 | Meaning                                          |
| -------------------- | ------------------------------------------------ |
| `--copy`             | Copy instead of symlink                          |
| `--force`            | Replace an existing destination                  |
| `--dataset-dir PATH` | Override cache dir (same as `$D4RL_DATASET_DIR`) |
| `--source PATH`      | Non-default demo HDF5                            |


## 3. Train dynamics

```bash
cd /workspace/OfflineRL-Kit
python3 run_example/run_dynamics.py --task mountaincar-human-v0 --seed 1 --track
```

Optional: `--track` for W&B.

Output:

```text
/workspace/OfflineRL-Kit/models/dynamics-ensemble/1/mountaincar-human-v0/
```

## 4. Train the LEQ agent

Requires dynamics weights from step 3 at:

```text
../OfflineRL-Kit/models/dynamics-ensemble/<seed>/mountaincar-human-v0/
```

For seeds `1`–`5`, `train_LEQ.py` loads that seed’s dynamics folder; for other
seeds it falls back to seed `1`. Use the same `--seed` you used for dynamics.

```bash
cd /workspace/LEQ
PYTHONPATH='.' python3 train/train_LEQ.py --env_name=mountaincar-human-v0 --expectile 0.5 --seed 1
```

Optional flags match the other LEQ runs (e.g. `--wandb_key`, `--save_dir`).

## Checklist

1. `download_from_dropbox.py` → demos HDF5 present
2. `install_mountain_car_dataset.py` → file visible under `~/.d4rl/datasets/`
3. `copy_dynamic_script_to_offline_rl.sh` → `run_example/run_dynamics.py`
4. `python3 run_example/run_dynamics.py --task mountaincar-human-v0 --seed 1`
5. `PYTHONPATH='.' python3 train/train_LEQ.py --env_name=mountaincar-human-v0 --expectile 0.5 --seed 1`

