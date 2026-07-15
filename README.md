# Tackling Long-Horizon Tasks with Model-based Offline Reinforcement Learning

This repository contains the official implementation of [Tackling Long-Horizon Tasks with Model-based Offline Reinforcement Learning](https://kwanyoungpark.github.io/LEQ/) by [Kwanyoung Park](https://kwanyoungpark.github.io/) and [Youngwoon Lee](https://youngwoon.github.io/).

If you use this code for your research, please consider citing our paper:

```
@article{park2024tackling,
  title={Tackling Long-Horizon Tasks with Model-based Offline Reinforcement Learning},
  author={Kwanyoung Park and Youngwoon Lee},
  journal={arXiv Preprint arxiv:2407.00699},
  year={2024}
}
```

## How to run the code

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)
- An NVIDIA GPU with a driver compatible with CUDA 12.2

### Setup with Docker

The provided Docker image (`nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04`) ships with Python 3.10, the system libraries needed for headless MuJoCo rendering (EGL, GLEW, GLFW, Mesa), and all Python dependencies from `requirements.txt`, JAX (CUDA 12), OfflineRL-Kit (Gymnasium MuJoCo v5 + native `mujoco`, no `mujoco-py`), and pinned `numpy`/`scipy` versions. No conda, venv, or other virtual environment is required.

**1. Build the image**

From the repository root:

```bash
cd docker
# Optionally edit USER, USER_ID, and GROUP_ID in build.sh to match your host user.
./build.sh
```

**2. Create and enter the container**

```bash
./create_container.sh
```

This starts an interactive container named `offline_rl_leq` with the repo mounted at `/workspace/LEQ`. Python dependencies are already installed in the image during `docker build`.

**Container management**


| Action                           | Command                                       |
| -------------------------------- | --------------------------------------------- |
| Re-attach to a stopped container | `cd docker && ./resume_container.sh`          |
| Start in the background          | `cd docker && ./resume_container_detached.sh` |
| Stop the container               | `cd docker && ./stop_container.sh`            |
| Remove the container             | `cd docker && ./remove_container.sh`          |


All training commands below should be run from `/workspace/LEQ` inside the container.

### Pretrain world model

For training the world model, we use the training script of [OfflineRL-Kit](https://github.com/yihaosun1124/mobile/tree/main).

For convenience, we provide `run_dynamics.py` that can be utilized to train the model with OfflineRL-Kit. Run these commands inside the container:

```bash
cd /workspace
git clone --branch bringing_it_up_to_date https://github.com/mohammad200h/OfflineRL-Kit.git
cd OfflineRL-Kit
pip install -e .
cp ../LEQ/run_dynamics.py run_example/run_dynamics.py
```

Offline datasets are loaded from the original D4RL HDF5 files; evaluation uses Gymnasium MuJoCo v5 with the native `mujoco` package (no `mujoco-py` / legacy `d4rl` install required).

Now, you can train the model with the `run_dynamics.py`. For example, you can run the command as below:

```bash
python3 run_example/run_dynamics.py --track --task walker2d-medium-replay-v2 --seed 3 --track
```

### Run training

Run from `/workspace/LEQ` inside the container.

#### LEQ

```bash
cd /workspace/LEQ
PYTHONPATH='.' python3 train/train_LEQ.py --env_name=walker2d-medium-replay-v2 --expectile 0.5 --seed 3
```

#### MOBILEQ (Please refer to the ablation study section of the paper for details)

```bash
cd /workspace/LEQ
PYTHONPATH='.' python train/train_MOBILEQ.py --env_name=Hopper-v3-medium --beta 1.0
```

#### MOBILE (Jax implementation of [Sun et al.](https://github.com/yihaosun1124/mobile/tree/main))

```bash
cd /workspace/LEQ
PYTHONPATH='.' python train/train_MOBILE.py --env_name=antmaze-large-play-v2 --beta 1.0
```

## References

- The implementation is based on [IQL](https://github.com/ikostrikov/implicit_q_learning/).
- MOBILE implementation is from [OfflineRLKit](https://github.com/yihaosun1124/OfflineRL-Kit).

