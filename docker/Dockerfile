
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository universe

RUN apt-get update && apt-get install -y \
    git mesa-utils \
    python3.10 python3.10-dev python3.10-distutils python3-pip \
    libegl1-mesa libgl1-mesa-glx libglib2.0-0 libfuse2 \
    wget unzip patchelf \
    libosmesa6-dev libglfw3 libglew-dev gcc g++ \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=egl

COPY requirements.txt /tmp/requirements.txt

# Pin JAX to 0.4.14 so it stays compatible with flax==0.7.2 and numpy==1.23.0
# (unpinned jax[cuda12] pulls a version that needs np.dtypes from NumPy >=1.25).
# MuJoCo / Gymnasium come from OfflineRL-Kit (and requirements.txt); no mujoco-py or d4rl.
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    && python3 -m pip install -r /tmp/requirements.txt \
    && python3 -m pip install "jax[cuda12_pip]==0.4.14" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && python3 -m pip install --no-deps \
        flax==0.7.2 orbax-checkpoint==0.2.3 optax==0.1.7 chex==0.1.82 \
    && python3 -m pip install numpy==1.23.0 scipy==1.10.1

WORKDIR /workspace

#####OFFLINE RL KIT INSTALLATION #########
ARG OFFLINE_RL_KIT_CACHEBUST
RUN cd /workspace && \
    echo "OfflineRL-Kit cache bust: ${OFFLINE_RL_KIT_CACHEBUST}" && \
    rm -rf OfflineRL-Kit && \
    git clone --branch bringing_it_up_to_date https://github.com/mohammad200h/OfflineRL-Kit.git && \
    cd OfflineRL-Kit && pip install -e .

# Load WANDB_API_KEY from mounted .env on interactive shells
RUN printf '%s\n' \
    'if [ -f /workspace/LEQ/docker/.env ]; then' \
    '  set -a' \
    '  . /workspace/LEQ/docker/.env' \
    '  set +a' \
    'fi' \
    >> /etc/bash.bashrc

CMD ["/bin/bash"]
