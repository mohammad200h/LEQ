## This code is the direct reimplementation of the world model in OfflineRLKit
## (https://github.com/yihaosun1124/OfflineRL-Kit/blob/main/offlinerlkit/dynamics/ensemble_dynamics.py)

from typing import Optional, Sequence, Tuple, Callable
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from common import Model, PRNGKey


def softplus(x):
    return jnp.logaddexp(x, 0)


def soft_clamp(x, _min, _max):
    x = _max - softplus(_max - x)
    x = _min + softplus(x - _min)
    return x


def decode_reward_logits(logits: jnp.ndarray, bin_centers: jnp.ndarray) -> jnp.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = jax.nn.softmax(logits, axis=-1)
    symlog_val = jnp.tensordot(probs, bin_centers, axes=([-1], [0]))
    return jnp.sign(symlog_val) * jnp.expm1(jnp.abs(symlog_val))


class EnsembleLinear(nn.Module):
    input_dim: int
    output_dim: int
    num_ensemble: int
    use_norm: bool = False

    def setup(self):
        self.weight = self.param(
            "kernel",
            nn.initializers.glorot_normal(),
            (self.num_ensemble, self.input_dim, self.output_dim),
        )
        self.bias = self.param(
            "bias",
            nn.initializers.glorot_normal(),
            (self.num_ensemble, 1, self.output_dim),
        )
        if self.use_norm:
            self.norm = nn.LayerNorm(epsilon=1e-05)

    def __call__(self, x: jnp.ndarray):
        x = jnp.einsum("nbi,nij->nbj", x, self.weight)
        if self.use_norm:
            x = self.norm(x)
        else:
            x = x + self.bias
        return x


class EnsembleWorldModel(nn.Module):
    num_models: int
    num_elites: int
    hidden_dims: Sequence[int]
    obs_dim: int
    action_dim: int
    reward_mode: str = "gaussian_joint"
    num_reward_bins: int = 255
    dropout_rate: Optional[float] = None
    use_norm: bool = False

    def setup(self):
        hidden_dims = (self.obs_dim + self.action_dim,) + self.hidden_dims
        self.layers = [
            EnsembleLinear(
                hidden_dims[i - 1], hidden_dims[i], self.num_models, self.use_norm
            )
            for i in range(1, len(hidden_dims))
        ]
        if self.reward_mode == "twohot":
            self.last_layer = EnsembleLinear(
                self.hidden_dims[-1], 2 * self.obs_dim, self.num_models
            )
            self.reward_head = EnsembleLinear(
                self.hidden_dims[-1], self.num_reward_bins, self.num_models
            )
            logvar_dim = self.obs_dim
        else:
            self.last_layer = EnsembleLinear(
                self.hidden_dims[-1], 2 * (self.obs_dim + 1), self.num_models
            )
            self.reward_head = None
            logvar_dim = self.obs_dim + 1
        self.min_logvar = self.param(
            "min_logvar", nn.initializers.zeros, (logvar_dim,)
        )
        self.max_logvar = self.param(
            "max_logvar", nn.initializers.zeros, (logvar_dim,)
        )

    def __call__(
        self,
        z: jnp.ndarray,
        training: bool = False,
    ):
        if len(z.shape) == 2:
            z = z[None, :, :].repeat(self.num_models, axis=0)
        for layer in self.layers:
            z = layer(z)
            z = nn.swish(z)
        if self.reward_mode == "twohot":
            dynamics = self.last_layer(z)
            mean, logvar = jnp.split(dynamics, 2, axis=-1)
            logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
            reward_logits = self.reward_head(z)
            return mean, logvar, reward_logits
        z = self.last_layer(z)
        mean, logvar = z[:, :, : self.obs_dim + 1], z[:, :, self.obs_dim + 1 :]
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar


class EnsembleDynamicModel(nn.Module):
    model: nn.Module
    elites: Tuple[int]
    terminal_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    output_all: bool = False
    clip_extremes: bool = False
    reward_mean: bool = False
    reward_mode: str = "gaussian_joint"
    bin_centers: jnp.ndarray = None

    def setup(self):
        self.scaler = self.param(
            "scaler",
            nn.initializers.ones,
            (2, self.model.obs_dim + self.model.action_dim),
        )
        self.reward_scaler = self.param("reward_scaler", nn.initializers.zeros, (2,))

    def __call__(
        self,
        key: PRNGKey,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        training: bool = False,
        model_idxs: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        key1, key2 = jax.random.split(key)
        shapes = observations.shape[:-1]
        observations = observations.reshape(-1, observations.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])

        z = jnp.concatenate([observations, actions], axis=1)
        z = (z - self.scaler[0]) / self.scaler[1]

        if self.reward_mode == "twohot":
            delta_mean, delta_logvar, reward_logits = self.model(z)
            std = jnp.sqrt(jnp.exp(delta_logvar))
            delta_samples = delta_mean + jax.random.normal(key1, delta_mean.shape) * std
            num_models, batch_size, _ = delta_samples.shape
            if model_idxs is None:
                model_idxs = jax.random.choice(key2, jnp.array(self.elites), (1, batch_size, 1))
            if self.reward_mean:
                next_obs = observations[None, :, :] + delta_samples
                next_obs = jnp.take_along_axis(next_obs, model_idxs, axis=0)[0]
                elite_logits = reward_logits[jnp.array(self.elites)]
                reward = decode_reward_logits(
                    elite_logits.mean(axis=0), self.bin_centers
                )
            else:
                samples = jnp.take_along_axis(delta_samples, model_idxs, axis=0)[0]
                next_obs = observations + samples
                logits = jnp.take_along_axis(reward_logits, model_idxs, axis=0)[0]
                reward = decode_reward_logits(logits, self.bin_centers)
            reward = reward * self.reward_scaler[0] + self.reward_scaler[1]
            terminal = self.terminal_fn(observations, actions, next_obs).squeeze(1)
            next_obs = next_obs.reshape((*shapes, -1))
            reward = reward.reshape(shapes)
            terminal = terminal.reshape(shapes)
            info = {"raw_reward": reward}
            return next_obs, reward, terminal, info

        mean, logvar = self.model(z)
        next_obs = mean[:, :, :-1] + observations[None, :, :]
        mean = jnp.concatenate([next_obs, mean[:, :, -1:]], axis=2)
        std = jnp.sqrt(jnp.exp(logvar))
        ensemble_samples = mean + jax.random.normal(key1, mean.shape) * std
        num_models, batch_size, _ = ensemble_samples.shape

        if self.output_all:
            samples = ensemble_samples
            next_obs = samples[..., :-1]
            reward = samples[..., -1] * self.reward_scaler[0] + self.reward_scaler[1]
            terminal = jax.vmap(self.terminal_fn, in_axes=(None, None, 0), out_axes=0)(
                observations, actions, next_obs
            ).squeeze(-1)
            shapes = (samples.shape[0], *shapes)
            next_obs = next_obs.reshape((*shapes, -1))
            reward = reward.reshape(shapes)
            terminal = terminal.reshape(shapes)
        else:
            if model_idxs is None:
                model_idxs = jax.random.choice(key2, jnp.array(self.elites), (1, batch_size, 1))
            if self.reward_mean:
                next_obs = jnp.take_along_axis(
                    ensemble_samples[..., :-1], model_idxs, axis=0
                )[0]
                reward = (
                    ensemble_samples[..., -1].mean(axis=0) * self.reward_scaler[0]
                    + self.reward_scaler[1]
                )
            else:
                samples = jnp.take_along_axis(ensemble_samples, model_idxs, axis=0)[0]
                next_obs = samples[..., :-1]
                reward = (
                    samples[..., -1] * self.reward_scaler[0] + self.reward_scaler[1]
                )
            terminal = self.terminal_fn(observations, actions, next_obs).squeeze(1)
            next_obs = next_obs.reshape((*shapes, -1))
            reward = reward.reshape(shapes)
            terminal = terminal.reshape(shapes)

        if self.clip_extremes:
            obs_scaler = self.scaler[:, : self.model.obs_dim]
            next_obs = jnp.clip(
                next_obs,
                obs_scaler[0] - 15.0 * obs_scaler[1],
                obs_scaler[0] + 15.0 * obs_scaler[1],
            )
        info = {"raw_reward": reward}
        return next_obs, reward, terminal, info


import torch


def _load_reward_config(model_path: str) -> dict:
    path = os.path.join(model_path, "reward_config.json")
    if not os.path.isfile(path):
        return {"reward_mode": "gaussian_joint"}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_world_model(
    model_path,
    obs_dim,
    action_dim,
    reward_scaler,
    termination_fn,
    deterministic=False,
    output_all=False,
    clip_extremes=False,
):
    observations = jax.device_put(np.zeros((1, obs_dim)))
    actions = jax.device_put(np.zeros((1, action_dim)))

    reward_cfg = _load_reward_config(model_path)
    reward_mode = reward_cfg.get("reward_mode", "gaussian_joint")
    encoder_cfg = reward_cfg.get("encoder", {})
    num_reward_bins = int(encoder_cfg.get("num_bins", 255))
    symlog_min = float(encoder_cfg.get("symlog_min", -20.0))
    symlog_max = float(encoder_cfg.get("symlog_max", 20.0))
    bin_centers = np.linspace(symlog_min, symlog_max, num_reward_bins, dtype=np.float32)

    mu = np.load(os.path.join(model_path, "mu.npy"))
    std = np.load(os.path.join(model_path, "std.npy"))
    ckpt = torch.load(
        os.path.join(model_path, "dynamics.pth"), map_location=torch.device("cpu")
    )
    ckpt = {k: v.cpu().numpy() for (k, v) in ckpt.items()}
    elites = ckpt["elites"]
    scaler = (np.array(mu), np.array(std))
    scaler = jax.device_put(scaler)
    reward_scaler = np.stack(reward_scaler, axis=0)
    reward_scaler = jax.device_put(reward_scaler)

    num_models = int(ckpt["backbones.0.weight"].shape[0])
    num_layers = sum(
        1 for k in ckpt if k.startswith("backbones.") and k.endswith(".weight")
    )
    # Infer MLP width/depth from the checkpoint (search may use [200,200] or [200]x4).
    # EnsembleLinear weights are (n_ensemble, in_dim, out_dim) — use out_dim.
    hidden_dims = tuple(
        int(ckpt[f"backbones.{i}.weight"].shape[2]) for i in range(num_layers)
    )
    model_def = EnsembleWorldModel(
        num_models,
        len(elites),
        hidden_dims,
        obs_dim,
        action_dim,
        reward_mode=reward_mode,
        num_reward_bins=num_reward_bins,
        dropout_rate=None,
    )
    model_def = EnsembleDynamicModel(
        model_def,
        tuple(int(x) for x in elites),
        termination_fn,
        output_all=output_all,
        clip_extremes=clip_extremes,
        reward_mode=reward_mode,
        bin_centers=jax.device_put(bin_centers),
    )

    model_key = PRNGKey(42)
    model = Model.create(
        model_def, inputs=[model_key, model_key, observations, actions], tx=None
    )

    ckpt_jax = {}
    for i in range(num_layers):
        ckpt_jax[f"layers_{i}"] = {
            "kernel": ckpt[f"backbones.{i}.weight"],
            "bias": ckpt[f"backbones.{i}.bias"],
        }
    ckpt_jax["last_layer"] = {
        "kernel": ckpt["output_layer.weight"],
        "bias": ckpt["output_layer.bias"],
    }
    if reward_mode == "twohot":
        ckpt_jax["reward_head"] = {
            "kernel": ckpt["reward_head.weight"],
            "bias": ckpt["reward_head.bias"],
        }
    ckpt_jax["min_logvar"] = ckpt["min_logvar"]
    ckpt_jax["max_logvar"] = ckpt["max_logvar"]
    ckpt_jaxs = {"model": ckpt_jax}
    ckpt_jaxs["scaler"] = jnp.concatenate(scaler, axis=0)
    ckpt_jaxs["reward_scaler"] = jnp.stack(reward_scaler, axis=0)
    ckpt_jaxs["elites"] = elites
    ckpt_jaxs = jax.tree_util.tree_map(lambda x: jax.device_put(x), ckpt_jaxs)
    model = model.replace(params=ckpt_jaxs)
    return model, scaler
