from typing import Dict, List, Optional

import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import pickle as pkl
import gymnasium as gym
import numpy as np
import copy
import time
import cv2
from tqdm import tqdm
from functools import partial

from common import PRNGKey, Model


def _reset_obs(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def _step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated or truncated, info
    obs, reward, done, info = result
    return obs, reward, done, info


def _try_render(env) -> Optional[np.ndarray]:
    """Return an RGB frame if the env supports rendering, else None."""
    try:
        frame = env.render()
    except Exception:
        return None
    if frame is None:
        return None
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        return None
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def _write_video(path: str, frames: List[np.ndarray], fps: int = 30) -> None:
    if not frames:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved eval video: {path} ({len(frames)} frames)")


@jax.jit
def step_imagine(
    key: PRNGKey,
    model_eval: Model,
    obs: jnp.ndarray,
    action: jnp.ndarray,
    states: jnp.ndarray,
):
    is_first = jnp.ones(obs.shape[0])
    next_states = model_eval(key, obs, action, is_first, states)
    return states


@partial(jax.jit, static_argnames=["action_dim", "state_dim"])
def step_imagine_first(
    key: PRNGKey, model_eval: Model, obs: jnp.ndarray, action_dim: int, state_dim: int
):
    batch_size = obs.shape[0]
    action = jnp.zeros((batch_size, action_dim))
    is_first = jnp.zeros(batch_size)
    states = jnp.zeros((batch_size, state_dim))
    next_states = model_eval(key, obs, action, is_first, states)
    return states


def evaluate(
    seed: int,
    agent: nn.Module,
    envs: List[gym.Env],
    video_path: str,
    step: int,
    model_eval=None,
    debug=False,
    record_video: bool = False,
    video_fps: int = 30,
) -> Dict[str, float]:
    """Roll out the agent in real envs; optionally save an mp4 of env 0.

    Video recording mirrors Gymnasium ``RecordVideo`` used in the PPO MountainCar
    expert script: ``render_mode='rgb_array'`` on the first eval env, frames
    captured each step, written under ``video_path``.
    """
    stats = {"return": [], "length": []}
    states, actions, rewards, observations = [], [], [], []

    _observations, dones = [], []
    num_episodes = len(envs)
    s = time.time()
    key = jax.device_put(PRNGKey(seed))
    video_frames: List[np.ndarray] = []
    record_video = bool(record_video and video_path)
    for env in envs:
        _observations.append(_reset_obs(env))
        observations.append([])
        dones.append(False)
        states.append([])
        actions.append([])
        rewards.append([])
    if record_video and envs:
        frame = _try_render(envs[0])
        if frame is not None:
            video_frames.append(frame)
    # print(observations)
    _observations = np.array(_observations)

    dones = np.array(dones, dtype=bool)
    for j in tqdm(range(10000)):
        if np.all(dones):
            break
        if model_eval is not None:
            key, rng = jax.random.split(key)
            if j == 0:
                action_dim, state_dim = env.action_space.shape[-1], 32 * 32 + 200
                _states = step_imagine_first(
                    key, model_eval, _observations, action_dim, state_dim
                )
            else:
                _states = step_imagine(
                    key, model_eval, _observations, _actions, jax.device_put(_states)
                )
            _states = jax.device_get(_states)
        else:
            _states = _observations
        _actions = agent.sample_actions(key, _states, temperature=0.0)
        _actions = np.array(_actions)
        for i in range(len(envs)):
            if dones[i]:
                continue
            observations[i].append(np.copy(_observations[i]))
            states[i].append(np.copy(_states[i]))
            actions[i].append(np.copy(_actions[i]))
            obs, reward, done, info = _step_env(envs[i], _actions[i])
            _observations[i] = obs
            rewards[i].append(reward)
            if record_video and i == 0:
                frame = _try_render(envs[i])
                if frame is not None:
                    video_frames.append(frame)
            if done:
                dones[i] = True
                stats["return"].append(info["episode"]["return"])
                stats["length"].append(info["episode"]["length"])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    observations = np.concatenate(observations, axis=0)
    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)

    if record_video:
        out_file = os.path.join(video_path, f"eval-{step}.mp4")
        _write_video(out_file, video_frames, fps=video_fps)
        if video_frames:
            stats["video_file"] = out_file

    if debug:
        _states, _actions = jax.device_put(states), jax.device_put(actions)
        q_values = agent.critic(_states, _actions)
        q_values = jax.device_get(q_values)
        trajectory = (observations, states, actions, rewards)
        print("Saving to:", video_path, step)
        np.save(os.path.join(video_path, f"q_values_{step}.npz"), q_values)
        with open(os.path.join(video_path, f"traj_{step}.pkl"), "wb") as F:
            pkl.dump(trajectory, F)
    return stats
