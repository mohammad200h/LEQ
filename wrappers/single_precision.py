import copy

import numpy as np


def _is_dict_space(space) -> bool:
    return hasattr(space, "spaces")


def _is_box_space(space) -> bool:
    return (
        hasattr(space, "low")
        and hasattr(space, "high")
        and hasattr(space, "shape")
        and not _is_dict_space(space)
    )


class SinglePrecision:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

        if _is_box_space(env.observation_space):
            obs_space = env.observation_space
            # Keep the same space type as the wrapped env when possible.
            self.observation_space = type(obs_space)(
                obs_space.low, obs_space.high, obs_space.shape, dtype=np.float32
            )
        elif _is_dict_space(env.observation_space):
            obs_spaces = copy.copy(env.observation_space.spaces)
            for k, v in obs_spaces.items():
                obs_spaces[k] = type(v)(v.low, v.high, v.shape, dtype=np.float32)
            self.observation_space = type(env.observation_space)(obs_spaces)
        else:
            raise NotImplementedError

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if isinstance(observation, np.ndarray):
            return observation.astype(np.float32)
        elif isinstance(observation, dict):
            observation = copy.copy(observation)
            for k, v in observation.items():
                observation[k] = v.astype(np.float32)
            return observation
        return observation

    def reset(self, **kwargs):
        try:
            result = self.env.reset(**kwargs)
        except TypeError:
            result = self.env.reset()
        if isinstance(result, tuple):
            obs, info = result
            return self.observation(obs), info
        return self.observation(result), {}

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            return (
                self.observation(observation),
                reward,
                terminated,
                truncated,
                info,
            )
        observation, reward, done, info = result
        return self.observation(observation), reward, done, info

    def seed(self, seed: int):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return self.reset(seed=seed)
