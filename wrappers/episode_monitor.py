import time

import numpy as np

from wrappers.common import TimeStep


class EpisodeMonitor:
    """A class that computes episode returns and lengths."""

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._reset_stats()
        self.total_timesteps = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray) -> TimeStep:
        result = self.env.step(action)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
        else:
            observation, reward, done, info = result
            terminated, truncated = bool(done), False

        done = terminated or truncated
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self.env, "get_normalized_score"):
                info["episode"]["return"] = (
                    self.env.get_normalized_score(info["episode"]["return"]) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, **kwargs):
        self._reset_stats()
        if seed is not None:
            kwargs["seed"] = seed
        try:
            result = self.env.reset(**kwargs)
        except TypeError:
            # Legacy gym / NeoRL reset without kwargs.
            result = self.env.reset()
        if isinstance(result, tuple):
            return result
        return result, {}

    def seed(self, seed: int):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return self.reset(seed=seed)
