"""PPO repositioning policy (stable-baselines3) with a heuristic fallback."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from surgeshield.ml.rl.env import RepositioningEnv, evaluate_policy, greedy_policy, residual_to_levels

log = logging.getLogger(__name__)


def train_ppo(timesteps: int, out_path: Path, seed: int = 0, **env_kw):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    venv = make_vec_env(lambda: RepositioningEnv(seed=None, residual=True, **env_kw), n_envs=8, seed=seed)
    model = PPO("MlpPolicy", venv, n_steps=256, batch_size=512, n_epochs=8, gamma=0.95, gae_lambda=0.95,
                learning_rate=3e-4, ent_coef=0.002, policy_kwargs={"net_arch": [128, 128]}, seed=seed, verbose=0)
    model.learn(total_timesteps=timesteps)
    model.save(str(out_path))
    ppo_fn = lambda o: residual_to_levels(o, model.predict(o, deterministic=True)[0])  # noqa: E731
    metrics = {
        "no_incentives": evaluate_policy(lambda o: np.zeros(len(o[:-2]) // 5, int), **env_kw),
        "greedy_heuristic": evaluate_policy(greedy_policy, **env_kw),
        "ppo_residual": evaluate_policy(ppo_fn, **env_kw),
    }
    return model, metrics


class RepositioningPolicy:
    def __init__(self, model_path: Path | None = None):
        self.model = None
        if model_path and Path(model_path).exists():
            try:
                from stable_baselines3 import PPO
                self.model = PPO.load(str(model_path), device="cpu")
            except Exception as e:  # pragma: no cover
                log.warning("could not load PPO policy (%s); using heuristic", e)

    @property
    def kind(self) -> str:
        return "ppo_residual" if self.model is not None else "greedy_heuristic"

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.model is not None:
            return residual_to_levels(obs, self.model.predict(obs, deterministic=True)[0])
        return greedy_policy(obs)
