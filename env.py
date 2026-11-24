"""Gymnasium environment: choose a per-zone driver incentive level every 15 minutes.

Observation (5*N + 2): per zone [expected demand now, forecast next step, idle supply,
                       forecast gap next step, previous bonus] + [time, budget used]
Action: MultiDiscrete([len(bonus_levels)] * N) -> relocation bonus ($ per driver who moves into the zone)
Reward: (margin * served - incentive spend - penalty * unmet demand) / 100
The lookahead lets the policy *pre-position* supply before a forecast shock.
Episodes randomise start time and inject a random demand shock so the policy
learns *when* and *where* incentives pay for themselves.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from surgeshield.data.synthetic import BASE_PER_HOUR, HOURLY, MIDTOWN, WEEKEND_HOURLY
from surgeshield.geo import ZONES
from surgeshield.ml.rl.market_sim import N, MarketSim

BASE = np.array([BASE_PER_HOUR[z.venue] * (1.3 if z.id in MIDTOWN else 1.0) for z in ZONES]) / 4.0
SCALE = 20.0


def normal_demand(slot: int, weekend: bool) -> np.ndarray:
    prof = WEEKEND_HOURLY if weekend else HOURLY
    return prof[(slot // 4) % 24] * BASE


def build_obs(expected: np.ndarray, expected_next: np.ndarray, idle: np.ndarray, prev_bonus: np.ndarray,
              t_frac: float, budget_frac: float) -> np.ndarray:
    return np.concatenate([expected / SCALE, expected_next / SCALE, idle / SCALE, (expected_next - idle) / SCALE,
                           prev_bonus / 8.0, [t_frac, budget_frac]]).astype(np.float32)


class RepositioningEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = 16, bonus_levels=(0, 2, 5, 8), budget_usd: float = 3000.0,
                 margin: float = 6.0, unmet_penalty: float = 5.0, seed: int | None = None, residual: bool = False):
        super().__init__()
        self.T, self.budget, self.residual = episode_steps, budget_usd, residual
        self.n_levels = len(bonus_levels)
        self.sim = MarketSim(tuple(float(b) for b in bonus_levels), margin, unmet_penalty, seed)
        # residual RL: the agent outputs {-1, 0, +1} adjustments to the ops heuristic per zone
        self.action_space = spaces.MultiDiscrete([3 if residual else self.n_levels] * N)
        self.observation_space = spaces.Box(-10, 10, shape=(5 * N + 2,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    def _expected(self, t: int) -> np.ndarray:
        slot = (self.start_slot + t) % 96
        d = normal_demand(slot, self.weekend)
        if self.shock_start <= t < self.shock_start + self.shock_len:
            d = d.copy()
            d[self.shock_zones] *= self.shock_mult
        return d

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        r = self._rng
        self.start_slot = int(r.integers(0, 96))
        self.weekend = bool(r.random() < 0.3)
        self.shock_zones = r.choice(N, size=int(r.integers(1, 4)), replace=False)
        self.shock_mult = float(r.uniform(1.0, 5.0)) if r.random() < 0.8 else 1.0
        self.shock_start = int(r.integers(0, self.T - 2))
        self.shock_len = int(r.integers(2, 6))
        if options:
            for k, v in options.items():
                setattr(self, k, v)
        self.t, self.spent = 0, 0.0
        self.sim.reset(self._expected(0))
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        self._last_obs = build_obs(self._expected(self.t), self._expected(self.t + 1), self.sim.idle, self.sim.bonus,
                         self.t / self.T, self.spent / self.budget)
        return self._last_obs

    def step(self, action):
        action = np.asarray(action)
        if self.residual:
            action = residual_to_levels(self._last_obs, action, self.n_levels)
        if self.spent >= self.budget:
            action = np.zeros_like(action)
        res = self.sim.step(self._expected(self.t), action)
        self.spent += res.incentive_cost
        overspend = max(0.0, self.spent - self.budget)
        reward = (res.reward - 1.0 * overspend) / 100.0
        self.t += 1
        info = {"served": res.served.sum(), "unmet": res.unmet.sum(), "cost": res.incentive_cost,
                "wait": float(np.average(res.wait_min, weights=np.maximum(res.demand, 1e-6)))}
        return self._obs(), reward, self.t >= self.T, False, info


def greedy_policy(obs: np.ndarray) -> np.ndarray:
    """Heuristic baseline ops teams use: incentivise zones whose demand exceeds idle supply."""
    gap = obs[3 * N:4 * N] * SCALE
    return np.digitize(gap, [4, 10, 18])


def residual_to_levels(obs: np.ndarray, delta: np.ndarray, n_levels: int = 4) -> np.ndarray:
    return np.clip(greedy_policy(obs) + np.asarray(delta, int) - 1, 0, n_levels - 1)


def evaluate_policy(policy_fn, episodes: int = 200, seed: int = 123, **env_kw) -> dict:
    env = RepositioningEnv(seed=seed, **env_kw)
    R, served, unmet, cost, wait = [], [], [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, ep_r = False, 0.0
        while not done:
            obs, r, done, _, info = env.step(policy_fn(obs))
            ep_r += r
            served.append(info["served"]); unmet.append(info["unmet"]); cost.append(info["cost"]); wait.append(info["wait"])
        R.append(ep_r)
    tot = np.sum(served) + np.sum(unmet)
    return {"mean_return": float(np.mean(R)), "fulfillment_rate": float(np.sum(served) / tot),
            "incentive_usd_per_episode": float(np.sum(cost) / episodes), "mean_wait_min": float(np.mean(wait))}
