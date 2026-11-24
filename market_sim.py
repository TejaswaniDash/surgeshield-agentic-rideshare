"""Zone-level two-sided marketplace simulator (15-minute steps).

Shared by RL training (gym env) and the live demo (monitoring loop), so the
policy is evaluated in the same dynamics it was trained in.

Dynamics per step
  1. idle drivers reposition: organic drift toward high demand/supply zones
     + incentive-driven moves (logit choice over bonus vs. travel distance)
  2. incentives pull extra drivers online
  3. Poisson demand arrives; served = min(demand, idle supply); unmet demand abandons
  4. served drivers are busy 1-2 steps then reappear at trip destinations (gravity model)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from surgeshield.geo import ZONES, distance_matrix

DIST = np.array(distance_matrix())
N = len(ZONES)
AIRPORT = np.array([z.venue == "airport" for z in ZONES])
_attract = np.where(AIRPORT, 3.0, 1.0)
P_DEST = _attract[None, :] * np.exp(-DIST / 6.0)
np.fill_diagonal(P_DEST, P_DEST.diagonal() * 0.3)
P_DEST /= P_DEST.sum(1, keepdims=True)
REACH = (DIST <= 3.0) | (AIRPORT[None, :] & (DIST <= 25))  # where an idle driver can reposition to in one step


def wait_minutes(demand: np.ndarray, supply: np.ndarray) -> np.ndarray:
    ratio = demand / np.maximum(supply, 1.0)
    return 3.2 + 3.2 * np.maximum(ratio - 1.0, 0) ** 1.1


@dataclass
class StepResult:
    demand: np.ndarray
    served: np.ndarray
    unmet: np.ndarray
    idle_before: np.ndarray
    wait_min: np.ndarray
    incentive_cost: float
    reward: float


@dataclass
class MarketSim:
    bonus_levels: tuple[float, ...] = (0.0, 2.0, 5.0, 8.0)
    margin: float = 6.0
    unmet_penalty: float = 5.0
    seed: int | None = None
    idle: np.ndarray = field(default_factory=lambda: np.zeros(N))
    busy: list[np.ndarray] = field(default_factory=lambda: [np.zeros(N), np.zeros(N)])
    bonus: np.ndarray = field(default_factory=lambda: np.zeros(N))

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self, expected_demand: np.ndarray, supply_factor: float = 1.25) -> None:
        self.idle = np.maximum(expected_demand * supply_factor, 1.0)
        self.busy = [expected_demand * 0.6, expected_demand * 0.3]
        self.bonus = np.zeros(N)

    def _choice(self, expected_demand: np.ndarray, bonus: np.ndarray) -> np.ndarray:
        pressure = np.clip((expected_demand - self.idle) / np.maximum(self.idle, 1), -1, 3)
        util = 0.6 * bonus[None, :] + 0.6 * pressure[None, :] - 0.8 * DIST
        util = np.where(REACH, util, -np.inf)
        np.fill_diagonal(util, 0.6 * bonus + 0.6 * np.maximum(pressure, 0) + 1.5)  # staying is sticky
        util -= util.max(1, keepdims=True)
        p = np.exp(util)
        return p / p.sum(1, keepdims=True)

    def _reposition(self, expected_demand: np.ndarray, bonus: np.ndarray) -> np.ndarray:
        """Move idle drivers; return incentive-*induced* arrivals per zone (what we pay relocation bonuses on)."""
        movable = self.idle * 0.35
        p_org = self._choice(expected_demand, np.zeros(N))
        p_inc = self._choice(expected_demand, bonus) if bonus.any() else p_org
        flows = movable[:, None] * p_inc
        induced = np.clip(movable[:, None] * (p_inc - p_org), 0, None)
        np.fill_diagonal(induced, 0)
        self.idle = self.idle - movable + flows.sum(0)
        return induced.sum(0)

    def step(self, expected_demand: np.ndarray, bonus_idx: np.ndarray | None = None,
             demand_override: np.ndarray | None = None) -> StepResult:
        """bonus = one-off relocation incentive ($ per driver who moves into / logs on in the zone)."""
        bonus = np.zeros(N) if bonus_idx is None else np.asarray(self.bonus_levels)[np.asarray(bonus_idx, int)]
        self.bonus = bonus
        induced = self._reposition(expected_demand, bonus)
        self.idle += self.busy.pop(0)                         # drivers finishing trips
        self.busy.append(np.zeros(N))
        # airports draw on a large outside pool (FHV holding lots, drivers in Queens/Brooklyn)
        organic_on = self.rng.poisson(0.07 * expected_demand + AIRPORT * 0.25 * np.maximum(expected_demand - self.idle, 0))
        induced_on = self.rng.poisson(0.45 * bonus * np.where(AIRPORT, 4.0, 1.0))  # incentive pulls drivers from outside / offline
        self.idle += organic_on + induced_on
        induced = induced + induced_on
        self.idle *= 0.95                                     # some log off
        demand = demand_override if demand_override is not None else self.rng.poisson(np.maximum(expected_demand, 0))
        idle_before = self.idle.copy()
        served = np.minimum(demand, np.floor(self.idle))
        unmet = demand - served
        self.idle -= served
        # served drivers become busy and reappear at destinations in 1-2 steps
        dest = served @ P_DEST
        self.busy[0] += 0.55 * dest
        self.busy[1] += 0.45 * dest
        self.last_induced = induced
        cost = float((bonus * induced).sum())
        reward = float(self.margin * served.sum() - cost - self.unmet_penalty * unmet.sum())
        return StepResult(demand, served, unmet, idle_before, wait_minutes(demand, idle_before), cost, reward)
