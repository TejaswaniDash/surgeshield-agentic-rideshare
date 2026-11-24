"""Repositioning agent - reinforcement learning (residual PPO) decides per-zone relocation bonuses;
a model-based rollout in the market simulator estimates impact and sizes the budget commitment."""
from __future__ import annotations

import re

import numpy as np

from surgeshield.a2a.protocol import AgentSkill
from surgeshield.agents.base import BaseAgent
from surgeshield.config import get_settings
from surgeshield.geo import ZONE_IDS, ZONE_INDEX, zone_name
from surgeshield.llm.client import get_llm
from surgeshield.llm.prompts import RATIONALE_SYSTEM
from surgeshield.ml.rl.env import build_obs
from surgeshield.ml.rl.market_sim import MarketSim
from surgeshield.schemas import Assessment, Disruption, Plan, ZoneAction
from surgeshield.world import get_world


class RepositioningAgent(BaseAgent):
    name = "repositioner"
    description = "Plans driver relocation incentives with a PPO policy and simulates their impact."
    skills = [AgentSkill(id="plan", name="Plan repositioning", tags=["reinforcement-learning", "simulation"],
                         description="Return per-zone relocation bonuses, expected impact and a budget commitment. "
                                     "Accepts feedback/constraints from compliance for re-planning.")]

    def skill_plan(self, disruption: dict, assessment: dict, feedback: list[str] | None = None,
                   version: int = 1, cycle: int = 0) -> dict:
        cfg = get_settings()
        w = get_world()
        d, a = Disruption.model_validate(disruption), Assessment.model_validate(assessment)
        feedback = feedback or []
        levels_usd = np.array(cfg.rl.bonus_levels_usd, float)
        base = w.baseline_demand(d.detected_at)
        live = w.live.get(d.incident_id)
        exp_now, exp_next = base.copy(), base.copy()
        idle = live.sim.idle.copy() if live else base * 1.25
        for f in a.forecasts:
            i = ZONE_INDEX[f.zone_id]
            exp_now[i], exp_next[i] = f.expected_requests[0], f.expected_requests[1]
            idle[i] = f.idle_supply
        obs = build_obs(exp_now, exp_next, idle, np.zeros(len(ZONE_IDS)), 0.0, 0.0)
        lv = w.rl_policy.act(obs)
        mask = np.zeros(len(ZONE_IDS), bool)
        mask[[ZONE_INDEX[f.zone_id] for f in a.forecasts]] = True
        lv = np.where(mask, lv, 0)

        # --- constraints coming back from the compliance loop (reflection / re-planning)
        budget_cap = cfg.policy.incentive_hard_budget_usd
        wav_zones: list[int] = []
        for fb in feedback:
            if m := re.search(r"budget to ≤ \$([\d,]+)", fb):
                budget_cap = min(budget_cap, float(m.group(1).replace(",", "")))
            if m := re.search(r"WAV bonus in zones \[([\d, ]+)\]", fb):
                wav_zones += [int(x) for x in m.group(1).split(",") if int(x) not in wav_zones]

        def rollout(levels: np.ndarray) -> dict:
            sim = MarketSim(tuple(levels_usd), seed=7)
            sim.reset(base)
            sim.idle = idle.copy()
            fc = {f.zone_id: f.expected_requests for f in a.forecasts}
            served = unmet = cost = 0.0
            waits = []
            induced = np.zeros(len(ZONE_IDS))
            zi = mask
            for h in range(4):
                e = base.copy()
                for z, seq in fc.items():
                    e[ZONE_INDEX[z]] = seq[h]
                r = sim.step(e, levels if h < 3 else np.zeros_like(levels), demand_override=np.round(e))
                served += r.served[zi].sum(); unmet += r.unmet[zi].sum(); cost += r.incentive_cost
                induced += sim.last_induced
                waits.append(float(np.average(r.wait_min[zi], weights=np.maximum(r.demand[zi], 1e-6))))
            return {"fulfillment": served / max(served + unmet, 1), "wait": float(np.mean(waits)), "cost": cost,
                    "induced": induced}

        res = rollout(lv)
        while res["cost"] * 1.25 > budget_cap and lv.max() > 0:  # shave the most expensive zone first
            lv[int(np.argmax(lv))] -= 1
            res = rollout(lv)
        noop = rollout(np.zeros_like(lv))
        budget = float(np.ceil(res["cost"] * 1.25 / 100) * 100)  # commitment incl. 25% headroom

        actions = []
        for f in a.forecasts:
            i = ZONE_INDEX[f.zone_id]
            actions.append(ZoneAction(zone_id=f.zone_id, zone_name=zone_name(f.zone_id),
                                      relocation_bonus_usd=float(levels_usd[lv[i]]),
                                      expected_drivers_in=round(float(res["induced"][i]), 1),
                                      wav_bonus_usd=10.0 if f.zone_id in wav_zones else 0.0))
        if wav_zones:
            budget += 200.0
        plan = Plan(plan_id=f"{d.incident_id}-c{cycle}-p{version}", version=version, actions=actions,
                    incentive_budget_usd=budget, surge_cap_applied=1.0, policy=w.rl_policy.kind,
                    expected_fulfillment=round(res["fulfillment"], 3), expected_wait_min=round(res["wait"], 1),
                    feedback_addressed=feedback)
        ctx = {"actions": [x.model_dump() for x in actions], "policy": plan.policy, "budget": budget,
               "expected_wait": plan.expected_wait_min, "feedback": feedback}
        prompt = (f"Plan: {plan.model_dump_json()}\nNo-action baseline: wait {noop['wait']:.1f} min, "
                  f"fulfillment {noop['fulfillment']:.0%}\nFeedback addressed: {feedback}")
        plan.rationale = get_llm().generate("plan_rationale", RATIONALE_SYSTEM, prompt, context=ctx).text
        return {"output": {"plan": plan.model_dump(),
                           "counterfactual": {"no_action_wait": round(noop["wait"], 1),
                                              "no_action_fulfillment": round(noop["fulfillment"], 3)}},
                "summary": f"plan v{version}: ${budget:,.0f}, wait {res['wait']:.1f} (vs {noop['wait']:.1f} no-action)"}

