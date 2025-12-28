from __future__ import annotations

"""
Minimal training scaffold: population + opponent-pool + annealed GA over macro-role biases.

Why this is the "simple feasible" option:
- The Transformer stays small and stable (macro-role prediction only).
- The GA only evolves a tiny vector of biases (5 floats) with simulated annealing.
- Adversarial aspect comes from evaluating each individual against a growing opponent pool.

This script intentionally does NOT try to be a full RL framework. Treat it as a starting point.
"""

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .constants import Roles
from .policies import MacroParams
from .sim_env import CTFSim
from .transformer_model import TransformerConfig, build_model, require_torch


@dataclass(slots=True)
class Individual:
    params: MacroParams
    fitness: float = 0.0


def _play_episode(seed: int, left_params: MacroParams, right_params: MacroParams) -> Tuple[int, int]:
    sim = CTFSim(seed=seed)
    sim.reset()

    # For GA evaluation we only use heuristics + macro params; the Transformer is trained separately.
    # This keeps evaluation fast and stable.
    for _step in range(400):  # hard cap
        if sim.done:
            break

        # Heuristic macro policy: return if carrying; rescue if prisoners; else steal nearest.
        l_actions = _macro_heuristic(sim, team="L", params=left_params)
        r_actions = _macro_heuristic(sim, team="R", params=right_params)
        sim.step(l_actions, r_actions)

    res = sim.result()
    return res.l_score, res.r_score


def _macro_heuristic(sim: CTFSim, *, team: str, params: MacroParams) -> Dict[str, str]:
    assert sim.map is not None
    my_players = [p for p in sim.players if p.team == team]
    opp_players = [p for p in sim.players if p.team != team]
    my_target = sim._static(team).target_tiles[0]
    my_prison = sim._static(team).prison_tiles[0]

    num_prisoners = sum(1 for p in my_players if p.in_prison)
    want_rescue = (num_prisoners > 0) and (params.rescue_bias >= 0.0)

    actions: Dict[str, str] = {}
    for p in my_players:
        if p.in_prison:
            continue
        if p.has_flag:
            dest = my_target
        elif want_rescue:
            dest = my_prison
        else:
            f = sim.nearest_enemy_flag(team, p.pos)
            dest = f.pos if f else my_target

        extra = []
        if params.avoid_opponents:
            # Avoid opponents only when in enemy half.
            middle = sim.width / 2.0
            x, _y = p.pos
            my_left = team == "L"
            in_my_half = (float(x) < middle) if my_left else (float(x) >= middle)
            if not in_my_half:
                extra = [o.pos for o in opp_players if not o.in_prison]

        actions[p.name] = sim.best_step_towards(p.pos, dest, extra_blocked=extra)
    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--games-per-individual", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="backend/ctf_ai_runs")
    args = parser.parse_args()

    pop_n = int(args.population)
    gens = int(args.generations)
    games = int(args.games_per_individual)
    rng = random.Random(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure torch is available for the Transformer part (even though GA eval doesn't require it).
    try:
        require_torch()
    except Exception as exc:
        raise SystemExit(
            "Missing PyTorch. Install it, then rerun.\n"
            "Example: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    # Initialize population with random biases.
    population: List[Individual] = []
    for _ in range(pop_n):
        population.append(
            Individual(
                params=MacroParams(
                    steal_bias=rng.uniform(-0.2, 0.2),
                    return_bias=rng.uniform(-0.2, 0.2),
                    rescue_bias=rng.uniform(-0.2, 0.2),
                    chase_bias=rng.uniform(-0.2, 0.2),
                    defend_bias=rng.uniform(-0.2, 0.2),
                    avoid_opponents=True,
                )
            )
        )

    # Opponent pool: start with a neutral baseline (zeros).
    opponent_pool: List[MacroParams] = [MacroParams()]

    for gen in range(gens):
        # Annealing temperature (high -> wide mutation, low -> fine-tune)
        t = max(0.05, math.exp(-gen / max(1.0, gens / 3.0)))
        print(f"Generation {gen+1}/{gens} (T={t:.3f})")

        # Evaluate population against sampled opponents
        for ind in population:
            score = 0.0
            for g in range(games):
                opp = rng.choice(opponent_pool)
                seed = int(args.seed) + gen * 10_000 + g * 100 + rng.randint(0, 99)
                l_score, r_score = _play_episode(seed, ind.params, opp)
                score += float(l_score - r_score)
            ind.fitness = score / float(games)

        population.sort(key=lambda it: it.fitness, reverse=True)
        best = population[0]
        print(f"  best fitness={best.fitness:.3f} params={best.params}")

        # Add best to opponent pool (adversarial pressure)
        opponent_pool.append(best.params)
        opponent_pool = opponent_pool[-32:]

        # Save checkpoint params
        with open(out_dir / f"gen_{gen:04d}_best_params.json", "w", encoding="utf-8") as handle:
            json.dump(best.params.__dict__, handle, indent=2, sort_keys=True)

        # Reproduce (elitism + SA-GA over small param vector)
        elites = population[: max(2, pop_n // 6)]
        next_pop: List[Individual] = [Individual(params=e.params, fitness=0.0) for e in elites]

        def mutate(p: MacroParams) -> MacroParams:
            def n(x: float) -> float:
                return float(x + rng.gauss(0.0, 0.15 * t))

            return MacroParams(
                steal_bias=n(p.steal_bias),
                return_bias=n(p.return_bias),
                rescue_bias=n(p.rescue_bias),
                chase_bias=n(p.chase_bias),
                defend_bias=n(p.defend_bias),
                avoid_opponents=p.avoid_opponents,
            )

        while len(next_pop) < pop_n:
            a = rng.choice(elites).params
            b = rng.choice(elites).params
            child = MacroParams(
                steal_bias=(a.steal_bias + b.steal_bias) / 2.0,
                return_bias=(a.return_bias + b.return_bias) / 2.0,
                rescue_bias=(a.rescue_bias + b.rescue_bias) / 2.0,
                chase_bias=(a.chase_bias + b.chase_bias) / 2.0,
                defend_bias=(a.defend_bias + b.defend_bias) / 2.0,
                avoid_opponents=True,
            )
            child = mutate(child)
            next_pop.append(Individual(params=child))

        population = next_pop

    print(f"Done. Best params written to: {out_dir}")


if __name__ == "__main__":
    main()

