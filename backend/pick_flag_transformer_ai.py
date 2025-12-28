import argparse
import asyncio
import os
from typing import Any, Dict

try:
    from ctf_ai.policies import MacroParams, TransformerMacroPolicy  # type: ignore
    from lib.game_engine import run_game_server  # type: ignore
except Exception:  # pragma: no cover
    from backend.ctf_ai.policies import MacroParams, TransformerMacroPolicy  # type: ignore
    from backend.lib.game_engine import run_game_server  # type: ignore


AI: TransformerMacroPolicy


def start_game(req: Dict[str, Any]) -> None:
    AI.start(req)


def plan_next_actions(req: Dict[str, Any]) -> Dict[str, str]:
    return AI.act(req)


def game_over(_req: Dict[str, Any]) -> None:
    print("Game Over!")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int)
    parser.add_argument(
        "--model",
        default=os.environ.get("CTF_TF_MODEL", ""),
        help="Path to a torch model state_dict (.pt). If missing or torch not installed, falls back to heuristics.",
    )
    parser.add_argument(
        "--params",
        default=os.environ.get("CTF_TF_PARAMS", ""),
        help="Optional JSON with macro-role biases (for SA-GA tuning).",
    )
    args = parser.parse_args()

    params = MacroParams.from_json(args.params) if args.params else None
    global AI
    AI = TransformerMacroPolicy(model_path=args.model, params=params)

    print(f"AI backend running on port {args.port} ...")
    if args.model:
        print(f"Transformer model: {args.model}")
    else:
        print("Transformer model: (none) -> heuristic fallback")

    await run_game_server(int(args.port), start_game, plan_next_actions, game_over)


if __name__ == "__main__":
    asyncio.run(main())
